import logging
from typing import Dict, Tuple, List, Any, Union
from IPython import embed
import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric, BLEU
from allennlp.nn.beam_search import BeamSearch


logger = logging.getLogger(__name__)

class CopyNetSeq2Seq(Model):

    def __init__(self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2SeqEncoder, attention: Attention, copy_token: str = "@COPY@", namespace: str = "tokens", target_embedding_dim: int = 50, decoder_output_dim = 100, max_decoding_steps: int=30, beam_size: int=4, initializer: InitializerApplicator = InitializerApplicator()):

        super().__init__(vocab)
        self._namespace = namespace
        self._src_start_index = self.vocab.get_token_index(START_SYMBOL, self._namespace)
        self._src_end_index = self.vocab.get_token_index(END_SYMBOL, self._namespace)
        # self._start_index = self.vocab.get_token_index(START_SYMBOL, self._namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._namespace)
        self._oov_index = self.vocab.get_token_index(self.vocab._oov_token, self._namespace)  
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._namespace)
        self._copy_index = self.vocab.add_token_to_namespace(copy_token, self._namespace)
        self._start_index = self.vocab.add_token_to_namespace(START_SYMBOL, self._namespace)

        self._bleu1 = BLEU(ngram_weights = (1,0,0,0),exclude_indices={self._pad_index, self._end_index, self._start_index})
        self._bleu2 = BLEU(ngram_weights = (0,1,0,0),exclude_indices={self._pad_index, self._end_index, self._start_index})
        self._bleu3 = BLEU(ngram_weights = (0,0,1,0),exclude_indices={self._pad_index, self._end_index, self._start_index})
        self._bleu4 = BLEU(ngram_weights = (0,0,0,1),exclude_indices={self._pad_index, self._end_index, self._start_index})
        self._bleu_all = BLEU(exclude_indices={self._pad_index, self._end_index, self._start_index})

        self._target_vocab_size = self.vocab.get_vocab_size(self._namespace)

        self._source_embedder = embedder
        self._target_embedder = Embedding(self._target_vocab_size, target_embedding_dim)

        self._encoder = encoder

        self.encoder_output_dim = self._encoder.get_output_dim()
        self.decoder_output_dim = decoder_output_dim

        self._final_encoder_projection_layer = Linear(self.encoder_output_dim, self.decoder_output_dim)

        self._decoder_cell = LSTMCell(target_embedding_dim + self.encoder_output_dim, self.decoder_output_dim)
        self._attention = attention

        self._output_generation_layer = Linear(target_embedding_dim + self.encoder_output_dim, self._target_vocab_size)
        # self._output_copying_layer = Linear(self.encoder_output_dim, self.decoder_output_dim)

        self._beam_search = BeamSearch(self._end_index, max_steps=max_decoding_steps, beam_size=beam_size)

        self._epoch_iter = 0

        initializer(self)


    def _decoder_step(self, last_predictions: torch.Tensor, source_mask, decoder_hidden, encoder_outputs, decoder_context) -> Dict[str, torch.Tensor]:
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_mask = source_mask
        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)
        # shape: (group_size, max_input_sequence_length)
        attentive_weights = self._attention(
                decoder_hidden, encoder_outputs, encoder_outputs_mask)
        # shape: (group_size, encoder_output_dim)
        attentive_read = util.weighted_sum(encoder_outputs, torch.softmax(attentive_weights, dim=1))
        # shape: (group_size, encoder_output_dim)
        # selective_read = util.weighted_sum(encoder_outputs, selective_weights)
        # shape: (group_size, target_embedding_dim + encoder_output_dim * 2)
        decoder_input = torch.cat((embedded_input, attentive_read), -1)
        # shape: (group_size, decoder_input_dim)
        # projected_decoder_input = self._input_projection_layer(decoder_input)

        decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_context))

        return decoder_hidden, decoder_context, attentive_weights, decoder_input


    def _gather_final_log_probs(self,generation_log_probs, copy_log_probs, state):

        _, source_length = state["source_to_target"].size()
        source_token_ids = state["source_tokens"][:,1:-1]
        modified_log_probs_list = [generation_log_probs]

        for i in range(source_length):
            copy_log_probs_slice = copy_log_probs[:, i]
            source_to_target_slice = state["source_to_target"][:, i]
            copy_log_probs_to_add_mask = (source_to_target_slice != self._oov_index).float()
            copy_log_probs_to_add = copy_log_probs_slice + (copy_log_probs_to_add_mask + 1e-45).log()
            copy_log_probs_to_add = copy_log_probs_to_add.unsqueeze(-1)
            selected_generation_log_probs = generation_log_probs.gather(1, source_to_target_slice.unsqueeze(-1))
            combined_scores = util.logsumexp(torch.cat((selected_generation_log_probs, copy_log_probs_to_add), dim=1))
            generation_log_probs.scatter_(-1, source_to_target_slice.unsqueeze(-1), combined_scores.unsqueeze(-1))
            if i < (source_length - 1):
                source_future_occurences = (source_token_ids[:, (i+1):] == source_token_ids[:, i].unsqueeze(-1)).float()
                future_copy_log_probs = copy_log_probs[:, (i+1):] + (source_future_occurences + 1e-45).log()
                combined = torch.cat((copy_log_probs_slice.unsqueeze(-1), future_copy_log_probs), dim=-1)
                copy_log_probs_slice = util.logsumexp(combined)
            if i > 0:
                source_previous_occurences = source_token_ids[:, 0:i] == source_token_ids[:, i].unsqueeze(-1)
                duplicate_mask = (source_previous_occurences.sum(dim=-1) == 0).float()
                copy_log_probs_slice = copy_log_probs_slice + (duplicate_mask + 1e-45).log()
            left_over_copy_log_probs = copy_log_probs_slice + (1.0 - copy_log_probs_to_add_mask + 1e-45).log()
            modified_log_probs_list.append(left_over_copy_log_probs.unsqueeze(-1))


        modified_log_probs = torch.cat(modified_log_probs_list, dim=-1)

        return modified_log_probs

    def take_search_step(self, last_predictions, state):

        group_size, source_length = state["source_to_target"].size()
        only_copied_mask = (last_predictions >= self._target_vocab_size).long()
        copy_input_choices = only_copied_mask.new_full((group_size,), fill_value=self._copy_index)
        # input_choices = last_predictions * (1 - only_copied_mask) + copy_input_choices * only_copied_mask
        input_choices = last_predictions

        state['decoder_hidden'],state['decoder_context'], attentive_weights, state['decoder_input'] = self._decoder_step(input_choices, state['source_mask'],state['decoder_hidden'],state['encoder_outputs'],state['decoder_context'])
        generation_scores = self._output_generation_layer(state['decoder_input'])
        # copy_scores = self._get_copy_scores(state['encoder_outputs'], state['decoder_hidden'])
        copy_scores = attentive_weights[:,1:-1]
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)

        copy_mask = state["source_mask"][:,1:-1].float()
        mask = torch.cat((generation_scores.new_full(generation_scores.size(), 1.0), copy_mask), dim=-1)

        log_probs = util.masked_log_softmax(all_scores, mask)

        generation_log_probs, copy_log_probs = log_probs.split([self._target_vocab_size, source_length], dim=-1)

        state["copy_log_probs"] = copy_log_probs

        final_log_probs = self._gather_final_log_probs(generation_log_probs, copy_log_probs, state)

        return final_log_probs, state


    def _gather_extended_gold_tokens(self,
                                     target_tokens: torch.Tensor,
                                     source_token_ids: torch.Tensor,
                                     target_token_ids: torch.Tensor) -> torch.LongTensor:
        """
        Modify the gold target tokens relative to the extended vocabulary.

        For gold targets that are OOV but were copied from the source, the OOV index
        will be changed to the index of the first occurence in the source sentence,
        offset by the size of the target vocabulary.

        Parameters
        ----------
        target_tokens : ``torch.Tensor``
            Shape: `(batch_size, target_sequence_length)`.
        source_token_ids : ``torch.Tensor``
            Shape: `(batch_size, trimmed_source_length)`.
        target_token_ids : ``torch.Tensor``
            Shape: `(batch_size, target_sequence_length)`.

        Returns
        -------
        torch.Tensor
            Modified `target_tokens` with OOV indices replaced by offset index
            of first match in source sentence.
        """
        batch_size, target_sequence_length = target_tokens.size()
        trimmed_source_length = source_token_ids.size(1)
        # Only change indices for tokens that were OOV in target vocab but copied from source.
        # shape: (batch_size, target_sequence_length)
        oov = (target_tokens == self._oov_index)
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        expanded_source_token_ids = source_token_ids\
            .unsqueeze(1)\
            .expand(batch_size, target_sequence_length, trimmed_source_length)
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        expanded_target_token_ids = target_token_ids\
            .unsqueeze(-1)\
            .expand(batch_size, target_sequence_length, trimmed_source_length)
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        matches = (expanded_source_token_ids == expanded_target_token_ids)
        # shape: (batch_size, target_sequence_length)
        copied = (matches.sum(-1) > 0)
        # shape: (batch_size, target_sequence_length)
        mask = (oov & copied).long()
        # shape: (batch_size, target_sequence_length)
        first_match = ((matches.cumsum(-1) == 1) * matches).argmax(-1)
        # shape: (batch_size, target_sequence_length)
        new_target_tokens = target_tokens * (1 - mask) + (first_match.long() + self._target_vocab_size) * mask
        return new_target_tokens



    @overrides
    def forward(self, source_tokens, target_tokens: Dict[str, torch.LongTensor] = None):

        '''
        Map the input tokens to embeddings. source_mask contains the map needed for padding
        '''
        # embed()

        embedded_input = self._source_embedder(source_tokens)
        source_mask = util.get_text_field_mask(source_tokens)

        batch_size = source_mask.size()[0]


        '''
        Pass the embedded_input to the encoder ((bi-)LSTM) and get the outputs
        '''

        encoder_outputs = self._encoder(embedded_input, source_mask)
        final_encoder_output = util.get_final_encoder_states(encoder_outputs, source_mask, self._encoder.is_bidirectional())

        '''
        Initialize decoder state
        '''

        source_to_target = source_tokens['tokens'][:,1:-1]

        decoder_hidden = torch.tanh(self._final_encoder_projection_layer(final_encoder_output))
        decoder_context = encoder_outputs.new_zeros(batch_size, self.decoder_output_dim)

        input_choices = source_tokens['tokens'][:,0].new_full((batch_size,), fill_value=self._start_index)

        output_dict = {}

        if(not self.training):
            # predictions = self._forward_beam_search(state)

            batch_size, source_length = source_mask.size()
            # trimmed_source_length = source_length - 2
            copy_log_probs = (decoder_hidden.new_zeros((batch_size, source_length - 2)) + 1e-45).log()
            start_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)
            state = {'source_to_target': source_to_target, 'source_mask': source_mask, 'decoder_hidden': decoder_hidden, 'encoder_outputs': encoder_outputs, 'decoder_context': decoder_context, 'source_tokens': source_tokens['tokens']}
            all_top_k_predictions, log_probabilities = self._beam_search.search(start_predictions, state, self.take_search_step)
            output_dict = {'predictions': all_top_k_predictions, 'predicted_log_probs': log_probabilities}
            if target_tokens:
                top_k_predictions = output_dict["predictions"]
                best_predictions = top_k_predictions[:, 0, :]
                best_predictions = ((1 - (best_predictions == self._end_index)).float()*best_predictions.float()).long()
                gold_tokens = self._gather_extended_gold_tokens(target_tokens['tokens'], source_tokens['tokens'][:,1:-1], target_tokens['tokens'])
                gold_tokens = ((1 - (gold_tokens == self._end_index)).float()*gold_tokens.float()).long()
                self._bleu1(best_predictions, gold_tokens[:,1:])
                self._bleu2(best_predictions, gold_tokens[:,1:])
                self._bleu3(best_predictions, gold_tokens[:,1:])
                self._bleu4(best_predictions, gold_tokens[:,1:])
                self._bleu_all(best_predictions, gold_tokens[:,1:])
                    # embed()
                    # print(self._tensor_metric.get_metric())
                    # embed()
            self._epoch_iter += 1
            #if(self._epoch_iter % 14 == 0):
            #     # embed()
            # if(self._epoch_iter % (self._test_size/batch_size) == 0):
               # print("BLEU1:", self._bleu1.get_metric())
               # print("BLEU2:", self._bleu2.get_metric())
               # print("BLEU3:", self._bleu3.get_metric())
               # print("BLEU4:", self._bleu4.get_metric())
               # print("BLEU ALL:", self._bleu_all.get_metric())


        target_sequence_length = target_tokens['tokens'].size()[1]

        num_decoding_steps = target_sequence_length - 1

        '''
        Continue here from self._forward_loss
        '''


        copy_input_choices = source_mask.new_full((batch_size,), fill_value=self._copy_index)
        copy_mask = source_mask[:,1:-1].float()
        target_to_source = source_tokens['tokens'][:,1:-1].new_zeros(copy_mask.size())

        generation_scores_mask = final_encoder_output.new_full((batch_size, self._target_vocab_size),
                                                                  fill_value=1.0)

        '''
        Compute the log likelihood loss by first calculating the prediction at each step, computing the ll loss at the step and adding them 
        '''

        step_log_likelihoods = []

        for timestep in range(num_decoding_steps):
            input_choices = target_tokens['tokens'][:, timestep]
            # embed()
            # target_to_source = source_tokens['tokens'] == target_tokens['tokens'][:, timestep].unsqueeze(-1)
            # copied = ((input_choices == self._oov_index) & (target_to_source.sum(-1) > 0)).long()
            if(timestep < num_decoding_steps - 1):
                # copied = (target_to_source.sum(-1) > 0).long()
                # copied = ((input_choices == self._oov_index) & (target_to_source.sum(-1) > 0)).long()
                # input_choices = input_choices * (1 - copied) + copy_input_choices * copied
                target_to_source = source_tokens['tokens'][:,1:-1] == target_tokens['tokens'][:, timestep+1].unsqueeze(-1)

            '''
            Decoder Step function
            '''

            # embedded_input = self._target_embedder(input_choices)

            # attentive_weights = self._attention(decoder_hidden, encoder_outputs, source_mask.float())
            # attentive_read = util.weighted_sum(encoder_outputs, attentive_weights)
            # decoder_input = torch.cat((embedded_input, attentive_read), -1)
            # projected_decoder_input = self._input_projection_layer(decoder_input)

            # decoder_hidden, decoder_context = self._decoder_cell(projected_decoder_input, (decoder_hidden, decoder_context))

            decoder_hidden, decoder_context, attentive_weights, decoder_input = self._decoder_step(input_choices, source_mask, decoder_hidden, encoder_outputs, decoder_context)

            '''
            Get the generation scores, which is the probability of each token for prediction at this step
            '''

            generation_scores = self._output_generation_layer(decoder_input)


            '''
            Get copy scores for each token in the source sentence, excluding the start and end tokens
            '''

            # trimmed_encoder_outputs = encoder_outputs[:, 1:-1]

            # copy_projection = self._output_copying_layer(encoder_outputs)

            # copy_projection = torch.tanh(copy_projection)
            # copy_scores = copy_projection.bmm(decoder_hidden.unsqueeze(-1)).squeeze(-1)

            # embed()

            # copy_scores = self._get_copy_scores(encoder_outputs, decoder_hidden)
            copy_scores = attentive_weights[:,1:-1]

            # step_target_tokens = target_tokens['tokens'][:, timestep + 1]


            # step_log_likelihood, selective_weights = self._get_ll_contrib(generation_scores, generation_scores_mask, copy_scores, step_target_tokens,target_to_source,copy_mask)

            '''
            We now compute the log likelihood using the generation scores and the copy scores
            '''
            

            _, target_size = generation_scores.size()

            mask = torch.cat((generation_scores_mask, copy_mask), dim=- 1)
            all_scores = torch.cat((generation_scores, copy_scores), dim=-1)

            log_probs = util.masked_log_softmax(all_scores, mask)
            copy_log_probs = log_probs[:, target_size:] + (target_to_source.float() + 1e-45).log()
            # embed()

            step_target_tokens = target_tokens['tokens'][:,timestep+1]

            # selective_weights = util.masked_softmax(log_probs[:, target_size:], target_to_source)
            gen_mask = ((step_target_tokens != self._oov_index) | (target_to_source.sum(-1) == 0)).float()
            log_gen_mask = (gen_mask + 1e-45).log().unsqueeze(-1)
            generation_log_probs = log_probs.gather(1, step_target_tokens.unsqueeze(1)) + log_gen_mask
            combined_gen_and_copy = torch.cat((generation_log_probs, copy_log_probs), dim=-1)
            step_log_likelihood = util.logsumexp(combined_gen_and_copy)


            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

            '''
            The input now changes
            '''



        log_likelihoods = torch.cat(step_log_likelihoods, 1)

        target_mask = util.get_text_field_mask(target_tokens)

        target_mask = target_mask[:,1:].float()
        log_likelihood = (log_likelihoods * target_mask).sum(dim=-1)
        loss = - log_likelihood.sum() / batch_size

        output_dict['loss'] = loss
        output_dict['metadata']= [{'source_tokens': source_tokens, 'target_tokens': target_tokens}]


        return output_dict


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Finalize predictions.

        After a beam search, the predicted indices correspond to tokens in the target vocabulary
        OR tokens in source sentence. Here we gather the actual tokens corresponding to
        the indices.
        """
        predicted_tokens = self._get_predicted_tokens(output_dict["predictions"],
                                                      output_dict["metadata"])
        output_dict["predicted_tokens"] = predicted_tokens
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            #all_metrics.update(self._bleu_all.get_metric(reset=reset))
            #all_metrics.update(self._bleu2.get_metric(reset=reset))
            #all_metrics.update(self._bleu3.get_metric(reset=reset))
            #all_metrics.update(self._bleu4.get_metric(reset=reset))
            all_metrics.update(self._bleu1.get_metric(reset=reset))
              # type: ignore
            # if self._token_based_metric is not None:
            #     all_metrics.update(self._token_based_metric.get_metric(reset=reset))  # type: ignore
        return all_metrics
