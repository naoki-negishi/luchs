from typing import Optional

import torch
from luchs.instances import ENC_labels, SNLIBatchInstance
from torch import nn
from torch.nn.functional import normalize
from torch.nn.init import kaiming_uniform_
from torch.nn.utils.rnn import pad_sequence
from transformers import (BertModel, BertPooler, RobertaConfig, RobertaModel,
                          RobertaTokenizer)

RoBERTaBase = "roberta-base"


class BaseCLSModel(nn.Module):
    def __init__(
        self,
        config: RobertaConfig,
        base_model: nn.Module,
        device: torch.device,
        # embed_dropout: float = 0.0,
        # layer_norm_eps: float = 1e-5,
    ) -> None:
        super(BaseCLSModel, self).__init__()
        self.config = config
        self.device = device

        # Embedding model
        self.tokenizer = RobertaTokenizer.from_pretrained(RoBERTaBase)
        self.encoder = base_model
        # self.layer_norm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        # self.embed_dropout_layer = nn.Dropout(self.embed_dropout)

        # Classifier
        self.embed_dim = config.hidden_size
        self.entail_classifier = nn.Linear(self.embed_dim, len(ENC_labels))

        # Loss function
        self.loss_f = nn.CrossEntropyLoss()

    def forward(self, batch_instance) -> torch.Tensor:
        """Calculate the loss"""
        entail_label_scores = self.prediction(batch_instance)

        assert self.loss_function
        loss = self.loss_function(batch_instance, entail_label_scores)

        return loss

    def loss_function(
        self,
        batch_instance: SNLIBatchInstance,
        entail_label_scores: torch.Tensor,
    ) -> torch.Tensor:
        gold_labels = batch_instance["gold_label"].to(self.device)
        loss = self.loss_f(entail_label_scores, gold_labels)

        return loss

    def prediction(self, batch_instance: SNLIBatchInstance) -> torch.Tensor:
        premise_sentence = batch_instance["premise"]
        hypothesis_sentence = batch_instance["hypothesis"]

        input_ids: torch.LongTensor = self.tokenizer.build_inputs_with_special_tokens(
            premise_sentence, hypothesis_sentence, padding=True, return_tensors="pt"
        ).to(self.device)
        batch_size, seq_length = input_ids.shape

        output = self.encoder(input_ids)
        # last_hidden = output.last_hidden_state[:, 0]
        # last_hidden = self.layer_norm(last_hidden)
        # last_hidden = self.embed_dropout_layer(last_hidden)
        # assert last_hidden.shape == torch.Size([batch_size, self.embed_dim])

        pooled_output = output.pooler_output
        assert pooled_output.shape == torch.Size([batch_size, self.embed_dim])

        # entail_label_scores = torch.sigmoid(
        #     self.entail_classifier(last_hidden)
        # )
        entail_label_scores = torch.sigmoid(self.entail_classifier(pooled_output))
        assert entail_label_scores.size() == [batch_size, 3]

        return entail_label_scores

    def inference(self, batch_instance) -> list[int]:
        with torch.no_grad():
            entail_label_scores = self.prediction(batch_instance)

            batch_size = batch_instance["sentence1"].size(0)
            assert entail_label_scores.size() == [batch_size, 3]

            entail_output = torch.argmax(entail_label_scores, dim=2).tolist()
            assert entail_output.size() == [batch_size]

            return entail_output


class BaseBiEncoderModel(nn.Module):
    """Base model using two RoBERTa models for encode

    Given two sentence, this model embeds them into two vectors.
    Then, classyfies whether the premise entails the hypothesis or not.

    The parameters to be trained are as follows:
        (1) two BERT models for encode
            - the mention + source sentence
            - candidate entity + its description (on Wikipedia)
        (2) a linear layer for classification
            - the input dimension is the dimension of BERT's output
    """

    def __init__(
        self,
        config: RobertaConfig,
        device: torch.device,
        # embed_dropout: float = 0.0,
        # layer_norm_eps: float = 1e-5,
        pooling_type: str = "cls",
    ) -> None:
        super(BaseBiEncoderModel, self).__init__()
        self.config = config
        self.device = device
        self.pooling_type = pooling_type

        # Modules for BiEncoder
        self.tokenizer = RobertaTokenizer.from_pretrained(RoBERTaBase)
        self.premise_encoder = BertModel.from_pretrained(RoBERTaBase)
        self.hypothesis_encoder = BertModel.from_pretrained(RoBERTaBase)
        # self.embed_dropout = embed_dropout
        # self.layer_norm_eps = layer_norm_eps
        # self.layer_norm = nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        # self.embed_dropout_layer = nn.Dropout(self.embed_dropout)

        # Modules for classification
        self.embed_dim = config.hidden_size
        self.entail_classifier = nn.Linear(self.embed_dim, len(ENC_labels))

        # Loss function
        self.loss_f = nn.CrossEntropyLoss()

    def forward(self, batch_instance) -> torch.Tensor:
        """Calculate the loss"""
        entail_label_scores = self.prediction(batch_instance)

        assert self.loss_function
        loss = self.loss_function(batch_instance, entail_label_scores, self.device)

        return loss

    def loss_function(
        self,
        batch_instance: SNLIBatchInstance,
        entail_label_scores: torch.Tensor,
    ) -> torch.Tensor:
        gold_labels = batch_instance["label"].to(self.device)
        loss = self.loss_f(entail_label_scores, gold_labels)

        return loss

    def prediction(self, batch_instance: SNLIBatchInstance) -> torch.Tensor:
        """
        Args:
            * batch_instance:

        Returns:
            * premise_cls_embedding:
                Tensor with shape ('batch_size', 'embbeding_dim')
            * hypothesis_cls_embeddings:
                Tensor with shape ('batch_size', 'embbeding_dim')
        """
        premise_sentence = batch_instance["sentence1"]
        hypothesis_sentence = batch_instance["sentence2"]

        # embed premise sentence
        premise_input: torch.LongTensor = (
            self.tokenizer.build_inputs_with_special_tokens(
                premise_sentence, padding=True, return_tensors="pt"
            ).to(self.device)
        )
        premise_batch_size = premise_input.size(0)

        premise_output = self.premise_encoder(premise_input)
        premise_last_hidden = premise_output.last_hidden_state[:, 0]
        # premise_last_hidden = self.layer_norm(premise_last_hidden)
        # premise_last_hidden = self.embed_dropout_layer(premise_last_hidden)
        # assert premise_last_hidden.shape == torch.Size(
        #     [premise_batch_size, premise_len, self.embed_dim]
        # )
        pooled_premise_output = premise_output.pooler_output
        assert pooled_premise_output.shape == torch.Size(
            [premise_batch_size, self.embed_dim]
        )

        # embed hypothesis sentence
        hypothesis_input: torch.LongTensor = (
            self.tokenizer.build_inputs_with_special_tokens(
                hypothesis_sentence, padding=True, return_tensors="pt"
            ).to(self.device)
        )
        hypothesis_batch_size = hypothesis_input.size(0)

        hypothesis_output = self.hypothesis_encoder(hypothesis_input)
        hypothesis_last_hidden = hypothesis_output.last_hidden_state[:, 0]
        # hypothesis_last_hidden = self.layer_norm(hypothesis_last_hidden)
        # hypothesis_last_hidden = self.embed_dropout_layer(hypothesis_last_hidden)
        # assert hypothesis_last_hidden.shape == torch.Size(
        #     [hypothesis_batch_size, hypothesis_len, self.embed_dim]
        # )
        pooled_hypothesis_output = hypothesis_output.pooler_output
        assert pooled_hypothesis_output.shape == torch.Size(
            [hypothesis_batch_size, self.embed_dim]
        )

        assert premise_input.size() == hypothesis_input.size()

        if self.pooling_type == "cls":
            # There is a linear+activation layer after CLS representation
            entail_label_scores = torch.sigmoid(
                self.entail_classifier(pooled_premise_output, pooled_hypothesis_output)
            )
            assert entail_label_scores.size() == [premise_batch_size, 3]

            return entail_label_scores

        elif self.pooling_type == "cls_before_pooler":
            entail_label_scores = torch.sigmoid(
                self.entail_classifier(
                    premise_last_hidden,
                    hypothesis_last_hidden
                )
            )
            assert entail_label_scores.size() == [premise_batch_size, 3]

            return entail_label_scores
        # elif self.pooling_type == "avg":
        #     return (
        #         (last_hidden * batch["attention_mask"].unsqueeze(-1)).sum(1)
        #         / batch["attention_mask"].sum(-1).unsqueeze(-1)
        #     ).cpu()
        else:
            raise NotImplementedError

    def inference(self, batch_instance) -> list[int]:
        with torch.no_grad():
            entail_label_scores = self.prediction(batch_instance)

            batch_size = batch_instance["premise"].size(0)
            assert entail_label_scores.size() == [batch_size, 3]

            entail_output = torch.argmax(entail_label_scores, dim=2).tolist()

            return entail_output


class SyntacticClassifier(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2
    ) -> None:
        """class for syntactic classifier

        Parameters
        ----------
        input_dim : int
            The dimension of input vector
        hidden_dim : int
            The dimension of hidden layer
        output_dim : int
            The dimension of output
        dropout : float, optional
            The dropout rate, by default 0.2
        """
        super(SyntacticClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        kaiming_uniform_(self.linear1.weight)
        kaiming_uniform_(self.linear2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Parameters
        ----------
        x : torch.Tensor
            The input vector

        Returns
        -------
        torch.Tensor
            The output vector
        """
        x = self.linear2(self.dropout(self.relu(self.layer_norm(self.linear1(x)))))
        return x


# class HolCCGModel(nn.Module):
#     def __init__(
#         self,
#         num_word_cat: int,
#         num_phrase_cat: int,
#         encoder_model: nn.Module,
#         tokenizer: nn.Module,
#         model_dim: int,
#         dropout: float,
#         vector_norm: int,
#         device: torch.device,
#         loss_function: Optional[LossFunction] = None,
#         bert_model: Optional[RoBERTaModel] = None,
#     ):
#         """class for HolCCG
#
#         Parameters
#         ----------
#         num_word_cat : int
#             The number of word category. This is used for building HolCCG's syntactic classifier
#         num_phrase_cat : int
#             The number of phrase category. This is used for building HolCCG's syntactic classifier
#         encoder : nn.Module
#             Text encoder used for HolCCG
#         tokenizer : nn.Module
#             Tokenizer used for HolCCG
#         model_dim : int
#             The dimension of HolCCG's representation
#         dropout : float
#             The dropout rate
#         vector_norm : int
#             The maximum norm of HolCCG's representation
#         device : torch.device
#             The device to use for HolCCG
#         """
#         super(HolCCG, self).__init__()
#         self.num_word_cat = num_word_cat
#         self.num_phrase_cat = num_phrase_cat
#
#         self.encoder_model = encoder_model
#         self.tokenizer = tokenizer
#         self.model_dim = model_dim
#
#         self.vector_norm = vector_norm
#
#         # the list which to record the modules to set separated learning rate
#         self.base_modules = []
#         self.base_params = []
#
#         self.linear = nn.Linear(self.model_dim, self.model_dim)
#         kaiming_uniform_(self.linear.weight)
#         self.base_modules.append(self.linear)
#         self.word_classifier = SyntacticClassifier(
#             self.model_dim, self.model_dim, self.num_word_cat, dropout=dropout
#         )
#         self.phrase_classifier = SyntacticClassifier(
#             self.model_dim, self.model_dim, self.num_phrase_cat, dropout=dropout
#         )
#         self.span_classifier = SyntacticClassifier(
#             self.model_dim, self.model_dim, 2, dropout=dropout
#         )
#         self.base_modules.append(self.word_classifier)
#         self.base_modules.append(self.phrase_classifier)
#         self.base_modules.append(self.span_classifier)
#         for module in self.base_modules:
#             for params in module.parameters():
#                 self.base_params.append(params)
#         self.base_params = iter(self.base_params)
#         self.device = device
#
#         self.loss_function = loss_function
#
#     def forward(self, batch_instance) -> torch.Tensor:
#         output = self.prediction(batch_instance)
#
#         assert self.loss_function
#         loss = self.loss_function(batch_instance, output, self.device)
#
#         return loss
#
#     def prediction(self, batch_instance) -> torch.Tensor:
#         num_node = batch[0]
#         sentence = batch[1]
#         original_position = batch[2]
#         composition_info = batch[3]
#         batch_label = batch[4]
#         word_split = batch[5]
#         random_num_node = batch[6]
#         random_composition_info = batch[7]
#         random_original_position = batch[8]
#         random_negative_node_id = batch[9]
#
#         vector_list, lengths = self.encode(sentence, word_split)
#
#         # compose word vectors and fed them into FFNN
#         original_vector = self.set_leaf_node_vector(
#             num_node, vector_list, lengths, original_position
#         )
#         # compose word vectors for randomly generated trees
#         random_vector = self.set_leaf_node_vector(
#             random_num_node, vector_list, lengths, random_original_position
#         )
#         original_vector_shape = original_vector.shape
#         random_vector_shape = random_vector.shape
#         original_vector = original_vector.view(-1, self.model_dim)
#         random_vector = random_vector.view(-1, self.model_dim)
#         vector = torch.cat((original_vector, random_vector))
#         original_vector = vector[
#             : original_vector_shape[0] * original_vector_shape[1], :
#         ].view(original_vector_shape[0], original_vector_shape[1], self.model_dim)
#         random_vector = vector[
#             original_vector_shape[0] * original_vector_shape[1] :, :
#         ].view(random_vector_shape[0], random_vector_shape[1], self.model_dim)
#         composed_vector = self.compose(original_vector, composition_info)
#         random_composed_vector = self.compose(random_vector, random_composition_info)
#         word_vector, phrase_vector, word_label, phrase_label = self.devide_word_phrase(
#             composed_vector, batch_label, original_position
#         )
#         span_vector, span_label = self.extract_span_vector(
#             phrase_vector, random_composed_vector, random_negative_node_id
#         )
#
#         word_output = self.word_classifier(word_vector)
#         phrase_output = self.phrase_classifier(phrase_vector)
#         span_output = self.span_classifier(span_vector)
#         return (
#             word_output,
#             phrase_output,
#             span_output,
#             word_label,
#             phrase_label,
#             span_label,
#         )
#
#     def encode(self, sentence: list[str], word_split: list[list[tuple]]) -> Tuple:
#         """Encoding sentence into word vectors
#
#         Parameters
#         ----------
#         sentence : List[str]
#             The sentence to encode
#         word_split : List[List[Tuple]]
#             The word split information
#
#         Returns
#         -------
#         Tuple
#             The word vectors and their corresponding lengths
#         """
#
#         input_ids = self.tokenizer(sentence, padding=True, return_tensors="pt").to(
#             self.device
#         )
#         # remove CLS
#         word_vector = self.encoder(**input_ids).last_hidden_state[:, 1:-1]
#
#         word_vector_list = []
#         lengths = []
#         for vector, info in zip(word_vector, word_split):
#             temp = []
#             for start_idx, end_idx in info:
#                 temp.append(torch.mean(vector[start_idx:end_idx], dim=0))
#             word_vector_list.append(torch.stack(temp))
#             lengths.append(len(temp))
#
#         word_vector = pad_sequence(word_vector_list, batch_first=True)
#         word_vector = self.linear(word_vector)
#         word_vector = self.vector_norm * normalize(word_vector, dim=-1)
#
#         lengths = torch.tensor(lengths, device=torch.device("cpu"))
#
#         return word_vector, lengths
