import torch
import torch.nn as nn
from modelling.layers import *
from modelling.templates import SequenceTemplate


# embeddings
class Embedding(BaseModule):

    def __init__(self,
                 vocab_size,
                 embedding_dim=300,
                 dropout=0.,
                 padding_idx=0,
                 embedding_matrix=None,
                 embedding_layer=None,
                 freeze_pre_trained_embeddings=True):

        super(Embedding, self).__init__()

        self.num_embeddings = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.freeze_pre_trained_embeddings = freeze_pre_trained_embeddings

        if embedding_layer is not None:

            self.embedding = embedding_layer

        else:

            self.embedding = torch.nn.Embedding(num_embeddings=self.num_embeddings,
                                                embedding_dim=self.embedding_dim,
                                                padding_idx=self.padding_idx)

            if embedding_matrix is not None:

                embedding_matrix = torch.Tensor(embedding_matrix)
                self.embedding = self.embedding.from_pretrained(embeddings=embedding_matrix,
                                                                freeze=self.freeze_pre_trained_embeddings)

                self.training = False

    def forward(self, x, scale=None):

        if self.dropout and self.training:

            mask = self.embedding.weight.data.new().resize_((self.embedding.weight.size(0), 1))
            mask = mask.bernoulli_(1 - self.dropout)
            mask = mask.expand_as(self.embedding.weight) / (1 - self.dropout)

            masked_embedding_matrix = mask * self.embedding.weight
        else:
            masked_embedding_matrix = self.embedding.weight
        if scale:
            masked_embedding_matrix = scale.expand_as(masked_embedding_matrix) * masked_embedding_matrix

        masked_embedding_matrix = masked_embedding_matrix.to(self.device)

        x = torch.nn.functional.embedding(x,
                                          masked_embedding_matrix,
                                          self.embedding.padding_idx,
                                          self.embedding.max_norm,
                                          self.embedding.norm_type,
                                          self.embedding.scale_grad_by_freq,
                                          self.embedding.sparse)

        return x

    def extra_repr(self):

        report = [self.model_report, '(dropout): Dropout(p={})'.format(self.dropout)]

        return '\n'.join(report)


class PositionalEmbedding(BaseModule):

    def __init__(self,
                 embedding_layer,
                 sequence_length,
                 dropout=0.):

        super(PositionalEmbedding, self).__init__()

        self.token_embedding = embedding_layer

        self.sequence_length = sequence_length
        self.dropout = dropout

        self.embedding_dim = self.token_embedding.embedding_dim
        self.padding_idx = self.token_embedding.padding_idx

        self._start = 0

        if self.token_embedding.padding_idx is not None:
            self._start += 1

        self.positional_embedding = Embedding(vocab_size=self.sequence_length + self._start,
                                              embedding_dim=self.embedding_dim,
                                              dropout=self.dropout,
                                              padding_idx=self.padding_idx)

    def forward(self, x, padding_mask=None):

        position_ids = torch.arange(start=self._start, end=x.size(1) + self._start, dtype=torch.long, device=x.device)

        position_ids = position_ids.unsqueeze(0).expand(x.size(0), x.size(1))

        if padding_mask is not None:
            position_ids *= padding_mask
        elif self.token_embedding.padding_idx is not None:
            position_ids = position_ids * (x != 0).long()

        token_embeddings = self.token_embedding(x)
        position_embedding = self.positional_embedding(position_ids)

        x = token_embeddings + position_embedding

        return x


class EmbeddingAggregation(BaseModule):

    def __init__(self,
                 embedding_size,
                 sequence_length,
                 n_layers,
                 n_combinations,
                 base_projection_layer=LinearWithActivation):

        super(EmbeddingAggregation, self).__init__()

        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.n_layers = n_layers
        self.n_combinations = n_combinations

        self.model = SequenceTemplate(
            LayersAggregation(n_combinations=self.n_combinations),
            WeightedAggregation(n_layers=self.n_layers // self.n_combinations),
            base_projection_layer(self.embedding_size * self.n_combinations, self.embedding_size)
        )

    def forward(self, x):

        x = self.model(x)

        return x


class Attention1DEmbedding(BaseModule):

    def __init__(self, embedding_layer, embedding_dim, sequence_length):

        super(Attention1DEmbedding, self).__init__()

        self.embedding_layer = embedding_layer

        self.embedding_dim = self.embedding_layer.embedding_dim
        self.padding_idx = self.embedding_layer.padding_idx

        self.attention_pooling = AttentionPooling(embedding_dim=embedding_dim, sequence_length=sequence_length)

    def forward(self, x):

        x = self.embedding_layer(x)

        x = self.attention_pooling(x)

        return x


class Attention2DEmbedding(BaseModule):

    def __init__(self, embedding_layer, embedding_dim, sequence_length):

        super(Attention2DEmbedding, self).__init__()

        self.embedding_layer = Attention1DEmbedding(embedding_layer=embedding_layer,
                                                    embedding_dim=embedding_dim,
                                                    sequence_length=sequence_length)

        self.embedding_dim = self.embedding_layer.embedding_dim
        self.padding_idx = self.embedding_layer.padding_idx

    def forward(self, x):

        output = []

        for sample in x:

            sample = self.embedding_layer(sample)

            sample = sample.unsqueeze(0)

            output.append(sample)

        output = torch.cat(output)

        return output


class PositionAttentionWordCharEmbedding(BaseModule):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 token_sequence_max_length,
                 char_sequence_max_length,
                 dropout=0.,
                 padding_idx=0,
                 embedding_matrix=None,
                 embedding_layer=None,
                 freeze_pre_trained_embeddings=True):

        super(PositionAttentionWordCharEmbedding, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.token_sequence_max_length = token_sequence_max_length
        self.char_sequence_max_length = char_sequence_max_length

        self.dropout = dropout
        self.padding_idx = padding_idx

        self.embedding = PositionalEmbedding(
            embedding_layer=Attention2DEmbedding(
                embedding_layer=PositionalEmbedding(
                    embedding_layer=Embedding(embedding_dim=self.embedding_dim,
                                              vocab_size=self.vocab_size,
                                              dropout=self.dropout,
                                              embedding_matrix=embedding_matrix,
                                              embedding_layer=embedding_layer,
                                              freeze_pre_trained_embeddings=freeze_pre_trained_embeddings),
                    sequence_length=self.char_sequence_max_length),
                embedding_dim=self.embedding_dim,
                sequence_length=self.char_sequence_max_length),
            sequence_length=self.token_sequence_max_length)

    def forward(self, x, word_padding_mask=None):

        x = self.embedding(x, padding_mask=word_padding_mask)

        return x


class MeanAndCNNEmbedding(BaseModule):

    def __init__(self,
                 embedding_dim,
                 sequence_length,
                 activation_function=torch.nn.ReLU()):

        super(MeanAndCNNEmbedding, self).__init__()

        self.mean_embedding = Mean(dim=2)

        self.cnn = CNN(input_size=embedding_dim,
                       out_channels=embedding_dim,
                       kernel_size_convolution=1,
                       sequence_length=sequence_length,
                       activation_function=activation_function,
                       global_pooling=True)

    def forward(self, x):

        x_mean = self.mean_embedding(x)
        x_cnn = torch.cat([self.cnn(sample).unsqueeze(0) for sample in x])

        x = x_mean + x_cnn

        return x


class CNNEmbedding(BaseModule):

    def __init__(self,
                 embedding_dim,
                 cnn_channels,
                 sequence_length,
                 fc_projection=512,
                 kernel_size_convolutions=(2, 3, 4),
                 fc_base_layer=Highway,
                 fc_residual=False,
                 activation_function=torch.nn.ReLU(),
                 activation_function_output=None,
                 fc_last_layer_linear=False):

        super(CNNEmbedding, self).__init__()

        self.cnns = torch.nn.ModuleList(modules=[CNN(input_size=embedding_dim,
                                                     out_channels=cnn_channels,
                                                     kernel_size_convolution=kernel_size_convolution,
                                                     sequence_length=sequence_length,
                                                     activation_function=activation_function,
                                                     global_pooling=True)
                                                 for kernel_size_convolution in kernel_size_convolutions])

        self.fully_connected = FullyConnected(sizes=(cnn_channels * len(kernel_size_convolutions), fc_projection),
                                              base_layer=fc_base_layer,
                                              activation_function=activation_function,
                                              activation_function_output=activation_function_output,
                                              residual=fc_residual,
                                              last_layer_linear=fc_last_layer_linear)

    def forward(self, x):

        x = torch.cat([torch.cat([self.fully_connected(token).unsqueeze(0) for token in
                                  torch.cat([cnn(sample) for cnn in self.cnns], dim=1).unsqueeze(0)])
                       for sample in x])

        return x


# other models
class DAN(BaseModule):

    def __init__(self,
                 sizes=(300, 128, 3),
                 fc_base_layer=LinearWithActivation,
                 fc_last_layer_linear=True,
                 activation_function=torch.nn.ReLU(),
                 activation_function_output=nn.LogSoftmax(dim=1)):
        """
        Deep Average Network
        from
        Deep Unordered Composition Rivals Syntactic Methods for Text Classification
        https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf
        """

        super(DAN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.name = 'DAN'

        self.mean_embedding = Mean(dim=1)

        self.fully_connected = FullyConnected(sizes=sizes,
                                              base_layer=fc_base_layer,
                                              activation_function=activation_function,
                                              activation_function_output=activation_function_output,
                                              last_layer_linear=fc_last_layer_linear).to(self.device)

    def forward(self, x):

        x = self.mean_embedding(x)

        x = self.fully_connected(x)

        return x


class RNNAttentionCNNModel(BaseModule):

    def __init__(self,
                 rnn_layer,
                 cnn_layer,
                 fc_layer=None):

        super(RNNAttentionCNNModel, self).__init__()

        self.rnn = rnn_layer

        self.self_attention = ScaledDotProductAttention()

        self.cnn = cnn_layer

        self.fully_connected = fc_layer

    def forward(self, x):

        x = self.rnn(x)

        x, _ = self.self_attention(x, x, x)

        x = self.cnn(x)

        if self.fully_connected is not None:
            x = self.fully_connected(x)

        return x


class Inception(BaseModule):

    def __init__(self,
                 cnn_in_channels=300,
                 cnn_out_channels=256,
                 sequence_length=64,
                 kernel_size_convolutions=(2, 3, 4),
                 activation_function=torch.nn.ReLU()):

        super(Inception, self).__init__()

        self.cnns = torch.nn.ModuleList(modules=[CNN(input_size=cnn_in_channels,
                                                     out_channels=cnn_out_channels,
                                                     kernel_size_convolution=kernel_size_convolution,
                                                     sequence_length=sequence_length,
                                                     global_pooling=True,
                                                     activation_function=activation_function)
                                                 for kernel_size_convolution in kernel_size_convolutions])

    def forward(self, x):

        if x.size(0) > 1:

            x = torch.cat([cnn(x) for cnn in self.cnns], dim=1)

        else:

            x = torch.cat([cnn(x) for cnn in self.cnns])

            x = x.unsqueeze(0)

        return x


class SelfAttention(BaseModule):

    def __init__(self, temperature=None, attention_dropout=0.):

        super(SelfAttention, self).__init__()

        self.attention_layer = ScaledDotProductAttention(temperature=temperature,
                                                         return_attention=False,
                                                         attention_dropout=attention_dropout)

    def forward(self, x):

        x = self.attention_layer(x, x, x)

        return x


class TransformerEncoder(BaseModule):

    def __init__(self, n_head, model_dim, inner_dim, key_dim, value_dim, dropout=0.1, return_attention=False):

        super(TransformerEncoder, self).__init__()

        self.attention = MultiHeadAttention(n_head=n_head, model_dim=model_dim, key_dim=key_dim,
                                            value_dim=value_dim, dropout=dropout, return_attention=True)

        self.position_wise_feed_forward = PositionWiseFeedForward(size_in=model_dim,
                                                                  size_out=inner_dim,
                                                                  dropout=dropout)

        self.return_attention = return_attention

    def forward(self, encoder_input, non_pad_mask=None, self_attention_mask=None):

        encoder_output, encoder_self_attn = self.attention(encoder_input,
                                                           encoder_input,
                                                           encoder_input,
                                                           mask=self_attention_mask)

        if non_pad_mask is not None:
            encoder_output *= non_pad_mask

        encoder_output = self.position_wise_feed_forward(encoder_output)

        if non_pad_mask is not None:
            encoder_output *= non_pad_mask

        if self.return_attention:
            return encoder_output, encoder_self_attn
        else:
            return encoder_output
