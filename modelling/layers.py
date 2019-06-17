import numpy as np
from collections import defaultdict
from functools import reduce
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# basics
class BaseModule(torch.nn.Module):

    def __init__(self):

        super(BaseModule, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.is_training = True

    def forward(self, *x):

        raise NotImplementedError

    @staticmethod
    def _n_beautify(n):
        reversed_n = ''.join(str(n)[::-1])
        return ','.join([reversed_n[f - 3:f] for f in range(3, len(reversed_n) + 3, 3)])[::-1]

    @property
    def layers_names(self):

        layers_names = []

        for param_name in self.state_dict().keys():

            param_name = param_name.replace('.weight', '').replace('.bias', '')

            if param_name not in layers_names:
                layers_names.append(param_name)

        return layers_names

    @property
    def param_report(self):

        trainable_n_params = sum([reduce(lambda x, y: x * y, param.size()) for param
                                  in self.parameters() if param.requires_grad])

        non_trainable_n_params = sum([reduce(lambda x, y: x * y, param.size()) for param
                                      in self.parameters() if not param.requires_grad])

        report = []

        if trainable_n_params > 0:
            report = ['Trainable — {}'.format(self._n_beautify(trainable_n_params))]

        if non_trainable_n_params > 0:
            report += ['Non trainable — {}'.format(self._n_beautify(non_trainable_n_params))]

        report = ' | '.join(report)

        return report

    @property
    def model_report(self):

        report = list()

        layers_names = self.layers_names

        if len(layers_names) > 1:
            report.append('Layers: {}'.format(len(layers_names)))

        param_report = self.param_report

        if param_report:
            report.append(self.param_report)

        if not self.training:
            report.append('Training: {}'.format(self.training))

        report = '\n'.join(report)

        return report

    def extra_repr(self):
        return self.model_report

    @staticmethod
    def empty_repr():
        return ''

    def freeze(self, n_last_params_unfreeze=0):

        # TODO n_last_params_unfreeze less risky

        for n, param in enumerate(self.parameters()):

            if n <= len(list(self.parameters())) - n_last_params_unfreeze - 1:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.train(mode=False)

        self.is_training = False if n_last_params_unfreeze != 0 else True

    def unfreeze(self):

        for n, param in enumerate(self.parameters()):

            param.requires_grad = True

        self.train(mode=True)

        self.is_training = True


class Mean(BaseModule):

    def __init__(self, dim=1):

        super(Mean, self).__init__()

        self.dim = dim

    def forward(self, x):

        x = x.mean(dim=self.dim)

        return x


class Flatten(BaseModule):

    def __init__(self, dim=1):

        super(Flatten, self).__init__()

        self.dim = dim

    def forward(self, x):

        return x.flatten(self.dim)


# activation functions
class GELU(BaseModule):

    def __init__(self):
        """
        Gaussian Error Linear Unit implementation
        https://arxiv.org/pdf/1606.08415.pdf
        Used in transformer
        """

        super(GELU, self).__init__()

        self.const_1 = torch.Tensor([0.044715]).to(self.device)
        # self.const_2 = torch.pow(torch.Tensor([2 / 3.141592653589793]), 0.5)
        self.const_2 = torch.Tensor([(2 / 3.141592653589793) ** 0.5]).to(self.device)

    def forward(self, x):

        return 0.5 * x * (1 + torch.tanh(self.const_1 * (x + self.const_2 * torch.pow(x, 3))))

    def extra_repr(self):

        return ''


# useful tools
class Residual(BaseModule):

    def __init__(self, layer, residual_layer=None, dropout_layer=None, normalization_layer=None):

        super(Residual, self).__init__()

        self.layer = layer

        self.residual_layer = residual_layer
        self.dropout_layer = dropout_layer

        self.normalization_layer = normalization_layer

    def forward(self, *x):

        if isinstance(x, (list, tuple)):
            residual = x[0]
        else:
            residual = x

        if self.residual_layer is not None:
            residual = self.residual_layer(residual)

        layer_output = self.layer(*x)

        if isinstance(layer_output, (list, tuple)):

            if self.dropout_layer is not None:
                layer_output[0] = self.dropout_layer(layer_output[0])

            output = layer_output[0] + residual

            if self.normalization_layer:
                output = self.normalization_layer(output)

            layer_output = (output, ) + layer_output[1:]

        else:

            if self.dropout_layer is not None:
                layer_output = self.dropout_layer(layer_output)

            layer_output += residual

            if self.normalization_layer:
                layer_output = self.normalization_layer(layer_output)

        return layer_output

    def extra_repr(self):

        if isinstance(self.layer, BaseModule) and self.residual_layer is None:
            self.layer.extra_repr = self.empty_repr

            if self.residual_layer is not None:
                self.residual_layer.extra_repr = self.empty_repr

            return self.model_report
        else:
            return self.model_report


class LayersAggregation(BaseModule):

    def __init__(self, n_combinations):

        super(LayersAggregation, self).__init__()

        self.n_combinations = n_combinations

    def forward(self, x):

        batch_size, sequence_length, n_layers, embedding_size = x.size()

        # batch_size, sequence_length, n_layers, embedding_size -> n_layers, batch_size, sequence_length, embedding_size
        x = x.permute(2, 0, 1, 3)

        output = torch.zeros((n_layers // self.n_combinations,
                              batch_size,
                              sequence_length,
                              embedding_size * self.n_combinations),
                             device=self.device)

        for n, i in enumerate(range(0, n_layers, self.n_combinations)):

            current_x = torch.cat([x[i + n, :] for n in range(self.n_combinations)], dim=-1)

            output[n] = current_x

        # n_layers, batch_size, sequence_length, embedding_size -> batch_size, sequence_length, n_layers, embedding_size
        output = output.permute(1, 2, 0, 3)

        return output


class WeightedAggregation(BaseModule):

    def __init__(self, n_layers):

        super(WeightedAggregation, self).__init__()

        self.weights = torch.nn.Parameter(torch.rand(n_layers))
        self.function = torch.nn.Softmax(dim=0)

    def forward(self, x):

        batch_size, sequence_length, n_layers, embedding_size = x.size()

        weights = self.function(self.weights)

        weights = weights.unsqueeze(0).repeat(sequence_length, 1)
        weights = weights.unsqueeze(0).repeat(batch_size, 1, 1)
        weights = weights.unsqueeze(3).expand_as(x)

        x *= weights

        x = x.sum(-2)

        return x


# dropouts
class LockedDropout(BaseModule):

    def __init__(self):

        super(LockedDropout, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, dropout=0.5):

        if not self.training or not dropout:
            return x

        if len(x.size()) == 2:
            mask = x.data.new(1, x.size(1)).bernoulli_(1 - dropout).to(self.device)
        elif len(x.size()) == 3:
            mask = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout).to(self.device)
        else:
            raise ValueError("Unavailable dimension. Only 2 and 3 support")

        # maybe it not necessary
        mask.requires_grad = False

        mask = mask / (1 - dropout)

        mask = mask.expand_as(x)

        x = mask * x

        return x


class WeightDrop(BaseModule):

    def __init__(self, module, weights, dropout=0., variational=False):

        super(WeightDrop, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational

        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):

        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:

            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', torch.nn.Parameter(w.data))

    def _setweights(self):

        for name_w in self.weights:

            raw_w = getattr(self.module, name_w + '_raw')

            if self.variational:

                mask = torch.nn.Parameter(torch.ones(raw_w.size(0), 1)).to(self.device)
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w

            else:

                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)

            setattr(self.module, name_w, w)

    def extra_repr(self):

        report = [self.model_report, '(weightdrop): p={}'.format(self.dropout)]

        return '\n'.join(report)

    def forward(self, *args):

        self._setweights()

        return self.module.forward(*args)


class MultiSampleDropout(BaseModule):

    """
    Implementation https://arxiv.org/pdf/1905.09788.pdf
    """

    def __init__(self, layer, dropout_probabilities=(0.1, 0.1), mean_before_layer=False):

        super(MultiSampleDropout, self).__init__()

        self.dropout_probabilities = dropout_probabilities

        for n, p in enumerate(self.dropout_probabilities):
            if not 0. < p < 1.:
                raise ValueError("dropout probability has to be between 0 and 1, "
                                 "but got {} at {} position".format(p, n))

        self.dropout = LockedDropout()

        self.layer = layer

        self.mean_before_layer = mean_before_layer

    def forward(self, x):

        if not self.is_training:
            return self.layer(x)

        if self.mean_before_layer:
            x = [self.dropout(x, dropout=p).unsqueeze(1) for p in self.dropout_probabilities]
        else:
            x = [self.layer(self.dropout(x, dropout=p)).unsqueeze(1) for p in self.dropout_probabilities]

        x = torch.cat(x, 1).mean(1)

        if self.mean_before_layer:
            x = self.layer(x)

        return x

    def extra_repr(self):

        return '(dropout_probabilities): ({})'.format(', '.join([str(p) for p in self.dropout_probabilities]))


# fully connected layers
class LinearWithActivation(BaseModule):

    def __init__(self, size_in, size_out, activation_function=torch.nn.ReLU()):

        super(LinearWithActivation, self).__init__()

        self.linear = torch.nn.Linear(size_in, size_out)
        self.activation_function = activation_function

    def forward(self, x):

        x = self.linear(x)
        x = self.activation_function(x)

        return x


class Highway(BaseModule):

    def __init__(self, size_in, size_out, activation_function=torch.nn.ReLU()):

        super(Highway, self).__init__()

        self.nonlinear = torch.nn.Linear(size_in, size_out)
        self.linear = torch.nn.Linear(size_in, size_out)
        self.gate = torch.nn.Linear(size_in, size_out)

        self.gate_function = torch.nn.Sigmoid()
        self.activation_function = activation_function

    def forward(self, x):
        """
        :param x: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
        f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
        and ⨀ is element-wise multiplication
        """

        gate = self.gate(x)
        gate = self.gate_function(gate)

        nonlinear = self.nonlinear(x)
        nonlinear = self.activation_function(nonlinear)

        linear = self.linear(x)

        x = gate * nonlinear + (1 - gate) * linear

        return x


# layers
class BasePositionWiseFeedForward(BaseModule):

    """
    A two-feed-forward-layer module from transformer
    """

    def __init__(self,
                 size_in,
                 size_out,
                 activation_function=GELU()):

        super(BasePositionWiseFeedForward, self).__init__()

        self.layer_1 = torch.nn.Conv1d(size_in, size_out, 1)  # position-wise
        self.layer_2 = torch.nn.Conv1d(size_out, size_in, 1)  # position-wise

        self.activation_function = activation_function

    def forward(self, x):

        output = x.transpose(1, 2)

        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.layer_2(output)

        output = output.transpose(1, 2)

        return output


class PositionWiseFeedForward(BaseModule):

    """
    A two-feed-forward-layer module from transformer
    """

    def __init__(self,
                 size_in,
                 size_out,
                 dropout=0.1,
                 activation_function=GELU()):

        super(PositionWiseFeedForward, self).__init__()

        self.layer = Residual(layer=BasePositionWiseFeedForward(size_in=size_in,
                                                                size_out=size_out,
                                                                activation_function=activation_function),
                              dropout_layer=torch.nn.Dropout(dropout),
                              normalization_layer=torch.nn.LayerNorm(size_in))

    def forward(self, x):

        output = self.layer(x)

        return output


class FullyConnected(BaseModule):

    def __init__(self,
                 sizes,
                 base_layer=LinearWithActivation,
                 activation_function=torch.nn.ReLU(),
                 activation_function_output=None,
                 residual=False,
                 last_layer_linear=False):
        """
        Simple FC layer. All you need is set sizes
        """

        super(FullyConnected, self).__init__()

        self.sizes = list(sizes)

        self.base_layer = base_layer
        self.base_last_layer = LinearWithActivation if last_layer_linear else self.base_layer

        activation_function = activation_function

        if activation_function_output is None:
            activation_function_output = activation_function

        self.input_size = self.sizes[0]
        self.output_size = self.sizes[-1]

        layers = []

        for n in range(len(self.sizes[:-1])):

            if isinstance(sizes[n], (int,)):

                out_features = self.sizes[n+1] if type(self.sizes[n+1]) == int else self.sizes[n+2]

                if n == len(self.sizes) - 2:
                    current_activation_function = activation_function_output
                else:
                    current_activation_function = activation_function

                if n == len(self.sizes[:-1]) - 1:
                    current_base_layer = self.base_last_layer
                else:
                    current_base_layer = self.base_layer

                layer = current_base_layer(size_in=self.sizes[n],
                                           size_out=out_features,
                                           activation_function=current_activation_function)

                if residual and self.sizes[n] == out_features:
                    layer = Residual(layer=layer)

            elif isinstance(sizes[n], (float,)):

                if last_layer_linear:
                    assert (n < len(self.sizes) - 2)

                layer = torch.nn.Dropout(p=sizes[n])

            else:

                raise ValueError('Only int or float in sizes')

            layers.append(layer)

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):

        x = self.model(x)

        return x


class RNN(BaseModule):

    def __init__(self,
                 input_size,
                 hidden_size=256,
                 base_rnn=torch.nn.LSTM,
                 layers=1,
                 dropout_input=0.5,
                 dropout_hidden=0.5,
                 dropout_output=0.5,
                 weight_drop=0.1,
                 bidirectional=False,
                 residual=False,
                 last_layer_without_dropout=False,
                 output_last_state=False,
                 output_need_states=False):
        """
        Simple RNN layer. All you need is set input_size and hidden_size
        """

        super(RNN, self).__init__()

        assert (hidden_size % 2 == 0)

        self.input_size = input_size
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size

        self.base_rnn = base_rnn
        self.layers = layers

        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.dropouts = [self.dropout_input, self.dropout_hidden, self.dropout_output]

        self.weight_drop = weight_drop

        self.residual = residual
        self.last_layer_without_dropout = last_layer_without_dropout

        self.output_last_state = output_last_state
        self.output_need_states = output_need_states

        self.locked_dropout = LockedDropout()

        rnn_layers = []

        for n in range(layers):

            if n > 0:
                current_input_size = self.hidden_size
                if bidirectional:
                    current_input_size *= 2
            else:
                current_input_size = self.input_size

            layer = base_rnn(input_size=current_input_size,
                             hidden_size=self.hidden_size,
                             num_layers=1,
                             bidirectional=bidirectional)

            if self.weight_drop:
                layer = WeightDrop(module=layer, weights=['weight_hh_l0'], dropout=self.weight_drop)

            if self.residual and current_input_size == self.hidden_size:
                layer = Residual(layer=layer)

            rnn_layers.append(layer)

        self.rnn = torch.nn.ModuleList(rnn_layers)

    def transpose_states(self, states):

        if isinstance(states, (list, tuple)):
            states = [self.transpose_states(states=current_states) for current_states in states]
        else:
            states = states.transpose(0, 1).contiguous()

        return states

    def init_states(self, batch_size):

        weight = next(self.parameters()).data

        if self.base_rnn in [torch.nn.LSTM]:
            return [(weight.new(batch_size, 1, self.hidden_size).zero_().to(self.device),
                     weight.new(batch_size, 1, self.hidden_size).zero_().to(self.device)) for _ in range(self.layers)]
        elif self.base_rnn in [torch.nn.GRU]:
            return [weight.new(batch_size, 1, self.hidden_size).zero_().to(self.device) for _ in range(self.layers)]

    def extra_repr(self):

        dropout_report = [
            '({}): p={}'.format(dropout, getattr(self, dropout))
            for dropout in ['dropout_input', 'dropout_hidden', 'dropout_output']
        ]

        report = [self.model_report] + dropout_report

        return '\n'.join(report)

    def forward(self, x, states=None, x_lengths=None):
        """
        Pack padded sequence if set x_lengths if you have batch we variable length for better perfomance
        """

        output_states = []

        x = self.locked_dropout(x=x, dropout=self.dropout_input)

        # (B,L,D) -> (L,B,D)
        x = x.transpose(0, 1)
        if states is not None:
            states = self.transpose_states(states)

        if x_lengths is not None:
            x = pack_padded_sequence(x, x_lengths)

        for n, layer in enumerate(self.rnn):

            if states is not None:
                x, current_states = layer(x, states[n])
            else:
                x, current_states = layer(x)

            output_states.append(current_states)

            if n != self.layers - 1:
                x = self.locked_dropout(x=x, dropout=self.dropout_hidden)

        x = self.locked_dropout(x=x, dropout=self.dropout_output)

        if x_lengths is not None:
            x, _ = pad_packed_sequence(x)

        # (L,B,D) -> (B,L,D)
        x = x.transpose(0, 1)
        output_states = self.transpose_states(output_states)

        if self.output_last_state:
            return x[:, -1, :]
        elif self.output_need_states:
            return x, output_states
        else:
            return x


class CNN(BaseModule):

    def __init__(self,
                 input_size,
                 out_channels,
                 kernel_size_convolution,
                 sequence_length,
                 kernel_size_pool=None,
                 global_pooling=False,
                 convolution_stride=1,
                 convolution_padding=0,
                 convolution_padding_same=False,
                 convolution_dilation=1,
                 convolution_groups=1,
                 base_pool_layer=torch.nn.MaxPool1d,
                 pool_stride=1,
                 pool_padding=0,
                 declared_pool_layer=None,
                 activation_function=torch.nn.ReLU()):
        """
        Simple CNN1D layer. All you need is set input_size, out_channels and kernel_size_convolution
        """

        super(CNN, self).__init__()

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.out_channels = out_channels
        self.global_pooling = global_pooling

        if convolution_padding_same:
            convolution_padding = kernel_size_convolution - 1

        self.convolution_layer = torch.nn.Conv1d(in_channels=self.input_size,
                                                 out_channels=self.out_channels,
                                                 kernel_size=kernel_size_convolution,
                                                 stride=convolution_stride,
                                                 padding=convolution_padding,
                                                 dilation=convolution_dilation,
                                                 groups=convolution_groups)

        self.activation_function = activation_function

        if declared_pool_layer is not None:
            self.pool_layer = declared_pool_layer
        elif base_pool_layer is not None:

            if self.global_pooling:
                kernel_size_pool = self.sequence_length - kernel_size_convolution + 1
            elif kernel_size_pool is None:
                kernel_size_pool = kernel_size_convolution

            self.pool_layer = base_pool_layer(kernel_size=kernel_size_pool,
                                              stride=pool_stride,
                                              padding=pool_padding)
        else:
            self.pool_layer = None

    def forward(self, x, x_lengths=None):
        """
        return correct batch with (batch_size x seq_len x input_size)  sizes
        """

        # Turn (batch_size x seq_len x input_size) into (batch_size x input_size x seq_len) for CNN
        x = x.transpose(1, 2)

        x = self.convolution_layer(x)

        x = self.activation_function(x)

        if self.pool_layer is not None:
            x = self.pool_layer(x)

        if self.global_pooling:
            x = x.squeeze()
        else:
            # Turn (batch_size x input_size x seq_len) into (batch_size x seq_len x input_size)
            x = x.transpose(1, 2)

        return x


# attentions
class ScaledDotProductAttention(BaseModule):
    """
    From Transformer
    """

    def __init__(self, temperature=None, return_attention=False, attention_dropout=0.1):

        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.return_attention = return_attention

        self.dropout = torch.nn.Dropout(p=attention_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, query, keys, values, mask=None):

        attention = torch.bmm(query, keys.transpose(1, 2))

        if self.temperature is not None:
            attention = torch.div(attention, self.temperature)

        if mask is not None:
            attention = attention.masked_fill(mask, -np.inf)

        attention = self.softmax(attention)

        attention = self.dropout(attention)

        output = torch.bmm(attention, values)

        if self.return_attention:
            return output, attention
        else:
            return output


class MultiHeadAttention(BaseModule):
    """
    From transformer
    """

    def __init__(self, n_head, model_dim, key_dim, value_dim, dropout=0.1, return_attention=False):

        super().__init__()

        self.n_head = n_head
        self.model_dim = model_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.query_projection = torch.nn.Linear(self.model_dim, self.n_head * self.key_dim)
        self.key_projection = torch.nn.Linear(self.model_dim, self.n_head * self.key_dim)
        self.value_projection = torch.nn.Linear(self.model_dim, self.n_head * self.value_dim)

        torch.nn.init.normal_(self.query_projection.weight, mean=0, std=np.sqrt(2.0 / (self.model_dim + self.key_dim)))
        torch.nn.init.normal_(self.key_projection.weight, mean=0, std=np.sqrt(2.0 / (self.model_dim + self.key_dim)))
        torch.nn.init.normal_(self.value_projection.weight, mean=0, std=np.sqrt(2.0 / (self.model_dim + self.value_dim)))

        self.attention = ScaledDotProductAttention(temperature=np.power(self.key_dim, 0.5), return_attention=True)
        self.layer_norm = torch.nn.LayerNorm(self.model_dim)

        self.linear = torch.nn.Linear(self.n_head * self.value_dim, self.model_dim)
        torch.nn.init.xavier_normal_(self.linear.weight)

        self.dropout = torch.nn.Dropout(dropout)

        self.return_attention = return_attention

    def forward(self, query, key, value, mask=None):

        batch_size = query.size(0)

        sequence_length_query = query.size(1)
        sequence_length_key = key.size(1)
        sequence_length_value = value.size(1)

        residual = query

        query = self.query_projection(query).view(batch_size, sequence_length_query, self.n_head, self.key_dim)
        key = self.key_projection(key).view(batch_size, sequence_length_key, self.n_head, self.key_dim)
        value = self.value_projection(value).view(batch_size, sequence_length_value, self.n_head, self.value_dim)

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, sequence_length_query, self.key_dim)
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, sequence_length_key, self.key_dim)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, sequence_length_value, self.value_dim)

        mask = mask.repeat(self.n_head, 1, 1) if mask is not None else mask
        output, attention = self.attention(query, key, value, mask=mask)

        output = output.view(self.n_head, batch_size, sequence_length_query, self.value_dim)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, sequence_length_query, -1)

        output = self.dropout(self.linear(output))
        output = self.layer_norm(output + residual)

        if self.return_attention:
            return output, attention
        else:
            return output

    def extra_repr(self):

        report = [self.model_report, 'Heads: {}'.format(self.n_head)]

        return '\n'.join(report)


class AttentionPooling(BaseModule):

    def __init__(self, embedding_dim, sequence_length, base_pool_layer=torch.nn.MaxPool1d):

        super(AttentionPooling, self).__init__()

        self.attention = ScaledDotProductAttention(temperature=embedding_dim)

        self.pool_layer = base_pool_layer(kernel_size=sequence_length)

    def forward(self, x):

        x = self.attention(x, x, x)

        x = x.transpose(1, 2)

        x = self.pool_layer(x).squeeze()

        return x


# losses
class SplitCrossEntropyLoss(BaseModule):

    """
    SplitCrossEntropyLoss calculates an approximate softmax
    """

    def __init__(self, hidden_size, splits, verbose=False):

        # We assume splits is [0, split1, split2, N] where N >= |V|
        # For example, a vocab of 1000 words may have splits [0] + [100, 500] + [inf]

        super(SplitCrossEntropyLoss, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hidden_size = hidden_size
        self.splits = [0] + splits + [100 * 1000000]
        self.nsplits = len(self.splits) - 1
        self.stats = defaultdict(list)
        self.verbose = verbose

        # Each of the splits that aren't in the head require a pretend token, we'll call them tombstones
        # The probability given to this tombstone is the probability of selecting an item from the represented split
        if self.nsplits > 1:
            self.tail_vectors = torch.nn.Parameter(torch.zeros(self.nsplits - 1, hidden_size)).to(self.device)
            self.tail_bias = torch.nn.Parameter(torch.zeros(self.nsplits - 1)).to(self.device)

    def logprob(self, weight, bias, hiddens, splits=None, softmaxed_head_res=None):

        # First we perform the first softmax on the head vocabulary and the tombstones

        if softmaxed_head_res is None:

            start, end = self.splits[0], self.splits[1]
            head_weight = None if end - start == 0 else weight[start:end]
            head_bias = None if end - start == 0 else bias[start:end]

            # We only add the tombstones if we have more than one split
            if self.nsplits > 1:
                head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
                head_bias = self.tail_bias if head_bias is None else torch.cat([head_bias, self.tail_bias])

            # Perform the softmax calculation for the word vectors in the head for all splits
            # We need to guard against empty splits as torch.cat does not like random lists

            head_res = torch.nn.functional.linear(hiddens, head_weight, bias=head_bias)
            softmaxed_head_res = torch.nn.functional.log_softmax(head_res, dim=-1)

        if splits is None:
            splits = list(range(self.nsplits))

        results = []

        for idx in splits:

            # For those targets in the head (idx == 0) we only need to return their loss
            if idx == 0:
                results.append(softmaxed_head_res[:, :-(self.nsplits - 1)])

            # If the target is in one of the splits, the probability is the p(tombstone) * p(word within tombstone)
            else:
                start, end = self.splits[idx], self.splits[idx + 1]

                tail_weight = weight[start:end]
                tail_bias = bias[start:end]

                # Calculate the softmax for the words in the tombstone
                tail_res = torch.nn.functional.linear(hiddens, tail_weight, bias=tail_bias)

                # Then we calculate p(tombstone) * p(word in tombstone)
                # Adding is equivalent to multiplication in log space
                head_entropy = (softmaxed_head_res[:, -idx]).contiguous()
                tail_entropy = torch.nn.functional.log_softmax(tail_res, dim=-1)

                results.append(head_entropy.view(-1, 1) + tail_entropy)

        if len(results) > 1:
            return torch.cat(results, dim=1)

        return results[0]

    def split_on_targets(self, hiddens, targets):

        # Split the targets into those in the head and in the tail
        split_targets = []
        split_hiddens = []

        # Determine to which split each element belongs (for each start split value, add 1 if equal or greater)
        # This method appears slower at least for WT-103 values for approx softmax

        # This is equally fast for smaller splits as method below but scales linearly
        mask = None

        for idx in range(1, self.nsplits):
            partial_mask = targets >= self.splits[idx]
            mask = mask + partial_mask if mask is not None else partial_mask

        for idx in range(self.nsplits):

            # If there are no splits, avoid costly masked select
            if self.nsplits == 1:
                split_targets, split_hiddens = [targets], [hiddens]
                continue

            # If all the words are covered by earlier targets, we have empties so later stages don't freak out
            if sum(len(t) for t in split_targets) == len(targets):
                split_targets.append([])
                split_hiddens.append([])
                continue

            # Are you in our split?
            tmp_mask = mask == idx

            split_targets.append(targets.masked_select(tmp_mask))
            split_hiddens.append(hiddens.masked_select(
                tmp_mask.unsqueeze(1).expand_as(hiddens)).view(-1, hiddens.size(1)))

        return split_targets, split_hiddens

    def forward(self, weight, bias, hiddens, targets, verbose=False):

        if self.verbose or verbose:
            for idx in sorted(self.stats):
                print('{}: {}'.format(idx, int(np.mean(self.stats[idx]))), end=', ')
            print()

        total_loss = None

        if len(hiddens.size()) > 2:
            hiddens = hiddens.view(-1, hiddens.size(2))

        split_targets, split_hiddens = self.split_on_targets(hiddens, targets)

        # First we perform the first softmax on the head vocabulary and the tombstones
        start, end = self.splits[0], self.splits[1]

        head_weight = None if end - start == 0 else weight[start:end]
        head_bias = None if end - start == 0 else bias[start:end]

        # We only add the tombstones if we have more than one split
        if self.nsplits > 1:
            head_weight = self.tail_vectors if head_weight is None else torch.cat([head_weight, self.tail_vectors])
            head_bias = self.tail_bias if head_bias is None else torch.cat([head_bias, self.tail_bias])

        # Perform the softmax calculation for the word vectors in the head for all splits
        # We need to guard against empty splits as torch.cat does not like random lists
        combo = torch.cat([split_hiddens[i] for i in range(self.nsplits) if len(split_hiddens[i])])

        all_head_res = torch.nn.functional.linear(combo, head_weight, bias=head_bias)
        softmaxed_all_head_res = torch.nn.functional.log_softmax(all_head_res, dim=-1)

        if self.verbose or verbose:
            self.stats[0].append(combo.size()[0] * head_weight.size()[0])

        running_offset = 0

        for idx in range(self.nsplits):

            # If there are no targets for this split, continue
            if len(split_targets[idx]) == 0:
                continue

            # For those targets in the head (idx == 0) we only need to return their loss
            if idx == 0:
                softmaxed_head_res = softmaxed_all_head_res[running_offset:running_offset + len(split_hiddens[idx])]
                entropy = -torch.gather(softmaxed_head_res, dim=1, index=split_targets[idx].view(-1, 1))

            # If the target is in one of the splits, the probability is the p(tombstone) * p(word within tombstone)
            else:
                softmaxed_head_res = softmaxed_all_head_res[running_offset:running_offset + len(split_hiddens[idx])]

                if self.verbose or verbose:
                    start, end = self.splits[idx], self.splits[idx + 1]
                    tail_weight = weight[start:end]
                    self.stats[idx].append(split_hiddens[idx].size()[0] * tail_weight.size()[0])

                # Calculate the softmax for the words in the tombstone
                tail_res = self.logprob(weight, bias, split_hiddens[idx],
                                        splits=[idx], softmaxed_head_res=softmaxed_head_res)

                # Then we calculate p(tombstone) * p(word in tombstone)
                # Adding is equivalent to multiplication in log space
                head_entropy = softmaxed_head_res[:, -idx]

                # All indices are shifted - if the first split handles [0,...,499]
                # then the 500th in the second split will be 0 indexed
                indices = (split_targets[idx] - self.splits[idx]).view(-1, 1)

                # Warning: if you don't squeeze, you get an N x 1 return, which acts oddly with broadcasting
                tail_entropy = torch.gather(torch.nn.functional.log_softmax(tail_res, dim=-1),
                                            dim=1, index=indices).squeeze()

                entropy = -(head_entropy + tail_entropy)

            running_offset += len(split_hiddens[idx])

            total_loss = entropy.float().sum() if total_loss is None else total_loss + entropy.float().sum()

        return (total_loss / len(targets)).type_as(weight)


# optimizers
class ScheduledOptim:

    """
    A simple wrapper class for learning rate scheduling from transformer
    """

    def __init__(self, optimizer, model_dim, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(model_dim, -0.5)

    def step_and_update_lr(self):
        """
        Step with the inner optimizer
        :return:
        """
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """
        Zero out the gradients by the inner optimizer
        :return:
        """
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        """
        Learning rate scheduling per step
        :return:
        """

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
