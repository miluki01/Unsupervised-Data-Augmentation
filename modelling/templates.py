import torch
import torch.nn as nn
from modelling.layers import BaseModule


class SequenceTemplate(BaseModule):

    def __init__(self, *models):

        super(SequenceTemplate, self).__init__()

        self.model = torch.nn.Sequential(*models).to(self.device)

    def forward(self, x):

        x = self.model(x)

        return x


class EmbeddingWithPaddingTemplate(BaseModule):

    def __init__(self, embedding, model):

        super(EmbeddingWithPaddingTemplate, self).__init__()

        self.embedding = embedding.to(self.device)
        self.model = model.to(self.device)

    def forward(self, x, padding_mask=None):

        x = self.embedding(x, padding_mask)

        x = self.model(x)

        return x


class RNNLanguageModelTemplate(BaseModule):

    def __init__(self,
                 embedding_layer,
                 rnn_model,
                 token_prediction_model,
                 loss=torch.nn.NLLLoss(ignore_index=0),
                 teacher_forcing_ratio=0.5,
                 teacher_forcing_inference_ratio=0.):

        super(RNNLanguageModelTemplate, self).__init__()

        self.embedding_layer = embedding_layer.to(self.device)
        self.rnn_model = rnn_model.to(self.device)
        self.token_prediction_model = token_prediction_model.to(self.device)
        self.loss = loss.to(self.device)

        if hasattr(self.rnn_model, 'output_need_states'):
            setattr(self.rnn_model, 'output_need_states', True)

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_inference_ratio = teacher_forcing_inference_ratio

    def forward(self, x):

        states = self.rnn_model.init_states(batch_size=x.size(0))

        current_step = x[:, 0, ].unsqueeze(1)

        loss = 0

        for n_step in range(1, x.size(1) - 1):

            current_step = self.embedding_layer(current_step)

            prediction, states = self.rnn_model(current_step, states)

            prediction = prediction.squeeze()

            token_prediction = self.token_prediction_model(prediction)

            next_step = x[:, n_step+1, ]
            loss += self.loss(token_prediction, next_step)

            if torch.rand(1) > self.teacher_forcing_ratio and self.training:
                current_step = token_prediction.argmax(dim=1).unsqueeze(1)
            else:
                current_step = x[:, n_step, ].unsqueeze(1)

        return loss

    def inference(self, x):

        states = self.rnn_model.init_states(batch_size=x.size(0))

        rnn_outputs = []

        current_step = x[:, 0, ].unsqueeze(1)

        for n_step in range(x.size(1)):

            embedded_current_step = self.embedding_layer(current_step)

            prediction, states = self.rnn_model(embedded_current_step, states)

            rnn_outputs.append(prediction)

            prediction = prediction.squeeze()

            token_prediction = self.token_prediction_model(prediction)

            if torch.rand(1) > self.teacher_forcing_inference_ratio \
                    and self.training and self.teacher_forcing_inference_ratio:
                current_step = token_prediction.argmax(dim=1).unsqueeze(1)
            else:
                current_step = x[:, n_step, ].unsqueeze(1)

        return rnn_outputs


class TransferLearningTemplate(BaseModule):

    def __init__(self,
                 pre_trained_model,
                 training_model,
                 pre_trained_freeze=True):

        super(TransferLearningTemplate, self).__init__()

        self.pre_trained_model = pre_trained_model.to(self.device)
        self.training_model = training_model.to(self.device)

        self.pre_trained_freeze = pre_trained_freeze

        if self.pre_trained_freeze:
            self.pre_trained_model.freeze()

    def forward(self, x):

        x = self.pre_trained_model(x)
        x = self.training_model(x)

        return x
