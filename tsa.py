import numpy as np
import torch
from modelling.layers import BaseModule


class TrainingSignalAnnealing:

    def __init__(self, total_steps: int, n_classes: int, current_step: int = 0, schedule_type: str = 'linear'):

        self.total_steps = total_steps
        self.n_classes = n_classes
        self.n_classes_smoother = 1 / self.n_classes

        self.schedule_type = schedule_type
        self.schedule_function = None
        self.__check_schedule_type()

        self.current_step = current_step

    def __check_schedule_type(self):

        if self.schedule_type == 'linear':
            self.schedule_function = self.linear_scaling
        elif self.schedule_type == 'log':
            self.schedule_function = self.log_scaling
        elif self.schedule_type == 'exp':
            self.schedule_function = self.exp_scaling
        else:
            raise ValueError('Available scheduling types: linear, log and exp')

    def linear_scaling(self, t: int):

        return t / self.total_steps * (1 - self.n_classes_smoother) + self.n_classes_smoother

    def log_scaling(self, t: int):

        return (1 - np.exp(- 5 * t / self.total_steps)) * (1 - self.n_classes_smoother) + self.n_classes_smoother

    def exp_scaling(self, t: int):

        return np.exp((t / self.total_steps - 1) * 5) * (1 - self.n_classes_smoother) + self.n_classes_smoother

    def __iter__(self):

        for n in range(self.total_steps):

            self.current_step = n

            yield self.calculate_step(self.current_step)

    def update(self, n: int = 1):

        if n <= 0:
            raise ValueError('n must be greater than zero')

        for _ in range(n):

            prob = self.calculate_step(self.current_step)

            self.current_step += 1

        return prob

    def calculate_step(self, step: int):

        if step >= self.total_steps:
            step = self.total_steps

        return self.schedule_function(step)

    def restart(self):

        self.current_step = 0

    @property
    def current_prob(self):

        return self.calculate_step(step=self.current_step)


class TSAScheduleCrossEntropy(BaseModule):

    def __init__(self, total_steps: int, n_classes: int, current_step: int = 0, schedule_type: str = 'linear'):

        super().__init__()

        self.tsa = TrainingSignalAnnealing(total_steps=total_steps, n_classes=n_classes,
                                           current_step=current_step, schedule_type=schedule_type)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, predictions, targets):

        current_max_prob = self.tsa.update()

        activated = self.activation(predictions)

        indexes = activated[torch.arange(predictions.size(0)), targets] < current_max_prob

        predictions = predictions[indexes]
        targets = targets[indexes]

        return predictions, targets


class TSAScheduleBinaryCrossEntropy(BaseModule):

    def __init__(self, total_steps: int, n_classes: int, current_step: int = 0, schedule_type: str = 'linear'):

        super().__init__()

        self.tsa = TrainingSignalAnnealing(total_steps=total_steps, n_classes=n_classes,
                                           current_step=current_step, schedule_type=schedule_type)

        self.criterion = torch.nn.BCELoss()
        self.activation = torch.nn.Sigmoid()

    def forward(self, predictions, targets):

        current_max_prob = self.tsa.update()

        activated = self.activation(predictions)

        activated = torch.cat([1 - activated, activated], dim=1)

        indexes = activated[torch.arange(predictions.size(0)), targets.squeeze().long()] < current_max_prob

        predictions = predictions[indexes]
        targets = targets[indexes]

        return predictions, targets
