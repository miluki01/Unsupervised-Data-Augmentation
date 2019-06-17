import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import List, Optional, Union, Generator, Any, Type, Tuple
import torch
from tools.data import Data, RankerSentencePieceData
import os
import shutil
from overrides import overrides


class GeneralWrapper:

    def __init__(self,
                 data: Type[Data],
                 optimizer: Any,
                 batch_size: Optional[int] = None,
                 total_accumulative_steps: int = 0,
                 model_name: str = 'model',
                 use_cuda_if_available: bool = True,
                 log_dir: str = './'):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if use_cuda_if_available else torch.device('cpu')

        self.data = data
        self.optimizer = optimizer

        self.total_accumulative_steps = total_accumulative_steps
        self.batch_size = batch_size if batch_size is not None else self.data.batch_size

        self.model_name = model_name
        self._log_dir = log_dir + self.model_name + '/' if log_dir[-1] == '/' else log_dir + '/' + self.model_name + '/'
        self._model_path = self.path_to_model()
        self._images_dir = self._log_dir + 'images/'
        self._model_dir = self._log_dir + 'model/'
        self._log_file = self._log_dir + 'log.txt'
        self.create_log_dir()

        self.epochs_passed = 0

        self.train_losses = []
        self.validation_losses = []

        self.moving_average_train_loss = None

        self.train_epoch_mean_losses = []

        self.best_metrics = []

        self.train_best_mean_loss = np.inf
        self.validation_best_mean_loss = np.inf

    def create_log_dir(self) -> None:

        shutil.rmtree(path=self._log_dir, ignore_errors=True)
        os.makedirs(self._images_dir, exist_ok=True)
        os.makedirs(self._model_dir, exist_ok=True)

    def path_to_model(self, postfix: str = '') -> str:

        if postfix:
            path = self._log_dir + self.model_name + '_' + postfix + '.pth'
        else:
            path = self._log_dir + self.model_name + '.pth'

        return path

    def plot(self,
             data: Any,
             title: Optional[str] = None,
             x_label: str = 'iter',
             y_label: str = 'loss',
             fig_size: Tuple[int] = (16, 14),
             save: bool = False):

        plt.figure(figsize=fig_size)
        plt.plot(data)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.ylim(bottom=0)
        plt.grid()

        if save:
            plt.savefig(self._images_dir + title + '.png')

    def batch_passing(self, batch):

        raise NotImplementedError

    def compute_loss(self, batch):

        raise NotImplementedError

    def prediction(self, batch):

        raise NotImplementedError

    def train(self, *args, **kwargs):

        raise NotImplementedError

    def validate(self, *args, **kwargs):

        raise NotImplementedError

    def save_model(self, *args, **kwargs):

        raise NotImplementedError


class OneModelWrapper(GeneralWrapper):

    def __init__(self,
                 model: Any,
                 loss: Any,
                 data: Type[Data],
                 optimizer: Any,
                 batch_size: Optional[int] = None,
                 total_accumulative_steps: int = 0,
                 alpha_moving_average: float = 0.9,
                 model_name: str = 'model',
                 log_dir: str = './'):

        super(OneModelWrapper, self).__init__(data=data,
                                              optimizer=optimizer,
                                              batch_size=batch_size,
                                              total_accumulative_steps=total_accumulative_steps,
                                              model_name=model_name,
                                              log_dir=log_dir)
        self.model = model
        self.loss = loss

        self.alpha_moving_average = alpha_moving_average

    @overrides
    def save_model(self, postfix: str = ''):

        # TODO add more info

        torch.save(obj=self.model, f=self.path_to_model(postfix=postfix))

    def _compute_moving_average_train_loss(self):

        if not self.train_losses:
            return 'undefined'

        if self.moving_average_train_loss is None:
            self.moving_average_train_loss = self.train_losses[-1]

        left_part = self.alpha_moving_average * self.moving_average_train_loss
        right_part = (1. - self.alpha_moving_average) * self.train_losses[-1]

        self.moving_average_train_loss = left_part + right_part

        return self.moving_average_train_loss

    @overrides
    def batch_passing(self, batch, training: bool = True):

        if training:

            if not self.model.is_training:
                self.model.unfreeze()

            self.optimizer.zero_grad()

            loss = self.compute_loss(batch)

            loss.backward()

            self.optimizer.step()

        else:

            if self.model.is_training:
                self.model.freeze()

            with torch.no_grad():

                loss = self.compute_loss(batch)

        loss = loss.item()

        return loss

    def epoch_loop(self,
                   training: bool = False,
                   batch_size: Optional[int] = None,
                   progress_bar: Optional[Any] = None,
                   verbose: bool = True):

        data_type = 'train' if training else 'validation'
        losses_data = self.__getattribute__('{}_losses'.format(data_type))

        batch_size = batch_size if batch_size is not None else self.batch_size

        if progress_bar is None:
            verbose = False

        batch_losses = []

        accumulative_step = 0
        samples_passed = 0
        loss = 0

        self.optimizer.zero_grad()

        if not self.model.is_training:
            self.model.unfreeze()

        for batch in self.data.batch_generator(data_type=data_type, batch_size=batch_size):

            # loss = self.batch_passing(batch=batch, training=training)

            loss += self.compute_loss(batch)

            if accumulative_step != self.total_accumulative_steps:
                accumulative_step += 1
                samples_passed += batch_size
                continue

            if self.total_accumulative_steps > 0:
                loss /= self.total_accumulative_steps

            loss.backward()

            self.optimizer.step()

            self.optimizer.zero_grad()

            loss = loss.item()

            batch_losses.append(loss)
            losses_data.append(loss)

            if verbose:

                progress_bar.update(samples_passed)

                postfix_dict = {
                    'loss': np.mean(batch_losses)
                }

                if training:
                    postfix_dict['moving_average'] = self._compute_moving_average_train_loss()

                progress_bar.set_postfix(**postfix_dict)

            accumulative_step = 0
            samples_passed = 0

        return batch_losses

    @property
    def extra_report(self):

        return ''

    @property
    def epoch_report(self):

        report = list()

        report.append(
            'Epoch: {} | Train Loss: {:.3f} | Validation Loss: {:.3f}'.format(
                self.epochs_passed,
                self.train_epoch_mean_losses[-1],
                self.validation_losses[-1]
            )
        )

        report.append(self.extra_report)

        report = '\n'.join(report)

        return report

    @overrides
    def train(self,
              epochs=5,
              batch_size=None,
              early_stopping=False,
              data_type_early_stopping='validation',
              save_best=True,
              verbose=True):

        stop_message = ''

        for n_epoch in range(1, epochs + 1):

            progress_bar = tqdm(total=len(self.data.train),
                                desc='Train Epoch {}'.format(self.epochs_passed + 1),
                                disable=not verbose)

            train_batch_losses = self.epoch_loop(training=True,
                                                 batch_size=batch_size,
                                                 progress_bar=progress_bar,
                                                 verbose=verbose)

            self.train_losses.extend(train_batch_losses)
            self.train_epoch_mean_losses.append(np.mean(train_batch_losses))

            progress_bar.close()

            progress_bar = tqdm(total=len(self.data.validation),
                                desc='Validation Epoch {}'.format(self.epochs_passed + 1),
                                disable=not verbose)

            validation_batch_losses = self.epoch_loop(training=False,
                                                      batch_size=batch_size,
                                                      progress_bar=progress_bar,
                                                      verbose=verbose)

            progress_bar.close()

            self.validation_losses.append(np.mean(validation_batch_losses))

            self.epochs_passed += 1

            self.save_model(postfix='last')

            if verbose:
                print(self.epoch_report)

            for data_type in ['train', 'validation']:

                current_loss = self.__getattribute__(f'{data_type}_losses')[-1]
                best_loss = self.__getattribute__(f'{data_type}_best_mean_loss')

                if current_loss < best_loss:

                    self.__setattr__(f'{data_type}_best_mean_loss', current_loss)

                    if data_type == data_type_early_stopping and early_stopping:
                        self.save_model(postfix='best')
                        stop_message = 'Early stopping'

            if stop_message:
                if verbose:
                    print(stop_message)

    @overrides
    def validate(self, batch_size: Optional[int] = None):

        validation_batch_losses = self.epoch_loop(training=False, batch_size=batch_size)

        return validation_batch_losses

    @overrides
    def compute_loss(self, batch):

        raise NotImplementedError

    @overrides
    def prediction(self, batch):

        raise NotImplementedError


class ClassificationWrapper(OneModelWrapper):

    def __init__(self,
                 model: Any,
                 loss: Any,
                 data: Type[Data],
                 optimizer: Any,
                 batch_size: Optional[int] = None,
                 total_accumulative_steps: int = 0,
                 alpha_moving_average: float = 0.9,
                 model_name: str = 'model',
                 log_dir: str = './'):

        super(ClassificationWrapper, self).__init__(model=model,
                                                    data=data,
                                                    optimizer=optimizer,
                                                    loss=loss,
                                                    batch_size=batch_size,
                                                    total_accumulative_steps=total_accumulative_steps,
                                                    alpha_moving_average=alpha_moving_average,
                                                    model_name=model_name,
                                                    log_dir=log_dir)

    def compute_loss(self, batch):

        x, y = batch

        prediction = self.model(x)

        target = torch.Tensor(y).unsqueeze(1).to(self.device)

        loss = self.loss(prediction, target)

        return loss

    def prediction(self, batch):

        prediction = self.model(batch)

        return prediction


class TripletRankingWrapper(OneModelWrapper):

    def __init__(self,
                 model: Any,
                 data: Type[RankerSentencePieceData],
                 optimizer: Any,
                 loss: Any = torch.nn.TripletMarginLoss(),
                 batch_size: Optional[int] = None,
                 generate_negatives_type: str = 'random',
                 use_hard_negatives: bool = False,
                 hard_negatives_batch_multiplier: int = 8,
                 alpha_moving_average: float = 0.9,
                 model_name: str = 'model',
                 log_dir: str = './'):

        super(TripletRankingWrapper, self).__init__(model=model,
                                                    data=data,
                                                    optimizer=optimizer,
                                                    loss=loss,
                                                    batch_size=batch_size,
                                                    alpha_moving_average=alpha_moving_average,
                                                    model_name=model_name,
                                                    log_dir=log_dir)

        self.generate_negatives_type = generate_negatives_type

        self.use_hard_negatives = use_hard_negatives
        self.hard_negatives_batch_multiplier = hard_negatives_batch_multiplier

    def encoding(self, batch, training=False):

        # if training and not self.model.is_training:
        #     self.model.unfreeze()
        # elif not training and self.model.is_training:
        #     self.model.freeze()

        batch = torch.LongTensor(batch).to(self.device)

        encoded_batch = self.model(batch)

        return encoded_batch

    @staticmethod
    def batch_cosine_similarity(batch_1, batch_2):

        similarity = torch.matmul(batch_1, batch_2.t())

        query_norms = torch.matmul(batch_1, batch_1.t()).pow(0.5).diag().unsqueeze(1)
        negatives_norms = torch.matmul(batch_2, batch_2.t()).pow(0.5).diag().unsqueeze(1)

        norms = torch.matmul(query_norms, negatives_norms.t())

        similarity = torch.div(similarity, norms)

        return similarity

    def generate_hard_negative(self, query_vectorized, negatives_candidate):

        negatives_candidate_vectorized = self.encoding(negatives_candidate, training=False)

        similarity = self.batch_cosine_similarity(batch_1=query_vectorized, batch_2=negatives_candidate_vectorized)

        negatives_candidate = negatives_candidate[similarity.argmax(dim=1)]

        negatives_candidate_vectorized = self.encoding(negatives_candidate, training=True)

        return negatives_candidate_vectorized

    def generate_negative(self, batch):

        k_samples = len(batch[0])

        if self.use_hard_negatives:
            k_samples *= self.hard_negatives_batch_multiplier

        negatives_candidate = self.data.get_random_batch(k_samples=k_samples)

        return negatives_candidate

    def encoding_negative_batch(self, negatives_candidate, query_vectorized):

        if self.use_hard_negatives:
            output = self.generate_hard_negative(query_vectorized=query_vectorized,
                                                 negatives_candidate=negatives_candidate)
        else:
            output = self.encoding(batch=negatives_candidate)

        return output

    @overrides
    def compute_loss(self, batch):

        query_vectorized = self.encoding(batch[0], training=True)
        positive_candidate_vectorized = self.encoding(batch[1], training=True)

        negative_candidate = self.generate_negative(batch=batch)

        negative_candidate_vectorized = self.encoding_negative_batch(negatives_candidate=negative_candidate,
                                                                     query_vectorized=query_vectorized)

        loss = self.loss(query_vectorized, positive_candidate_vectorized, negative_candidate_vectorized)

        return loss

    @overrides
    def prediction(self, batch):

        output = self.encoding(batch=batch, training=False)

        output = output.detach().cpu().numpy()

        return output


class MailRankerWrapper(TripletRankingWrapper):

    def __init__(self,
                 model: Any,
                 data: Type[RankerSentencePieceData],
                 optimizer: Any,
                 loss: Any = torch.nn.TripletMarginLoss(),
                 batch_size: Optional[int] = None,
                 negative_same_category: bool = False,
                 generate_negatives_type: str = 'random',
                 use_hard_negatives: bool = False,
                 hard_negatives_batch_multiplier: int = 8,
                 alpha_moving_average: float = 0.9,
                 model_name: str = 'model',
                 log_dir: str = './'):

        super(MailRankerWrapper, self).__init__(model=model,
                                                data=data,
                                                optimizer=optimizer,
                                                loss=loss,
                                                batch_size=batch_size,
                                                generate_negatives_type=generate_negatives_type,
                                                use_hard_negatives=use_hard_negatives,
                                                hard_negatives_batch_multiplier=hard_negatives_batch_multiplier,
                                                alpha_moving_average=alpha_moving_average,
                                                model_name=model_name,
                                                log_dir=log_dir)

        self.negative_same_category = negative_same_category

    # def compute_loss(self, query, candidate, category, is_train_batch=True):
    #
    #     self.optimizer.zero_grad()
    #
    #     query = self.encoding(batch=query)
    #
    #     samples_k = query.shape[0]
    #     sample_k_exact_match = True
    #
    #     if self.is_hard_negatives:
    #         sample_k_exact_match = False
    #         samples_k *= self.hard_negatives_batch_multiplier
    #
    #     if self.negatives_from_another_category:
    #         negative_candidate = self.dataset.get_batch_from_another_category_ids(
    #             category_ids=category,
    #             samples_k=samples_k,
    #              sample_k_exact_match=sample_k_exact_match)
    #     else:
    #         negative_candidate = self.dataset.get_random_batch(samples_k=samples_k)
    #
    #     if not is_train_batch:
    #         positive_candidate = self.encoding(batch=candidate)
    #         negative_candidate = self.encoding(batch=negative_candidate)
    #     else:
    #         with torch.no_grad():
    #             positive_candidate = self.encoding(batch=candidate)
    #             negative_candidate = self.encoding(batch=negative_candidate)
    #
    #     if self.is_hard_negatives:
    #
    #         similarity = torch.matmul(query, negative_candidate.t())
    #
    #         query_norms = torch.matmul(query, query.t()).pow(0.5).diag().unsqueeze(1)
    #         negatives_norms = torch.matmul(negative_candidate, negative_candidate.t()).pow(0.5).diag().unsqueeze(1)
    #
    #         norms = torch.matmul(query_norms, negatives_norms.t())
    #
    #         similarity = similarity / norms
    #
    #         negative_candidate = negative_candidate[similarity.argmax(dim=1)]
    #
    #     loss = self.loss(query, positive_candidate, negative_candidate)
    #
    #     return loss

    @overrides
    def generate_negative(self, batch):

        query, _, category = batch

        # n_samples = len(query)
        # k_samples_by_category = 1
        #
        # if self.use_hard_negatives:
        #     n_samples *= self.hard_negatives_batch_multiplier
        #     k_samples_by_category *= self.hard_negatives_batch_multiplier

        if self.negative_same_category:
            negatives_candidates = self.data.get_random_batch_by_category_batch(category_ids=category)
        else:
            negatives_candidates = self.data.get_random_batch_by_random_categories(category_ids=category)

        return negatives_candidates
