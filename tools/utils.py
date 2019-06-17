import numpy as np
import logging
from tqdm import tqdm
import io
import copy
import os
import torch
from typing import List, Optional, Union, Generator, Any
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import sentencepiece as sp
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.tokenize import word_tokenize, wordpunct_tokenize


class Perplexity:

    def __init__(self):

        self.losses = []

    def update(self, loss, is_batch=False):

        if is_batch:
            self.losses.extend(loss)
        else:
            self.losses.append(loss)

    @property
    def score(self):

        score = self.calculate(losses=self.losses)

        return score

    def reset(self):

        self.losses = []

    def calculate(self, losses):

        score = np.log(np.sum(losses) / len(losses))

        return score


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """

    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):

        super(TqdmToLogger, self).__init__()

        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):

        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


def count_lines(filename, chunk_size=1 << 13):

    with open(filename) as file:

        return sum(chunk.count('\n') for chunk in iter(lambda: file.read(chunk_size), ''))


class LRFinder:
    """Learning rate range test.
    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.
    Arguments:
        model (torch.nn.Module): wrapped model.
        optimizer (torch.optim.Optimizer): wrapped optimizer where the defined learning
            is assumed to be the lower boundary of the range test.
        criterion (torch.nn.Module): wrapped loss function.
        device (str or torch.device, optional): a string ("cpu" or "cuda") with an
            optional ordinal for the device type (e.g. "cuda:X", where is the ordinal).
            Alternatively, can be an object representing the device on which the
            computation will take place. Default: None, uses the same device as `model`.
        memory_cache (boolean): if this flag is set to True, `state_dict` of model and
            optimizer will be cached in memory. Otherwise, they will be saved to files
            under the `cache_dir`.
        cache_dir (string): path for storing temporary files. If no path is specified,
            system-wide temporary directory is used.
            Notice that this parameter will be ignored if `memory_cache` is True.
    Example:
        >>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
        >>> lr_finder.range_test(dataloader, end_lr=100, num_iter=100)
    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    fastai/lr_find: https://github.com/fastai/fastai
    """

    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 device=None,
                 memory_cache=True,
                 cache_dir=None):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.history = {"lr": [], "loss": []}
        self.best_lr = None
        self.best_loss = None
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir

        # Save the original state of the model and optimizer so they can be restored if
        # needed
        self.model_device = next(self.model.parameters()).device
        self.state_cacher = StateCacher(memory_cache, cache_dir=cache_dir)
        self.state_cacher.store('model', self.model.state_dict())
        self.state_cacher.store('optimizer', self.optimizer.state_dict())

        # If device is None, use the same as the model
        if device:
            self.device = device
        else:
            self.device = self.model_device

    def reset(self):
        """Restores the model and optimizer to their initial states."""
        self.model.load_state_dict(self.state_cacher.retrieve('model'))
        self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer'))
        self.model.to(self.model_device)

    def range_test(
            self,
            wrapper,
            use_validation_loss=False,
            end_lr=0.05,
            num_iter=100,
            step_mode="exp",
            smooth_f=0.05,
            diverge_th=5,
    ):
        """Performs the learning rate range test.
        Arguments:
            wrapper () some wrapper
            use_validation_loss (bool) use or not to use
            # train_loader (torch.utils.data.DataLoader): the training set data laoder.
            # val_loader (torch.utils.data.DataLoader, optional): if `None` the range test
            #     will only use the training loss. When given a data loader, the model is
            #     evaluated after each iteration on that dataset and the evaluation loss
            #     is used. Note that in this mode the test takes significantly longer but
            #     generally produces more precise results. Default: None.
            end_lr (float, optional): the maximum learning rate to test. Default: 10.
            num_iter (int, optional): the number of iterations over which the test
                occurs. Default: 100.
            step_mode (str, optional): one of the available learning rate policies,
                linear or exponential ("linear", "exp"). Default: "exp".
            smooth_f (float, optional): the loss smoothing factor within the [0, 1[
                interval. Disabled if set to 0, otherwise the loss is smoothed using
                exponential smoothing. Default: 0.05.
            diverge_th (int, optional): the test is stopped when the loss surpasses the
                threshold:  diverge_th * best_loss. Default: 5.
        """
        # Reset test results
        self.history = {"lr": [], "loss": []}
        self.best_lr = None
        self.best_loss = None

        # Move the model to the proper device
        self.model.to(self.device)

        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1[")

        iterator = iter(wrapper.data)
        for iteration in tqdm(range(num_iter)):

            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(wrapper.data)
                batch = next(iterator)

            loss = wrapper.batch_passing(batch=batch, training=True)

            if use_validation_loss:
                loss = np.mean([wrapper.batch_passing(batch=batch, training=False) for batch in wrapper.data])

            # Update the learning rate
            lr_schedule.step()
            self.history["lr"].append(lr_schedule.get_lr()[0])

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                self.best_loss = loss
                self.best_lr = self.history['lr'][-1]
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history["loss"][-1]
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_lr = self.history['lr'][-1]

            # Check if the loss has diverged; if it has, stop the test
            self.history["loss"].append(loss)
            if loss > diverge_th * self.best_loss:
                print("Stopping early, the loss has diverged")
                break

    def plot(self, skip_start=10, skip_end=5, log_lr=True):
        """Plots the learning rate range test.
        Arguments:
            skip_start (int, optional): number of batches to trim from the start.
                Default: 10.
            skip_end (int, optional): number of batches to trim from the start.
                Default: 5.
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
        """

        if skip_start < 0:
            raise ValueError("skip_start cannot be negative")
        if skip_end < 0:
            raise ValueError("skip_end cannot be negative")

        # Get the data to plot from the history dictionary. Also, handle skip_end=0
        # properly so the behaviour is the expected
        lrs = self.history["lr"]
        losses = self.history["loss"]
        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        # Plot loss as a function of the learning rate
        plt.plot(lrs, losses)
        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.show()


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float, optional): the initial learning rate which is the lower
            boundary of the test. Default: 10.
        num_iter (int, optional): the number of iterations over which the test
            occurs. Default: 100.
        last_epoch (int): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class StateCacher(object):

    def __init__(self, in_memory, cache_dir=None):

        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile
            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError('Given `cache_dir` is not a valid directory.')

        self.cached = {}

    def store(self, key, state_dict):
        
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, 'state_{}_{}.pt'.format(key, id(self)))
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):

        if key not in self.cached:
            raise KeyError('Target {} was not cached.'.format(key))

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError('Failed to load state in {}. File does not exist anymore.'.format(fn))
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in `cache_dir` before
        this instance being destroyed."""
        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])


class SentenceEncoder:

    SYMBOLS = ')(=+%$@#!&)№\"\'?.,]+'

    SYMBOLS_MAPPER = {
        '`': '"',
        "''": '"'
    }

    def __init__(self, model, state_dict_path, tokenizer_path, max_sequence_length=64):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self.model.load_state_dict(torch.load(state_dict_path))
        self.model.freeze()

        self.tokenizer_path = tokenizer_path
        self.tokenizer = sp.SentencePieceProcessor()
        self.tokenizer.load(filename=self.tokenizer_path)

        self.max_sequence_length = max_sequence_length

    @staticmethod
    def _sequence_padding(sequence: Union[List[Union[int, str]], List[np.ndarray], np.ndarray],
                          max_sequence_length: int,
                          value: Union[int, str]) -> np.ndarray:

        sequence = sequence[:max_sequence_length]

        if len(sequence) < max_sequence_length:
            for _ in range((max_sequence_length - len(sequence))):
                sequence.append(value)

        sequence = np.array(sequence)

        return sequence

    def padding(self, sequence: Union[List[int], np.ndarray]) -> np.ndarray:

        sequence = self._sequence_padding(sequence=sequence,
                                          max_sequence_length=self.max_sequence_length,
                                          value=self.tokenizer.pad_id())

        sequence = np.array(sequence)

        return sequence

    def __call__(self, text):

        text = ' '.join(self.prepare(text=text))

        ids_text = self.tokenizer.encode_as_ids(text)

        ids_text = np.array([self.padding(sequence=ids_text)])

        ids_text = torch.LongTensor(ids_text).to(self.device)

        text_vector = self.model(ids_text)[0].detach().cpu().numpy()

        return text_vector

    def compare_texts(self, text_1, text_2):

        text1_vector = self.__call__(text=text_1)
        text2_vector = self.__call__(text=text_2)

        similarity = cosine_similarity(text1_vector.reshape(1, -1), text2_vector.reshape(1, -1))[0][0]

        return similarity

    def prepare(self, text):

        text = text.lower()

        text = re.sub(r'http\S+|www.\S+|^www.\/\/.*[\r\n]*|^http?:\/\/.*[\r\n]*|^https?:\/\/.*[\r\n]*',
                      ' ссылка ',
                      text,
                      flags=re.MULTILINE)

        text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9.)(=+%$@#!&)№\"?.,]+", r" ", text)

        text = text.replace('ё', 'e')

        text = re.sub('[0-9]{5,}', '^ ', text)

        tokens = wordpunct_tokenize(text)

        if tokens and tokens[-1] == '.':
            tokens = tokens[:-1]

        correct_tokens = []

        for i in range(len(tokens)):

            tmp_correct_token = ''

            last_char = ''

            for n in range(len(tokens[i])):

                if tokens[i][n].isalpha():
                    tmp_correct_token += tokens[i][n]
                else:
                    if tokens[i][n] != last_char:
                        tmp_correct_token += tokens[i][n]
                        last_char = tokens[i][n]

            if correct_tokens and tmp_correct_token == list(set(correct_tokens[-1]))[0]:
                correct_tokens[-1] = correct_tokens[-1] + tmp_correct_token
            else:
                correct_tokens.append(self.SYMBOLS_MAPPER.get(tmp_correct_token, tmp_correct_token))

        return correct_tokens
