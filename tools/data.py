import random
import numpy as np
from tqdm import tqdm
import math
import json
from typing import List, Optional, Union, Generator, Any, Dict
import sentencepiece as sp
from overrides import overrides
from tools.utils import count_lines


class Data:

    def __init__(self,
                 batch_size: int = 32,
                 max_sequence_length: int = 64,
                 validation_split_ratio: float = 0.05,
                 shuffle: bool = True,
                 drop_first_line: bool = False,
                 verbose: bool = True):

        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.validation_split_ratio = validation_split_ratio
        self.shuffle = shuffle
        self.drop_first_line = drop_first_line
        self.verbose = verbose

        self.train = []
        self.validation = []

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

    def batch_padding(self, batch: Union[List[List[Any]], List[np.ndarray], np.ndarray]) -> np.ndarray:

        batch = [self.padding(sequence=sample) for sample in batch]

        batch = np.array(batch)

        return batch

    def load(self,
             file_path: str,
             verbose: bool = False) -> None:

        try:
            verbose = self.__getattribute__('verbose')
        except AttributeError:
            pass

        total = count_lines(filename=file_path)

        data = []

        n = 0

        progress_bar = tqdm(total=total, desc='Loading', disable=not verbose)

        with open(file_path) as f:

            while True:

                line = f.readline().strip()

                if n == 0 and self.drop_first_line:
                    n += 1
                    continue

                if not line or (total is not None and n >= total):
                    break

                n += 1

                processed_line = self.line_processing(line=line)

                if processed_line:
                    data.append(processed_line)

                if verbose:
                    progress_bar.update()

        if verbose:
            progress_bar.close()

        if self.shuffle:
            random.shuffle(data)

        validation_start = len(data) - int(len(data) * self.validation_split_ratio)

        self.train = data[:validation_start]
        self.validation = data[validation_start:]

    def batch_generator(self,
                        data_type: str = 'train',
                        batch_size: Optional[int] = None) -> Generator[Any, None, None]:

        data = self.__getattribute__(data_type)
        batch_size = batch_size if batch_size is not None else self.batch_size

        for n_batch in range(math.ceil(len(data) / batch_size)):

            batch = data[n_batch * batch_size:(n_batch + 1) * batch_size]

            batch = self.batch_processing(batch)

            yield batch

    def __iter__(self):

        return self.batch_generator()

    @staticmethod
    def read_json_line(line: str):

        line = line.strip()

        data = json.loads(line)

        return data

    def line_processing(self, line: str) -> Any:

        raise NotImplementedError

    def padding(self, sequence: Union[List[Any], np.ndarray]) -> np.ndarray:

        raise NotImplementedError

    def batch_processing(self, batch: Any) -> Any:

        raise NotImplementedError


# class VocabularyData(Data):
#
#     def __init__(self,
#                  token2index: Optional[Dict[str, int]] = None,
#                  target2index: Optional[Dict[str, int]] = None,
#                  batch_size: int = 32,
#                  max_sequence_length: int = 64,
#                  validation_split_ratio: float = 0.01,
#                  shuffle: bool = True,
#                  verbose: bool = True):
#
#         super().__init__(batch_size=batch_size,
#                          max_sequence_length=max_sequence_length,
#                          validation_split_ratio=validation_split_ratio,
#                          shuffle=shuffle,
#                          verbose=verbose)
#
#         self.token2index = token2index if token2index is not None else {}
#         self.index2token = {value: key for key, value in self.token2index.items()}
#         self.collect_vocab = False if token2index else True
#
#         self.target2index = target2index if target2index is not None else {}
#         self.index2target = {value: key for key, value in self.target2index.items()}
#         self.collect_target_vocab = False if target2index else True
#
#     def line_processing(self, line: str) -> Any:
#
#         raise NotImplementedError
#
#     def padding(self, sequence: Union[List[Any], np.ndarray]) -> np.ndarray:
#
#         raise NotImplementedError
#
#     def batch_processing(self, batch: Any) -> Any:
#
#         raise NotImplementedError


class SentencePieceData(Data):

    def __init__(self,
                 tokenizer_path: str,
                 batch_size: int = 32,
                 max_sequence_length: int = 64,
                 validation_split_ratio: float = 0.01,
                 shuffle: bool = True,
                 verbose: bool = True):

        super(SentencePieceData, self).__init__(batch_size=batch_size,
                                                max_sequence_length=max_sequence_length,
                                                validation_split_ratio=validation_split_ratio,
                                                shuffle=shuffle,
                                                verbose=verbose)

        self.tokenizer_path = tokenizer_path
        self.tokenizer = sp.SentencePieceProcessor()
        self.tokenizer.load(filename=self.tokenizer_path)

        self.train = []
        self.validation = []

    @property
    def vocab_size(self) -> int:

        return self.tokenizer.get_piece_size()

    @overrides
    def padding(self, sequence: Union[List[int], np.ndarray]) -> np.ndarray:

        sequence = self._sequence_padding(sequence=sequence,
                                          max_sequence_length=self.max_sequence_length,
                                          value=self.tokenizer.pad_id())

        return sequence

    def decode_sample(self, sample: Union[List[int], np.ndarray]) -> str:

        sample = [int(index) for index in sample]

        sample = self.tokenizer.decode_ids(input=sample)

        return sample

    def decode_batch(self, batch: Union[List[np.ndarray], List[int], np.ndarray]) -> List[str]:

        batch = [self.decode_sample(sample=sample) for sample in batch]

        return batch

    @overrides
    def line_processing(self, line: str) -> Any:

        raise NotImplementedError

    @overrides
    def batch_processing(self, batch: Any) -> Any:

        raise NotImplementedError


class LanguageModelSentencePieceData(SentencePieceData):

    def __init__(self,
                 tokenizer_path: str,
                 batch_size: int = 32,
                 max_sequence_length: int = 256,
                 validation_split_ratio: float = 0.01,
                 shuffle: bool = True,
                 verbose: bool = True):

        super(LanguageModelSentencePieceData, self).__init__(tokenizer_path=tokenizer_path,
                                                             batch_size=batch_size,
                                                             max_sequence_length=max_sequence_length,
                                                             validation_split_ratio=validation_split_ratio,
                                                             shuffle=shuffle,
                                                             verbose=verbose)

    @overrides
    def line_processing(self, line: str) -> List[int]:

        return self.tokenizer.encode_as_ids(input=line)

    @overrides
    def batch_processing(self, batch: Any) -> np.ndarray:

        batch = self.batch_padding(batch=batch)

        return batch


class RankerSentencePieceData(SentencePieceData):

    def __init__(self,
                 tokenizer_path: str,
                 id2indexed_path: str,
                 batch_size: int = 32,
                 max_sequence_length: int = 64,
                 validation_split_ratio: float = 0.01,
                 shuffle: bool = True,
                 verbose: bool = True):

        super(RankerSentencePieceData, self).__init__(tokenizer_path=tokenizer_path,
                                                      batch_size=batch_size,
                                                      max_sequence_length=max_sequence_length,
                                                      validation_split_ratio=validation_split_ratio,
                                                      shuffle=shuffle,
                                                      verbose=verbose)

        self.id2indexed = {}
        self.ids = []
        self.collect_id2indexed(file=id2indexed_path)

    def collect_id2indexed(self, file):

        self.id2indexed = {}

        progress_bar = tqdm(desc='Collect id2indexed', total=count_lines(file), disable=not self.verbose)

        with open(file=file, mode='r') as f:

            n = 0

            while True:

                line = f.readline().strip()

                if not line:
                    break

                sample = json.loads(line)

                self.id2indexed[n] = sample

                n += 1
                progress_bar.update()

        self.ids = list(range(len(self.id2indexed)))

        progress_bar.close()

    @overrides
    def line_processing(self, line: str) -> List[int]:

        return [int(sample) for sample in line.split(',')]

    def ids2processed(self, batch: Union[np.ndarray, List[int]]) -> np.ndarray:

        batch = [self.id2indexed[sample] for sample in batch]
        batch = self.batch_padding(batch=batch)

        return batch

    @overrides
    def batch_processing(self, batch: Any) -> List[np.ndarray]:

        query_batch = [sample[0] for sample in batch]
        candidate_batch = [sample[1] for sample in batch]

        query_batch = self.ids2processed(batch=query_batch)
        candidate_batch = self.ids2processed(batch=candidate_batch)

        return [np.array(query_batch), np.array(candidate_batch)]

    def get_random_batch(self, k_samples: int = 2048):

        random_ids = random.sample(population=self.ids, k=k_samples)

        random_batch = self.ids2processed(batch=random_ids)

        return random_batch


class MailRankerSentencePieceData(RankerSentencePieceData):

    def __init__(self,
                 tokenizer_path: str,
                 id2indexed_path: str,
                 batch_size: int = 32,
                 max_sequence_length: int = 64,
                 validation_split_ratio: float = 0.01,
                 shuffle: bool = True,
                 verbose: bool = True):

        super(MailRankerSentencePieceData, self).__init__(tokenizer_path=tokenizer_path,
                                                          id2indexed_path=id2indexed_path,
                                                          batch_size=batch_size,
                                                          max_sequence_length=max_sequence_length,
                                                          validation_split_ratio=validation_split_ratio,
                                                          shuffle=shuffle,
                                                          verbose=verbose)

        self.category2id = {}
        self.category_id2text_ids = {}
        self.category_ids = []
        self._category2id_collection_status = True

    def collect_category2id(self, file: str) -> None:

        with open(file=file, mode='r') as f:
            categories_data = json.loads(f.read())

        for n, category in enumerate(categories_data):

            self.category2id[category] = n
            self.category_ids.append(n)

        self._category2id_collection_status = False

    def set_category2id(self, category):

        if self._category2id_collection_status:
            if category in self.category2id:
                index = self.category2id[category]
            else:
                index = len(self.category2id)
                self.category2id[category] = index
                self.category_ids.append(index)
        else:
            index = self.category2id.get(category)

        if index is None:
            return index

        if not self.category_id2text_ids.get(index):
            self.category_id2text_ids[index] = list()

        return index

    @overrides
    def line_processing(self, line: str) -> Optional[List[int]]:

        line = line.strip().split(',')

        query = line[0]
        candidate = line[1]
        category = line[2]

        query_id = int(query)
        candidate_id = int(candidate)

        category_id = self.set_category2id(category)

        if category_id is None:
            return category_id

        ids = [query_id, candidate_id]

        self.category_id2text_ids[category_id].extend(ids)

        ids.append(category_id)

        return ids

    @overrides
    def batch_processing(self, batch: Any) -> List[np.ndarray]:

        query_batch = [sample[0] for sample in batch]
        candidate_batch = [sample[1] for sample in batch]

        query_batch = self.ids2processed(batch=query_batch)
        candidate_batch = self.ids2processed(batch=candidate_batch)

        category_batch = [sample[2] for sample in batch]

        output_batch = [query_batch, candidate_batch, category_batch]

        output_batch = [np.array(part) for part in output_batch]

        return output_batch

    def get_random_batch_by_category(self, category_id, k_samples: int = 2048):

        random_ids = random.sample(population=self.category_id2text_ids[category_id], k=k_samples)

        random_batch = self.ids2processed(batch=random_ids)

        return random_batch

    def get_random_batch_by_category_batch(self, category_ids):

        random_ids = []

        for cat_id in category_ids:
            current_data = self.category_id2text_ids[cat_id]
            current_random_id = random.choice(current_data)
            random_ids.append(current_random_id)

        random_batch = self.ids2processed(batch=random_ids)

        return random_batch

    def get_random_batch_by_random_categories(self, category_ids):

        random_ids = []

        for _ in category_ids:
            cat_id = random.choice(self.category_ids)
            current_data = self.category_id2text_ids[cat_id]
            current_random_id = random.choice(current_data)
            random_ids.append(current_random_id)

        random_batch = self.ids2processed(batch=random_ids)

        return random_batch


class JigsawData:

    def __init__(self, bert_model, batch_size=32):

        self.bert = bert_model

        self.batch_size = batch_size

        self.train = None
        self.validation = None
        self.test = None

    def load(self, train_filename, test_filename, **csv_kwargs):

        import pandas as pd
        from sklearn.model_selection import train_test_split

        data = pd.read_csv(train_filename, **csv_kwargs)

        data = data.sample(frac=1)

        data['binary_target'] = (data.target >= 0.5).astype('int')

        self.train, self.validation = train_test_split(data, stratify=data.binary_target, test_size=0.05)

        self.test = pd.read_csv(test_filename, **csv_kwargs)

    def batch_generator(self,
                        data_type: str = 'train',
                        batch_size: Optional[int] = None) -> Generator[Any, None, None]:

        data = self.__getattribute__(data_type)
        batch_size = batch_size if batch_size is not None else self.batch_size

        for n_batch in range(math.ceil(len(data) / batch_size)):

            batch = data[n_batch * batch_size:(n_batch + 1) * batch_size]

            batch = self.batch_processing(batch, data_type=data_type)

            yield batch

    def __iter__(self):

        return self.batch_generator()

    def batch_processing(self, batch: Any, data_type: str = 'train') -> Any:

        if data_type in ['train', 'validation']:

            x = self.bert(batch.comment_text, is_batch=True)

            y = np.array(batch.binary_target)

            return x, y

        elif data_type in ['test']:

            return self.bert(batch.comment_text, is_batch=True)
