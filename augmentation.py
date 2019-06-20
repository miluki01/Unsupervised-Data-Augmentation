import numpy as np
import faiss
import random
from typing import List, Optional, Union, Generator, Any, Dict


# class TfIdfAugmentation:
#
#     def __init__(self, vocab_path, matrix_path, token2prob, k_top_words=3):
#
#         with open(vocab_path) as f:
#             self.vocab = f.read().split('\n')
#
#         self._exist_words = set(self.vocab)
#
#         self.word_matrix = np.load(matrix_path)
#
#         self.token2prob = {key: value for key, value in token2prob.items() if key in self._exist_words}
#         self.min_prob = min(token2prob.values())
#
#         self.k_top_words = k_top_words
#
#         self.index = faiss.IndexFlatL2(self.word_matrix.shape[1])
#         self.index.add(self.word_matrix)
#
#     def get_word_index(self, word):
#
#         if word in self._exist_words:
#             return self.vocab.index(word)
#
#     def get_word_vector(self, word):
#
#         index = self.get_word_index(word)
#
#         if index:
#             return self.word_matrix[index]
#
#     def get_similar(self, word, k=1):
#
#         k += 1
#
#         vector = self.get_word_vector(word)
#
#         if vector is None:
#             return None
#
#         _, indexes = self.index.search(x=np.array([vector]), k=k)
#
#         indexes = indexes[0][1:]
#
#         return [self.vocab[n] for n in indexes]
#
#     def get_replace_prob(self, word):
#
#         return self.token2prob.get(word, self.min_prob)
#
#     def replace_word(self, word):
#
#         index = self.get_word_index(word)
#
#         if index is None:
#             return word
#
#         if self.get_replace_prob(word) > random.random():
#             similar_word = random.choice(self.get_similar(word, k=self.k_top_words))
#             return similar_word
#         else:
#             return word
#
#     def replace_sentence(self, tokens):
#
#         vectors = [np.expand_dims(self.get_word_vector(word), 0) for word in tokens if
#                    word in self._exist_words]
#
#         if not vectors:
#             return tokens
#
#         vectors = np.concatenate(vectors)
#
#         _, indexes = self.index.search(x=vectors, k=self.k_top_words + 1)
#
#         if self.k_top_words == 1:
#             replaced_words = iter([index for index in indexes[:, 1]])
#         else:
#             replaced_words = iter([random.choice(index) for index in indexes[:, 1:]])
#
#         replaced = [self.vocab[next(replaced_words)] if word in self._exist_words else word for word in tokens]
#
#         return replaced
#
#     def replace_batch(self, batch):
#
#         return [self.replace_sentence(tokens=sample) for sample in batch]
#
#     @staticmethod
#     def _sequence_padding(sequence: Union[List[Union[int, str]], List[np.ndarray], np.ndarray],
#                           max_sequence_length: int,
#                           value: Union[int, str]) -> np.ndarray:
#
#         sequence = sequence[:max_sequence_length]
#
#         if len(sequence) < max_sequence_length:
#             for _ in range((max_sequence_length - len(sequence))):
#                 sequence.append(value)
#
#         sequence = np.array(sequence)
#
#         return sequence
#
#     def indexing(self, tokens, max_len=32):
#
#         indexes = [self.get_word_index(tok) for tok in tokens]
#         indexes = [index for index in indexes if index]
#
#         indexes = self._sequence_padding(sequence=indexes, max_sequence_length=max_len, value=0)
#
#         return indexes
#
#     def batch_indexing(self, batch, max_len=32):
#
#         return np.array([self.indexing(tokens=sample, max_len=max_len) for sample in batch])
#
#     def __call__(self, tokens):
#
#         return self.replace_sentence(tokens=tokens)


# class Augmentator:
#
#     def __init__(self, indexes_matrix, max_n=5):
#
#         self.indexes_matrix = indexes_matrix
#         self.max_n = max_n
#
#         if not (0 < self.max_n < self.indexes_matrix.shape[1]):
#             raise ValueError('max_n must set in range 0 and {}'.format(indexes_matrix.shape[1]))
#
#     def replace(self, indexes):
#
#         replaced_indexes = self.indexes_matrix[indexes, :self.max_n]
#
#         if self.max_n == 1:
#             replaced_indexes = replaced_indexes[:, 0]
#         else:
#             random_choice = np.random.randint(replaced_indexes.shape[1], size=replaced_indexes.shape[0])
#             replaced_indexes = replaced_indexes[np.arange(replaced_indexes.shape[0]), random_choice]
#
#         return replaced_indexes


class TfIdfAugmentation:

    def __init__(self, indexes_matrix, index2prob, k_top_words=500):

        self.indexes_matrix = indexes_matrix

        self.index2prob = {key: value for key, value in index2prob.items()}
        self.min_prob = min(index2prob.values())

        self.k_top_words = k_top_words

        self.get_replace_prob = np.vectorize(self._get_replace_prob)

    def _get_replace_prob(self, index):

        return self.index2prob.get(index, self.min_prob)

    def replace_without_probs(self, indexes):

        replaced_indexes = self.indexes_matrix[indexes, :self.k_top_words]

        if self.k_top_words == 1:
            replaced_indexes = replaced_indexes[:, 0]
        else:
            random_choice = np.random.randint(replaced_indexes.shape[1], size=replaced_indexes.shape[0])
            replaced_indexes = replaced_indexes[np.arange(replaced_indexes.shape[0]), random_choice]

        return replaced_indexes

    def replace(self, indexes: np.ndarray):

        padding_mask = (indexes != 0).astype(int)

        replace_probs = self.get_replace_prob(indexes)

        random_sampling = np.random.rand(indexes.shape[0])

        replaced_indexes = indexes[replace_probs >= random_sampling]

        indexes[replace_probs >= random_sampling] = self.replace_without_probs(indexes=replaced_indexes)

        indexes *= padding_mask

        return indexes

    def replace_batch(self, batch):

        return np.array([self.replace(indexes=sample) for sample in batch])
