import numpy as np
import faiss
import random
from typing import List, Optional, Union, Generator, Any, Dict


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
