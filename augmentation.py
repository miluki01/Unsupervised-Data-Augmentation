import numpy as np
import faiss
import random


class TfIdfAugmentation:

    def __init__(self, vocab_path, matrix_path, token2prob, k_top_words=3):

        with open(vocab_path) as f:
            self.vocab = f.read().split('\n')

        self._exist_words = set(self.vocab)

        self.word_matrix = np.load(matrix_path)

        self.token2prob = {key: value for key, value in token2prob.items() if key in self._exist_words}
        self.min_prob = min(token2prob.values())

        self.k_top_words = k_top_words

        self.index = faiss.IndexFlatL2(self.word_matrix.shape[1])
        self.index.add(self.word_matrix)

    def get_word_index(self, word):

        if word in self._exist_words:
            return self.vocab.index(word)

    def get_word_vector(self, word):

        index = self.get_word_index(word)

        if index:
            return self.word_matrix[index]

    def get_similar(self, word, k=1):

        k += 1

        vector = self.get_word_vector(word)

        if vector is None:
            return None

        _, indexes = self.index.search(x=np.array([vector]), k=k)

        indexes = indexes[0][1:]

        return [self.vocab[n] for n in indexes]

    def get_replace_prob(self, word):

        return self.token2prob.get(word, self.min_prob)

    def replace_word(self, word):

        index = self.get_word_index(word)

        if index is None:
            return word

        if self.get_replace_prob(word) > random.random():
            similar_word = random.choice(self.get_similar(word, k=self.k_top_words))
            return similar_word
        else:
            return word

    def replace_sentence(self, tokens):

        vectors = [np.expand_dims(self.get_word_vector(word), 0) for word in tokens if
                   word in self._exist_words]

        if not vectors:
            return tokens

        vectors = np.concatenate(vectors)

        _, indexes = self.index.search(x=vectors, k=self.k_top_words + 1)

        if self.k_top_words == 1:
            replaced_words = iter([random.choice(index) for index in indexes[:, 1:]])
        else:
            replaced_words = iter([index for index in indexes[:, 0]])

        replaced = [self.vocab[next(replaced_words)] if word in self._exist_words else word for word in tokens]

        return replaced

    def __call__(self, tokens):

        return self.replace_sentence(tokens=tokens)
