import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


class Bert:

    def __init__(self, model_name='bert-base-uncased', n_combine=0, max_length=256):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_name = model_name
        self.max_length = max_length

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=True)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.n_layers = 24 if 'large' in self.model_name else 12
        self.n_combine = n_combine

        self.embedding_size = 1024 if 'large' in self.model_name else 768

        self.pad_index = 0

    def tokenize(self, text):

        text = '[CLS] ' + text

        tokens = self.tokenizer.tokenize(text)
        segments_ids = [0 for _ in range(len(tokens))]

        indexed = self.tokenizer.convert_tokens_to_ids(tokens)

        return indexed, segments_ids

    @staticmethod
    def _sequence_padding(sequence, max_sequence_length, value):

        sequence = sequence[:max_sequence_length]

        if len(sequence) < max_sequence_length:
            for _ in range((max_sequence_length - len(sequence))):
                sequence.append(value)

        return sequence

    def __call__(self, text, is_batch=False, need_cls_token=False):

        if is_batch:

            indexed = []
            segments_ids = []

            max_len = 0

            for sample in text:
                current_indexed, current_segments_ids = self.tokenize(sample)
                indexed.append(current_indexed)
                segments_ids.append(current_segments_ids)

                if len(current_indexed) > max_len:
                    max_len = len(current_indexed)

            max_len = max_len if max_len <= 512 else 512

            indexed = [self._sequence_padding(sequence=sample,
                                              max_sequence_length=self.max_length,
                                              value=self.pad_index)
                       for sample in indexed]

            segments_ids = [self._sequence_padding(sequence=sample,
                                                   max_sequence_length=self.max_length,
                                                   value=self.pad_index)
                            for sample in segments_ids]

        else:
            indexed, segments_ids = self.tokenize(text)
            indexed = [indexed]
            segments_ids = [segments_ids]

        tokens_tensor = torch.tensor(indexed).to(self.device)
        segments_tensors = torch.tensor(segments_ids).to(self.device)

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

        encoded_layers = torch.cat([layer.unsqueeze(0) for layer in encoded_layers])

        encoded_layers = encoded_layers.permute(1, 2, 0, 3)

        return encoded_layers