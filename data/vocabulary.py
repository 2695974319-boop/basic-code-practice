import torch


class Vocabulary:
    PAD = "<pad>"
    SOS = "<sos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self._add_special_tokens()

    def _add_special_tokens(self):
        for token in (self.PAD, self.SOS, self.EOS, self.UNK):
            self.add_token(token)

    def add_special_tokens(self):
        self._add_special_tokens()

    def add_token(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def add_text(self, text):
        for ch in self.normalize_text(text):
            self.add_token(ch)

    def normalize_text(self, text):
        return str(text).replace(" ", "").strip()

    def get_pad_id(self):
        return self.token2idx[self.PAD]

    def get_sos_id(self):
        return self.token2idx[self.SOS]

    def get_eos_id(self):
        return self.token2idx[self.EOS]

    def get_unk_id(self):
        return self.token2idx[self.UNK]

    def length(self):
        return len(self.token2idx)

    def __len__(self):
        return self.length()

    def encode(self, text, max_len):
        text = self.normalize_text(text)
        ids = [self.get_sos_id()]
        ids += [self.token2idx.get(ch, self.get_unk_id()) for ch in text[:max_len - 2]]
        ids += [self.get_eos_id()]

        if len(ids) < max_len:
            ids += [self.get_pad_id()] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
            ids[-1] = self.get_eos_id()

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        chars = []
        for idx in ids:
            token = self.idx2token.get(int(idx), self.UNK)
            if token in (self.PAD, self.SOS):
                continue
            if token == self.EOS:
                break
            chars.append(token)
        return "".join(chars)
