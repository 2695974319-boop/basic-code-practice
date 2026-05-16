import random

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from .vocabulary import Vocabulary


class CoupletDataset(Dataset):
    def __init__(self, pairs, vocab, max_len):
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        up, down = self.pairs[idx]
        return self.vocab.encode(up, self.max_len), self.vocab.encode(down, self.max_len)


class CoupletDataModule:
    def __init__(self, config):
        self.config = config
        self.vocab = Vocabulary()
        self.train_pairs = []
        self.valid_pairs = []

    def setup(self):
        raw_dataset = load_dataset(self.config.dataset_name, split="train")
        pairs = []
        max_needed = min(self.config.max_samples, len(raw_dataset))

        for i in range(max_needed):
            item = raw_dataset[i]
            up, down = self._extract_pair(item)
            up = self.vocab.normalize_text(up)
            down = self.vocab.normalize_text(down)

            if 0 < len(up) <= self.config.max_len - 2 and 0 < len(down) <= self.config.max_len - 2:
                pairs.append((up, down))
                self.vocab.add_text(up)
                self.vocab.add_text(down)

        random.Random(self.config.seed).shuffle(pairs)
        valid_size = max(1, int(len(pairs) * self.config.valid_ratio))
        self.valid_pairs = pairs[:valid_size]
        self.train_pairs = pairs[valid_size:]

        print(
            f"Total pairs: {len(pairs)} "
            f"Train: {len(self.train_pairs)} "
            f"Valid: {len(self.valid_pairs)} "
            f"Vocab: {self.vocab.length()}"
        )

    def _extract_pair(self, item):
        if "up" in item and "down" in item:
            return item["up"], item["down"]
        if "input" in item and "output" in item:
            return item["input"], item["output"]
        if "source" in item and "target" in item:
            return item["source"], item["target"]
        raise KeyError(f"No fields in {list(item.keys())}")

    def train_dataloader(self):
        return DataLoader(
            CoupletDataset(self.train_pairs, self.vocab, self.config.max_len),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

    def valid_dataloader(self):
        return DataLoader(
            CoupletDataset(self.valid_pairs, self.vocab, self.config.max_len),
            batch_size=self.config.batch_size,
            shuffle=False,
        )
