import torch
from torch.utils.data import DataLoader
from pathlib import Path

class CoupletTrainer:
    def __init__(self, model, config, vocab, device):
        """
        model: TransformerCoupletModel / Seq2SeqCoupletModel / Seq2SeqGRUModel
        config: TrainConfig
        vocab: Vocabulary
        device: torch.device
        """
        self.model = model.to(device)
        self.config = config
        self.vocab = vocab
        self.device = device

        # 优先使用 AdamW / Adam 根据需要
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # CrossEntropyLoss，忽略 PAD token
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab.get_pad_id())
        self._uses_teacher_forcing = hasattr(self.model, "decoder")

    # 单轮训练
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for src, tgt in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            self.optimizer.zero_grad()
            teacher_forcing_ratio = self.config.teacher_forcing_ratio if self._uses_teacher_forcing else 0.0
            logits, _ = self.model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)

            # target: 去掉 <sos>
            target = tgt[:, 1:]

            loss = self.criterion(logits.reshape(-1, logits.size(-1)),
                                  target.reshape(-1))
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, len(dataloader))

    # 单轮验证
    def evaluate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        for src, tgt in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            logits, _ = self.model(src, tgt, teacher_forcing_ratio=0.0)
            target = tgt[:, 1:]

            loss = self.criterion(logits.reshape(-1, logits.size(-1)),
                                  target.reshape(-1))
            total_loss += loss.item()

        return total_loss / max(1, len(dataloader))

    # 整体训练
    def fit(self, train_loader, valid_loader, save_path: Path):
        """
        train_loader: DataLoader
        valid_loader: DataLoader
        save_path: Path to save best model
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        best_valid_loss = float("inf")

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            valid_loss = self.evaluate_epoch(valid_loader)

            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f}")

            # 保存最优模型
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save(save_path)
                print(f"Saved best model to {save_path}")

    # 保存模型
    def save(self, path: Path):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "token2idx": self.vocab.token2idx,
            "idx2token": self.vocab.idx2token,
        }
        torch.save(checkpoint, path)

    # 加载模型权重
    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
