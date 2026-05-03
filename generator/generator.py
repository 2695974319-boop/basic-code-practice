import torch

class CoupletGenerator:
    def __init__(self, model, vocab, config, device):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.config = config
        self.device = device

    def generate(self, up_line):
        with torch.no_grad():
            src = self.vocab.encode(up_line, self.config.max_len).unsqueeze(0).to(self.device)
            logits, _ = self.model(src, tgt=None, teacher_forcing_ratio=0.0, max_len=self.config.max_len-1)
            pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
            return self.vocab.decode(pred_ids)