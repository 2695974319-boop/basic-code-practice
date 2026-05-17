import random

import torch
import torch.nn as nn

from models.lstm_model import AttentionDecoderLSTM


class BertSeq2SeqCoupletModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        sos_id,
        eos_id,
        idx2token,
        bert_model_name,
        embed_size,
        hidden_size,
        dropout,
        attention_type="bahdanau",
        freeze_bert=True,
        local_files_only=False,
        bert_max_len=64,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BERT mode requires transformers. Install it with: "
                "pip install transformers"
            ) from exc

        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.idx2token = {int(k): v for k, v in idx2token.items()}
        self.bert_max_len = bert_max_len
        self.freeze_bert = freeze_bert

        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_model_name,
            local_files_only=local_files_only,
        )
        self.bert = AutoModel.from_pretrained(
            bert_model_name,
            local_files_only=local_files_only,
        )

        bert_hidden = self.bert.config.hidden_size
        self.memory_projection = nn.Linear(bert_hidden, hidden_size)
        self.hidden_projection = nn.Linear(bert_hidden, hidden_size)
        self.cell_projection = nn.Linear(bert_hidden, hidden_size)
        self.decoder = AttentionDecoderLSTM(
            vocab_size,
            embed_size,
            hidden_size,
            pad_id,
            dropout,
            attention_type,
        )

        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            self.bert.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bert:
            self.bert.eval()
        return self

    def _decode_src_texts(self, src):
        texts = []
        special_ids = {self.pad_id, self.sos_id, self.eos_id}
        for row in src.detach().cpu().tolist():
            chars = []
            for idx in row:
                if idx in special_ids:
                    continue
                token = self.idx2token.get(int(idx), "")
                if token.startswith("<") and token.endswith(">"):
                    continue
                chars.append(token)
            texts.append("".join(chars) or self.tokenizer.unk_token)
        return texts

    def encode(self, src):
        texts = self._decode_src_texts(src)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.bert_max_len,
            return_tensors="pt",
        ).to(src.device)

        if self.freeze_bert:
            with torch.no_grad():
                bert_output = self.bert(**encoded)
        else:
            bert_output = self.bert(**encoded)

        encoder_outputs = self.memory_projection(bert_output.last_hidden_state)
        pooled = getattr(bert_output, "pooler_output", None)
        if pooled is None:
            pooled = bert_output.last_hidden_state[:, 0, :]

        hidden = torch.tanh(self.hidden_projection(pooled)).unsqueeze(0)
        cell = torch.tanh(self.cell_projection(pooled)).unsqueeze(0)
        src_mask = encoded["attention_mask"].eq(0)
        return {
            "encoder_outputs": encoder_outputs,
            "state": (hidden, cell),
            "src_mask": src_mask,
        }

    def decode_step(self, input_token, state, encoder_cache):
        return self.decoder.forward_step(
            input_token,
            state,
            encoder_cache["encoder_outputs"],
            encoder_cache["src_mask"],
        )

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5, max_len=None):
        batch_size = src.size(0)
        encoder_cache = self.encode(src)
        state = encoder_cache["state"]
        decode_len = tgt.size(1) - 1 if tgt is not None else max_len if max_len is not None else src.size(1)
        input_token = torch.full((batch_size, 1), self.sos_id, dtype=torch.long, device=src.device)
        logits_list = []
        attentions = []

        for t in range(decode_len):
            logits, state, attn_weights = self.decode_step(input_token, state, encoder_cache)
            logits_list.append(logits)
            attentions.append(attn_weights)
            if tgt is not None and random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t + 1].unsqueeze(1)
            else:
                input_token = logits.argmax(dim=-1).detach()

        logits = torch.cat(logits_list, dim=1)
        attentions = torch.cat(attentions, dim=1)
        return logits, attentions
