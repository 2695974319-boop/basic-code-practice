import torch
import torch.nn.functional as F

class CoupletGenerator:
    def __init__(self, model, vocab, config, device):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.config = config
        self.device = device

    def _get_next_log_probs(self, src, prefix_ids):
        tgt = torch.tensor([prefix_ids + [self.vocab.get_pad_id()]], dtype=torch.long, device=self.device)
        logits, _ = self.model(src, tgt=tgt, teacher_forcing_ratio=0.0)
        next_logits = logits[:, -1, :].squeeze(0)

        penalty = float(self.config.repetition_penalty)
        if penalty > 1.0 and len(prefix_ids) > 1:
            seen_tokens = set(prefix_ids[1:])
            for token_id in seen_tokens:
                if next_logits[token_id] > 0:
                    next_logits[token_id] = next_logits[token_id] / penalty
                else:
                    next_logits[token_id] = next_logits[token_id] * penalty

        return F.log_softmax(next_logits, dim=-1)

    def _violates_no_repeat_ngram(self, prefix_ids, next_token_id):
        n = int(self.config.no_repeat_ngram_size)
        if n <= 1:
            return False
        if len(prefix_ids) < n - 1:
            return False

        candidate = prefix_ids + [next_token_id]
        new_ngram = tuple(candidate[-n:])
        for i in range(len(candidate) - n):
            if tuple(candidate[i:i + n]) == new_ngram:
                return True
        return False

    def _rerank_score(self, token_ids, log_prob_sum, desired_len):
        text_ids = [
            t for t in token_ids
            if t not in (self.vocab.get_sos_id(), self.vocab.get_eos_id(), self.vocab.get_pad_id())
        ]
        length_gap = abs(len(text_ids) - desired_len)
        unique_count = len(set(text_ids))
        repetition_count = len(text_ids) - unique_count
        return log_prob_sum - 0.6 * length_gap - 0.2 * repetition_count

    def generate(self, up_line):
        with torch.no_grad():
            src = self.vocab.encode(up_line, self.config.max_len).unsqueeze(0).to(self.device)
            desired_len = len(self.vocab.normalize_text(up_line))
            desired_len = max(1, min(desired_len, self.config.max_len - 2))
            beam_width = max(1, int(self.config.beam_width))
            sos_id = self.vocab.get_sos_id()
            eos_id = self.vocab.get_eos_id()

            beams = [([sos_id], 0.0, False)]
            max_decode_steps = min(self.config.max_len - 1, desired_len + 2)

            for step in range(max_decode_steps):
                all_candidates = []
                for token_ids, log_prob_sum, finished in beams:
                    if finished:
                        all_candidates.append((token_ids, log_prob_sum, True))
                        continue

                    log_probs = self._get_next_log_probs(src, token_ids)
                    if step < desired_len - 1:
                        log_probs[eos_id] = float("-inf")

                    topk_vals, topk_ids = torch.topk(log_probs, k=beam_width)
                    for val, token_id in zip(topk_vals.tolist(), topk_ids.tolist()):
                        if self._violates_no_repeat_ngram(token_ids, token_id):
                            continue
                        next_ids = token_ids + [token_id]
                        next_finished = token_id == eos_id
                        all_candidates.append((next_ids, log_prob_sum + val, next_finished))

                if not all_candidates:
                    break

                all_candidates.sort(
                    key=lambda item: self._rerank_score(item[0], item[1], desired_len),
                    reverse=True,
                )
                beams = all_candidates[:beam_width]

                if all(item[2] for item in beams):
                    break

            best_ids, _, _ = max(
                beams,
                key=lambda item: self._rerank_score(item[0], item[1], desired_len),
            )
            if best_ids[-1] != eos_id:
                best_ids = best_ids + [eos_id]
            return self.vocab.decode(best_ids)
