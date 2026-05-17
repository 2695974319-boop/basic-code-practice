import torch
import torch.nn.functional as F

from metrics.reward_scorer import reward_scorer_from_config


class CoupletGenerator:
    def __init__(self, model, vocab, config, device):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.config = config
        self.device = device
        self.reward_scorer = reward_scorer_from_config(config)

    def _cfg(self, name, default):
        return getattr(self.config, name, default)

    def repeat_pattern(self, items):
        first_seen = {}
        pattern = []
        for item in items:
            if item not in first_seen:
                first_seen[item] = len(first_seen)
            pattern.append(first_seen[item])
        return pattern

    def repeat_pattern_text(self, text):
        return self.repeat_pattern(list(self.vocab.normalize_text(text)))

    def _content_ids(self, token_ids):
        special_ids = {
            self.vocab.get_sos_id(),
            self.vocab.get_eos_id(),
            self.vocab.get_pad_id(),
        }
        return [token_id for token_id in token_ids if token_id not in special_ids]

    def _apply_token_filters(self, logits):
        filtered = logits.clone()
        filtered[self.vocab.get_pad_id()] = float("-inf")
        filtered[self.vocab.get_sos_id()] = float("-inf")
        filtered[self.vocab.get_unk_id()] = float("-inf")
        return filtered

    def _apply_repetition_penalty(self, logits, prefix_ids):
        penalty = float(self._cfg("repetition_penalty", 1.0))
        if penalty <= 1.0 or len(prefix_ids) <= 1:
            return logits

        adjusted = logits.clone()
        for token_id in set(self._content_ids(prefix_ids)):
            if adjusted[token_id] > 0:
                adjusted[token_id] = adjusted[token_id] / penalty
            else:
                adjusted[token_id] = adjusted[token_id] * penalty
        return adjusted

    def _upper_char_token_ids(self, upper_text, desired_len):
        token_ids = set()
        for char in self.vocab.normalize_text(upper_text)[:desired_len]:
            token_id = self.vocab.token2idx.get(char)
            if token_id is not None:
                token_ids.add(token_id)
        return token_ids

    def _apply_upper_overlap_penalty(self, logits, upper_char_ids):
        penalty = float(self._cfg("upper_overlap_penalty", 1.0))
        if penalty <= 1.0 or not upper_char_ids:
            return logits

        adjusted = logits.clone()
        for token_id in upper_char_ids:
            if adjusted[token_id] > 0:
                adjusted[token_id] /= penalty
            else:
                adjusted[token_id] *= penalty
        return adjusted

    def _violates_upper_char_overlap(self, next_token_id, upper_char_ids, required_id):
        if not upper_char_ids or next_token_id not in upper_char_ids:
            return False
        if required_id is not None and next_token_id == required_id:
            return False
        return bool(self._cfg("forbid_upper_chars", False))

    def _content_length(self, token_ids):
        return len(self._content_ids(token_ids))

    def _apply_length_constraints(self, log_probs, prefix_ids, desired_len):
        """Hard constraint: lower line must have exactly desired_len content chars."""
        eos_id = self.vocab.get_eos_id()
        content_len = self._content_length(prefix_ids)

        if content_len < desired_len:
            log_probs = log_probs.clone()
            log_probs[eos_id] = float("-inf")
        elif content_len >= desired_len:
            log_probs = torch.full_like(log_probs, float("-inf"))
            log_probs[eos_id] = 0.0
        return log_probs

    def _get_next_log_probs(self, encoder_cache, prefix_ids, state, upper_char_ids=None):
        input_token = torch.tensor([[prefix_ids[-1]]], dtype=torch.long, device=self.device)

        if hasattr(self.model, "decode_step"):
            logits, next_state, _ = self.model.decode_step(input_token, state, encoder_cache)
            next_logits = logits[:, -1, :].squeeze(0)
        elif hasattr(self.model, "decode_from_memory"):
            tgt = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
            logits = self.model.decode_from_memory(encoder_cache, tgt)
            next_state = None
            next_logits = logits[:, -1, :].squeeze(0)
        else:
            raise TypeError("Model must provide decode_step or decode_from_memory.")

        next_logits = self._apply_token_filters(next_logits)
        next_logits = self._apply_repetition_penalty(next_logits, prefix_ids)
        if upper_char_ids:
            next_logits = self._apply_upper_overlap_penalty(next_logits, upper_char_ids)
        return F.log_softmax(next_logits, dim=-1), next_state

    def _violates_no_repeat_ngram(self, prefix_ids, next_token_id):
        n = int(self._cfg("no_repeat_ngram_size", 0))
        if n <= 1 or len(prefix_ids) < n - 1:
            return False

        candidate = prefix_ids + [next_token_id]
        new_ngram = tuple(candidate[-n:])
        for i in range(len(candidate) - n):
            if tuple(candidate[i:i + n]) == new_ngram:
                return True
        return False

    def _required_repeat_token_id(self, prefix_ids, upper_pattern):
        content_ids = self._content_ids(prefix_ids)
        pos = len(content_ids)
        if pos >= len(upper_pattern):
            return None

        current_group = upper_pattern[pos]
        for prev_pos, group in enumerate(upper_pattern[:pos]):
            if group == current_group:
                return content_ids[prev_pos]
        return None

    def _violates_repeat_pattern(self, prefix_ids, next_token_id, upper_pattern):
        if next_token_id == self.vocab.get_eos_id():
            return False

        candidate_ids = self._content_ids(prefix_ids + [next_token_id])
        pos = len(candidate_ids) - 1
        if pos < 0 or pos >= len(upper_pattern):
            return False

        for prev_pos in range(pos):
            upper_same = upper_pattern[pos] == upper_pattern[prev_pos]
            lower_same = candidate_ids[pos] == candidate_ids[prev_pos]
            if upper_same != lower_same:
                return bool(self._cfg("repeat_pattern_hard", True))
        return False

    def _repeat_pattern_penalty(self, token_ids, upper_pattern):
        content_ids = self._content_ids(token_ids)
        limit = min(len(content_ids), len(upper_pattern))
        penalty = 0

        for i in range(limit):
            for j in range(i):
                upper_same = upper_pattern[i] == upper_pattern[j]
                lower_same = content_ids[i] == content_ids[j]
                if upper_same != lower_same:
                    penalty += 1
        return penalty

    def _upper_overlap_count(self, token_ids, upper_char_ids):
        if not upper_char_ids:
            return 0
        return sum(1 for token_id in self._content_ids(token_ids) if token_id in upper_char_ids)

    def _rerank_score(self, token_ids, log_prob_sum, desired_len, upper_pattern, upper_text=None, upper_char_ids=None):
        text_ids = self._content_ids(token_ids)
        length_gap = abs(len(text_ids) - desired_len)
        unique_count = len(set(text_ids))
        repetition_count = len(text_ids) - unique_count
        pattern_penalty = self._repeat_pattern_penalty(token_ids, upper_pattern)
        upper_overlap = self._upper_overlap_count(token_ids, upper_char_ids)
        score = (
            log_prob_sum
            - 0.6 * length_gap
            - 0.2 * repetition_count
            - float(self._cfg("repeat_pattern_weight", 0.8)) * pattern_penalty
            - float(self._cfg("upper_overlap_rerank_weight", 0.8)) * upper_overlap
        )
        reward_weight = float(self._cfg("generation_reward_weight", 0.0))
        if reward_weight > 0.0 and upper_text is not None:
            lower_text = self.vocab.decode(token_ids)
            reward = self.reward_scorer.score(upper_text, lower_text)
            score += reward_weight * max(1, desired_len) * reward
        return score

    def _candidate_token_ids(self, log_probs, prefix_ids, upper_pattern, beam_width, desired_len):
        eos_id = self.vocab.get_eos_id()
        content_len = self._content_length(prefix_ids)

        if content_len >= desired_len:
            return [eos_id]

        pool_size = min(
            log_probs.numel(),
            max(beam_width, beam_width * int(self._cfg("repeat_pattern_candidate_multiplier", 6))),
        )
        topk_ids = torch.topk(log_probs, k=pool_size).indices.tolist()
        candidate_ids = [
            token_id for token_id in dict.fromkeys(topk_ids) if token_id != eos_id
        ]

        required_id = self._required_repeat_token_id(prefix_ids, upper_pattern)
        if required_id is not None and required_id not in candidate_ids:
            candidate_ids.append(required_id)

        return candidate_ids

    def generate(self, up_line):
        with torch.no_grad():
            src = self.vocab.encode(up_line, self.config.max_len).unsqueeze(0).to(self.device)
            encoder_cache = self.model.encode(src)

            normalized_up = self.vocab.normalize_text(up_line)
            desired_len = max(1, min(len(normalized_up), self.config.max_len - 2))
            upper_char_ids = self._upper_char_token_ids(normalized_up, desired_len)
            upper_pattern = self.repeat_pattern(list(normalized_up[:desired_len]))
            beam_width = max(1, int(self.config.beam_width))
            sos_id = self.vocab.get_sos_id()
            eos_id = self.vocab.get_eos_id()

            initial_state = encoder_cache.get("state")
            beams = [([sos_id], 0.0, False, initial_state)]
            max_decode_steps = min(self.config.max_len - 1, desired_len + 1)

            for step in range(max_decode_steps):
                all_candidates = []

                for token_ids, log_prob_sum, finished, state in beams:
                    if finished:
                        all_candidates.append((token_ids, log_prob_sum, True, state))
                        continue

                    log_probs, next_state = self._get_next_log_probs(
                        encoder_cache, token_ids, state, upper_char_ids
                    )
                    log_probs = self._apply_length_constraints(log_probs, token_ids, desired_len)

                    required_id = self._required_repeat_token_id(token_ids, upper_pattern)
                    for token_id in self._candidate_token_ids(
                        log_probs, token_ids, upper_pattern, beam_width, desired_len
                    ):
                        if not torch.isfinite(log_probs[token_id]):
                            continue
                        if self._violates_upper_char_overlap(token_id, upper_char_ids, required_id):
                            continue
                        if token_id != required_id and self._violates_no_repeat_ngram(token_ids, token_id):
                            continue
                        if self._violates_repeat_pattern(token_ids, token_id, upper_pattern):
                            continue

                        next_ids = token_ids + [token_id]
                        next_finished = (
                            token_id == eos_id
                            and self._content_length(token_ids) == desired_len
                        )
                        all_candidates.append(
                            (next_ids, log_prob_sum + log_probs[token_id].item(), next_finished, next_state)
                        )

                if not all_candidates:
                    break

                all_candidates.sort(
                    key=lambda item: self._rerank_score(
                        item[0],
                        item[1],
                        desired_len,
                        upper_pattern,
                        normalized_up,
                        upper_char_ids,
                    ),
                    reverse=True,
                )
                beams = all_candidates[:beam_width]

                if all(item[2] for item in beams):
                    break

            def _beam_rank(item):
                token_ids, log_prob_sum, finished, _ = item
                score = self._rerank_score(
                    token_ids,
                    log_prob_sum,
                    desired_len,
                    upper_pattern,
                    normalized_up,
                    upper_char_ids,
                )
                if self._content_length(token_ids) != desired_len:
                    score -= 1e9
                if not finished:
                    score -= 1e6
                return score

            best_ids, _, _, _ = max(beams, key=_beam_rank)
            if self._content_length(best_ids) == desired_len and best_ids[-1] != eos_id:
                best_ids = best_ids + [eos_id]
            return self.vocab.decode(best_ids)
