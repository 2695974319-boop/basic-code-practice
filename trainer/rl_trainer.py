from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from metrics.reward_scorer import CoupletRewardScorer, reward_scorer_from_config
from trainer.trainer import CoupletTrainer


class RLCoupletTrainer(CoupletTrainer):
    """
    Supervised + REINFORCE fine-tuning.

    The supervised loss keeps the model close to the paired couplet corpus,
    while the policy loss pushes sampled lower lines toward higher reward.
    """

    def __init__(self, model, config, vocab, device, reward_scorer=None):
        super().__init__(model, config, vocab, device)
        rl_lr = getattr(config, "rl_learning_rate", None)
        if rl_lr is None:
            rl_lr = config.learning_rate * 0.2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=rl_lr)
        self.reward_scorer = reward_scorer or reward_scorer_from_config(config)
        self._moving_baseline = None

    def train_epoch_rl(self, dataloader):
        self.model.train()
        totals = {
            "loss": 0.0,
            "ce_loss": 0.0,
            "policy_loss": 0.0,
            "entropy": 0.0,
            "reward": 0.0,
        }

        for src, tgt in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            self.optimizer.zero_grad()

            ce_loss = self._supervised_loss(src, tgt)
            sampled_ids, log_prob, entropy, lengths = self._decode_content(src, sample=True)
            rewards = self._batch_rewards(src, sampled_ids, tgt)
            advantage = self._advantage(rewards)

            log_prob = log_prob / lengths.clamp_min(1.0)
            policy_loss = -(advantage.detach() * log_prob).mean()
            entropy = (entropy / lengths.clamp_min(1.0)).mean()

            loss = (
                float(getattr(self.config, "rl_supervised_loss_weight", 0.5)) * ce_loss
                + float(getattr(self.config, "rl_policy_loss_weight", 1.0)) * policy_loss
                - float(getattr(self.config, "rl_entropy_weight", 0.01)) * entropy
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float(getattr(self.config, "rl_grad_clip", 1.0)),
            )
            self.optimizer.step()

            batch_size = src.size(0)
            totals["loss"] += loss.item() * batch_size
            totals["ce_loss"] += ce_loss.item() * batch_size
            totals["policy_loss"] += policy_loss.item() * batch_size
            totals["entropy"] += entropy.item() * batch_size
            totals["reward"] += rewards.mean().item() * batch_size

        count = max(1, len(dataloader.dataset))
        return {name: value / count for name, value in totals.items()}

    def evaluate_reward_epoch(self, dataloader):
        self.model.eval()
        total_reward = 0.0
        total_count = 0
        max_batches = int(getattr(self.config, "rl_valid_batches", 0))

        with torch.no_grad():
            for batch_index, (src, tgt) in enumerate(dataloader):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                sampled_ids, _, _, _ = self._decode_content(src, sample=False)
                rewards = self._batch_rewards(src, sampled_ids, tgt)
                total_reward += rewards.sum().item()
                total_count += src.size(0)

        return total_reward / max(1, total_count)

    def fit_rl(self, train_loader, valid_loader, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        best_reward = float("-inf")

        for epoch in range(1, int(getattr(self.config, "rl_epochs", 3)) + 1):
            stats = self.train_epoch_rl(train_loader)
            valid_reward = self.evaluate_reward_epoch(valid_loader)
            valid_loss = self.evaluate_epoch(valid_loader)

            print(
                "RL Epoch "
                f"{epoch:03d} | loss={stats['loss']:.4f} "
                f"| ce={stats['ce_loss']:.4f} "
                f"| policy={stats['policy_loss']:.4f} "
                f"| reward={stats['reward']:.4f} "
                f"| valid_reward={valid_reward:.4f} "
                f"| valid_loss={valid_loss:.4f}"
            )

            if valid_reward > best_reward:
                best_reward = valid_reward
                self.save(save_path)
                print(f"Saved best RL model to {save_path}")

        return best_reward

    def _supervised_loss(self, src, tgt):
        teacher_forcing_ratio = (
            self.config.teacher_forcing_ratio if self._uses_teacher_forcing else 0.0
        )
        logits, _ = self.model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        target = tgt[:, 1:]
        return self.criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

    def _decode_content(self, src, sample):
        desired_lengths = self._content_lengths(src)
        batch_size = src.size(0)
        max_steps = int(desired_lengths.max().item())
        max_steps = max(1, min(max_steps, self.config.max_len - 2))

        encoder_cache = self.model.encode(src)
        state = encoder_cache.get("state")
        input_token = torch.full(
            (batch_size, 1),
            self.vocab.get_sos_id(),
            dtype=torch.long,
            device=self.device,
        )
        tgt_input = input_token

        generated = []
        log_probs = []
        entropies = []

        for step in range(max_steps):
            if hasattr(self.model, "decode_step"):
                logits, state, _ = self.model.decode_step(input_token, state, encoder_cache)
                next_logits = logits[:, -1, :]
            elif hasattr(self.model, "decode_from_memory"):
                logits = self.model.decode_from_memory(encoder_cache, tgt_input)
                next_logits = logits[:, -1, :]
            else:
                raise TypeError("Model must provide decode_step or decode_from_memory.")

            filtered_logits = self._filter_content_logits(next_logits)
            filtered_logits = self._apply_upper_overlap_penalty_batch(filtered_logits, src)
            if sample:
                token, token_log_prob, token_entropy = self._sample_from_logits(filtered_logits)
            else:
                log_prob_dist = F.log_softmax(filtered_logits, dim=-1)
                token = torch.argmax(log_prob_dist, dim=-1)
                token_log_prob = log_prob_dist.gather(1, token.unsqueeze(1)).squeeze(1)
                token_entropy = torch.zeros_like(token_log_prob)

            active = (step < desired_lengths).float()
            token_for_output = torch.where(
                active.bool(),
                token,
                torch.full_like(token, self.vocab.get_pad_id()),
            )
            generated.append(token_for_output)
            log_probs.append(token_log_prob * active)
            entropies.append(token_entropy * active)

            input_token = token.unsqueeze(1)
            if hasattr(self.model, "decode_from_memory") and not hasattr(self.model, "decode_step"):
                tgt_input = torch.cat([tgt_input, input_token], dim=1)

        return (
            torch.stack(generated, dim=1),
            torch.stack(log_probs, dim=1).sum(dim=1),
            torch.stack(entropies, dim=1).sum(dim=1),
            desired_lengths.float(),
        )

    def _sample_from_logits(self, logits):
        top_k = int(getattr(self.config, "rl_sample_top_k", 0))
        if top_k > 0 and top_k < logits.size(-1):
            values, indices = torch.topk(logits, k=top_k, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            logits = filtered.scatter(1, indices, values)

        distribution = Categorical(logits=logits)
        token = distribution.sample()
        return token, distribution.log_prob(token), distribution.entropy()

    def _filter_content_logits(self, logits):
        filtered = logits.clone()
        special_ids = [
            self.vocab.get_pad_id(),
            self.vocab.get_sos_id(),
            self.vocab.get_eos_id(),
            self.vocab.get_unk_id(),
        ]
        filtered[:, special_ids] = float("-inf")
        return filtered

    def _upper_char_token_ids_from_src(self, src_row):
        special_ids = {
            self.vocab.get_pad_id(),
            self.vocab.get_sos_id(),
            self.vocab.get_eos_id(),
        }
        return {
            int(token_id)
            for token_id in src_row.tolist()
            if int(token_id) not in special_ids
        }

    def _apply_upper_overlap_penalty_batch(self, logits, src):
        penalty = float(getattr(self.config, "rl_upper_overlap_penalty", 1.0))
        if penalty <= 1.0:
            return logits

        adjusted = logits.clone()
        for batch_index in range(src.size(0)):
            upper_ids = self._upper_char_token_ids_from_src(src[batch_index])
            for token_id in upper_ids:
                if adjusted[batch_index, token_id] > 0:
                    adjusted[batch_index, token_id] /= penalty
                else:
                    adjusted[batch_index, token_id] *= penalty
        return adjusted

    def _content_lengths(self, token_batch):
        special = (
            token_batch.eq(self.vocab.get_pad_id())
            | token_batch.eq(self.vocab.get_sos_id())
            | token_batch.eq(self.vocab.get_eos_id())
        )
        return (~special).sum(dim=1).clamp_min(1)

    def _batch_rewards(self, src, sampled_ids, tgt):
        uppers = self._decode_texts(src)
        lowers = self._decode_texts(sampled_ids)
        references = self._decode_texts(tgt)
        reward_values = self.reward_scorer.score_many(uppers, lowers, references)
        return torch.tensor(reward_values, dtype=torch.float, device=self.device)

    def _decode_texts(self, token_batch):
        token_batch = token_batch.detach().cpu().tolist()
        return [self.vocab.decode(token_ids) for token_ids in token_batch]

    def _advantage(self, rewards):
        momentum = float(getattr(self.config, "rl_baseline_momentum", 0.9))
        batch_mean = rewards.mean().detach()
        if self._moving_baseline is None:
            self._moving_baseline = batch_mean
        else:
            self._moving_baseline = (
                momentum * self._moving_baseline + (1.0 - momentum) * batch_mean
            )
        advantage = rewards - self._moving_baseline

        if bool(getattr(self.config, "rl_normalize_advantage", True)) and rewards.numel() > 1:
            std = advantage.std(unbiased=False)
            if torch.isfinite(std) and std.item() > 1e-6:
                advantage = advantage / (std + 1e-6)

        return advantage
