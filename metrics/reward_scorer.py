import math
import re

from .lm_fluency_scorer import LMFluencyScorer


class CoupletRewardScorer:
    """
    Rule-based reward for couplet generation.

    Scores are normalized to roughly [0, 1]. The total reward is a weighted
    average, so it is suitable for REINFORCE-style fine-tuning.
    """

    DEFAULT_WEIGHTS = {
        "length": 1.2,
        "tone": 1.2,
        "repeat_pattern": 1.0,
        "position_category": 0.7,
        "pos_alignment": 0.8,
        "punctuation": 0.4,
        "fluency": 0.8,
        "semantic_fluency": 1.0,
        "no_cross_repeat": 1.0,
        "imagery": 0.8,
        "reference": 0.5,
    }

    PUNCTUATION = set("，。！？；：、,.!?;:")

    FALLBACK_TONE_CLASS = {
        # ping
        "春": "ping", "风": "ping", "花": "ping", "山": "ping", "天": "ping",
        "江": "ping", "河": "ping", "湖": "ping", "海": "ze", "云": "ping",
        "月": "ze", "星": "ping", "人": "ping", "君": "ping", "心": "ping",
        "清": "ping", "明": "ping", "红": "ping", "青": "ping", "白": "ze",
        "寒": "ping", "年": "ping", "时": "ping", "来": "ping", "归": "ping",
        "飞": "ping", "开": "ping", "香": "ping", "长": "ping", "高": "ping",
        "东": "ping", "西": "ping", "南": "ping", "中": "ping", "知": "ping",
        "苏": "ping", "屠": "ping", "存": "ping", "啼": "ping",
        # ze
        "夏": "ze", "秋": "ping", "冬": "ping", "雪": "ze", "雨": "ze",
        "水": "ze", "地": "ze", "夜": "ze", "日": "ze", "草": "ze",
        "柳": "ze", "竹": "ze", "客": "ze", "酒": "ze", "梦": "ze",
        "绿": "ze", "碧": "ze", "黑": "ze", "冷": "ze", "暖": "ze",
        "岁": "ze", "去": "ze", "入": "ze", "出": "ze", "落": "ze",
        "小": "ze", "早": "ze", "北": "ze", "内": "ze", "己": "ze",
        "处": "ze", "似": "ze", "送": "ze", "故": "ze",
    }

    CATEGORY_GROUPS = {
        "nature": "山水云月风雨雪霜露花草柳松竹梅兰荷桂烟霞星日天江河湖海潮浪波泉溪",
        "season": "春夏秋冬寒暑暖凉霜雪雨晴",
        "time": "朝暮晨昏晓夜夕年岁时日月古今",
        "color": "红绿青蓝白黑黄紫碧金银翠丹朱",
        "place": "楼台亭阁门院城郭桥路村寺舟岸关塞",
        "person": "人客君子友臣翁童郎女",
        "motion": "来去归入出上下载飞落送迎看照听闻",
        "emotion": "喜悲愁恨爱怨闲幽清静孤",
        "number": "一二三四五六七八九十百千万双两半",
    }

    POS_GROUPS = {
        "n": "noun",
        "nr": "noun",
        "ns": "noun",
        "nt": "noun",
        "nz": "noun",
        "vn": "noun",
        "v": "verb",
        "vd": "verb",
        "vg": "verb",
        "a": "adj",
        "ad": "adj",
        "an": "adj",
        "d": "adv",
        "m": "num",
        "q": "num",
        "r": "pron",
        "p": "prep",
        "c": "conj",
        "u": "aux",
        "t": "time",
        "s": "place",
        "f": "dir",
    }

    def __init__(
        self,
        weights=None,
        use_pypinyin=True,
        use_pos_tagger=True,
        use_lm_fluency=True,
        lm_scorer=None,
        lm_model_name=None,
        lm_model_type="causal",
        lm_device=None,
        lm_local_files_only=False,
        lm_max_ppl=150.0,
        lm_blend_weight=0.85,
        lm_lazy_load=True,
    ):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self._char_categories = self._build_char_categories()
        self._pinyin = None
        self._style = None
        self._pos_cut = None
        self._lm_blend_weight = float(lm_blend_weight)
        self._lm_scorer = lm_scorer
        if use_lm_fluency and self._lm_scorer is None:
            self._lm_scorer = LMFluencyScorer(
                model_name=lm_model_name,
                model_type=lm_model_type,
                device=lm_device,
                max_ppl=lm_max_ppl,
                local_files_only=lm_local_files_only,
                lazy_load=lm_lazy_load,
            )
        if use_pypinyin:
            self._try_load_pypinyin()
        if use_pos_tagger:
            self._try_load_pos_tagger()

    def _try_load_pypinyin(self):
        try:
            from pypinyin import Style, pinyin
        except ImportError:
            return
        self._pinyin = pinyin
        self._style = Style.TONE3

    def _build_char_categories(self):
        char_categories = {}
        for category, chars in self.CATEGORY_GROUPS.items():
            for char in chars:
                char_categories.setdefault(char, set()).add(category)
        return char_categories

    def _try_load_pos_tagger(self):
        try:
            import jieba.posseg as posseg
        except ImportError:
            return
        self._pos_cut = posseg.cut

    def normalize(self, text):
        return str(text).replace(" ", "").strip()

    def content_chars(self, text):
        text = self.normalize(text)
        return [char for char in text if char not in self.PUNCTUATION]

    def punctuation_positions(self, text):
        return {
            index: char
            for index, char in enumerate(self.normalize(text))
            if char in self.PUNCTUATION
        }

    def repeat_pattern(self, chars):
        first_seen = {}
        pattern = []
        for char in chars:
            if char not in first_seen:
                first_seen[char] = len(first_seen)
            pattern.append(first_seen[char])
        return pattern

    def score(self, upper, lower, reference=None, lm_score=None):
        breakdown = self.breakdown(upper, lower, reference, lm_score=lm_score)
        total_weight = 0.0
        weighted_sum = 0.0
        for name, value in breakdown.items():
            weight = float(self.weights.get(name, 0.0))
            if weight <= 0.0:
                continue
            weighted_sum += weight * value
            total_weight += weight
        if total_weight <= 0.0:
            return 0.0
        return weighted_sum / total_weight

    def score_many(self, uppers, lowers, references=None):
        if references is None:
            references = [None] * len(uppers)
        lm_scores = [None] * len(lowers)
        if self._lm_scorer is not None:
            lm_scores = self._lm_scorer.score_batch(
                [self.normalize(lower) for lower in lowers]
            )
        return [
            self.score(upper, lower, reference, lm_score=lm_scores[index])
            for index, (upper, lower, reference) in enumerate(
                zip(uppers, lowers, references)
            )
        ]

    def breakdown(self, upper, lower, reference=None, lm_score=None):
        upper = self.normalize(upper)
        lower = self.normalize(lower)
        return {
            "length": self.length_score(upper, lower),
            "tone": self.tone_score(upper, lower),
            "repeat_pattern": self.repeat_pattern_score(upper, lower),
            "position_category": self.position_category_score(upper, lower),
            "pos_alignment": self.pos_alignment_score(upper, lower),
            "punctuation": self.punctuation_score(upper, lower),
            "fluency": self.fluency_score(lower, upper),
            "semantic_fluency": self.semantic_fluency_score(lower, lm_score=lm_score),
            "no_cross_repeat": self.no_cross_repeat_score(upper, lower),
            "imagery": self.imagery_score(upper, lower),
            "reference": self.reference_score(lower, reference),
        }

    def length_score(self, upper, lower):
        upper_chars = self.content_chars(upper)
        lower_chars = self.content_chars(lower)
        if not upper_chars:
            return 0.0
        gap = abs(len(upper_chars) - len(lower_chars))
        return self._clip01(1.0 - gap / max(1, len(upper_chars)))

    def tone_score(self, upper, lower):
        upper_chars = self.content_chars(upper)
        lower_chars = self.content_chars(lower)
        limit = min(len(upper_chars), len(lower_chars))
        if limit == 0:
            return 0.0

        compared = 0
        opposite = 0
        for index in range(limit):
            upper_tone = self.tone_class(upper_chars[index])
            lower_tone = self.tone_class(lower_chars[index])
            if upper_tone is None or lower_tone is None:
                continue
            compared += 1
            if upper_tone != lower_tone:
                opposite += 1

        parallel_score = opposite / compared if compared else 0.5

        upper_end = self.tone_class(upper_chars[-1])
        lower_end = self.tone_class(lower_chars[-1])
        if upper_end is None or lower_end is None:
            ending_score = 0.5
        elif upper_end == "ze" and lower_end == "ping":
            ending_score = 1.0
        elif upper_end != lower_end:
            ending_score = 0.8
        else:
            ending_score = 0.3

        return self._clip01(0.8 * parallel_score + 0.2 * ending_score)

    def tone_class(self, char):
        if self._pinyin is not None:
            values = self._pinyin(char, style=self._style, heteronym=False, errors="ignore")
            if values and values[0]:
                match = re.search(r"([1-5])$", values[0][0])
                if match:
                    tone = int(match.group(1))
                    if tone in (1, 2):
                        return "ping"
                    if tone in (3, 4):
                        return "ze"
        return self.FALLBACK_TONE_CLASS.get(char)

    def repeat_pattern_score(self, upper, lower):
        upper_pattern = self.repeat_pattern(self.content_chars(upper))
        lower_pattern = self.repeat_pattern(self.content_chars(lower))
        limit = min(len(upper_pattern), len(lower_pattern))
        if limit <= 1:
            return self.length_score(upper, lower)

        total = 0
        mismatch = 0
        for i in range(limit):
            for j in range(i):
                total += 1
                upper_same = upper_pattern[i] == upper_pattern[j]
                lower_same = lower_pattern[i] == lower_pattern[j]
                if upper_same != lower_same:
                    mismatch += 1
        return self._clip01(1.0 - mismatch / max(1, total))

    def position_category_score(self, upper, lower):
        upper_chars = self.content_chars(upper)
        lower_chars = self.content_chars(lower)
        limit = min(len(upper_chars), len(lower_chars))
        if limit == 0:
            return 0.0

        scores = []
        for index in range(limit):
            upper_categories = self._char_categories.get(upper_chars[index], set())
            lower_categories = self._char_categories.get(lower_chars[index], set())
            if not upper_categories and not lower_categories:
                scores.append(0.5)
            elif upper_categories & lower_categories:
                scores.append(1.0)
            elif upper_categories and lower_categories:
                scores.append(0.35)
            else:
                scores.append(0.5)
        return self._clip01(sum(scores) / len(scores))

    def pos_alignment_score(self, upper, lower):
        upper_pairs = self.pos_pairs(upper)
        lower_pairs = self.pos_pairs(lower)
        if not upper_pairs or not lower_pairs:
            # No POS dependency available: neutral fallback
            return 0.5

        limit = min(len(upper_pairs), len(lower_pairs))
        if limit == 0:
            return 0.0

        exact_match = 0.0
        coarse_match = 0.0
        for i in range(limit):
            upper_tag = upper_pairs[i][1]
            lower_tag = lower_pairs[i][1]
            if upper_tag == lower_tag:
                exact_match += 1.0
                coarse_match += 1.0
                continue
            if self._coarse_pos(upper_tag) == self._coarse_pos(lower_tag):
                coarse_match += 1.0

        length_penalty = abs(len(upper_pairs) - len(lower_pairs)) / max(1, len(upper_pairs))
        score = 0.65 * (exact_match / limit) + 0.35 * (coarse_match / limit)
        score *= (1.0 - 0.35 * length_penalty)
        return self._clip01(score)

    def pos_pairs(self, text):
        text = self.normalize(text)
        if not text:
            return []

        if self._pos_cut is None:
            return []

        pairs = []
        for item in self._pos_cut(text):
            word = str(item.word)
            flag = str(item.flag)
            cleaned = "".join(ch for ch in word if ch not in self.PUNCTUATION).strip()
            if cleaned:
                pairs.append((cleaned, flag))
        return pairs

    def _coarse_pos(self, tag):
        if not tag:
            return "x"
        if tag in self.POS_GROUPS:
            return self.POS_GROUPS[tag]
        for key, value in self.POS_GROUPS.items():
            if tag.startswith(key):
                return value
        return "x"

    def punctuation_score(self, upper, lower):
        upper_positions = self.punctuation_positions(upper)
        lower_positions = self.punctuation_positions(lower)
        if not upper_positions and not lower_positions:
            return 1.0
        keys = set(upper_positions) | set(lower_positions)
        matches = sum(
            1
            for key in keys
            if upper_positions.get(key) == lower_positions.get(key)
        )
        return self._clip01(matches / max(1, len(keys)))

    def fluency_score(self, lower, upper=None):
        lower_chars = self.content_chars(lower)
        if not lower_chars:
            return 0.0

        unique_ratio = len(set(lower_chars)) / len(lower_chars)
        repetition_penalty = max(0.0, 0.75 - unique_ratio)
        repeated_ngram_penalty = self._repeated_ngram_ratio(lower_chars, n=2)
        special_penalty = 1.0 if "<unk>" in lower or "<pad>" in lower else 0.0

        unexplained_repeat_penalty = 0.0
        if upper is not None:
            upper_pattern = self.repeat_pattern(self.content_chars(upper))
            lower_pattern = self.repeat_pattern(lower_chars)
            if len(upper_pattern) == len(lower_pattern):
                unexplained_repeat_penalty = 1.0 - self.repeat_pattern_score(upper, lower)

        cross_repeat_penalty = 0.0
        if upper is not None:
            cross_repeat_penalty = 1.0 - self.no_cross_repeat_score(upper, lower)

        score = (
            1.0
            - 0.35 * repetition_penalty
            - 0.25 * repeated_ngram_penalty
            - 0.15 * special_penalty
            - 0.20 * unexplained_repeat_penalty
            - 0.25 * cross_repeat_penalty
        )
        return self._clip01(score)

    def no_cross_repeat_score(self, upper, lower):
        """Higher when the lower line reuses fewer characters from the upper line."""
        upper_chars = self.content_chars(upper)
        lower_chars = self.content_chars(lower)
        if not lower_chars:
            return 0.0

        limit = min(len(upper_chars), len(lower_chars))
        position_overlap = sum(
            1 for index in range(limit) if upper_chars[index] == lower_chars[index]
        )
        position_score = 1.0 - position_overlap / max(1, len(lower_chars))

        upper_set = set(upper_chars)
        lower_set = set(lower_chars)
        set_overlap = len(lower_set & upper_set)
        set_score = 1.0 - set_overlap / max(1, len(lower_set))

        repeated_in_lower = sum(1 for char in lower_chars if char in upper_set)
        usage_score = 1.0 - repeated_in_lower / max(1, len(lower_chars))
        return self._clip01(0.4 * position_score + 0.35 * set_score + 0.25 * usage_score)

    def semantic_fluency_score(self, lower, lm_score=None):
        """
        Semantic fluency: primarily LM perplexity (lower PPL -> higher score),
        blended with jieba POS heuristics as fallback/auxiliary signal.
        """
        heuristic = self._semantic_fluency_heuristic(lower)
        if lm_score is None and self._lm_scorer is not None:
            lm_score = self._lm_scorer.score(self.normalize(lower))
        if lm_score is None:
            return heuristic

        blend = self._lm_blend_weight
        return self._clip01(blend * lm_score + (1.0 - blend) * heuristic)

    def _semantic_fluency_heuristic(self, lower):
        """Lightweight fallback via jieba segmentation/POS."""
        lower_chars = self.content_chars(lower)
        if not lower_chars:
            return 0.0

        pairs = self.pos_pairs(lower)
        if not pairs:
            unique_ratio = len(set(lower_chars)) / len(lower_chars)
            return self._clip01(0.55 + 0.45 * unique_ratio)

        words = [word for word, _ in pairs]
        coarse_tags = [self._coarse_pos(tag) for _, tag in pairs]

        covered_chars = sum(len(word) for word in words)
        coverage = min(1.0, covered_chars / max(1, len(lower_chars)))

        single_word_ratio = sum(1 for word in words if len(word) == 1) / max(1, len(words))
        cohesion = self._clip01(1.0 - max(0.0, single_word_ratio - 0.45) * 1.8)

        known_pos_ratio = sum(1 for tag in coarse_tags if tag != "x") / max(1, len(coarse_tags))
        meaningful_tags = {tag for tag in coarse_tags if tag in {"noun", "verb", "adj", "adv", "time", "place"}}
        tag_diversity = min(1.0, len(meaningful_tags) / 2.0)

        transition_score = self._pos_transition_score(coarse_tags)
        return self._clip01(
            0.30 * coverage
            + 0.25 * cohesion
            + 0.20 * known_pos_ratio
            + 0.15 * tag_diversity
            + 0.10 * transition_score
        )

    def _pos_transition_score(self, coarse_tags):
        if len(coarse_tags) <= 1:
            return 0.6

        plausible = {
            ("noun", "verb"),
            ("noun", "adj"),
            ("noun", "adv"),
            ("verb", "noun"),
            ("verb", "adj"),
            ("verb", "adv"),
            ("adj", "noun"),
            ("adv", "verb"),
            ("adv", "adj"),
            ("time", "noun"),
            ("time", "verb"),
            ("place", "noun"),
            ("place", "verb"),
        }
        valid = 0
        for left, right in zip(coarse_tags, coarse_tags[1:]):
            if left == "x" or right == "x":
                valid += 0.5
            elif left == right:
                valid += 0.7
            elif (left, right) in plausible:
                valid += 1.0
            else:
                valid += 0.45
        return valid / max(1, len(coarse_tags) - 1)

    def imagery_score(self, upper, lower):
        upper_domains = self._domains_for_text(upper)
        lower_domains = self._domains_for_text(lower)
        if not upper_domains and not lower_domains:
            return 0.5
        if not upper_domains or not lower_domains:
            return 0.35

        shared = len(upper_domains & lower_domains)
        harmony = shared / max(1, min(len(upper_domains), len(lower_domains)))
        coverage = min(len(lower_domains), len(upper_domains)) / max(1, len(upper_domains))
        return self._clip01(0.75 * harmony + 0.25 * coverage)

    def reference_score(self, lower, reference):
        if not reference:
            return 0.5

        lower_chars = self.content_chars(lower)
        ref_chars = self.content_chars(reference)
        if not lower_chars or not ref_chars:
            return 0.0

        lower_set = set(lower_chars)
        ref_set = set(ref_chars)
        overlap = len(lower_set & ref_set)
        precision = overlap / max(1, len(lower_set))
        recall = overlap / max(1, len(ref_set))
        if precision + recall <= 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        length = self._clip01(1.0 - abs(len(lower_chars) - len(ref_chars)) / max(1, len(ref_chars)))
        return self._clip01(0.8 * f1 + 0.2 * length)

    def _domains_for_text(self, text):
        domains = set()
        for char in self.content_chars(text):
            domains.update(self._char_categories.get(char, set()))
        return domains

    def _repeated_ngram_ratio(self, chars, n):
        if len(chars) < n * 2:
            return 0.0
        counts = {}
        for index in range(len(chars) - n + 1):
            ngram = tuple(chars[index:index + n])
            counts[ngram] = counts.get(ngram, 0) + 1
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        return self._clip01(repeated / max(1, len(chars) - n + 1))

    def _clip01(self, value):
        if math.isnan(value):
            return 0.0
        return max(0.0, min(1.0, float(value)))


def reward_scorer_from_config(config=None, weights=None, **kwargs):
    """Build a reward scorer from TrainConfig fields."""
    if config is None:
        return CoupletRewardScorer(weights=weights, **kwargs)

    return CoupletRewardScorer(
        weights=weights or getattr(config, "reward_weights", None),
        use_lm_fluency=getattr(config, "use_lm_fluency", True),
        lm_model_name=getattr(config, "lm_fluency_model_name", None),
        lm_model_type=getattr(config, "lm_fluency_model_type", "causal"),
        lm_device=getattr(config, "lm_fluency_device", None),
        lm_local_files_only=getattr(config, "lm_fluency_local_files_only", False),
        lm_max_ppl=getattr(config, "lm_fluency_max_ppl", 150.0),
        lm_blend_weight=getattr(config, "lm_fluency_blend_weight", 0.85),
        lm_lazy_load=getattr(config, "lm_fluency_lazy_load", True),
        **kwargs,
    )
