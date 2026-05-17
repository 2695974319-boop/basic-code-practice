import math


class LMFluencyScorer:
    """
    Score text fluency with a pretrained Chinese LM via perplexity (PPL).
    Lower PPL -> higher score in [0, 1].
    """

    DEFAULT_CAUSAL_MODEL = "uer/gpt2-chinese-cluecorpussmall"
    DEFAULT_MLM_MODEL = "bert-base-chinese"

    def __init__(
        self,
        model_name=None,
        model_type="causal",
        device=None,
        max_length=64,
        max_ppl=150.0,
        local_files_only=False,
        lazy_load=True,
    ):
        self.model_name = model_name
        self.model_type = (model_type or "causal").lower()
        self.device = device
        self.max_length = max_length
        self.max_ppl = max(1.0, float(max_ppl))
        self.local_files_only = local_files_only
        self.lazy_load = lazy_load

        self._model = None
        self._tokenizer = None
        self._resolved_device = None
        self._use_amp = False
        self._amp_dtype = None
        self._load_failed = False

        if not lazy_load:
            self._ensure_loaded()

    def available(self):
        return self._ensure_loaded()

    def _resolve_model_name(self):
        if self.model_name:
            return self.model_name
        if self.model_type == "mlm":
            return self.DEFAULT_MLM_MODEL
        return self.DEFAULT_CAUSAL_MODEL

    def _resolve_device(self):
        if self._resolved_device is not None:
            return self._resolved_device

        if self.device:
            self._resolved_device = self.device
            return self._resolved_device

        try:
            import torch
        except ImportError:
            self._resolved_device = "cpu"
            return self._resolved_device

        if torch.cuda.is_available():
            self._resolved_device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self._resolved_device = "mps"
        else:
            self._resolved_device = "cpu"
        return self._resolved_device

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return True
        if self._load_failed:
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            self._load_failed = True
            return False

        model_name = self._resolve_model_name()
        device = self._resolve_device()

        try:
            if self.model_type == "mlm":
                from transformers import BertForMaskedLM

                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=self.local_files_only,
                )
                self._model = BertForMaskedLM.from_pretrained(
                    model_name,
                    local_files_only=self.local_files_only,
                )
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=self.local_files_only,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    local_files_only=self.local_files_only,
                )
            self._model.to(device)
            self._model.eval()
            self._use_amp = False
            self._amp_dtype = None
            if device == "cuda":
                # Prefer bf16 on Ada GPUs like 4090 for fast and stable inference.
                if torch.cuda.is_bf16_supported():
                    self._use_amp = True
                    self._amp_dtype = torch.bfloat16
                else:
                    self._use_amp = True
                    self._amp_dtype = torch.float16
            print(f"[LMFluencyScorer] loaded {model_name} on device={device}")
            return True
        except Exception:
            self._load_failed = True
            self._model = None
            self._tokenizer = None
            return False

    def _normalize_text(self, text):
        return str(text).replace(" ", "").strip()

    def perplexity(self, text):
        text = self._normalize_text(text)
        if not text or not self._ensure_loaded():
            return None

        import torch

        device = self._resolve_device()
        if self.model_type == "mlm":
            return self._mlm_perplexity(text, device)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if inputs["input_ids"].size(1) < 2:
            return None

        with torch.inference_mode():
            if self._use_amp:
                with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                    outputs = self._model(**inputs, labels=inputs["input_ids"])
            else:
                outputs = self._model(**inputs, labels=inputs["input_ids"])
            loss = float(outputs.loss.item())

        return math.exp(loss)

    def _mlm_perplexity(self, text, device):
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = inputs["input_ids"].to(device)
        if input_ids.size(1) < 3:
            return None

        mask_id = getattr(self._tokenizer, "mask_token_id", None)
        if mask_id is None:
            return None

        total_loss = 0.0
        count = 0
        with torch.inference_mode():
            for pos in range(1, input_ids.size(1) - 1):
                masked_input = input_ids.clone()
                labels = torch.full_like(input_ids, -100)
                labels[0, pos] = input_ids[0, pos]
                masked_input[0, pos] = mask_id
                if self._use_amp:
                    with torch.autocast(device_type="cuda", dtype=self._amp_dtype):
                        outputs = self._model(masked_input, labels=labels)
                else:
                    outputs = self._model(masked_input, labels=labels)
                if outputs.loss is not None:
                    total_loss += float(outputs.loss.item())
                    count += 1

        if count == 0:
            return None
        return math.exp(total_loss / count)

    def ppl_to_score(self, ppl):
        if ppl is None or math.isnan(ppl) or math.isinf(ppl):
            return None
        return max(0.0, min(1.0, 1.0 - (math.log(max(ppl, 1.0)) / math.log(self.max_ppl))))

    def score(self, text):
        return self.ppl_to_score(self.perplexity(text))

    def score_batch(self, texts):
        return [self.score(text) for text in texts]
