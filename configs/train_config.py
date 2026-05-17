class TrainConfig:
    def __init__(
        self,
        model_type="lstm",
        dataset_name="wb14123/couplet",
        max_samples=50000,
        valid_ratio=0.05,
        max_len=32,
        batch_size=64,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        embed_size=256,
        hidden_size=512,
        dropout=0.1,
        attention_type="bahdanau",
        bert_model_name="bert-base-chinese",
        bert_freeze=True,
        bert_local_files_only=False,
        bert_max_len=64,
        epochs=15,
        learning_rate=0.0003,
        teacher_forcing_ratio=0.7,
        beam_width=5,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        repeat_pattern_weight=0.8,
        repeat_pattern_hard=True,
        repeat_pattern_candidate_multiplier=6,
        generation_reward_weight=0.45,
        upper_overlap_penalty=1.5,
        forbid_upper_chars=False,
        upper_overlap_rerank_weight=0.8,
        rl_upper_overlap_penalty=1.3,
        use_lm_fluency=True,
        lm_fluency_model_name=None,
        lm_fluency_model_type="causal",
        lm_fluency_device="cpu",
        lm_fluency_local_files_only=False,
        lm_fluency_max_ppl=150.0,
        lm_fluency_blend_weight=0.85,
        lm_fluency_lazy_load=True,
        use_rl=False,
        rl_epochs=3,
        rl_learning_rate=None,
        rl_supervised_loss_weight=0.5,
        rl_policy_loss_weight=1.0,
        rl_entropy_weight=0.01,
        rl_grad_clip=1.0,
        rl_sample_top_k=50,
        rl_baseline_momentum=0.9,
        rl_normalize_advantage=True,
        rl_valid_batches=0,
        reward_weights=None,
        seed=42,
        save_path="outputs/model.pt",
        rl_save_path=None,
    ):
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.valid_ratio = valid_ratio
        self.max_len = max_len
        self.batch_size = batch_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.attention_type = attention_type
        self.bert_model_name = bert_model_name
        self.bert_freeze = bert_freeze
        self.bert_local_files_only = bert_local_files_only
        self.bert_max_len = bert_max_len
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.beam_width = beam_width
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repeat_pattern_weight = repeat_pattern_weight
        self.repeat_pattern_hard = repeat_pattern_hard
        self.repeat_pattern_candidate_multiplier = repeat_pattern_candidate_multiplier
        self.generation_reward_weight = generation_reward_weight
        self.upper_overlap_penalty = upper_overlap_penalty
        self.forbid_upper_chars = forbid_upper_chars
        self.upper_overlap_rerank_weight = upper_overlap_rerank_weight
        self.rl_upper_overlap_penalty = rl_upper_overlap_penalty
        self.use_lm_fluency = use_lm_fluency
        self.lm_fluency_model_name = lm_fluency_model_name
        self.lm_fluency_model_type = lm_fluency_model_type
        self.lm_fluency_device = lm_fluency_device
        self.lm_fluency_local_files_only = lm_fluency_local_files_only
        self.lm_fluency_max_ppl = lm_fluency_max_ppl
        self.lm_fluency_blend_weight = lm_fluency_blend_weight
        self.lm_fluency_lazy_load = lm_fluency_lazy_load
        self.use_rl = use_rl
        self.rl_epochs = rl_epochs
        self.rl_learning_rate = rl_learning_rate
        self.rl_supervised_loss_weight = rl_supervised_loss_weight
        self.rl_policy_loss_weight = rl_policy_loss_weight
        self.rl_entropy_weight = rl_entropy_weight
        self.rl_grad_clip = rl_grad_clip
        self.rl_sample_top_k = rl_sample_top_k
        self.rl_baseline_momentum = rl_baseline_momentum
        self.rl_normalize_advantage = rl_normalize_advantage
        self.rl_valid_batches = rl_valid_batches
        self.reward_weights = reward_weights or {
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
        self.seed = seed
        self.save_path = save_path
        self.rl_save_path = rl_save_path
