from pathlib import Path
import sys
import torch
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.train_config import TrainConfig
from data.dataset import CoupletDataModule
from trainer.trainer import CoupletTrainer
from generator.generator import CoupletGenerator

from models.transformer_model import TransformerCoupletModel
from models.lstm_model import Seq2SeqCoupletModel as LSTMModel
from models.gru_model import Seq2SeqGRUModel as GRUModel

# 配置
MODEL_TYPE = "lstm"  # "transformer", "lstm", "gru"

config = TrainConfig()
config.save_path = str(Path("outputs") / f"{MODEL_TYPE}_best_model.pt")

# 设备选择：CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 设置随机种子
random.seed(config.seed)
torch.manual_seed(config.seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(config.seed)

# dataloader
data_module = CoupletDataModule(config)
data_module.setup()
train_loader = data_module.train_dataloader()
valid_loader = data_module.valid_dataloader()

# 初始化模型
if MODEL_TYPE == "transformer":
    model = TransformerCoupletModel(
        vocab_size=len(data_module.vocab),
        pad_id=data_module.vocab.get_pad_id(),
        sos_id=data_module.vocab.get_sos_id(),
        eos_id=data_module.vocab.get_eos_id(),
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_len=config.max_len
    )
elif MODEL_TYPE == "lstm":
    model = LSTMModel(
        vocab_size=len(data_module.vocab),
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        pad_id=data_module.vocab.get_pad_id(),
        sos_id=data_module.vocab.get_sos_id(),
        eos_id=data_module.vocab.get_eos_id(),
        dropout=config.dropout
    )
elif MODEL_TYPE == "gru":
    model = GRUModel(
        vocab_size=len(data_module.vocab),
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        pad_id=data_module.vocab.get_pad_id(),
        sos_id=data_module.vocab.get_sos_id(),
        eos_id=data_module.vocab.get_eos_id(),
        dropout=config.dropout
    )
else:
    raise ValueError("Unsupported MODEL_TYPE")

# Trainer
trainer = CoupletTrainer(model, config, data_module.vocab, device)
best_valid_loss = trainer.fit(train_loader, valid_loader, save_path=Path(config.save_path))
trainer.load(Path(config.save_path))
print(f"Loaded best model from {config.save_path} | best_valid_loss={best_valid_loss:.4f}")

# 推理生成
generator = CoupletGenerator(model, data_module.vocab, config, device)

test_lines = [
    "春风送暖入屠苏",
    "海内存知己",
    "山高月小",
    "绿柳迎春早",
    "年年岁岁花相似",
    "山山水水处处明",
    "风风雨雨送春归",
    "明月明年照故人",
    "处处莺啼处处春",
]

print("\n生成示例：")
for line in test_lines:
    down_line = generator.generate(line)
    print(f"上联：{line}")
    print(f"上联重复模式：{generator.repeat_pattern_text(line)}")
    print(f"下联：{down_line}")
    print(f"下联重复模式：{generator.repeat_pattern_text(down_line)}")
    print("-" * 30)
