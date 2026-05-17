在main.py中修改MODEL_TYPE = "transformer"/ "lstm"/ "gru"即可切换模型
需要的环境：pip install torch datasets
数据来源hugging face

强化学习版本：
1. 在 `scripts/main.py` 中取消 `config.use_rl = True` 的注释。
2. 流程会先用原来的交叉熵做监督训练，加载最优监督模型后再用 REINFORCE 微调。
3. 奖励函数在 `metrics/reward_scorer.py`，默认优化长度一致、平仄相对、重字结构、逐字类别对称、标点对齐、流畅度、意象和参考下联相似度。
4. 可以在 `configs/train_config.py` 的 `reward_weights` 中调高某项权重，例如更重视平仄就提高 `"tone"`，更重视意境就提高 `"imagery"`。
5. 生成阶段也会用同一套 reward 参与 beam search 重排，权重由 `generation_reward_weight` 控制。
