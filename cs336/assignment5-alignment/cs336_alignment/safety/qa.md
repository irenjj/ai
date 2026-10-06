```
cs336_alignment/
├── prompts_safety/             # 作业已提供的提示模板，保持原位置
└── safety/
    ├── __init__.py
    │
    ├── data/
    │   ├── __init__.py
    │   ├── sft_dataset.py      # SFT 数据集与批次加载
    │   └── hh_dataset.py       # HH 偏好数据加载
    │
    ├── evaluation/
    │   ├── __init__.py
    │   ├── mmlu.py             # MMLU 评估与答案解析
    │   ├── gsm8k.py            # GSM8K 评估与答案解析
    │   ├── alpaca_eval.py      # 生成并导出 AlpacaEval 所需结果
    │   └── simple_safety.py    # 生成并导出安全评估所需结果
    │
    ├── training/
    │   ├── __init__.py
    │   ├── sft.py              # SFT 训练入口
    │   ├── dpo_loss.py         # DPO 损失
    │   └── dpo.py              # DPO 训练入口
    │
    └── README.md              # 记录你的运行方式与文件用途
```