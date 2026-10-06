"""从已保存的模型配置和 checkpoint 调用现有文本生成函数。"""

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.nn_utils import generate
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer.model import TransformerLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=Path('data/tinystories/tokenizer'))
    parser.add_argument('--prompt', default='Once upon a time')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    tokenizer = Tokenizer.from_files(
        args.tokenizer / 'vocab.json', args.tokenizer / 'merges.json',
        json.loads((args.tokenizer / 'special_tokens.json').read_text()),
    )
    torch.manual_seed(args.seed)
    model = TransformerLM(
        **{k: config[k] for k in ('d_model', 'num_heads', 'd_ff', 'vocab_size',
                                 'context_length', 'num_layers', 'theta')},
        eps=config['norm_eps'], device=torch.device(args.device), dtype=torch.float32,
        use_rmsnorm=not config.get('no_rmsnorm', False),
    )
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    prompt = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=args.device)
    output = generate(model, prompt, args.max_new_tokens,
                      tokenizer.token_ids[b'<|endoftext|>'], config['context_length'],
                      args.temperature, args.top_p)
    print(tokenizer.decode(output.tolist()))


if __name__ == '__main__':
    main()
