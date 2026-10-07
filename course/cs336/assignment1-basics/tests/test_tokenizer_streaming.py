from cs336_basics.tokenizer import Tokenizer
from .test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH


def test_streaming_chunk_boundaries():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, ["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    text = "Héllo  world\n\n  we're🙃<|endoftext|><|endoftext|> done\n  "
    expected = tokenizer.encode(text)
    for chunk_size in range(1, 25):
        chunks = (text[i:i + chunk_size] for i in range(0, len(text), chunk_size))
        assert list(tokenizer.encode_iterable(chunks)) == expected


def test_tokenizer_serialization(tmp_path):
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, ["<|endoftext|>"])
    tokenizer.save(tmp_path)
    restored = Tokenizer.from_files(tmp_path / "vocab.json", tmp_path / "merges.json",
                                    ["<|endoftext|>"])
    text = "Hello, 世界🙃<|endoftext|>"
    assert restored.encode(text) == tokenizer.encode(text)
    assert restored.decode(restored.encode(text)) == text
