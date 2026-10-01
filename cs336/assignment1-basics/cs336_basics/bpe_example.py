def train_bpe(pretoken_counts, num_merges):
    pretoken_counts = dict(pretoken_counts)
    merges = []

    for step in range(num_merges):
        # 1. 按词频累计相邻 token pair 的次数
        word_pairs = {}

        for tokens, cnt in pretoken_counts.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                word_pairs[pair] = word_pairs.get(pair, 0) + cnt

        # 所有词都只剩一个 token 时, 无法继续合并
        if not word_pairs:
            break

        # 2 优先取最大频次, 并列时取最大的 pair
        best_pair, frequency = max(
            word_pairs.items(),
            key=lambda kv: (kv[1], kv[0]),
        )
        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        # 3. 在每个词中合并所有不重叠的匹配
        new_counts = {}

        for tokens, cnt in pretoken_counts.items():
            new_tokens = []
            i = 0

            while i < len(tokens):
                if tokens[i:i + 2] == best_pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            new_key = tuple(new_tokens)
            new_counts[new_key] = new_counts.get(new_key, 0) + cnt

        # 合并不改变词频综合
        assert sum(new_counts.values()) == sum(pretoken_counts.values())
        pretoken_counts = new_counts

        print(f"\n第 {step + 1} 轮")
        print(f"合并: {best_pair} -> {merged_token}, 频次: {frequency}")
        print(f"结果: {pretoken_counts}")

    return merges, pretoken_counts

def bpe_example():
    word_counts = {
        b"low": 5,
        b"lower": 2,
        b"widest": 3,
        b"newest": 6
    }

    # bytes 的切片结果还是 bytes
    # b"low" -> (b"l", b"o", b"w")
    # {(b'l', b'o', b'w'): 5, (b'l', b'o', b'w', b'e', b'r'): 2, (b'w', b'i', b'd', b'e', b's', b't'): 3, (b'n', b'e', b'w', b'e', b's', b't'): 6}
    pretoken_counts = {
        tuple(word[i: i+1] for i in range(len(word))): cnt
        for word, cnt in word_counts.items()
    }
    #print(pretoken_counts)

    merges, final_counts = train_bpe(pretoken_counts, num_merges=6)

    print("\n合并顺序: ")
    for pair in merges:
        print(pair, "->", pair[0] + pair[1])

    return merges, final_counts


if __name__ == "__main__":
    bpe_example()
