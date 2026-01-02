from collections import Counter, defaultdict

# -----------------------------
# Load corpus
# -----------------------------
from ml_chores.corpus_for_tokenizations import load_corpus
corpus = load_corpus()

# -----------------------------
# Hyperparameters
# -----------------------------
vocab_size = 10000
optimal_threshold = 3

# -----------------------------
# Initial symbol count
# -----------------------------
arr = []
for word in corpus:
    for ch in word:
        arr.append(ch)

no_of_unique_symbols = len(set(arr)) + 1  # +1 for <w>

# -----------------------------
# BPE training
# -----------------------------
curr_vocab = Counter()
curr_bigram = ""
curr_replacement = ""
merge_rules = []   # IMPORTANT: store merge rules in order
kd = 0

while no_of_unique_symbols < vocab_size:
    merged_vocab = Counter()

    # initialization
    if not curr_vocab:
        for word in corpus:
            chars = " ".join(word) + " <w>"
            curr_vocab[chars] += 1
        merged_vocab = curr_vocab.copy()
    else:
        for word, freq in curr_vocab.items():
            new_word = word.replace(curr_bigram, curr_replacement)
            merged_vocab[new_word] += freq

    curr_vocab = merged_vocab

    # count symbol pairs
    curr_pair_count = defaultdict(int)
    for word, freq in curr_vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            curr_pair_count[(symbols[i], symbols[i + 1])] += freq

    if not curr_pair_count:
        break

    best_pair = max(curr_pair_count, key=curr_pair_count.get)

    if curr_pair_count[best_pair] < optimal_threshold:
        break

    # apply merge
    curr_bigram = " ".join(best_pair)
    curr_replacement = "".join(best_pair)

    merge_rules.append(best_pair)   # STORE RULE

    if curr_replacement not in curr_vocab:
        no_of_unique_symbols += 1

    kd += 1

# -----------------------------
# Compression statistics
# -----------------------------
def total_tokens(vocab):
    return sum(len(word.split()) * freq for word, freq in vocab.items())

L_char = sum(len(w) + 1 for w in corpus)
L_bpe = total_tokens(curr_vocab)

compression_ratio = L_bpe / L_char
compression_gain = 1 - compression_ratio

print("compression gain:", compression_gain)
print("number of merges:", len(merge_rules))

# -----------------------------
# BPE Encode function
# -----------------------------
def encode(word, merge_rules):
    """
    Encode a word using learned BPE merge rules.

    word: string
    merge_rules: list of tuples [('l','o'), ('lo','w'), ...]

    returns: list of BPE tokens
    """

    # start from characters + end marker
    symbols = list(word) + ["<w>"]

    # replay merges in exact training order
    for a, b in merge_rules:
        i = 0
        new_symbols = []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(a + b)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols

    return symbols

# -----------------------------
# Example encodings
# -----------------------------
print(encode("lowest", merge_rules))
print(encode("unbelievable", merge_rules))
