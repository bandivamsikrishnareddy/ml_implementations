# Unigram Language Model Tokenizer — from scratch (hard EM)

import math
from ml_chores.corpus_for_tokenizations import load_corpus
corpus = load_corpus()

max_sub_len = 10
end_marker = "</w>"
epsilon = 1e-8
num_iters = 100

# -----------------------------
# 2. Build initial vocabulary
# -----------------------------
substr_freq = {}

for word in corpus:
    w = word + end_marker
    n = len(w)
    for i in range(n):
        for l in range(1, max_sub_len + 1):
            if i + l <= n:
                sub = w[i:i + l]
                substr_freq[sub] = substr_freq.get(sub, 0) + 1

# ensure all characters exist
for word in corpus:
    for ch in word + end_marker:
        substr_freq.setdefault(ch, 0)

vocab = list(substr_freq.keys())

# -----------------------------
# 3. Initialize probabilities
# -----------------------------
token_prob = {}
total = 0.0
for t in vocab:
    total += substr_freq[t] + epsilon

for t in vocab:
    token_prob[t] = (substr_freq[t] + epsilon) / total

# -----------------------------
# 4. EM iterations
# -----------------------------
# print(len(token_prob))
for _ in range(num_iters):

    # ---------- E-step ----------
    expected_count = {t: 0 for t in vocab}

    for word in corpus:
        w = word + end_marker
        n = len(w)

        dp = [-1e18] * (n + 1)
        back = [None] * (n + 1)
        dp[0] = 0.0

        for i in range(n):
            if dp[i] == -1e18:
                continue
            for l in range(1, max_sub_len + 1):
                j = i + l
                if j > n:
                    break
                tok = w[i:j]
                if tok not in token_prob:
                    continue
                score = dp[i] + math.log(token_prob[tok])
                if score > dp[j]:
                    dp[j] = score
                    back[j] = tok

        # backtrack
        idx = n
        while idx > 0:
            tok = back[idx]
            expected_count[tok] += 1
            idx -= len(tok)


    # ---------- M-step ----------
    total = 0.0
    for t in vocab:
        total += expected_count[t] + epsilon

    for t in vocab:
        token_prob[t] = (expected_count[t] + epsilon) / total

    # ---------- Pruning ----------
    new_vocab = []
    for t in vocab:
        if expected_count[t] > 0 or len(t) == 1 or t == end_marker:
            new_vocab.append(t)

    vocab = new_vocab

    # renormalize after pruning
    total = 0.0
    for t in vocab:
        total += token_prob[t]

    for t in vocab:
        token_prob[t] /= total
# print(len(token_prob))//
# -----------------------------
# 5. Final tokenizer (Viterbi)
# -----------------------------
def tokenize(word):
    w = word + end_marker
    n = len(w)
    dp = [-1e18] * (n + 1)
    back = [None] * (n + 1)
    dp[0] = 0.0

    for i in range(n):
        if dp[i] == -1e18:
            continue
        for l in range(1, max_sub_len + 1):
            j = i + l
            if j > n:
                break
            tok = w[i:j]
            if tok not in token_prob:
                continue
            score = dp[i] + math.log(token_prob[tok])
            if score > dp[j]:
                dp[j] = score
                back[j] = tok

    tokens = []
    idx = n
    while idx > 0:
        tok = back[idx]
        tokens.append(tok)
        idx -= len(tok)

    return tokens[::-1]

# -----------------------------
# 6. Example
# -----------------------------
print(tokenize("vamsikrishna"))

