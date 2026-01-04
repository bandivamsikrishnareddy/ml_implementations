import math
from collections import Counter

from black.trans import defaultdict

from ml_chores.corpus_for_tokenizations import load_corpus

corpus = load_corpus()
from collections import Counter, defaultdict

# ---------- corpus prep ----------
def prepare_corpus(words):
    vocab = Counter()
    for w in words:
        vocab[" ".join(list(w) + ["</w>"])] += 1
    return vocab

# ---------- counts ----------
def get_counts(word_vocab):
    unigram = defaultdict(int)
    bigram = defaultdict(int)

    for word, freq in word_vocab.items():
        symbols = word.split()
        for s in symbols:
            unigram[s] += freq
        for i in range(len(symbols) - 1):
            bigram[(symbols[i], symbols[i+1])] += freq

    return unigram, bigram

# ---------- scoring ----------
def best_wordpiece_pair(unigram, bigram, epsilon):
    best_pair = None
    best_score = epsilon

    for (x, y), c_xy in bigram.items():
        score = c_xy / (unigram[x] * unigram[y])
        if score > best_score:
            best_score = score
            best_pair = (x, y)

    return best_pair, best_score

# ---------- merge ----------
def merge_pair(word_vocab, pair):
    x, y = pair
    merged = x + y
    pattern = f"{x} {y}"

    new_vocab = Counter()
    for word, freq in word_vocab.items():
        new_vocab[word.replace(pattern, merged)] += freq

    return new_vocab, merged

# ---------- training loop ----------
def train_wordpiece(words, vocab_size=10000, epsilon=1e-8):
    word_vocab = prepare_corpus(words)

    symbol_vocab = set()
    for w in words:
        symbol_vocab.update(w)
    symbol_vocab.add("</w>")
    symbol_vocab.add("<unk>")

    while len(symbol_vocab) < vocab_size:
        unigram, bigram = get_counts(word_vocab)
        pair, score = best_wordpiece_pair(unigram, bigram, epsilon)

        if pair is None:
            break

        word_vocab, new_symbol = merge_pair(word_vocab, pair)
        symbol_vocab.add(new_symbol)

    return symbol_vocab, word_vocab


# ---------- run ----------
corpus = load_corpus()
symbols, final_vocab = train_wordpiece(
    corpus,
    vocab_size=8000,
    epsilon=1e-8
)

print("final vocab size:", len(symbols))
