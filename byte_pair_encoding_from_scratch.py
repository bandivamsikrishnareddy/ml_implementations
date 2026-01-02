from collections import Counter
from black.trans import defaultdict
no_of_unique_symbols = 0
vocab_size = 10000
corpus =corpus = [
    "low","lower","lowest",
    "new","newer","newest",
    "fast","faster","fastest",
    "slow","slower","slowest",
    "high","higher","highest",
    "wide","wider","widest",
    "strong","stronger","strongest",
    "bright","brighter","brightest",
    "dark","darker","darkest",
    "small","smaller","smallest",
    "large","larger","largest",
    "cold","colder","coldest",
    "warm","warmer","warmest",
    "long","longer","longest",
    "short","shorter","shortest",
    "early","earlier","earliest",
    "late","later","latest",
    "walk","walked","walking","walker",
    "run","ran","running","runner",
    "talk","talked","talking","talker",
    "read","reader","reading","readable",
    "play","played","playing","player",
    "compute","computer","computing","computation",
    "optimize","optimization","optimal","optimizer",
    "token","tokens","tokenize","tokenization",
    "compress","compression","compressed","compressor",
    "segment","segmentation","segmented","segmenter",
    "model","models","modeling","modeller",
    "train","trainer","training","trained",
    "predict","prediction","predictive","predictor",
    "classify","classification","classifier",
    "entropy","probability","distribution","frequency",
    "statistics","statistical","estimation","likelihood",
    "variance","covariance","correlation",
    "language","linguistic","linguistics","lexicon",
    "vocabulary","subword","character","alphabet",
    "syntax","semantic","semantics","pragmatics",
    "generalize","generalization","overfit","overfitting",
    "regularize","regularization","bias","variance_tradeoff",
    "embedding","embeddings","vector","vectors",
    "matrix","matrices","tensor","tensors",
    "sequence","sequential","sequencing",
    "attention","transformer","encoder","decoder",
    "scalable","scalability","efficient","efficiency",
    "robust","robustness","stability","stable",
    "quantize","quantization","quantized",
    "parallel","parallelism","distributed",
    "throughput","latency","bandwidth"
]
optimal_threshold = 3
arr =[]
for word in corpus:
    for i in word:
        arr.append(i)
no_of_unique_symbols = len(set(arr)) + 1
curr_vocab = Counter()
curr_bigram = ""
curr_replacement = ""
kd = 0
while no_of_unique_symbols < vocab_size:
    merged_vocab = Counter()
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
    curr_pair_count = defaultdict(int)
    for word, freq in curr_vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            curr_pair_count[(symbols[i], symbols[i + 1])] += freq
    if not curr_pair_count:
        break
    best_pair = max(curr_pair_count, key=curr_pair_count.get)
    if curr_pair_count[best_pair]<optimal_threshold:
        break
    curr_bigram = " ".join(best_pair)
    curr_replacement = "".join(best_pair)
    if curr_replacement not in curr_vocab:
        no_of_unique_symbols += 1
    print(kd, curr_vocab, best_pair, curr_pair_count[best_pair])
    kd += 1
def total_tokens(vocab):
    return sum(len(word.split()) * freq for word, freq in vocab.items())
L_char = sum(len(w) + 1 for w in corpus)
L_bpe = total_tokens(curr_vocab)
compression_ratio = L_bpe / L_char
compression_gain = 1 - compression_ratio
print(compression_gain)