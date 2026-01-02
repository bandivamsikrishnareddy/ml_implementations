corpus = ["low", "lower", "newest", "widest"]
max_sub_len = 5
end_marker = "</w>"
substr_freq = {}
for word in corpus:
    w = word + end_marker
    n = len(w)
    for i in range(n):
        for j in range(1,max_sub_len+1):
            if i + j <= n:
                sub = w[i:i+j]
                if sub in substr_freq:
                    substr_freq[sub]+=1
                else:
                    substr_freq[sub] =1
for word in corpus:
    w = word + end_marker
    for ch in w:
        if ch not in substr_freq:
            substr_freq[ch] = 0
vocab = list(substr_freq.keys())

print("vocab size:", len(vocab))
print("first 20 tokens:", vocab[:20])


epsilon = 1e-9
total = 0.0
for t in vocab:
    total += substr_freq[t] + epsilon
token_prob = {}
for t in vocab:
    token_prob[t] = (substr_freq[t] + epsilon) / total
s = 0.0
for p in token_prob.values():
    s += p

print("sum of probabilities:", s)
print("sample tokens with probs:", list(token_prob.items())[:10])
