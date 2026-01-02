# SINGLE CELL: small real corpus, plain text only

import os
import subprocess

BASE_DIR = "/Users/vamsi.k/Documents/Projects/ml_chores"
OUT = os.path.join(BASE_DIR, "alice.txt")
URL = "https://www.gutenberg.org/files/11/11-0.txt"  # Alice in Wonderland

os.makedirs(BASE_DIR, exist_ok=True)

if not os.path.exists(OUT):
    subprocess.run(["curl", "-L", "-o", OUT, URL], check=True)

corpus = []
with open(OUT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().lower()
        if not line or line.startswith("***"):
            continue
        corpus.extend(line.split())

print("words:", len(corpus))
print("sample:", corpus[:20])
import os

BASE_DIR = "/Users/vamsi.k/Documents/Projects/ml_chores"
CORPUS_FILE = os.path.join(BASE_DIR, "alice.txt")

def load_corpus():
    corpus = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if not line or line.startswith("***"):
                continue
            corpus.extend(line.split())
    return corpus
