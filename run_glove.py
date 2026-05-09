import json
import numpy as np
from numpy.linalg import norm

# --------------------------
# Step 1: 加载 tiny_glove.json 文件
# --------------------------
# 这里的路径就是关键！必须和你放文件的位置一致
with open("./datasets/tiny_glove.json", "r", encoding="utf-8") as f:
    glove = json.load(f)

print("词汇表大小:", len(glove))
print("前20个单词:", list(glove.keys())[:20])

# --------------------------
# Step 2: 工具函数
# --------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def get_vector(word):
    if word in glove:
        return np.array(glove[word])
    return None

def nearest_words(target_word, top_n=10):
    if target_word not in glove:
        return []
    target_vec = get_vector(target_word)
    scores = []
    for word in glove:
        if word == target_word:
            continue
        sim = cosine_similarity(target_vec, get_vector(word))
        scores.append((word, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# --------------------------
# Step 3: 基础测试
# --------------------------
print("\n===== 单词相似度对比 =====")
pairs = [
    ("king", "queen"),
    ("man", "woman"),
    ("doctor", "nurse"),
    ("king", "apple"),
    ("teacher", "rich")
]
for w1, w2 in pairs:
    v1, v2 = get_vector(w1), get_vector(w2)
    if v1 is not None and v2 is not None:
        print(f"{w1:10s} vs {w2:10s} → {cosine_similarity(v1, v2):.4f}")
    else:
        print(f"单词不存在: {w1} 或 {w2}")

print("\n===== 查找 king 的近义词 =====")
for word, score in nearest_words("king", top_n=10):
    print(f"{word:15s} {score:.4f}")

# --------------------------
# Step 4: 词向量算术 king - man + woman
# --------------------------
print("\n===== 词向量算术: king - man + woman =====")
result_vector = get_vector("king") - get_vector("man") + get_vector("woman")
scores = []
for word in glove:
    sim = cosine_similarity(result_vector, get_vector(word))
    scores.append((word, sim))
scores.sort(key=lambda x: x[1], reverse=True)
for word, score in scores[:10]:
    print(f"{word:15s} {score:.4f}")

# --------------------------
# Step 5: 句子向量 & 相似度
# --------------------------
def sentence_vector(sentence):
    words = sentence.lower().split()
    vectors = []
    for word in words:
        if word in glove:
            vectors.append(get_vector(word))
    if len(vectors) == 0:
        return np.zeros(50)
    return np.mean(vectors, axis=0)

print("\n===== 句子相似度 =====")
base_sentence = "king man"
base_vec = sentence_vector(base_sentence)
sentences = ["king queen", "man woman", "doctor nurse", "banana orange"]
print(f"基准句子: {base_sentence}")
for s in sentences:
    vec = sentence_vector(s)
    sim = cosine_similarity(base_vec, vec)
    print(f"{s:15s} → {sim:.4f}")