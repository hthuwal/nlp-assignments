import json
import pickle
from tqdm import tqdm
from collections import Counter
from nltk.corpus import stopwords
en_stop = set(stopwords.words('english'))

# if not os.path.exists("models/mf_word_dict"):
print("\nReading Data\n")
data = []
with open("../dataset/audio_train.json", "r") as f:
    for line in tqdm(f):
        data.append(json.loads(line))

print("\nExtracting x's and y's\n")
corpus = [(each["summary"] + " ") * 4 + each["reviewText"] for each in data]
y = [each["overall"] for each in data]

vocab = []
for each in tqdm(corpus):
    temp = each.lower().split()
    vocab += [i for i in temp if i not in en_stop]


vocab = Counter(vocab)

vocab = vocab.most_common()

vocab = vocab[0:15000]

word2idx = Counter()

for i, w in enumerate(vocab):
    word2idx[w[0]] = i + 1

pickle.dump(word2idx, open("word2idx", "wb"))
