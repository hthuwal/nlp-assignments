import json
import pickle
from nltk.tokenize import RegexpTokenizer
from gensim import models, corpora
from tqdm import tqdm
from multiprocessing import Pool
import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s : %(message)s', level=logging.INFO)

# initializing stemmer
tokenizer = RegexpTokenizer(r'\w+')

y_train = pickle.load(open("y_tri_cat", "rb"))

dictionary = None


def stem_stop(text):
    tokens = tokenizer.tokenize(text)
    return tokens


def doc2bowmul(text):
    global dictionary
    return dictionary.doc2bow(text)


# if not os.path.exists("stemmed_train.set"):
print("\nReading Data\n")
data = []
y = []
with open("dataset/audio_train.json", "r") as f:
    for line in tqdm(f):
        temp = json.loads(line)
        data.append((temp["summary"] + " ") * 3 + temp["reviewText"])
        y.append(temp["overall"])


print("Stemming and stopword removal")
pool = Pool(processes=8)
tokens = list(tqdm(pool.imap(stem_stop, data), total=len(data)))

print("Creating Dictitionary")
dictionary = corpora.Dictionary(tokens)

print("Creating corpora")
corpora = [dictionary.doc2bow(text) for text in tqdm(tokens)]

with open("dictionary", "wb") as f:
    pickle.dump(dictionary, f)

with open("corpora", "wb") as f:
    pickle.dump(corpora, f)

print("Training Ldamodel")
ldamodel = models.LdaMulticore(corpora, num_topics=100, id2word=dictionary, passes=1, workers=8)

with open("ldamodel", "wb") as f:
    pickle.dump(ldamodel, f)
