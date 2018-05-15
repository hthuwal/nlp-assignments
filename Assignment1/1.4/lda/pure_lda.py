import json
import os
import pickle
from nltk.tokenize import RegexpTokenizer
from gensim import models, corpora
from tqdm import tqdm
from multiprocessing import Pool
from scipy.stats import entropy
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# initializing stemmer
tokenizer = RegexpTokenizer(r'\w+')
dictionary = None


def stem_stop(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    return tokens


def doc2bowmul(text):
    global dictionary
    return dictionary.doc2bow(text)


def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M where M is the number of documents in the corpus
    """
    # lets keep with the p,q notation above
    p = query[None, :].T  # take transpose
    q = matrix.T  # transpose matrix
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def get_most_similar_documents(query, matrix, k=10):
    """
    This function implements the Jensen-Shannon distance above
    and retruns the top k indices of the smallest jensen shannon distances
    """
    sims = jensen_shannon(query, matrix)  # list of jensen shannon distances
    return sims.argsort()[:k]  # the top k positional index of the smallest Jensen Shannon distances


if not os.path.exists("ldamodel"):
    print("\nReading Data\n")
    data = []
    with open("../dataset/audio_train.json", "r") as f:
        for line in tqdm(f):
            temp = json.loads(line)
            data.append((temp["summary"] + " ") * 3 + temp["reviewText"])

    print("Stemming and stopword removal")
    pool = Pool(processes=8)
    tokens = list(tqdm(pool.imap(stem_stop, data), total=len(data)))

    print("Creating Dictionary")
    dictionary = corpora.Dictionary(tokens)

    print("Creating corpora")
    corpora = [dictionary.doc2bow(text) for text in tqdm(tokens)]

    with open("dictionary", "wb") as f:
        pickle.dump(dictionary, f)

    with open("corpora", "wb") as f:
        pickle.dump(corpora, f)

    print("Training Ldamodel")
    ldamodel = models.LdaMulticore(corpora, num_topics=15, id2word=dictionary, passes=1, workers=8)

    with open("ldamodel", "wb") as f:
        pickle.dump(ldamodel, f)

else:
    print("\nLoading LDA Model\n")
    ldamodel = pickle.load(open("ldamodel", "rb"))
    dictionary = pickle.load(open("dictionary", "rb"))
    l_c = pickle.load(open("corpora", "rb"))

    print("\nReading Unlabelled Data\n")
    data = []
    y_gold = []
    with open("../dataset/audio_dev.json", "r") as f:
        for line in tqdm(f):
            temp = json.loads(line)
            data.append((temp["summary"] + " ") * 3 + temp["reviewText"])
            y_gold.append(temp["overall"] - 1)

    print("Stemming and stopword removal")
    pool = Pool(processes=8)
    tokens = list(tqdm(pool.imap(stem_stop, data), total=len(data)))

    print("Creating corpora")
    ul_c = [dictionary.doc2bow(text) for text in tqdm(tokens)]

    del data
    del tokens

    doc_topic_dist = np.array([[tup[1] for tup in ldamodel.get_document_topics(each, minimum_probability=0)] for each in l_c])
    u_doc_topic_dist = np.array([[tup[1] for tup in ldamodel.get_document_topics(each, minimum_probability=0)] for each in ul_c])

    print("\nLoading Labels of labelled data\n")

    y = []
    with open("../dataset/audio_train.json", "r") as f:
        for line in tqdm(f):
            temp = json.loads(line)
            y.append(int(temp["overall"] - 1))

    u_y = []
    for each in u_doc_topic_dist:
        msd = get_most_similar_documents(np.array(each), doc_topic_dist, k=1)[0]
        u_y.append(y[msd])

    print(classification_report(y_gold, u_y))
