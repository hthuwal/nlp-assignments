import json
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

lemmatizer = WordNetLemmatizer()
analyzer = TfidfVectorizer().build_analyzer()


def my_analyzer(doc):
    return [lemmatizer.lemmatize(token) for token in analyzer(doc)]


tfidf_file = "tfidf_lemmetized.termdoc"
# tfidf_file="tfidf_unlemmetized.termdoc"

tfidf = None
y = None
if not os.path.exists(tfidf_file):
    print("\nReading Data\n")
    data = []
    with open("dataset/audio_train.json", "r") as f:
        for line in tqdm(f):
            data.append(json.loads(line))

    print("\nExtracting x's and y's\n")
    corpus = [each["reviewText"] for each in data]
    y = [-1 if each["overall"] == 1 or each["overall"] == 2 else 1 if each["overall"] == 4 or each["overall"] == 5 else 0 for each in data]

    # print("\nStemming\n")
    # corpus = getStemmedDocument(corpus)

    vectorizer = TfidfVectorizer(max_features=15000, analyzer=my_analyzer, input='content', binary=True, norm='l2', sublinear_tf=True, stop_words='english')

    print("\nCalculating TFIDF scores\n")
    tfidf = vectorizer.fit_transform(corpus)

    print("Dumping tfidf")
    with open("tfidf_lemmetized.termdoc", "wb") as f:
        pickle.dump(tfidf, f)

    with open("y_tri_cat", "wb") as f:
        pickle.dump(y, f)
else:
    print("Loading tfidf and labels")
    tfidf = pickle.load(open(tfidf_file, "rb"))
    y = pickle.load(open("y_tri_cat", "rb"))

    svc = LinearSVC(multi_class="ovr", verbose=1)
    print("Training svc")
    svc.fit(tfidf, y)
    del tfidf
    del y

    print("Loadint test data")
    data = []
    with open("dataset/audio_dev.json", "r") as f:
        for line in tqdm(f):
            data.append(json.loads(line))

    print("\nExtracting x's and y's\n")
    corpus = [each["reviewText"] for each in data]
    y = [-1 if each["overall"] == 1 or each["overall"] == 2 else 1 if each["overall"] == 4 or each["overall"] == 5 else 0 for each in data]

    vectorizer = TfidfVectorizer(max_features=15000, analyzer=my_analyzer, input='content', binary=True, norm='l2', sublinear_tf=True, stop_words='english')
    print("\nCalculating TFIDF scores\n")
    tfidf = vectorizer.fit_transform(corpus)
    svc.predict(tfidf)
    svc.score(corpus, y)
