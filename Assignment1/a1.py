import json
import os
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm

lemmatizer = WordNetLemmatizer()
analyzer = TfidfVectorizer().build_analyzer()
tokenizer = RegexpTokenizer(r'(?u)\b\w\w+\b')
en_stop = set(stopwords.words('english'))


def my_analyzer(doc):
    return [lemmatizer.lemmatize(token) for token in analyzer(doc)]


def linearsvc(tfidf, y):
    svc = LinearSVC(multi_class="ovr", verbose=1)
    print("Training svc")
    svc.fit(tfidf, y)
    return svc


lemma = True if len(sys.argv) == 2 and sys.argv[1] == "lemma" else False

if lemma:
    tfidf_file = "tfidf_lemmetized.termdoc"
else:
    tfidf_file = "tfidf_unlemmetized.termdoc"

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

    if lemma:
        vectorizer = TfidfVectorizer(max_features=15000, analyzer=my_analyzer, input='content', binary=True, norm='l2', stop_words='english')
    else:
        vectorizer = TfidfVectorizer(max_features=15000, input='content', binary=True, norm='l2', stop_words='english')

    print("\nCalculating TFIDF scores\n")
    tfidf = vectorizer.fit_transform(corpus)

    del data

    print("Dumping tfidf")
    with open(tfidf_file, "wb") as f:
        pickle.dump(tfidf, f)

    with open("y_tri_cat", "wb") as f:
        pickle.dump(y, f)

    with open("model.model", "wb") as f:
        pickle.dump(vectorizer, f)

else:

    print("Loading tfidf and labels")
    tfidf_train = pickle.load(open(tfidf_file, "rb"))
    y_train = pickle.load(open("y_tri_cat", "rb"))
    vectorizer = pickle.load(open("model.model", "rb"))
    model = linearsvc(tfidf_train, y_train)
    # model = runSVC(tfidf_train, y_train, "rbf", 1.0, 0.001)

    print("\nReading dev Data\n")
    data = []
    with open("dataset/audio_dev.json", "r") as f:
        for line in tqdm(f):
            data.append(json.loads(line))

    print("\nExtracting x's and y's\n")
    corpus = [each["reviewText"] for each in data]
    y_test = [-1 if each["overall"] == 1 or each["overall"] == 2 else 1 if each["overall"] == 4 or each["overall"] == 5 else 0 for each in data]

    del data

    print("Cacluating tfidf")
    tfidf_test = vectorizer.transform(corpus)
    del corpus

    y_pred = model.predict(tfidf_test)
    y_train_pred = model.predict(tfidf_train)
    del tfidf_test
    del tfidf_train

    print("Saving Predictions to file")
    with open("out.txt", "w") as f:
        for ys in y_pred:
            f.write("%d\n" % (ys))

    print(classification_report(y_train, y_train_pred))
    print(classification_report(y_test, y_pred))
