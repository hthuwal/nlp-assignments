import json
import os
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm

lemmatizer = WordNetLemmatizer()
analyzer = TfidfVectorizer().build_analyzer()
tokenizer = RegexpTokenizer(r'(?u)\b\w\w+\b')
en_stop = set(stopwords.words('english'))


def calculate_acc(y1, y2):
    count = 0
    for a, b in zip(y1, y2):
        if a == b:
            count += 1

    return "Accuracy: %0.2f" % (count * 100 / len(y1))


def my_analyzer(doc):
    return [lemmatizer.lemmatize(token) for token in analyzer(doc)]


def train(tfidf, y, model_name, overwrite=False):
    if os.path.exists(model_name) and not overwrite:
        print("Model Already Exists. Loading %s" % model_name)
        model = pickle.load(open(model_name, "rb"))
        return model

    print("Training %s" % model_name)
    if model_name == "linearsvc":
        model = LinearSVC(multi_class="ovr", verbose=1)
    elif model_name == "bNB":
        model = BernoulliNB()
    elif model_name == "mNB":
        model = MultinomialNB()
    elif model_name == "knn":
        model = KNeighborsClassifier()
    elif model_name == "preceptron":
        model = Perceptron(penalty="l2", verbose=1)
    elif model_name == "mlp":
        model = MLPClassifier(hidden_layer_sizes=[5 for i in range(10)], verbose=True)
    elif model_name == "pac":
        model = PassiveAggressiveClassifier(n_jobs=4, verbose=1)

    model.fit(tfidf, y)
    print("Dumping Model")
    pickle.dump(model, open(model_name, "wb"))
    return model


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

    model = train(tfidf_train, y_train, "pac")

    print("\nReading dev Data\n")
    data = []
    with open("dataset/audio_dev.json", "r") as f:
        for line in tqdm(f):
            data.append(json.loads(line))

    print("\nExtracting x's and y's\n")
    corpus = [each["reviewText"] for each in data]
    y_test = [-1 if each["overall"] == 1 or each["overall"] == 2 else 1 if each["overall"] == 4 or each["overall"] == 5 else 0 for each in data]

    del data

    print("Cacluating tfidf for dev data")
    tfidf_test = vectorizer.transform(corpus)
    del corpus

    print("Predicting dev data")
    y_pred = model.predict(tfidf_test)
    y_train_pred = model.predict(tfidf_train)
    del tfidf_test
    del tfidf_train

    print("Saving Predictions to file")
    with open("out.txt", "w") as f:
        for ys in y_pred:
            f.write("%d\n" % (ys))

    print("Report on Training Data")
    print(calculate_acc(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

    print("Report on Test Data")
    print(calculate_acc(y_test, y_pred))
    print(classification_report(y_test, y_pred))
