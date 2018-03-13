import pickle
import os
import json
from gensim import corpora, models
from sklearn.svm import LinearSVC
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
def calculate_acc(y1, y2):
    count = 0
    for a, b in zip(y1, y2):
        if a == b:
            count += 1

    return "Accuracy: %0.2f" % (count * 100 / len(y1))

def convert(ldamodel, dictionary, data):

    print("Converting each sample to bow and lda")
    data = [ldamodel[dictionary.doc2bow(each.split())] for each in tqdm(data)]

    print("Converting it into SVM's format")
    for i in tqdm(range(len(data))):
        text = dat[i]
        temp = [0 for i in range(100)]
        for a, b in text:
            temp[a] = b
        data[i] = temp

    return data


print("Loading ldamodel")
ldamodel = pickle.load(open("ldamodel", "rb"))
dictionary = pickle.load(open("dictionary", "rb"))

model_name = "libsvm_lda"

if not os.path.exists(model_name):
    print("Loading Train data")
    train_data = []
    train_y = []
    with open("dataset/audio_train.json", "r") as f:
        for line in tqdm(f):
            temp = json.loads(line)
            train_data.append((temp["summary"] + " ") * 3 + temp["reviewText"])
            train_y.append(int(temp["overall"]))


    train_x = convert(ldamodel, dictionary, train_data)
    del train_data
    model = LinearSVC(multi_class="ovr", verbose=1, penalty="l1", dual=False)
    print("Training Model")
    model.fit(train_x, train_y)

    print("Dumping Model")
    pickle.dump(model, open(model_name, "wb"))

else:
    print("Loading svm_model")
    model = pickle.load(model_name, "rb")

    print("Loading Test data")
    test_data = []
    test_y = []
    with open("dataset/audio_dev.json", "r") as f:
        for line in tqdm(f):
            temp = json.loads(line)
            test_data.append((temp["summary"] + " ") * 3 + temp["reviewText"])
            test_y.append(int(temp["overall"]))

    test_x = convert(ldamodel, dictionary, test_data)
    del test_data
    print("Predict")
    y_pred = model.predict(test_x)

    test_y = [1 if i == 1 or i == 2 else 5 if i == 4 or i == 5 else 3 for i in test_y]
    y_pred = [1 if i == 1 or i == 2 else 5 if i == 4 or i == 5 else 3 for i in y_pred]

    print("Report on Test Data")
    print(calculate_acc(test_y, y_pred))
    print(f1_score(test_y, y_pred, average="macro"))
    print(classification_report(test_y, y_pred))


