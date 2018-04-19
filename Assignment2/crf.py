import random
import sklearn_crfsuite
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
from sklearn.grid_search import RandomizedSearchCV
import scipy

random.seed(64)


def read_data(file):
    data = []
    labels = []
    with open(file, "r", errors='replace') as f:
        temp_data = []
        temp_labels = []
        for line in f:
            if line.strip() == "":
                data.append(temp_data)
                labels.append(temp_labels)
                temp_data = []
                temp_labels = []
            else:
                temp_data.append(line.strip().split()[0])
                temp_labels.append(line.strip().split()[1])

    return data, labels


def w2f(sentence, i):
    word = sentence[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sentence[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sentence) - 1:
        word1 = sentence[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features


def data2features(data):
    temp_data = []
    for sentence in data:
        temp_sentence = [w2f(sentence, i) for i in range(len(sentence))]
        temp_data.append(temp_sentence)

    return temp_data


def gridsearch():
    labels = ['T', 'D']
    # define fixed parameters and parameters to search
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        #     'algorithm': ['lbfgs', 'l2sgd', 'ap', 'pa', 'arow'],
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }

    # use the same metric for evaluation
    f1_scorer = make_scorer(metrics.flat_f1_score,
                            average='macro', labels=labels)

    # search
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_scorer)
    rs.fit(train_data, train_labels)
    return rs.best_params_, rs.best_estimator_, rs.best_score_


train_data, train_labels = read_data("train.txt")
train_data = data2features(train_data)

data = list(zip(train_data, train_labels))
random.shuffle(data)
train_data, train_labels = zip(*data)

dev_data, dev_labels = train_data[3001:], train_labels[3001:]
train_data, train_labels = train_data[:3001], train_labels[:3001]

# best params: {'c1': 0.18518537861478376, 'c2': 0.00752984879117378}

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=200,
    all_possible_transitions=True
)
crf.fit(train_data, train_labels)

labels = list(crf.classes_)
labels.remove('O')

y_pred = crf.predict(dev_data)
metrics.flat_f1_score(dev_labels, y_pred,
                      average='macro', labels=labels)

print(metrics.flat_classification_report(y_pred, dev_labels, labels))
