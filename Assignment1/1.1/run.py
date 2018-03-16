import json
import pickle
import sys
from sklearn.svm import LinearSVC


def convert(ldamodel, dictionary, data):

    print("Converting each sample to bow and lda")
    data = [ldamodel[dictionary.doc2bow(each.split)] for each in tqdm(data)]

    print("Converting it into SVM's format")
    for i in tqdm(range(len(data))):
        text = dat[i]
        temp = [0 for i in range(100)]
        for a, b in text:
            temp[a] = b
        data[i] = temp

    return data

input_file = sys.argv[1]
output_file = sys.argv[2]

print("Loading Test Set")
test_data = []
test_y = []
with open(input_file, "r") as f:
    i = 0
    for line in tqdm(f):
        i += 1
        sys.stdout.write("\r\x1b[K" + "%d" % (i))
        sys.stdout.flush()

        temp = json.loads(line)
        test_data.append((temp["summary"] + " ") * 3 + temp["reviewText"])
        test_y.append(int(temp["overall"]))

print ("Loading Model")
model = pickle.load(open("2017MCS2074.model", "rb"))

test_x = convert(ldamodel, dictionary, test_data)
del test_data
print("Predict")
y_pred = model.predict(test_x)

y_pred = [1 if i == 1 or i == 2 else 5 if i == 4 or i == 5 else 3 for i in y_pred]

with open(output_file, "w") as f:
    for y in y_pred:
        f.write("%d\n" %y)
