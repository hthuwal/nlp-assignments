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
