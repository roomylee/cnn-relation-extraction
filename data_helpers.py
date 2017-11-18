import numpy as np
import pandas as pd
import nltk


def convertFile(filepath, outputpath):
    data = []
    lines = [line.strip() for line in open(filepath)]
    for idx in range(0, len(lines), 4):
        id = lines[idx].split("\t")[0]
        label = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.replace("<e1>", " _e1_ ").replace("</e1>", " _/e1_ ")
        sentence = sentence.replace("<e2>", " _e2_ ").replace("</e2>", " _/e2_ ")
        tokens = nltk.word_tokenize(sentence)

        tokens.remove('_/e1_')
        tokens.remove('_/e2_')

        e1 = tokens.index("_e1_")
        del tokens[e1]

        e2 = tokens.index("_e2_")
        del tokens[e2]

        sentence = " ".join(tokens)

        data.append([id, sentence, e1, e2, label])

    df = pd.DataFrame(data=data, columns=["id", "sentence", "e1_pos", "e2_pos", "label"])
    df.to_csv(outputpath, index=False)


if __name__ == "__main__":
    trainFile = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    testFile = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'

    convertFile(trainFile, "data/train.csv")
    convertFile(testFile, "data/test.csv")

    print("Train / Test file created")



def load_data_and_labels(path):
    # read training data from CSV file
    data = pd.read_csv(path)

    print('data({0[0]},{0[1]})'.format(data.shape))
    print(data.head())

    images = data.iloc[:, 1:].values
    images = images.astype(np.float)

    # convert from [0:255] => [0.0:1.0]
    images = np.multiply(images, 1.0 / 255.0)
    print('images({0[0]},{0[1]})'.format(images.shape))

    image_size = images.shape[1]
    print('image_size => {0}'.format(image_size))

    # in this case all images are square
    image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint16)
    print('image_width => {0}\nimage_height => {1}'.format(image_width, image_height))

    labels_flat = data['label'].values.ravel()
    print('labels_flat({0})'.format(len(labels_flat)))
    print('labels_flat[{0}] => {1}'.format(10, labels_flat[10]))

    labels_count = np.unique(labels_flat).shape[0]
    print('labels_count => {0}'.format(labels_count))

    # convert class labels from scalars to one-hot vectors
    # 0 => [1 0 0 0 0 0 0 0 0 0]
    # 1 => [0 1 0 0 0 0 0 0 0 0]
    # ...
    # 9 => [0 0 0 0 0 0 0 0 0 1]
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    IMAGE_TO_DISPLAY = 10
    print('labels({0[0]},{0[1]})'.format(labels.shape))
    print('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY, labels[IMAGE_TO_DISPLAY]))

    return images, labels

def load_test_data(path):
    # read training data from CSV file
    data = pd.read_csv(path)

    print('data({0[0]},{0[1]})'.format(data.shape))
    print(data.head())

    images = data.values
    images = images.astype(np.float)

    # convert from [0:255] => [0.0:1.0]
    images = np.multiply(images, 1.0 / 255.0)
    print('images({0[0]},{0[1]})'.format(images.shape))

    image_size = images.shape[1]
    print('image_size => {0}'.format(image_size))

    # in this case all images are square
    image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint16)
    print('image_width => {0}\nimage_height => {1}'.format(image_width, image_height))

    return images

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
