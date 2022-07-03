import sys
from os import path
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from load_tweets import load_train_test
from preprocess import preprocess
from datetime import datetime


def make_tokenizer(sentences_train):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences_train)
    vocabulary_len = len(tokenizer.word_index)
    print(f"Vocabulary size: {vocabulary_len}")
    return tokenizer


def create_embedding_matrix(filepath, word_index, embedding_dim):
    voc_size = len(word_index) + 1
    embedding_mat = np.zeros((voc_size, embedding_dim))

    with open(filepath) as f:
        next(f)
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_mat[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    nonzero = np.count_nonzero(np.count_nonzero(embedding_mat, axis=1))
    voc_coverage = round(100 * nonzero / voc_size, 2)
    print(f"{voc_coverage}% vocabulary coverage")

    return embedding_mat


def create_cnn_model(num_filters, kernel_size, vocab_size, embedding_matrix, embedding_dim, maxlen):
    model = Sequential()
    model.add(
        layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_cnn(x_train, y_train, vocab_size, embedding_matrix, embedding_dim, maxlen):
    class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
    param_grid_cnn = {
        'num_filters': [32, 64, 128],
        'kernel_size': [3, 5, 7],
        'vocab_size': [vocab_size],
        'embedding_matrix': [embedding_matrix],
        'embedding_dim': [embedding_dim],
        'maxlen': [maxlen],
        'class_weight': [{0: class_weight[0], 1: class_weight[1]}]
    }

    clf = KerasClassifier(create_cnn_model, epochs=3, batch_size=10)
    cnn_grid = GridSearchCV(estimator=clf, param_grid=param_grid_cnn, refit=True, verbose=3, n_jobs=4)
    cnn_grid.fit(x_train, y_train)

    return cnn_grid


def create_lstm_model(units, activation, vocab_size, embedding_matrix, embedding_dim, maxlen):
    lstm_model = Sequential()
    lstm_model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
    lstm_model.add(layers.LSTM(units, activation=activation, return_sequences=True, recurrent_dropout=0.1))
    lstm_model.add(layers.Flatten())
    lstm_model.add(layers.Dense(1, activation='sigmoid'))
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return lstm_model


def train_lstm(x_train, y_train, vocab_size, embedding_matrix, embedding_dim, maxlen):
    class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
    param_grid_lstm = {
        'units': [32, 64, 128],
        'activation': ['relu', 'tanh'],
        'vocab_size': [vocab_size],
        'embedding_matrix': [embedding_matrix],
        'embedding_dim': [embedding_dim],
        'maxlen': [maxlen],
        'class_weight': [{0: class_weight[0], 1: class_weight[1]}]
    }

    clf = KerasClassifier(create_lstm_model, epochs=3, batch_size=10)
    lstm_grid = GridSearchCV(estimator=clf, param_grid=param_grid_lstm, refit=True, verbose=3, n_jobs=3)
    lstm_grid.fit(x_train, y_train)

    return lstm_grid


def evaluate_model(model, x_test, y_test):
    # Se usa el modelo para predecir las categorías de los tweets del conjunto de test
    y_pred = model.predict(x_test)
    return classification_report(y_test, y_pred, digits=4)


def save_results(model, results, results_file, save_model):
    with open(path.join("models", results_file + ".txt"), 'w') as f:
        for result in results:
            f.write(result)

    if save_model:
        model.best_estimator_.model.save(path.join("models", results_file + ".h5"))


def train_models(model_type, train_file, test_file, embeddings_path, embedding_dim, preprocess_tweets=True, save_model=True):
    train, test = load_train_test(train_path=train_file, test_path=test_file)
    y_train = train.label
    y_test = test.label

    if preprocess_tweets:
        # Preprocesa los tweets
        sentences_train = train.sentence.apply(preprocess)
        sentences_test = test.sentence.apply(preprocess)
        results_file = train_file.split('.')[0] + "_preprocess_" + datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        sentences_train = train.sentence
        sentences_test = test.sentence
        results_file = train_file.split('.')[0] + "_no_preprocess_" + datetime.now().strftime("%Y%m%d%H%M%S")

    tokenizer = make_tokenizer(sentences_train)
    x_train = tokenizer.texts_to_sequences(sentences_train)
    x_test = tokenizer.texts_to_sequences(sentences_test)

    maxlen = max(map(len, x_train))
    x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
    x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

    embedding_matrix = create_embedding_matrix(embeddings_path, tokenizer.word_index, embedding_dim)
    vocab_size = len(tokenizer.word_index) + 1

    if model_type == 0:

        # CNN
        print("\n---------\n CNN \n---------\n")
        cnn = train_cnn(x_train, y_train, vocab_size, embedding_matrix, embedding_dim, maxlen)
        results = str(cnn.best_params_) + "\n"
        results += evaluate_model(cnn, x_test, y_test)
        results_file = "CNN_" + results_file
        save_results(cnn, results, results_file, save_model)

    elif model_type == 1:

        # LSTM
        print("\n---------\n LSTM \n---------\n")
        lstm = train_lstm(x_train, y_train, vocab_size, embedding_matrix, embedding_dim, maxlen)
        results = str(lstm.best_params_) + "\n"
        results += evaluate_model(lstm, x_test, y_test)
        results_file = "LSTM_" + results_file
        save_results(lstm, results, results_file, save_model)


if __name__ == '__main__':

    emb_path = "cc.es.300.vec"
    emb_dim = 300

    model_type = int(sys.argv[1])
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    preprocess_tweets = sys.argv[4] == "True"

    train_models(model_type, train_file, test_file, emb_path, emb_dim, preprocess_tweets=True)
