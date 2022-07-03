from os import path
import sys
from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from load_tweets import load_train_test
from preprocess import preprocess
from datetime import datetime


BATCH_SIZE = 16
N_EPOCHS = 3
BETO_PATH = "dccuchile/bert-base-spanish-wwm-uncased"
ROBERTA_PATH = "PlanTL-GOB-ES/roberta-base-bne"
BERTIN_PATH = "bertin-project/bertin-roberta-base-spanish"


def encode_sentences(tokenizer, sentences, y):
    encodings = tokenizer(list(sentences.values), truncation=True, padding=True)
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), list(y.values)))
    return dataset


def train_model(model, tokenizer, sentences_train, y_train):
    train_dataset = encode_sentences(tokenizer, sentences_train, y_train)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
    weight_dic = {0: class_weight[0], 1: class_weight[1]}
    model.fit(train_dataset.batch(BATCH_SIZE), epochs=N_EPOCHS, batch_size=BATCH_SIZE, class_weight=weight_dic)


def evaluate_model(model, tokenizer, sentences_test, y_test):
    test_dataset = encode_sentences(tokenizer, sentences_test, y_test)

    preds = model.predict(test_dataset.batch(BATCH_SIZE)).logits
    predicted_class_id = tf.math.argmax(preds, axis=-1)
    return classification_report(y_test, predicted_class_id, digits=4)


def predict_tweet(text, mod, tok):
    encodings = tok([text], truncation=True, padding=True)
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))
    preds = mod.predict(dataset.batch(1)).logits
    print(preds)
    predicted_class_id = int(tf.math.argmax(preds, axis=-1)[0])
    return predicted_class_id


def loop_predict(mod, tok):
    text = input("Tweet: ")
    while text:
        print(predict_tweet(text, mod, tok))
        text = input("Tweet: ")


def train_beto(sentences_train, y_train, sentences_test, y_test, save_model, results_file):
    # Carga el modelo preentrenado
    beto = TFAutoModelForSequenceClassification.from_pretrained(BETO_PATH, num_labels=2, from_pt=True)
    beto_tokenizer = AutoTokenizer.from_pretrained(BETO_PATH, do_lower_case=True, truncation=True, padding=True)

    train_model(beto, beto_tokenizer, sentences_train, y_train)
    results = evaluate_model(beto, beto_tokenizer, sentences_test, y_test)

    save_results(beto, results, "beto_" + results_file, save_model)
    

def train_roberta(sentences_train, y_train, sentences_test, y_test, save_model, results_file):
    # Carga el modelo preentrenado
    roberta = TFAutoModelForSequenceClassification.from_pretrained(ROBERTA_PATH, num_labels=2, from_pt=True)
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH, truncation=True, padding=True)

    train_model(roberta, roberta_tokenizer, sentences_train, y_train)
    results = evaluate_model(roberta, roberta_tokenizer, sentences_test, y_test)

    save_results(roberta, results, "roberta_" + results_file, save_model)


def train_bertin(sentences_train, y_train, sentences_test, y_test, save_model, results_file):
    # Carga el modelo preentrenado
    bertin = TFAutoModelForSequenceClassification.from_pretrained(BERTIN_PATH, num_labels=2, from_pt=True)
    bertin_tokenizer = AutoTokenizer.from_pretrained(BERTIN_PATH, truncation=True, padding=True)

    train_model(bertin, bertin_tokenizer, sentences_train, y_train)
    results = evaluate_model(bertin, bertin_tokenizer, sentences_test, y_test)

    save_results(bertin, results, "bertin_" + results_file, save_model)


def save_results(model, results, results_file, save_model):
    with open(path.join("models", results_file + ".txt"), 'w') as f:
        for result in results:
            f.write(result)

    if save_model:
        model.save_pretrained(path.join("models", results_file))


def train_models(model_type, train_file, test_file, preprocess_tweets, save_models=True):
    # Carga los tweets de los ficheros
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

    if model_type == 0:
        train_beto(sentences_train, y_train, sentences_test, y_test, save_models, results_file)
    elif model_type == 1:
        train_roberta(sentences_train, y_train, sentences_test, y_test, save_models, results_file)
    elif model_type == 2:
        train_bertin(sentences_train, y_train, sentences_test, y_test, save_models, results_file)


if __name__ == '__main__':

    model_type = int(sys.argv[1])
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    preprocess_tweets = sys.argv[4] == "True"

    train_models(model_type, train_file, test_file, preprocess_tweets)
