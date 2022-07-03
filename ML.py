import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from joblib import dump
from load_tweets import load_train_test
from preprocess import preprocess
from datetime import datetime


def make_vectorizer(sentences_train):
    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    vectorizer.fit(sentences_train)
    vocabulary_len = len(vectorizer.vocabulary_)
    print(f"Vocabulary size: {vocabulary_len}")
    return vectorizer


# SVC
def train_svc(x_train, y_train):
    class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
    classifier_svc = SVC(class_weight={0: class_weight[0], 1: class_weight[1]})
    # Hiperparámetros
    param_grid_svc = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf']
    }

    grid_svc = GridSearchCV(classifier_svc, param_grid_svc, refit=True, verbose=3, n_jobs=-1)
    grid_svc.fit(x_train, y_train)
    # Mejor ajuste de hiperparámetros
    print(grid_svc.best_params_)

    return grid_svc


# Random Forest
def train_rf(x_train, y_train):
    class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
    classifier_rf = RandomForestClassifier(class_weight={0: class_weight[0], 1: class_weight[1]})
    # Hiperparámetros
    param_grid_rf = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    grid_rf = GridSearchCV(classifier_rf, param_grid_rf, refit=True, verbose=3, n_jobs=-1)
    grid_rf.fit(x_train, y_train)
    # Mejor ajuste de hiperparámetros
    print(grid_rf.best_params_)

    return grid_rf


# Logisitc Regression
def train_lr(x_train, y_train):
    class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
    classifier_lr = LogisticRegression(class_weight={0: class_weight[0], 1: class_weight[1]})
    # Hiperparámetros
    param_grid_lr = {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['liblinear']
    }

    grid_lr = GridSearchCV(classifier_lr, param_grid_lr, refit=True, verbose=3, n_jobs=-1)
    grid_lr.fit(x_train, y_train)
    # Mejor ajuste de hiperparámetros
    print(grid_lr.best_params_)

    return grid_lr


def evaluate_model(model, x_test, y_test):
    # Se usa el modelo para predecir las categorías de los tweets del conjunto de test
    y_pred = model.predict(x_test)
    return classification_report(y_test, y_pred, digits=4)


def write_results(preprocess_tweets, results, train_file):
    if preprocess_tweets:
        results_file = "ML_" + train_file.split('.')[0] + "_results_preprocess" + datetime.now().strftime("%Y%m%d%H%M%S") + ".txt"
    else:
        results_file = "ML_" + train_file.split('.')[0] + "_results_no_preprocess" + datetime.now().strftime("%Y%m%d%H%M%S") + ".txt"

    with open(results_file, 'w') as f:
        for result in results:
            f.write(result)


def train_models(train_file, test_file, preprocess_tweets=True, save_models=True):
    # Carga los tweets de los ficheros
    train, test = load_train_test(train_path=train_file, test_path=test_file)
    y_train = train.label
    y_test = test.label

    if preprocess_tweets:
        # Preprocesa los tweets
        sentences_train = train.sentence.apply(preprocess)
        sentences_test = test.sentence.apply(preprocess)
    else:
        sentences_train = train.sentence
        sentences_test = test.sentence

    # Crea un vocabulario con las palabras que aparecen en las frases de entrenamiento
    vectorizer = make_vectorizer(sentences_train)

    # Crea vectores de tipo BOW
    x_train = vectorizer.transform(sentences_train)
    x_test = vectorizer.transform(sentences_test)

    # SVC
    print("\n---------\n SVC \n---------\n")
    svc = train_svc(x_train, y_train)
    # RF
    print("\n---------\n RF \n---------\n")
    rf = train_rf(x_train, y_train)
    # LG
    print("\n---------\n LR \n---------\n")
    lg = train_lr(x_train, y_train)

    # Guarda los modelos
    if save_models:
        svc_file = "SVC_"
        rf_file = "RF_"
        lg_file = "LR_"
        if preprocess_tweets:
            svc_file += "preprocess_"
            rf_file += "preprocess_"
            lg_file += "preprocess_"
        date = datetime.now().strftime("%Y%m%d%H%M%S")
        svc_file += train_file.split('.')[0] + date + ".pkl"
        rf_file += train_file.split('.')[0] + date + ".pkl"
        lg_file += train_file.split('.')[0] + date + ".pkl"
        dump(svc.best_estimator_, svc_file)
        dump(rf.best_estimator_, rf_file)
        dump(lg.best_estimator_, lg_file)

    # Resultados
    results = [
        "\n---------\n SVC \n---------\n" +
        str(svc.best_params_) + '\n' +
        evaluate_model(svc, x_test, y_test),
        "\n---------\n RF \n---------\n" +
        str(rf.best_params_) + '\n' +
        evaluate_model(rf, x_test, y_test),
        "\n---------\n LR \n---------\n" +
        str(lg.best_params_) + '\n' +
        evaluate_model(lg, x_test, y_test)
    ]
    write_results(preprocess_tweets, results, train_file)


if __name__ == '__main__':

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    preprocess_tweets = sys.argv[3] == "True"

    train_models(train_file, test_file, preprocess_tweets)
