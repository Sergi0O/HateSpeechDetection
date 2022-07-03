import pandas as pd


# Carga los datos de entrenamiento y test
def load_train_test(train_path="train.tsv", test_path="test.tsv"):

    train = pd.read_csv(train_path, names=['id', 'sentence', 'label'], sep='\t')
    test = pd.read_csv(test_path, names=['id', 'sentence', 'label'], sep='\t')

    print(f"\nTrain tweets: {len(train)}")
    print(train.label.value_counts())
    print(f"\nTest tweets: {len(test)}")
    print(test.label.value_counts())

    return train, test
