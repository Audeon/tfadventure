import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def load_and_parse_data(train_url, eval_url, **kwargs):

    try:
        dftrain = pd.read_csv(str(train_url).strip())
    except Exception as err:
        raise Exception(str(err))

    try:
        dfeval = pd.read_csv(str(eval_url).strip())
    except Exception as err:
        raise Exception(str(err))

    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')

    cata_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

    num_columns = ['age', 'fare']

    feature_columns = [] # Theta

    for feature_name in cata_columns:
        vocabulary = dftrain[feature_name].unique() # Gets list of unique values from each colum
        print(vocabulary)
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in num_columns:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    print(feature_columns)

    return { 'dftrain' : dftrain, 'dfeval': dfeval, 'y_train': y_train, 'y_eval': y_eval, 'cata_columns': cata_columns,
             'num_columns': num_columns, 'feature_columns': feature_columns}

def make_input_fn(data_df, label_df, num_epocs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # Create tensorflow dataset object with labels
        if shuffle:
            ds = ds.shuffle(1000) # Randomizes the order of the data
        ds = ds.batch(batch_size).repeat(num_epocs) # Splits dataset into batchs of 32 and repeats the process for each epoc.
        return ds
    return input_function

def check_people(person_num, result_data, dfeval, y_eval):
    per = int(person_num)
    print(f"Person[{per}]: {dfeval.loc[per]}")
    print(f"Chance of survival: {result_data[per]['probabilities'][1]}")
    print(f"Did they actually die (0 = Dead, 1 = Alive): {y_eval[per]}")

if __name__ == "__main__":

    url_train = './train.csv'
    url_eval = './eval.csv'

    all_data = load_and_parse_data(url_train, url_eval)

    dftrain = all_data['dftrain']
    dfeval = all_data['dfeval']
    y_train = all_data['y_train']
    y_eval = all_data['y_eval']
    feature_columns = all_data['feature_columns']
    # dftrain.age.hist(bins=20)
    # plt.show()
    #
    # print(dftrain.loc[0])
    # dftrain.sex.value_counts().plot(kind='barh')  # Bar Horizontal
    # plt.show()
    # dftrain['class'].value_counts().plot(kind='barh')
    # plt.show()
    # pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% Survived')
    # plt.show()

    train_input_fn = make_input_fn(dftrain, y_train, num_epocs=20) # Here we call the input function that was returned to get dataset object we can feed
    eval_input_fn = make_input_fn(dfeval, y_eval, num_epocs=1, shuffle=False)

    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

    linear_est.train(train_input_fn)
    result = linear_est.evaluate(eval_input_fn)

    print(f"Accuracy: {result['accuracy']}")

    result = list(linear_est.predict(eval_input_fn))

    check_people(0, result, dfeval, y_eval)