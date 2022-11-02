import glob
import logging
from matplotlib import pyplot
import numpy
import pandas
import tensorflow_decision_forests as tfdf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def read_data(path):
    """Read in all data from path, asterisks are allowed"""
    filenames = glob.glob(path)
    logger.info("Read input files")
    df = pandas.concat([pandas.read_csv(f)
                       for f in filenames], ignore_index=True)
    return df


def prepare_data(df):
    """Data preparation

    - make the label an integer
    - starttime/stoptime to datetime
    - derive quantities frrom starttime
    - make gender a category
    """
    logger.info("Prepare Data")
    label = "usertype"
    classes = ['Customer', 'Subscriber']
    df["usertype_int"] = df[label].map(classes.index)
    df['starttime'] = pandas.to_datetime(df['starttime'])
    df['stoptime'] = pandas.to_datetime(df['stoptime'])
    df['dayofweek'] = df['starttime'].dt.dayofweek
    df['weekofyear'] = df['starttime'].dt.isocalendar().week
    df['hour'] = df['starttime'].dt.hour
    categorical_feature_names = ['gender']
    for feature_name in categorical_feature_names:
        df[feature_name] = df[feature_name].astype(str)
    return df


def train(df, features, label, model_class):
    """Train the model provided in model_class"""
    logger.info("Train")
    columns = features + [label]
    df2 = df[columns]
    tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        df2, label, batch_size=1000, max_num_classes=2)
    model = model_class()
    model.fit(tf_dataset)
    print(model.summary())
    return model


def predict(model, df, features, label):
    """Predict on data from dataframe df with featrues and label."""
    logger.info("Predict")
    columns = features + [label]
    df2 = df[columns]
    tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        df2, label, batch_size=1000, max_num_classes=2)
    model.evaluate(tf_dataset)
    result = model.predict(tf_dataset)
    tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0, max_depth=3)
    return result


def split_dataset(df, test_ratio=0.30):
    """Splits a panda dataframe in two based on the ratio."""
    logger.info("Split train/test dataset")
    test_indices = numpy.random.rand(len(df)) < test_ratio
    return df[~test_indices], df[test_indices]


def main():
    df = read_data("data/sample.csv")
    df = prepare_data(df)
    df_train, df_test = split_dataset(df)
    # define features and model
    features = ['dayofweek', 'hour', 'tripduration',
                'start station name', 'end station name', 'birth year', 'gender']
    model_type = tfdf.keras.GradientBoostedTreesModel
    label = "usertype_int"
    # train
    model = train(df_train, features, label, model_type)
    # predict
    result = predict(model, df_test, features, label)
    # store results
    df_test['result'] = result
    df_test.to_csv(f"model/{model_type.__name__}")
    model.save(f"model/{model_type.__name__}.model")
    logger.info("Finished")


if __name__ == '__main__':
    main()
