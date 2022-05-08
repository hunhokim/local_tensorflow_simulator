import pandas as pd
import tensorflow as tf

def _data_to_tf_train_feature(data: list):
    if type(data) is not list:
        raise TypeError("The input should be a list.")

    data_type = type(data[0])
    if data_type == int:
        feature_list = tf.train.Int64List(value=data)
        feature = tf.train.Feature(int64_list=feature_list)
    elif data_type == float:
        feature_list = tf.train.FloatList(value=data)
        feature = tf.train.Feature(float_list=feature_list)
    else:
        feature_list = tf.train.BytesList(value=data)
        feature = tf.train.Feature(bytes_list=feature_list)

    return feature


def _pandas_df_to_tfrecords(df, output_path):
    dict_features = {}
    for key in df.keys():
        dict_features[key] = _data_to_tf_train_feature(list(df[key]))
    
    features = tf.train.Features(feature=dict_features)
    example = tf.train.Example(features=features)

    with tf.io.TFRecordWriter(output_path) as writer:
        writer.write(example.SerializeToString())


def _spark_df_to_tfrecords(df, output_path):
    pass


def df_to_tfrecords(df, output_path):
    if isinstance(df, pd.DataFrame):
        _pandas_df_to_tfrecords(df, output_path)
    elif isinstance(df, spark.sql.DataFrame):
        _spark_df_to_tfrecords(df, output_path)
    else:
        raise TypeError("df is not a valid type of DataFrame.")


