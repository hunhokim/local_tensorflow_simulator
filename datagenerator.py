import numbers
import warnings

import pandas as pd
import numpy as np
import yaml

class DataGenerator:
    def __init__(self, num_rows, path_yml):
        if type(num_rows) is not int:
            raise TypeError("num_rows should be a type int.")
        
        self._num_rows = num_rows

        with open(path_yml, 'r') as f:
            self._config = yaml.load(f, Loader=yaml.Loader)

    def _generate_a_feature(self, name):
        config = self._config[name]

        if config['type'] == 'numeric':
            min_value = config['min_value']
            max_value = config['max_value']

            if isinstance(min_value, numbers.Number) or isinstance(max_value, numbers.Number):
                raise TypeError("min_value and max_value in the configuration file should be numeric.")
            if min_value > max_value:
                raise ValueError("min_value should be smaller than max_value.")

            feature = np.linspace(min_value, max_value, self._num_rows)

        return feature

    def _make_input_features(self):
        dict_features = {}
        for name in self._config.keys():
            dict_features[name] = self._generate_a_feature(name)

        return dict_features

    def _make_feature_labels(self, function, dict_features):
        labels = function(dict_features)

        return labels

    def generate_data(self, function):
        if not callable(function):
            raise TypeError("function should be callable.")

        dict_features = self._make_input_features()
        dict_features['label'] = self._make_feature_labels(function, dict_features)

        return pd.DataFrame(dict_features)


def sigmoid_1d(dict_features: dict):
    if type(dict_features) is not dict:
        raise TypeError("dict_features should be a dictionary.")
    if len(dict_features) != 1:
        raise ValueError("dict_features should have only one key.")

    key = list(dict_features.keys())[0]
    x = dict_features[key]

    if type(x) is list:
        warnings.warn("The input is converted to numpy.ndarray.")
        x = np.array(x)
    if type(x) is not np.ndarray:
        raise TypeError("The input data should be numpy.ndarray.")

    y = 1 / (1 + np.exp(-x))

    return y
