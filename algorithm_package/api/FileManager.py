import numpy as np
import ast

class FileManager:
    absolute_path = '/'

    X_TRAIN = '{}/data/_data/final/train_data.csv'.format(absolute_path)
    Y_TARGET = '{}/data/_data/final/target_data.csv'.format(absolute_path)
    X_TEST = '{}/data/_data/final/test_train_data.csv'.format(absolute_path)
    Y_TEST = '{}/data/_data/final/test_target_data.csv'.format(absolute_path)

    ML_PATH = 'saved_ml_model/data_aggregation_model.h5'
    ML_PATH_TRAINED = 'saved_ml_model/data_aggregation_model_trained.h5'
    # ML_PATH = 'data_aggregation_model.h5'
    # ML_PATH_TRAINED = 'data_aggregation_model_trained.h5'

    ML_MODEL_TRAINING_ACCURACY = "training_report/data.txt"
    ML_MODEL_ACCURACY = "training_report/accuracy.txt"
    ML_MODEL_VALIDATION_ACCURACY = "training_report/val_accuracy.txt"

    def __init__(self):
        pass

    @staticmethod
    def get_accuracy_data():
        data_array = None
        with open(FileManager.ML_MODEL_VALIDATION_ACCURACY) as f:
            data_array = f.readline()

        data_array = np.array(ast.literal_eval(data_array))
        data_array = data_array.reshape(-1, 1)
        return data_array

    @staticmethod
    def get_val_accuracy_data():

        data_array = None
        with open(FileManager.ML_MODEL_VALIDATION_ACCURACY) as f:
            data_array = f.readline()

        data_array = np.array(ast.literal_eval(data_array))
        data_array = data_array.reshape(-1, 1)
        return data_array