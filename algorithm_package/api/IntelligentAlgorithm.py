import keras
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from algorithm_package.api.NeuralNetworkConfig import *
from algorithm_package.api.CombineData import *
from algorithm_package.api.FileManager import *
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


class IntelligentAlgorithm:

    def __init__(self):

        self.iaResult = None
        self.nnc = NeuralNetworkConfig()
        self.dataFrame = {}
        self.iDataComplete = None
        self.transformedData = None

        self.model = None
        self.targetData = None
        self.sampleData = None

        self.X_train = None
        self.X_train_test = None
        self.y_target = None
        self.y_target_test = None

        self.final_df = None
        self.final_target_DF = None,

        self.sampleData = pd.read_csv('../excel_package/data_files/data_file.csv')

        self.total_columns = [
            'classification',
            'variable',
            'obj_energy',
            'obj_bandwidth',
            'obj_latency',
            'field_size',
            'sink_distance',
            'atmosphere',
            'limit_energy',
            'limit_bandwidth',
            'limit_latency',
            'number_of_nodes',
            'aggregation_function',
            'physical_topology',
            'logical_topology',
            'sampling_rate',
            'required_connectivity',
            'best_technique'
        ]
        self.cat_columns = [
            'classification',
            'variable',
            'field_size',
            'sink_distance',
            'atmosphere',
            'number_of_nodes',
            'aggregation_function',
            'physical_topology',
            'logical_topology',
            'sampling_rate',
            'required_connectivity',
            'best_technique'
        ]
        self.real_columns = [
            'obj_energy',
            'obj_bandwidth',
            'obj_latency',
            'limit_energy',
            'limit_bandwidth',
            'limit_latency',
        ]

        self.total_columns = [
            'classification',
            'atmosphere',
            'field_size',
            'number_of_nodes',
            'physical_topology',
            'sampling_rate',
            'best_technique',
            'obj_energy',
            'obj_bandwidth',
            'obj_latency'
        ]
        self.cat_columns = [
            'classification',
            'atmosphere',
            'field_size',
            'physical_topology',
            'best_technique'
        ]
        self.real_columns = [
            'number_of_nodes',
            'sampling_rate',
            'obj_energy',
            'obj_bandwidth',
            'obj_latency'
        ]

        """
        self.binary = [
            'homogenous_nodes',
            'periodic_reporting',
            'event_reporting',
            'real_time_monitoring',
            'event_monitoring',
            'location_awareness',
            'node_mobility'
        ]
        """
        self.sampleData = self.sampleData[self.total_columns]

    def startIAProcess(self, simulationDataDF=None, rebuild=False):

        # process data
        self.processData()

        # build model rebuild = True - will rebuild model from scratch | False - will load from saved model
        self.buildKerasModel(True)

        # train model
        self.trainKerasModel()

        ######################################
        # self.seabornAnalysis()

    def startGridSearchProcess(self):
        # process data
        self.processData()

        # build model rebuild = True - will rebuild model from scratch | False - will load from saved model
        self.buildKerasModel(True)

        self.apply_scikit_learn()

    def processData(self):

        rebuild = True

        if rebuild:
            self.sampleData, self.targetData = self.preprocess(self.sampleData, self.real_columns, self.cat_columns)

            # CombineData.saveTrainTestData(self.sampleData, self.targetData)
        else:
            # use this when by-passing earlier code
            self.sampleData, self.targetData, = CombineData.readTrainTestData()

        # split data
        self.X_train, self.X_train_test, self.y_target, self.y_target_test = train_test_split(
            self.sampleData,
            self.targetData,
            train_size=0.8,
            shuffle=12,
            random_state=15)

        """
        CombineData.saveTransformedData( self.X_train, self.y_target, self.X_train_test, self.y_target_test)
        """


    # Intelligent Algorithm Code
    ############################################
    # process techniques data
    # filter techniques with values outside the acceptable ranges - [perform this filtering later - 13/08/2020]
    # then select the technique based on its leve of compliance
    ########################################

    # filter out all techniques with values that do not comply to this round
    # select the best technique and insert in the 'best_tech' column

    def preprocess(self, sampleData, real_columns, categorical_columns):

        selected_cat_data_DF = sampleData[categorical_columns]
        selected_real_sampleData = sampleData[real_columns]

        # Convert reals to numeric from string
        a = pd.to_numeric(selected_real_sampleData['obj_energy'])
        b = pd.to_numeric(selected_real_sampleData['obj_bandwidth'])
        c = pd.to_numeric(selected_real_sampleData['obj_latency'])
        d = pd.to_numeric(selected_real_sampleData['sampling_rate'])
        e = pd.to_numeric(selected_real_sampleData['number_of_nodes'])

        selected_real_sampleData = pd.DataFrame()
        selected_real_sampleData['obj_energy'] = a
        selected_real_sampleData['obj_bandwidth'] = b
        selected_real_sampleData['obj_latency'] = c
        selected_real_sampleData['sampling_rate'] = d
        selected_real_sampleData['number_of_nodes'] = e

        """ Skipping processing the Real Numbers"""

        standardScaler = StandardScaler()
        scaled_real_data_df1 = standardScaler.fit_transform(selected_real_sampleData)
        minMaxScaler = MinMaxScaler(feature_range=(0, 1))
        range_scaled_real_data_df1 = minMaxScaler.fit_transform(scaled_real_data_df1)

        encoded_real_data_DF = pd.DataFrame(range_scaled_real_data_df1, columns=['obj_energy', 'obj_bandwidth', 'obj_latency', 'sampling_rate', 'number_of_nodes'])

        # self.final_df = encoded_real_data_DF
        ############# END REAL DATA PROCESSING


        # Remove target column into separate variable
        data_target_DF = selected_cat_data_DF[['best_technique']]
        data_cat_DF = selected_cat_data_DF.drop(['best_technique'], axis=1)

        # Split out target for processing ######################################

        lbe = LabelEncoder()
        lbe.fit(['Leach', 'Heed', 'Pegasis', 'DBST', 'Directed Diffusion' ])

        encoded_target_DF = np.asarray(data_target_DF).ravel()
        encoded_target_DF = lbe.transform(encoded_target_DF)
        encoded_target_DF = np.asarray(encoded_target_DF).reshape(-1, 1)

        """ What am i doing here ? - For Target ?"""
        oneHot = OneHotEncoder(sparse=False)
        ohe_final_target_DF = oneHot.fit_transform(encoded_target_DF)
        self.final_target_DF = pd.DataFrame(ohe_final_target_DF)

        # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        # X_scaled = X_std * (max - min) + min

        ##### Process categorical data
        oneHot = OneHotEncoder(sparse=False)
        oneHot.fit(data_cat_DF)
        oneHotTransform = oneHot.transform(data_cat_DF)
        encoded_data_cat_DF = pd.DataFrame(oneHotTransform)

        # Real Features  ######################################
        # print(selected_real_sampleData)
        # selected_real_sampleData.to_csv("data_s.csv")

        self.final_df = encoded_real_data_DF

        print(self.final_df)
        self.final_df = self.final_df.join(encoded_data_cat_DF) #, lsuffix='_caller', rsuffix='_other')
        print(self.final_df)

        return self.final_df, self.final_target_DF

    def check_feature_correlation(self, simulationDataDF=None):
        # simulationDataDF = CombineData.readData()
        """
        for i, col in enumerate(['packetSize']):
            plt.figure(i)
            sns.catplot(x = col, y='energy', data=simulationDataDF, kind='point', aspect=2, )
        """

        sns.pairplot(simulationDataDF[['energy', 'bandwidth']])
        # sns.pairplot(energy)
        # sns.plt.show()

        exit()

    def buildKerasModel(self, rebuild=True):

        # try to load model first
        # model_exists, model = CombineData.readModel()

        #if rebuild or not model_exists:
        self.model = Sequential()

        self.model.add(Dense(self.nnc.hidden_layer_1, activation='relu', kernel_initializer=self.nnc.init[0], input_shape=(self.nnc.input_nodes,)))
        self.model.add(Dense(self.nnc.hidden_layer_2, activation='relu'))
        self.model.add(Dense(self.nnc.hidden_layer_3, activation='relu'))
        self.model.add(Dense(self.nnc.output_layer, activation='softmax'))
        # Use SGD algorithm - for testing
        self.model.compile(optimizer=self.nnc.optimizer, loss=self.nnc.loss, metrics=self.nnc.metrics)

        CombineData.saveModel(self.model)
        self.model.save(FileManager.ML_PATH)
        # else:
        #    self.model = model

        print(self.model.summary())
        print('Model has been saved...')

    def trainKerasModel(self):

        # trainDf, targetDf, trainTestDf, targetTestDF
        # X_train, y_target, X_test, y_test = CombineData.readTransformedData()
        # x_vals = np.asarray(X_train.values)
        # y_vals = np.asarray(y_target.values)

        history = self.model.fit(
            self.X_train.values, self.y_target.values,
            epochs=self.nnc.epochs,
            batch_size=self.nnc.batch,
            shuffle=True,
            verbose=2,
            validation_data=(self.X_train_test.values, self.y_target_test.values)
        )

        print('Accuracy: \n{}'.format(history.history['categorical_accuracy']))
        print('Validation Accuracy: \n{}'.format(history.history['val_categorical_accuracy']))

        test_error_rate = self.model.evaluate(self.X_train_test.values,self.y_target_test.values, verbose=0)
        print('Mean Squared Error (MSE): {}'.format(test_error_rate))

        self.model.save(FileManager.ML_PATH_TRAINED)
        # CombineData.saveModel(self.model)
        self.saveTrainingReport(None, history.history['categorical_accuracy'], history.history['val_categorical_accuracy'])

        # Plot the accuracy curve

    # API
    def saveTrainingReport(self, data, accuracy, val_accuracy):

        if data is not None:
            f = open(FileManager.ML_MODEL_TRAINING_ACCURACY, "w")
            f.write(data)
            f.close()
            # with open("training_report/data.json", "w") as json_file:
            #    json.dump(data, json_file)

        f = open(FileManager.ML_MODEL_ACCURACY, "w")
        f.write(str(accuracy))
        f.close()

        f = open(FileManager.ML_MODEL_VALIDATION_ACCURACY, "w")
        f.write(str(val_accuracy))
        f.close()

        self.plot_accuracy_graphs()
        print('All files written...')

    def plot_accuracy_graphs(self):

        fig, axes = plt.subplots()
        fig.tight_layout()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.3, hspace=0.2)

        accuracyDF = pd.DataFrame(FileManager.get_accuracy_data(), columns=['accuracy'])
        valAccuracyDF = pd.DataFrame(FileManager.get_val_accuracy_data(), columns=['val_accuracy'])

        plt.title('Validation Accuracy: Batch: {}, Optimiser: {}'.format(self.nnc.batch, 'rmsprop'))
        plt.legend('Accuracy', loc='upper left')
        plt.xlim(0, 250)
        # plt.ylim(0, 1.2)
        axes.set_ylabel('Val Accuracy')
        axes.set_xlabel('Epoch')

        ax = accuracyDF.plot(kind='line', y='accuracy', color='red', ax=axes, label='Accuracy')
        ax = valAccuracyDF.plot(kind='line', y='val_accuracy', color='blue', ax=ax, label='Val Accuracy')

        plt.show()

    def apply_scikit_learn(self):
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        validation_data = (self.X_train_test.values, self.y_target_test.values)

        # create model
        model = KerasClassifier(build_fn=self.model, verbose=0)

        # grid search epochs, batch size and optimizer
        optimizers = self.nnc.optimizers
        init = self.nnc.init
        epochs = self.nnc.epochs_array
        batches = self.nnc.batches_array

        param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid_result = grid.fit(self.X_train.values, self.y_target.values)

        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        exit()

    def model_predict(self, requirements):

        pass

        self.model = keras.models.load_model(FileManager.ML_PATH_TRAINED)
