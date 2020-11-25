import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense

import sqlite3
import pandas as pd
from openpyxl import *

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

        # self.sampleData = pd.read_csv('../excel_package/data_files/data_file.csv')
        self.sampleData = pd.read_excel('../excel_package/events_main_v.1.1.xlsx')
        print(self.sampleData)

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

        rebuild = True

        if rebuild:
            # techniques in range and best technique
            # CombineData.saveData(simulationDataDF)

            self.sampleData, self.targetData = self.preprocess(self.sampleData, self.real_columns, self.cat_columns)

            # add patch data to ensure all categorical options are covered
            # patch_data = self.dataHelper.getPatchDataDF()

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
        CombineData.saveTransformedData(
            self.X_train,
            self.y_target,
            self.X_train_test,
            self.y_target_test)
        """

        self.activateKerasModel()

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

        # Remove target column
        data_target_DF = selected_cat_data_DF[['best_technique']]
        data_cat_DF = selected_cat_data_DF.drop(['best_technique'], axis=1)

        # Split out target for processing ######################################

        oneHot = OneHotEncoder(sparse=False)

        lbe = LabelEncoder()
        lbe.fit(['Leach', 'Heed', 'Pegasis', 'DBST', 'Directed Diffusion'])

        encoded_target_DF = np.asarray(data_target_DF).ravel()
        encoded_target_DF = lbe.transform(encoded_target_DF)
        encoded_target_DF = np.asarray(encoded_target_DF).reshape(-1, 1)

        """ What am i doing here ? - For Target ?"""
        ohe_final_target_DF = oneHot.fit_transform(encoded_target_DF)
        self.final_target_DF = pd.DataFrame(ohe_final_target_DF)

        # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        # X_scaled = X_std * (max - min) + min

        # Categorical features processing ######################################
        # use one hot encoder to convert to appropriate value

        # data_cat_DF.to_csv("data_cat.csv")

        # data_cat_DF = data_cat_DF.iloc[:, 0:6]

        oneHot = OneHotEncoder(sparse=False)
        oneHot.fit(data_cat_DF)
        clb1 = oneHot.transform(data_cat_DF)
        # print(type(clb1))
        # print(clb1)
        encoded_data_cat_DF = pd.DataFrame(clb1)

        # Real Features  ######################################
        # print(selected_real_sampleData)
        # selected_real_sampleData.to_csv("data_s.csv")

        """ Skipping processing the Real Numbers"""
        """
        standardScaler = StandardScaler()
        scaled_real_data_df1 = standardScaler.fit_transform(selected_real_sampleData)

        minMaxScaler = MinMaxScaler(feature_range=(0, 1))
        range_scaled_real_data_df1 = minMaxScaler.fit_transform(scaled_real_data_df1)

        encoded_real_data_DF = pd.DataFrame(range_scaled_real_data_df1)

        self.final_df = encoded_real_data_DF
        """

        self.final_df = selected_real_sampleData
        print(self.final_df)
        self.final_df = self.final_df.join(encoded_data_cat_DF)  # , lsuffix='_caller', rsuffix='_other')
        print(self.final_df)

        return self.final_df, self.final_target_DF

    def activateKerasModel(self):

        #####  TRAINING SESSION COMMENCES HERE
        # use this when by-passing earlier code

        # rebuild = True - will rebuild model from scratch | False - will load from saved model
        self.buildKerasModel(True)

        self.trainKerasModel()

        ######################################
        # self.seabornAnalysis()

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

        # if rebuild or not model_exists:
        self.model = Sequential()

        self.model.add(Dense(self.nnc.hidden_layer_1, activation='relu', kernel_initializer='uniform',
                             input_shape=(self.nnc.input_nodes,)))
        self.model.add(Dense(self.nnc.hidden_layer_2, activation='relu'))
        self.model.add(Dense(self.nnc.hidden_layer_3, activation='relu'))
        self.model.add(Dense(self.nnc.output_layer, activation='softmax'))

        self.model.compile(self.nnc.optimizer, self.nnc.loss_function, self.nnc.metrics)

        # CombineData.saveModel(self.model)
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
            epochs=self.nnc.training_epochs,
            batch_size=100,
            shuffle=True,
            verbose=2,
            validation_data=(self.X_train_test.values, self.y_target_test.values)
        )

        print('Accuracy: \n{}'.format(history.history['accuracy']))
        print('Validation Accuracy: \n{}'.format(history.history['val_accuracy']))

        test_error_rate = self.model.evaluate(self.X_train_test.values, self.y_target_test.values, verbose=0)
        print('Mean Squared Error (MSE): {}'.format(test_error_rate))

        self.model.save(FileManager.ML_PATH_TRAINED)
        # CombineData.saveModel(self.model)
        self.saveTrainingReport(None, history.history['accuracy'], history.history['val_accuracy'])

        # Plot the accuracy curve

    def predictOption(self, requirements):
        pass

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

        plt.title('Accuracy and Validation Error')
        plt.legend('Accuracy', loc='upper left')
        plt.xlim(0, 10)
        plt.ylim(0, 1.2)
        axes.set_ylabel('Accuracy')
        axes.set_xlabel('Round')

        ax = accuracyDF.plot(kind='line', y='accuracy', color='red', ax=axes, label='Accuracy')
        # ax = valAccuracyDF.plot(kind='line', y='val_accuracy', color='blue', ax=ax, label='Val Accuracy')

        plt.show()

        print('')


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



class RuleEngine:

    def __init__(self):

        self.techniques = {
            RulesEnums.BestTechnique.LEACH: {'id': 49, 'name': 'Leach'},
            RulesEnums.BestTechnique.HEED: {'id': 50, 'name': 'Heed'},
            RulesEnums.BestTechnique.PEGASIS: {'id': 51, 'name': 'Pegasis'},
            RulesEnums.BestTechnique.DBST: {'id': 52, 'name': 'DBST'},
            RulesEnums.BestTechnique.DIRECTED_DIFFUSION: {'id': 53, 'name': 'Directed Diffusion'}
        }
        self.techniques_list = list(self.techniques)

        # rules = RulesEngine()
        # self.rules_pack = rules.rules_pack

        self.conn = None
        self.cursor = None
        self.question_lists = []

    def process_rules(self):

        self.process_rules_data()

    def process_rules_data(self):

        self.conn = sqlite3.connect('AlgorithmStore.db')
        self.cursor = self.conn.cursor()

        query = "Select ml_attribute_values.id, ml_attributes.attribute_name, ml_attributes.data_type, ml_attribute_values.at_value from ml_attributes, ml_attribute_values where ml_attributes.id=ml_attribute_values.attribute_id order by ml_attributes.id"
        self.cursor.execute(query)

        result_set = self.cursor.fetchall()

        data = {}
        for r in result_set:
            attrib_id = r[0]
            attrib = r[1]
            typ = r[2]
            val = r[3]
            if attrib not in data:
                data.update({attrib: {}})
                data[attrib].update({'ids': [], 'data': []})
            data[attrib]['ids'].append(attrib_id)
            data[attrib]['data'].append(val)

        self.conn.close()

        questions = {}

        data_keys = list(data)
        counter = 0

        fd1 = data_keys[0]
        for idi_1, d1 in zip(data[fd1]['ids'], data[fd1]['data']):
            questions.update({fd1: {"id": idi_1, "value": d1, "display": "{:20}: {:>20}".format(fd1, d1)}})
            fd2 = data_keys[1]
            for idi_2, d2 in zip(data[fd2]['ids'], data[fd2]['data']):
                questions.update({fd2: {"id": idi_2, "value": d2, "display": "{:20}: {:>20}".format(fd2, d2)}})
                fd3 = data_keys[2]
                for idi_3, d3 in zip(data[fd3]['ids'], data[fd3]['data']):
                    questions.update({fd3: {"id": idi_3, "value": d3, "display": "{:20}: {:>20}".format(fd3, d3)}})
                    fd4 = data_keys[3]
                    for idi_4, d4 in zip(data[fd4]['ids'], data[fd4]['data']):
                        questions.update({fd4: {"id": idi_4, "value": d4, "display": "{:20}: {:>20}".format(fd4, d4)}})
                        fd5 = data_keys[4]
                        for idi_5, d5 in zip(data[fd5]['ids'], data[fd5]['data']):
                            questions.update(
                                {fd5: {"id": idi_5, "value": d5, "display": "{:20}: {:>20}".format(fd5, d5)}})
                            fd6 = data_keys[5]
                            for idi_6, d6 in zip(data[fd6]['ids'], data[fd6]['data']):
                                questions.update(
                                    {fd6: {"id": idi_6, "value": d6, "display": "{:20}: {:>20}".format(fd6, d6)}})
                                fd7 = data_keys[6]
                                for idi_7, d7 in zip(data[fd7]['ids'], data[fd7]['data']):
                                    questions.update(
                                        {fd7: {"id": idi_7, "value": d7, "display": "{:20}: {:>20}".format(fd7, d7)}})
                                    fd8 = data_keys[7]
                                    for idi_8, d8 in zip(data[fd8]['ids'], data[fd8]['data']):
                                        questions.update({fd8: {"id": idi_8, "value": d8,
                                                                "display": "{:20}: {:>20}".format(fd8, d8)}})
                                        fd9 = data_keys[8]
                                        for idi_9, d9 in zip(data[fd9]['ids'], data[fd9]['data']):
                                            questions.update({fd9: {"id": idi_9, "value": d9,
                                                                    "display": "{:20}: {:>20}".format(fd9, d9)}})
                                            fd10 = data_keys[9]
                                            for idi_10, d10 in zip(data[fd10]['ids'], data[fd10]['data']):
                                                questions.update({fd10: {"id": idi_10, "value": d10,
                                                                         "display": "{:20}: {:>20}".format(fd10, d10)}})
                                                fd11 = data_keys[10]
                                                for idi_11, d11 in zip(data[fd11]['ids'], data[fd11]['data']):
                                                    questions.update({fd11: {"id": idi_11, "value": d11,
                                                                             "display": "{:20}: {:>20}".format(fd11,
                                                                                                               d11)}})
                                                    fd12 = data_keys[11]
                                                    for idi_12, d12 in zip(data[fd12]['ids'], data[fd12]['data']):
                                                        questions.update({fd12: {"id": idi_12, "value": d12,
                                                                                 "display": "{:20}: {:>20}".format(fd12,
                                                                                                                   d12)}})
                                                        fd13 = data_keys[12]
                                                        for idi_13, d13 in zip(data[fd13]['ids'], data[fd13]['data']):
                                                            questions.update({fd13: {"id": idi_13, "value": d13,
                                                                                     "display": "{:20}: {:>20}".format(
                                                                                         fd13, d13)}})
                                                            fd14 = data_keys[13]
                                                            for idi_14, d14 in zip(data[fd14]['ids'],
                                                                                   data[fd14]['data']):
                                                                questions.update({fd14: {"id": idi_14, "value": d14,
                                                                                         "display": "{:20}: {:>20}".format(
                                                                                             fd14, d14)}})
                                                                fd15 = data_keys[14]
                                                                for idi_15, d15 in zip(data[fd15]['ids'],
                                                                                       data[fd15]['data']):
                                                                    questions.update({fd15: {"id": idi_15, "value": d15,
                                                                                             "display": "{:20}: {:>20}".format(
                                                                                                 fd15, d15)}})
                                                                    fd16 = data_keys[15]
                                                                    for idi_16, d16 in zip(data[fd16]['ids'],
                                                                                           data[fd16]['data']):
                                                                        questions.update({fd16: {"id": idi_16,
                                                                                                 "value": d16,
                                                                                                 "display": "{:20}: {:>20}".format(
                                                                                                     fd16, d16)}})
                                                                        fd17 = data_keys[16]
                                                                        for idi_17, d17 in zip(data[fd17]['ids'],
                                                                                               data[fd17]['data']):

                                                                            questions.update({fd17: {"id": idi_17,
                                                                                                     "value": d17,
                                                                                                     "display": "{:20}: {:>20}".format(
                                                                                                         fd17, d17)}})

                                                                            questions["Best Technique"] = {"id": 0,
                                                                                                           "value": "",
                                                                                                           "display": ""}

                                                                            # Start processing from here on
                                                                            if not self.is_complete(questions):
                                                                                best_tech_selected_by_rules, questions = self.apply_rules_to_sample(
                                                                                    questions)

                                                                                if best_tech_selected_by_rules:
                                                                                    self.save_to_database(questions)
                                                                                else:
                                                                                    questions = self.get_response(
                                                                                        questions)
                                                                                    self.save_to_database(questions,
                                                                                                          True)

                                                                                # self.question_lists.append(questions)
                                                                            else:
                                                                                counter += 1
                                                                                print('{} completed'.format(counter))

    def is_complete(self, questions, manual_check=False):

        is_completed = False

        # 28 features
        query = "Select Count(*) from ml_attribute_combination where classification=? and atmosphere=? and field_size=? and sink_distance=? and number_of_nodes=? and aggregation_function=? and physical_topology=? and logical_topology=? and sampling_rate=? and required_connectivity=? and variable=? and obj_energy=? and obj_bandwidth=? and obj_latency=? and limit_energy=? and limit_bandwidth=? and limit_latency=?"
        if manual_check:
            query = "Select Count(*) from ml_attribute_combination_manual where classification=? and atmosphere=? and field_size=? and sink_distance=? and number_of_nodes=? and aggregation_function=? and physical_topology=? and logical_topology=? and sampling_rate=? and required_connectivity=? and variable=? and obj_energy=? and obj_bandwidth=? and obj_latency=? and limit_energy=? and limit_bandwidth=? and limit_latency=?"

        values_ = []
        for i, q in questions.items():
            if i == 'Best Technique':
                continue
            values_.append(q['id'])

        values_tuple = tuple(values_)
        self.conn = sqlite3.connect('AlgorithmStore.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute(query, values_tuple)

        # Fetch tuple containing number of affected rows
        (number_of_rows,) = self.cursor.fetchone()

        if number_of_rows > 0:
            is_completed = True

        self.conn.close()

        return is_completed

    # Main brain of the Rule Engine
    def apply_rules_to_sample(self, questions):

        best_tech_selected_by_rules = False

        #########################################################################################################

        # Process Rules Pack
        #########################################################################################################
        # questions[RulesEnums.Attributes.Logical_Topology]['value'] = 'Test'
        # print(questions)
        for rule in self.rules_pack:
            conditions = rule['conditions']
            settings = rule['settings']
            conditions_passed = True
            # check a set of conditions
            for attrib, value in conditions.items():
                if questions[attrib]['value'] != value:
                    conditions_passed = False
                    break

            if conditions_passed:
                # set a set of conditions
                for attrib, value in settings.items():
                    questions[attrib]['value'] = value
                    # if Best Technique Set
                    if attrib == RulesEnums.Attributes.Best_Technique:
                        vl = self.techniques[value]['id']
                        questions[attrib]['id'] = self.techniques[value]['id']
                        best_tech_selected_by_rules = True

            # print(questions)
        return best_tech_selected_by_rules, questions

    def get_response(self, questions):

        for k in questions.keys():
            print(questions[k]['display'])

        print('')
        print('1. Leach')
        print('2. Heed')
        print('3. Pegasis')
        print('4. DBST')
        print('5. Directed Diffusion')

        print("\nBest Technique: ? ")
        best_technique = input()

        vl = self.techniques[self.techniques_list(int(best_technique) - 1)]['id']
        questions['Best Technique']['id'] = self.techniques[self.techniques_list(int(best_technique) - 1)]['id']
        questions['Best Technique']['value'] = self.techniques[int(best_technique) - 1]['name']
        return questions

    def save_to_database(self, questions, include_manual=False):

        # Prepare tuples for query

        query = 'INSERT INTO ml_attribute_combination(classification, atmosphere, field_size, sink_distance, number_of_nodes, aggregation_function, physical_topology, logical_topology, sampling_rate, required_connectivity, variable, obj_energy, obj_bandwidth, obj_latency, limit_energy, limit_bandwidth, limit_latency, best_technique, completed) Values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'

        values_ = []
        for i, q in questions.items():
            values_.append(q['id'])
        values_.append(1)

        values_tuple = tuple(values_)

        self.conn = sqlite3.connect('AlgorithmStore.db')
        self.cursor = self.conn.cursor()

        self.cursor.execute(query, values_tuple)

        if include_manual and not self.is_complete(questions):
            query = 'INSERT INTO ml_attribute_combination_manual(classification, atmosphere, field_size, sink_distance, number_of_nodes, aggregation_function, physical_topology, logical_topology, sampling_rate, required_connectivity, variable, obj_energy, obj_bandwidth, obj_latency, limit_energy, limit_bandwidth, limit_latency, best_technique, completed) Values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
            self.cursor.execute(query, values_tuple)

        self.conn.commit()
        self.conn.close()

    # Returns dataFilteredDF containing unique records
    def read_excel_file(self):

        dataDF = pd.read_excel('events_main_v.1.1.xlsx')
        dataFilteredDF = dataDF.copy(deep=True)
        m_dataDF = dataDF.iloc[:, 1:]
        m_dataDF.to_csv('data_files/data_file.csv')

        self.conn = sqlite3.connect('AlgorithmStore.db')
        self.cursor = self.conn.cursor()

        query = 'Delete from ml_attribute_combination'
        self.cursor.execute(query)
        self.conn.commit()

        data_query = []
        data_values = []
        for index, data in dataDF.iterrows():
            query_values = list(data.iloc[1:])
            query = """
            Select Count(*) from ml_attribute_combination where 
            classification=(Select id from ml_attribute_values where at_value=?) 
            and variable=(Select id from ml_attribute_values where at_value=?)
            and obj_energy=(Select id from ml_attribute_values where at_value=?) and obj_bandwidth=(Select id from ml_attribute_values where at_value=?) and obj_latency=(Select id from ml_attribute_values where at_value=?) and field_size=(Select id from ml_attribute_values where at_value=?) and sink_distance=(Select id from ml_attribute_values where at_value=?) and atmosphere=(Select id from ml_attribute_values where at_value=?) and limit_energy=(Select id from ml_attribute_values where at_value=?) and limit_bandwidth=(Select id from ml_attribute_values where at_value=?) and limit_latency=(Select id from ml_attribute_values where at_value=?) and number_of_nodes=(Select id from ml_attribute_values where at_value=?) and aggregation_function=(Select id from ml_attribute_values where at_value=?) and physical_topology=(Select id from ml_attribute_values where at_value=?) and logical_topology=(Select id from ml_attribute_values where at_value=?) and sampling_rate=(Select id from ml_attribute_values where at_value=?) and required_connectivity=(Select id from ml_attribute_values where at_value=?) and best_technique=(Select id from ml_attribute_values where at_value=?)
            """

            self.cursor.execute(query, query_values)
            self.conn.commit()

            # Fetch tuple containing number of affected rows
            (number_of_rows,) = self.cursor.fetchone()
            query = """
                    Insert into ml_attribute_combination 
                    (classification, variable, obj_energy, obj_bandwidth, obj_latency, field_size, sink_distance, atmosphere, limit_energy, limit_bandwidth, limit_latency, number_of_nodes, aggregation_function, physical_topology, logical_topology, sampling_rate, required_connectivity, best_technique, completed) 
                    Values(
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?), 
                    (select id from ml_attribute_values where at_value=?),
                    ? )
                    """
            query_values.append(1)

            if index < 100:  # if number_of_rows <= 0:
                self.cursor.execute(query, query_values)
            else:
                dataFilteredDF.drop(index=index, inplace=True)

        self.conn.commit()
        self.conn.close()

        dataFilteredDF.to_csv('data_files/data_file_unique.csv')
        print('')


class RulesEngine:

    """
    Order of selections
    ###################
    Field_Size
    Sink_Distance
    Atmosphere
    No of Nodes
    Physical Topology
    Logical Topology

    """

    def __init__(self):

        self.rules_pack = [

            # Sink Distance == Field Size
            {'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.ROOM},
             'settings': {RulesEnums.Attributes.Sink_Distance: RulesEnums.SinkDistance.ROOM}},
            {'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.APARTMENT_COMPLEX},
             'settings': {RulesEnums.Attributes.Sink_Distance: RulesEnums.SinkDistance.APARTMENT_COMPLEX}},
            {'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY},
             'settings': {RulesEnums.Attributes.Sink_Distance: RulesEnums.SinkDistance.CITY}},
            {'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY_BLOCK},
             'settings': {RulesEnums.Attributes.Sink_Distance: RulesEnums.SinkDistance.CITY_BLOCK}},
            {'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.REGION},
             'settings': {RulesEnums.Attributes.Sink_Distance: RulesEnums.SinkDistance.REGION}},
            {'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.OCEAN},
             'settings': {RulesEnums.Attributes.Sink_Distance: RulesEnums.SinkDistance.OCEAN}},

            # Field Size == Num of Nodes

            # FieldSize = SinkDistance = Atmosphere
            {
                'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.OBJECT},
                'settings': {
                    RulesEnums.Attributes.Sink_Distance: RulesEnums.SinkDistance.OBJECT,
                    RulesEnums.Attributes.Sink_Distance: RulesEnums.Atmosphere.OBJECT
                }
            },
            {
                'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.OBJECT},
                'settings': {
                    RulesEnums.Attributes.Sink_Distance: RulesEnums.SinkDistance.OBJECT,
                    RulesEnums.Attributes.Sink_Distance: RulesEnums.Atmosphere.OBJECT
                }
            },

            # Physical Topology
            { 'conditions': { RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR },
                'settings': {RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER}
             },
            { 'conditions': { RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.LINEAR },
                'settings': {RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CHAIN}
             },
            { 'conditions': { RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.MESH },
                'settings': {RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.MESH}
             },
            { 'conditions': { RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.BUS },
                'settings': {RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CHAIN}
             },

            # Field Size
            { 'conditions': {'Number of Nodes': '1-10 nodes', 'Logical Topology': 'Cluster', 'Field Size': 'Room'},
                'settings': {'Best Technique': 'Leach'}
            },
            { 'conditions': {'Number of Nodes': '1-10 nodes', 'Logical Topology': 'Cluster', 'Field Size': 'Room'},
                'settings': {'Best Technique': 'Leach'}
            },

            # Air, Vaccum Space - Gas,

            # Field Size = No of Nodes
            { 'conditions': { RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.AIR },
                'settings': {'Logical Topology': 'Cluster'}
            }

        ]


class RulesEnums:

    class FieldSize:
        OBJECT = 'Object'
        ROOM = 'Room'
        APARTMENT_COMPLEX = 'Apartment Complex'
        CITY_BLOCK = 'City Block'
        CITY = 'City'
        REGION = 'Region'
        OCEAN = 'Ocean'

    class SinkDistance:
        OBJECT = 'Object'
        ROOM = 'Room'
        APARTMENT_COMPLEX = 'Apartment Complex'
        CITY_BLOCK = 'City Block'
        CITY = 'City'
        REGION = 'Region'
        OCEAN = 'Ocean'

    class NodeCount:
        _1_10_NODES = '1-10 nodes'
        _1_100_NODES = '1-100 nodes'
        _1_1000_NODES = '1-1000 nodes'
    """
    class NodeCount:
        GAS = 'Gas'
        LIQUID = 'Liquid'
        SOLID = 'Solid'
    """

    class Atmosphere:
        OBJECT = 'Object'
        AIR = 'Air'
        OVERGROUND = 'Overground'
        UNDERGROUND = 'Underground'
        UNDERWATER_SURFACE = 'Underwater Surface'
        UNDERWATER_DEEP = 'Underwater Deep'
        VACUUM_SPACE = 'Vacuum Space'

    class AggregationFunction:
        SUM = 'SUM'
        AVG = 'AVG'
        MIN = 'MIN'
        MAX = 'MAX'
        COUNT = 'COUNT'

    class PhysicalTopology:
        STAR = 'STAR'
        BUS = 'BUS'
        LINEAR = 'LINEAR'
        RING = 'RING'
        MESH = 'MESH'

    class LogicalTopology:
        CLUSTER = 'Cluster'
        CHAIN = 'Chain'
        TREE = 'Tree'
        MESH = 'Mesh'

    class BestTechnique:
        LEACH = 'Leach'
        HEED = 'Heed'
        PEGASIS = 'Pegasis'
        DBST = 'DBST'
        DIRECTED_DIFFUSION = 'Directed Diffusion'

    class Attributes:
        Classification = 'Classification'
        Atmosphere = 'Atmosphere'
        Sink_Distance = 'Sink Distance'
        Field_Size = 'Field Size'
        Number_of_Nodes = 'Number of Nodes'
        Aggregation_Function = 'Aggregation Function'
        Physical_Topology = 'Physical Topology'
        Logical_Topology = 'Logical Topology'
        Sampling_Rate = 'Sampling Rate'
        Required_Connectivity = 'Required Connectivity'
        Homogenous_Nodes = 'Homogenous Nodes'
        Periodic_Reporting = 'Periodic Reporting'
        Event_Reporting = 'Event Reporting'
        Realtime_Reporting = 'Realtime Reporting'
        Event_Monitoring = 'Event Monitoring'
        Location_Awareness = 'Location Awareness'
        Node_Mobility = 'Node Mobility'
        Medium = 'Medium'
        Energy_Objective = 'Energy Objective'
        Bandwidth_Objective = 'Bandwidth Objective'
        Latency_Objective = 'Latency Objective'
        Energy_Limit = 'Energy Limit'
        Bandwidth_Limit = 'Bandwidth Limit'
        Latency_Limit = 'Latency Limit'
        Variable = 'Variable'
        Best_Technique = 'Best Technique'


class NeuralNetworkConfig:

    def __init__(self):
        self.maximum_rounds = 100
        self.training_epochs = 1000
        self.batch_size = 1000
        self.optimizer = 'adam'
        self.loss_function = 'categorical_crossentropy'
        # self.loss_function = 'sparse_categorical_crossentropy'
        self.metrics = ['accuracy']

        self.input_nodes = 55
        self.hidden_layer_1 = 512
        self.hidden_layer_2 = 512
        self.hidden_layer_3 = 256
        self.output_layer = 5


ruleEngine = RuleEngine()
ruleEngine.read_excel_file()

# IntelligentAlgorithm