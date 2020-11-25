from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from xml.etree import ElementTree
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
import numpy as np
import os
from keras.models import load_model


class CombineData:

    def __init__(self):

        self.absolute_path = FileManager.absolute_path


        self.all_experiments = {}

        for k, v in self.technique_paths.items():
            exp_combination = []
            for experiment in self.experiments:
                tech_dict = {}
                tech_dict.update({'sim_data': '{}/{}/leach_data.dat'.format(v, experiment)})
                tech_dict.update({'out_bandwidth': '{}/{}/bandwidth.dat'.format(v, experiment)})
                tech_dict.update({'flow_data': '{}/{}/flowdata.xml'.format(v, experiment)})
                tech_dict.update({'out_delay': '{}/{}/delay.dat'.format(v, experiment)})
                tech_dict.update({'out_combined': '{}/{}/combined_data.csv'.format(v, experiment)})
                tech_dict.update({'losses_bitrates': '{}/{}/losses_bitrates.dat'.format(v, experiment)})

                exp_combination.append(tech_dict)

            self.all_experiments.update({k: exp_combination})


        self.columns = ['round', 'time', 'technique', 'energy', 'bandwidth', 'latency', 'numberOfNodes',
                        'maximumRounds', 'numberOfPackets', 'maximumHeads', 'fractionHeads', 'simulationTime',
                        'packetSize', 'communicationsRange', 'regionSize', 'initialNodeEnergy', 'txCurrentA',
                        'rxCurrentA', 'idleCurrentA', 'activeNodes', 'minNextNodeDistance',
                        'maxNextNodeDistance', 'avgNode2SinkDistance', 'avgNextNodeDistance', 'coverageRadius',
                        'avgNodeCoverage']
        self.columns_complete = ['round', 'time', 'technique', 'energy', 'bandwidth', 'latency', 'numberOfNodes',
                                 'maximumRounds', 'numberOfPackets', 'maximumHeads', 'fractionHeads', 'simulationTime',
                                 'packetSize', 'communicationsRange', 'regionSize', 'initialNodeEnergy', 'txCurrentA',
                                 'rxCurrentA', 'idleCurrentA', 'activeNodes', 'minNextNodeDistance',
                                 'maxNextNodeDistance', 'avgNode2SinkDistance', 'avgNextNodeDistance', 'coverageRadius', 'avgNodeCoverage',
                                 'technique2', 'energy2', 'bandwidth2', 'latency2',
                                 'technique3', 'energy3', 'bandwidth3', 'latency3',
                                 'technique4', 'energy4', 'bandwidth4', 'latency4',
                                 'technique5', 'energy5', 'bandwidth5', 'latency5']
        self.band_columns = ['round', 'bandwidth', 'band_cumm']

        self.input_file1 = None
        self.input_file2 = None
        self.output_file = None

        self.globalDF = pd.DataFrame(columns=self.columns_complete)
        self.model = None

    def get_combined_file(self):
        return self.combined_data_file

    def get_combined_filepath(self):
        return self.compiled_data_file

    def convert_data_to_DF(self):

        # transform data to DF
        # self.all_experiments contains 5 techniques x 3 data folders
        for k, v in self.all_experiments.items():
            experiments = v
            for experiment in experiments:

                try:
                    self.input_file1 = open(experiment['sim_data'], 'r', )
                    self.output_file = open(experiment['out_combined'], 'w')

                    columns = ['round', 'time', 'technique', 'energy', 'bandwidth', 'latency', 'numberOfNodes',
                               'maximumRounds', 'numberOfPackets', 'maximumHeads', 'fractionHeads',
                               'simulationTime', 'packetSize', 'communicationsRange', 'regionSize',
                               'initialNodeEnergy', 'txCurrentA', 'rxCurrentA', 'idleCurrentA',
                               'activeNodes', 'minNextNodeDistance', 'maxNextNodeDistance', 'avgNode2SinkDistance',
                               'avgNextNodeDistance', 'coverageRadius', 'avgNodeCoverage']
                    dataDF = pd.read_table(experiment['sim_data'], sep=",", usecols=columns)
                    print(dataDF)

                    dataDF.to_csv(experiment['out_combined'], index=False)

                except Exception as exception:
                    print(exception)

                finally:
                    self.input_file1.close()
                    # self.output_file.close()
        print('Combined file created...')

    def combine_metrics(self):

        for k, v in self.all_experiments.items():
            experiments = v
            for experiment in experiments:
                destination_file = experiment['out_combined']                            # final combined output

                dataDF = pd.read_table(experiment['sim_data'], sep=",")                  # energy
                bandwidthDF = pd.read_table(experiment['out_bandwidth'], sep=",")        # bandwidth

                dataDF.iloc[0:, 4] = bandwidthDF.iloc[0:, 1]

                dataDF.to_csv(destination_file)


        print('Metrics combination completed...')

    # combine data for all techniques to enable easy processing by IA
    def compile_all_data(self):

        # combinedDF = pd.DataFrame(columns=self.columns)
        columns = ['round', 'time', 'technique', 'energy', 'bandwidth', 'latency', 'numberOfNodes', 'maximumRounds',
         'numberOfPackets', 'maximumHeads', 'fractionHeads', 'simulationTime', 'packetSize', 'communicationsRange',
         'regionSize', 'initialNodeEnergy', 'txCurrentA', 'rxCurrentA', 'idleCurrentA',
         'activeNodes', 'minNextNodeDistance', 'maxNextNodeDistance', 'avgNode2SinkDistance', 'avgNextNodeDistance',
         'coverageRadius', 'avgNodeCoverage']

        dataTechDF = {
            'pegasis': pd.DataFrame(columns=columns),
            'leach': pd.DataFrame(columns=columns),
            'dbst': pd.DataFrame(columns=columns),
            'heed': pd.DataFrame(columns=columns),
            'directeddiffussion': pd.DataFrame(columns=self.columns)
        }
        self.globalDF = pd.DataFrame(columns=columns)
        for k, v in self.all_experiments.items():
            tech_name = k
            experiments = v
            for experiment in experiments:

                temp_table = pd.read_csv(experiment['out_combined'])
                dataTechDF[tech_name] = dataTechDF[tech_name].append(temp_table, ignore_index=True)


        dataTechDF['pegasis'] = dataTechDF['pegasis'].iloc[:, :-1]
        self.globalDF = self.globalDF.append(dataTechDF['pegasis'])

        # dataTechDF['leach'] = dataTechDF['leach'].iloc[:, :-1]
        self.globalDF['technique2'] = dataTechDF['leach'].iloc[:, 2:3]
        self.globalDF['energy2'] = dataTechDF['leach'].iloc[:, 3:4]
        self.globalDF['bandwidth2'] = dataTechDF['leach'].iloc[:, 4:5]
        self.globalDF['latency2'] = dataTechDF['leach'].iloc[:, 5:6]

        # dataTechDF['dbst'] = dataTechDF['dbst'].iloc[:-1, :]
        self.globalDF['technique3'] = dataTechDF['dbst'].iloc[:, 2:3]
        self.globalDF['energy3'] = dataTechDF['dbst'].iloc[:, 3:4]
        self.globalDF['bandwidth3'] = dataTechDF['dbst'].iloc[:, 4:5]
        self.globalDF['latency3'] = dataTechDF['dbst'].iloc[:, 5:6]

        # dataTechDF['heed'] = dataTechDF['heed'].iloc[:-1, :]
        self.globalDF['technique4'] = dataTechDF['heed'].iloc[:, 2:3]
        self.globalDF['energy4'] = dataTechDF['heed'].iloc[:, 3:4]
        self.globalDF['bandwidth4'] = dataTechDF['heed'].iloc[:, 4:5]
        self.globalDF['latency4'] = dataTechDF['heed'].iloc[:, 5:6]

        # dataTechDF['directeddiffussion'] = dataTechDF['directeddiffussion'].iloc[:-1, :]
        self.globalDF['technique5'] = dataTechDF['directeddiffussion'].iloc[:, 2:3]
        self.globalDF['energy5'] = dataTechDF['directeddiffussion'].iloc[:, 3:4]
        self.globalDF['bandwidth5'] = dataTechDF['directeddiffussion'].iloc[:, 4:5]
        self.globalDF['latency5'] = dataTechDF['directeddiffussion'].iloc[:, 5:6]

        self.globalDF.to_csv(self.combined_data_file)


    @classmethod
    def saveData(cls, dataDF):
        absolut = 'D:/Projects/PythonCharmProjects/_1_Research/Algorithm/data_storage'
        file_path = '{}/data/_data/_compiled/processed_data.csv'.format(absolut)
        dataDF.to_csv(file_path) # , index=False)

    @classmethod
    def readData(cls):
        absolut = 'D:/Projects/PythonCharmProjects/_1_Research/Algorithm/data_storage'
        file_path = '{}/data/_data/_compiled/processed_data.csv'.format(absolut)
        dataDF = pd.read_table(file_path, sep=',', index_col=0)
        return dataDF

    @classmethod
    def saveTrainTestData(cls, trainDf, testDF):
        absolut = 'D:/Projects/PythonCharmProjects/_1_Research/Algorithm/'
        train_path = '{}/data/training/train_data.csv'.format(absolut)
        test_path = '{}/data/training/test_data.csv'.format(absolut)
        trainDf.to_csv(train_path)
        testDF.to_csv(test_path)

    @classmethod
    def readTrainTestData(cls):
        absolut = 'D:/Projects/PythonCharmProjects/_1_Research/Algorithm/'
        train_path = '{}/data/training/train_data.csv'.format(absolut)
        test_path = '{}/data/training/test_data.csv'.format(absolut)
        trainDf = pd.read_table(train_path, sep=',', index_col=0)
        testDF = pd.read_table(test_path, sep=',', index_col=0)
        return trainDf, testDF

    @classmethod
    def saveFinalTrainData(cls, trainDf):
        absolut = 'D:/Projects/PythonCharmProjects/_1_Research/Algorithm/'
        train_path = '{}/data/_data/_compiled/final_training_data.csv'.format(absolut)
        trainDf.to_csv(train_path, index=False)

    @classmethod
    def readFinalTrainData(cls):
        absolut = 'D:/Projects/PythonCharmProjects/_1_Research/Algorithm/'
        train_path = '{}/data/_data/_compiled/final_training_data.csv'.format(absolut)
        trainDf = pd.read_table(train_path, sep=',', index_col=0)
        return trainDf

    @classmethod
    def saveTransformedData(cls, trainDf, targetDf, trainTestDf, targetTestDF):

        trainDf.to_csv(FileManager.X_TRAIN, index=False)
        targetDf.to_csv(FileManager.Y_TARGET, index=False)
        trainTestDf.to_csv(FileManager.X_TEST, index=False)
        targetTestDF.to_csv(FileManager.Y_TEST, index=False)

    @classmethod
    def readTransformedData(cls):

        trainDf = pd.read_csv(FileManager.X_TRAIN)
        targetDf = pd.read_csv(FileManager.Y_TARGET)
        trainTestDf = pd.read_csv(FileManager.X_TEST)
        targetTestDF = pd.read_csv(FileManager.Y_TEST)

        return trainDf, targetDf, trainTestDf, targetTestDF

    @classmethod
    def readModel(cls):

        model = None
        model_exists = False
        if os.path.isfile(FileManager.ML_PATH):
            model_exists = True
            model = load_model(FileManager.ML_PATH)

        return model_exists, model

    @classmethod
    def saveModel(cls, model):
        model.save(FileManager.ML_PATH)
        print('Model saved on last training...')

    def process_bandwidth_bitrate(self):
        self.absolute_path = FileManager.absolute_path

        for k, v in self.all_experiments.items():
            experiments = v
            for experiment in experiments:

                et = ET.parse(experiment['flow_data'])
                root = et.getroot()

                bitrates = []
                losses = []
                delays = []
                packets = []

                for child in root:
                    print(child.tag, child.attrib)

                # Packets / Bytes (packets * 100
                nodeIPs = {}
                flows = root.findall('Ipv4FlowClassifier/Flow')

                for flow in flows:
                    ipAddress = flow.attrib['sourceAddress']
                    packets = flow.findall('Dscp')[0].attrib['packets']

                    if ipAddress not in nodeIPs:
                        nodeIPs.update({ipAddress: []})
                    nodeIPs[ipAddress].append(packets)

                # round, throughput, bandwidth
                print(experiment['out_bandwidth'])
                ofilename = experiment['out_bandwidth']
                f = open(ofilename, "w")
                f.write('round, throughput, bandwidth, lostPackets')
                totalPacketSum = 0

                roundData = {}
                for k, v in nodeIPs.items():
                    rounds = len(v)
                    for d in range(0, rounds):

                        for r in range(len(v)):
                            if r not in roundData:
                                roundData.update({r: 0})
                            roundData[r] = roundData[r] + int(v[r])

                counter = 0
                packetSum = 0
                totalPacketSum = 0
                for k, v in roundData.items():
                    cuonter = counter + 1
                    packetSum = v
                    totalPacketSum = totalPacketSum + packetSum
                    f.write('\n{},{},{}'.format(counter, packetSum, totalPacketSum))
                f.close()

                ###########################################
                bitrates = []
                losses = []
                delays = []
                rxPacketsArr = []

                for flow in et.findall('FlowStats/Flow'):
                    # filteroutOLSR
                    for tpl in et.findall('Ipv4FlowClassifier/Flow'):
                        if tpl.get('flowId') == flow.get('flowId'):
                            break
                    if tpl.get('destinationPort') == '698':
                        continue

                    losses.append(int(flow.get('lostPackets')))
                    rxPackets = int(flow.get('rxPackets'))
                    rxPacketsArr.append(rxPackets)

                    if rxPackets == 0:
                        bitrates.append(0)
                    else:
                        t0 = float(flow.get('timeFirstRxPacket')[:-2])
                        t1 = float(flow.get('timeLastRxPacket')[:-2])

                        duration = (t1 - t0) * pow(10, -9)
                        if duration <= 0:
                            bitrates.append(0)
                        else:
                            bitrates.append(8 * int(flow.get('rxBytes')) / duration * pow(10, -3))
                        delays.append(float(flow.get('delaySum')[:-2]) * pow(10, -9) / rxPackets)

                # round, throughput, bandwidth
                brfilename = experiment['losses_bitrates']
                f = open(brfilename, "w")
                f.write('counter, losses, bitrates, receivedPackets')
                totalPacketSum = 0
                i = 0
                for a, b, c in zip(losses, bitrates, rxPacketsArr):
                    i = i + 1
                    f.write('\n{},{},{},{}'.format(i, a, b, c))
                f.close()

                # delay
                print(experiment['out_delay'])
                dfilename = experiment['out_delay']
                f = open(dfilename, "w")
                f.write('counter, delay')
                totalDelaySum = 0
                i = 0
                for d in delays:
                    i = i + 1
                    totalDelaySum = totalDelaySum + d
                    f.write('\n{},{}, {}'.format(i, d, totalDelaySum))
                f.close()


