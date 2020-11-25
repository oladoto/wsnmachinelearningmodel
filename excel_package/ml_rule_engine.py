import sqlite3
import pandas as pd
from excel_package.rules_pack import RulesEngine
from excel_package.rules_enums import RulesEnums
from openpyxl import *

class RuleEngine:

    def __init__(self):

        self.techniques = {
            RulesEnums.BestTechnique.LEACH: {'id': 49, 'name': 'Leach', 'Physical': 'Star', 'Logical': 'Cluster'},
            RulesEnums.BestTechnique.HEED: {'id': 50, 'name': 'Heed', 'Physical': 'Star', 'Logical': 'Cluster'},
            RulesEnums.BestTechnique.PEGASIS: {'id': 51, 'name': 'Pegasis', 'Physical': 'Bus', 'Logical': 'Chain'},
            RulesEnums.BestTechnique.DBST: {'id': 52, 'name': 'DBST', 'Physical': 'Star', 'Logical': 'Tree'},
            RulesEnums.BestTechnique.DIRECTED_DIFFUSION: {'id': 53, 'name': 'Directed Diffusion', 'Physical': 'Mesh', 'Logical': 'Mesh'}
        }
        self.techniques_list = list(self.techniques)

        self.valid_fields = ['Classification', 'Atmosphere', 'Field Size', 'Number of Nodes',
                             'Physical Topology', 'Sampling Rate',
                             'Required Connectivity', 'Energy Objective', 'Bandwidth Objective', 'Latency Objective',
                             'Energy Limit', 'Bandwidth Objective', 'Latency Objective',
                             'Best Technique']

        rules = RulesEngine()
        self.rules_pack = rules.rules_pack

        self.conn = None
        self.cursor = None
        self.question_lists = []
        self.select_fields = []

        self.conn = sqlite3.connect('AlgorithmStore.db')
        self.cursor = self.conn.cursor()
        self.process_rules()

        self.write_file()
        self.conn.close()

    def process_rules(self):

        query = "Delete from ml_attribute_combination_rules where manual<>1"
        self.cursor.execute(query)

        query = "Select ml_attribute_values.id, ml_attributes.attribute_name, ml_attributes.data_type, ml_attribute_values.at_value, ml_attributes.field_name from ml_attributes, ml_attribute_values where ml_attributes.id=ml_attribute_values.attribute_id and ml_attributes.included=1 order by ml_attributes.id"
        self.cursor.execute(query)

        result_set = self.cursor.fetchall()

        data = {}
        for r in result_set:
            attrib_id = r[0]
            attrib = r[1]
            typ = r[2]
            val = r[3]
            fld_name = r[4]
            if attrib not in data:
                data.update({attrib: {}})
                data[attrib].update({ 'field_name': '', 'ids': [], 'data': []})
            data[attrib]['field_name'] = fld_name
            data[attrib]['ids'].append(attrib_id)
            data[attrib]['data'].append(val)

        questions = {}

        data_keys = list(data)
        counter = 0

        fd1 = data_keys[0]
        for idi_1, d1 in zip(data[fd1]['ids'], data[fd1]['data']):
            questions.update({fd1: {"id": idi_1, "field_name": data[fd1]['field_name'], "value": d1, "display": "{:20}: {:>20}".format(fd1, d1)}})
            fd2 = data_keys[1]
            for idi_2, d2 in zip(data[fd2]['ids'], data[fd2]['data']):
                questions.update({fd2: {"id": idi_2, "field_name": data[fd2]['field_name'], "value": d2, "display": "{:20}: {:>20}".format(fd2, d2)}})
                fd3 = data_keys[2]
                for idi_3, d3 in zip(data[fd3]['ids'], data[fd3]['data']):
                    questions.update({fd3: {"id": idi_3, "field_name": data[fd3]['field_name'], "value": d3, "display": "{:20}: {:>20}".format(fd3, d3)}})
                    fd4 = data_keys[3]
                    for idi_4, d4 in zip(data[fd4]['ids'], data[fd4]['data']):
                        questions.update({fd4: {"id": idi_4, "field_name": data[fd4]['field_name'], "value": d4, "display": "{:20}: {:>20}".format(fd4, d4)}})
                        fd5 = data_keys[4]
                        for idi_5, d5 in zip(data[fd5]['ids'], data[fd5]['data']):
                            questions.update(
                                {fd5: {"id": idi_5, "field_name": data[fd5]['field_name'], "value": d5, "display": "{:20}: {:>20}".format(fd5, d5)}})
                            fd6 = data_keys[5]
                            for idi_6, d6 in zip(data[fd6]['ids'], data[fd6]['data']):
                                questions.update(
                                    {fd6: {"id": idi_6, "field_name": data[fd6]['field_name'], "value": d6, "display": "{:20}: {:>20}".format(fd6, d6)}})
                                fd7 = data_keys[6]
                                for idi_7, d7 in zip(data[fd7]['ids'], data[fd7]['data']):
                                    questions.update(
                                        {fd7: {"id": idi_7, "field_name": data[fd7]['field_name'], "value": d7, "display": "{:20}: {:>20}".format(fd7, d7)}})
                                    fd8 = data_keys[7]
                                    for idi_8, d8 in zip(data[fd8]['ids'], data[fd8]['data']):
                                        questions.update({fd8: {"id": idi_8, "field_name": data[fd8]['field_name'], "value": d8,
                                                                "display": "{:20}: {:>20}".format(fd8, d8)}})
                                        fd9 = data_keys[8]
                                        for idi_9, d9 in zip(data[fd9]['ids'], data[fd9]['data']):
                                            questions.update({fd9: {"id": idi_9, "field_name": data[fd9]['field_name'], "value": d9,
                                                                    "display": "{:20}: {:>20}".format(fd9, d9)}})
                                            fd10 = data_keys[9]
                                            for idi_10, d10 in zip(data[fd10]['ids'], data[fd10]['data']):
                                                questions.update({fd10: {"id": idi_10, "field_name": data[fd10]['field_name'], "value": d10,
                                                                         "display": "{:20}: {:>20}".format(fd10, d10)}})

                                                if (questions['Energy Objective']['value'] == '0.2' and questions['Bandwidth Objective']['value'] == '0.2' and questions['Latency Objective']['value'] == '0.2') or \
                                                        (questions['Field Size']['value'] == 'Room' and questions['Number of Nodes']['value'] == '1-1000 nodes') or \
                                                    (questions['Energy Objective']['value'] == '0.8' and questions['Latency Objective']['value'] == '0.8') or \
                                                    (questions['Bandwidth Objective']['value'] == '0.8' and questions['Latency Objective']['value'] == '0.8') or \
                                                    (questions['Energy Objective']['value'] == '0.8' and questions['Bandwidth Objective']['value'] == '0.8'):

                                                    continue
                                                questions["Best Technique"] = {"id": 0,
                                                                               "value": "",
                                                                               "field_name": "best_technique",
                                                                               "display": ""}

                                                # Start processing from here on
                                                if not self.is_complete(questions):
                                                    best_tech_selected_by_rules, questions = self.apply_rules_to_sample( questions)

                                                    if best_tech_selected_by_rules:
                                                        self.save_to_database(questions)

                                                    else:
                                                        continue
                                                        # questions = self.get_response( questions )
                                                        # self.save_to_database(questions, True)

                                                    counter += 1
                                                    # self.question_lists.append(questions)

                                                    print('{} completed'.format(counter))

    def is_complete(self, questions):

        is_completed = False

        # 28 features
        query = "Select Count(*) from ml_attribute_combination_rules where "

        first = True
        values_ = []
        for i, q in questions.items():
            if i == 'Best Technique':
                continue
            if not first:
                query = query + 'and'
            first = False
            query = query + ' ' + q['field_name'] + '=? '
            values_.append(q['value'])

        # append completed
        query = query + ' and completed=?'
        values_.append(1)

        values_tuple = tuple(values_)
        self.cursor.execute(query, values_tuple)

        # Fetch tuple containing number of affected rows
        (number_of_rows,) = self.cursor.fetchone()

        if number_of_rows > 0:
            is_completed = True

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
            or_conditions_passed = True
            # check a set of conditions
            for attrib, value in conditions.items():
                if attrib not in self.valid_fields:
                    continue
                if type(questions[attrib]['value']) is list:
                    for v in questions[attrib]['value']:
                        if v == value:
                            continue
                else:
                    if questions[attrib]['value'] != value:
                        conditions_passed = False
                        break

            if conditions_passed:
                # set a set of conditions
                for attrib, value in settings.items():
                    if attrib not in self.valid_fields:
                        continue
                    questions[attrib]['value'] = value
                    # if Best Technique Set
                    if attrib == RulesEnums.Attributes.Best_Technique:

                        tech_dict = self.techniques[value]

                        questions[attrib]['id'] = tech_dict['id']
                        questions[attrib]['value'] = tech_dict['name']
                        questions[attrib]['Physical Topology'] = tech_dict['Physical']
                        questions[attrib]['Logical Topology'] = tech_dict['Logical']
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

        tech_dict = self.techniques[self.techniques_list[int(best_technique) - 1]]

        questions['Best Technique']['id'] = tech_dict['id']
        questions['Best Technique']['value'] = tech_dict['name']
        questions['Physical Topology']['value'] = tech_dict['Physical']
        return questions

    def save_to_database(self, questions, done_manually=False):

        # Prepare tuples for query

        query = 'INSERT INTO ml_attribute_combination_rules('
        values_ = []
        for i, q in questions.items():
            values_.append(q['value'])
            query = query + q['field_name'] + ', '

        if done_manually:
            query = query + 'completed, manual) Values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'
            values_.append(1)
            values_.append(1)
        else:
            query = query + 'completed) Values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);'
            values_.append(1)
        values_tuple = tuple(values_)

        self.cursor.execute(query, values_tuple)

        self.conn.commit()

    def write_file(self):
        query = "SELECT classification, atmosphere, obj_energy, obj_bandwidth, obj_latency, field_size, number_of_nodes, physical_topology, sampling_rate, best_technique FROM ml_attribute_combination_rules;"

        self.cursor.execute(query)
        resultset = self.cursor.fetchall()
        data_set = {'classification': [], 'atmosphere': [], 'obj_energy': [], 'obj_bandwidth': [], 'obj_latency': [],
                    'field_size': [], 'number_of_nodes': [], 'physical_topology': [], 'sampling_rate': [],
                    'best_technique': []}
        titles_list = list(data_set)
        for r in resultset:
            data_set['classification'].append(r[0])
            data_set['atmosphere'].append(r[1])
            data_set['obj_energy'].append(r[2])
            data_set['obj_bandwidth'].append(r[3])
            data_set['obj_latency'].append(r[4])
            data_set['field_size'].append(r[5])
            data_set['number_of_nodes'].append(r[6])
            data_set['physical_topology'].append(r[7])
            data_set['sampling_rate'].append(r[8])
            data_set['best_technique'].append(r[9])

        dataDF = pd.DataFrame(data_set, columns=titles_list)
        dataDF.to_csv('data_files/data_file.csv')
        print(dataDF)

    # Returns dataFilteredDF containing unique records
    def read_excel_file(self):

        dataDF = pd.read_excel('events_main_v.1.1.xlsx')
        dataFilteredDF = dataDF.copy(deep=True)
        m_dataDF = dataDF.iloc[:, 1:]
        m_dataDF.to_csv('data_files/data_file.csv')

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

        dataFilteredDF.to_csv('data_files/data_file_unique.csv')
        print('')


"""

    def save_to_database(self, questions, done_manually=False):

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

        if done_manually and not self.is_complete(questions):
            query = 'INSERT INTO ml_attribute_combination_manual(classification, atmosphere, field_size, sink_distance, number_of_nodes, aggregation_function, physical_topology, logical_topology, sampling_rate, required_connectivity, variable, obj_energy, obj_bandwidth, obj_latency, limit_energy, limit_bandwidth, limit_latency, best_technique, completed) Values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
            self.cursor.execute(query, values_tuple)

        self.conn.commit()
        self.conn.close()

"""

"""

fd11 = data_keys[10]
for idi_11, d11 in zip(data[fd11]['ids'], data[fd11]['data']):
    questions.update({fd11: {"id": idi_11, "field_name": data[fd11]['field_name'], "value": d11,
                             "display": "{:20}: {:>20}".format(fd11,
                                                               d11)}})
    fd12 = data_keys[11]
    for idi_12, d12 in zip(data[fd12]['ids'], data[fd12]['data']):
        questions.update({fd12: {"id": idi_12, "field_name": data[fd12]['field_name'], "value": d12,
                                 "display": "{:20}: {:>20}".format(fd12,
                                                                   d12)}})
        fd13 = data_keys[12]
        for idi_13, d13 in zip(data[fd13]['ids'], data[fd13]['data']):
            questions.update({fd13: {"id": idi_13, "field_name": data[fd13]['field_name'], "value": d13,
                                     "display": "{:20}: {:>20}".format(
                                         fd13, d13)}})
            fd14 = data_keys[13]
            for idi_14, d14 in zip(data[fd14]['ids'],
                                   data[fd14]['data']):
                questions.update({fd14: {"id": idi_14, "field_name": data[fd14]['field_name'], "value": d14,
                                         "display": "{:20}: {:>20}".format(
                                             fd14, d14)}})
                fd15 = data_keys[14]
                for idi_15, d15 in zip(data[fd15]['ids'],
                                       data[fd15]['data']):
                    questions.update({fd15: {"id": idi_15, "field_name": data[fd15]['field_name'], "value": d15,
                                             "display": "{:20}: {:>20}".format(
                                                 fd15, d15)}})
                    fd16 = data_keys[15]
                    for idi_16, d16 in zip(data[fd16]['ids'],
                                           data[fd16]['data']):
                        questions.update({fd16: {"id": idi_16,
                                                 "field_name": data[fd16]['field_name'], "value": d16,
                                                 "display": "{:20}: {:>20}".format(
                                                     fd16, d16)}})
                        fd17 = data_keys[16]
                        for idi_17, d17 in zip(data[fd17]['ids'],
                                               data[fd17]['data']):

                            questions.update({fd17: {"id": idi_17,
                                                     "field_name": data[fd17]['field_name'], "value": d17,
                                                     "display": "{:20}: {:>20}".format(
                                                         fd17, d17)}})

"""