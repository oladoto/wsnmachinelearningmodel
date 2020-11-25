from old_codes.ParamValues import *

class DataLoader:

    def __init__(self):
        pass

    def loadEarthquake_Detection(self, loadFunction):
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.NAME, ParamValues.NAME_EARTHQUAKE.DETECTION)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.EVENT_STATE,
                     ParamValues.PARAM_EVENT_STATE_DETECTION)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.PRIMARY_FEATURE,
                     ParamValues.PARAM_PRIMARY_FEATURE_VIBRATION)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.SPREAD_RATE, ParamValues.PARAM_0_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.PRIMARY_MEDIUM,
                     ParamValues.PARAM_PRIMARY_MEDIUM_SOLID)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.AGGREG_ALGORITHM,
                     ParamValues.PARAM_AGGREG_ALGORITHM_MAX)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.FIELD_SIZE_CHANGE, ParamValues.PARAM_0_2)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.AGGREGATION_TYPE,
                     ParamValues.PARAM_AGGREGATION_TYPE_PUSH)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.OBJECTIVE_ENERGY, ParamValues.PARAM_0_8)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.OBJECTIVE_BANDWIDTH,
                     ParamValues.PARAM_0_2)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.OBJECTIVE_LATENCY, ParamValues.PARAM_0_2)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.MIN_ENERGY_CONSUMPTION,
                     ParamValues.PARAM_0_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.MAX_ENERGY_CONSUMPTION,
                     ParamValues.PARAM_0_6)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.MIN_BANDWIDTH_CONSUMPTION,
                     ParamValues.PARAM_0_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.MAX_BANDWIDTH_CONSUMPTION,
                     ParamValues.PARAM_0_6)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.MIN_LATENCY, ParamValues.PARAM_0_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.MAX_LATENCY, ParamValues.PARAM_0_6)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.REQUIRED_CONNECTIVITY,
                     ParamValues.PARAM_REQUIRED_CONN_PARTIAL)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.SAMPLING_RATE, ParamValues.PARAM_0_1)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.HOMOGENOUS_NODES, ParamValues.PARAM_1_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.COMMUNICATION_ALGORITHM,
                     ParamValues.PARAM_COMMS_ALGO_HIERARCHICAL)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.SENSING_TRIGGER,
                     ParamValues.PARAM_SENSING_TRIGGER_BY_EVENT)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.REPORTING_BY_PERIODIC,
                     ParamValues.PARAM_1_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.REPORTING_BY_EVENT,
                     ParamValues.PARAM_1_0)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.PERIODIC_MONITORING_BY_REALTIME,
                     ParamValues.PARAM_1_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.PERIODIC_MONITORING_BY_QUERY,
                     ParamValues.PARAM_0_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.PERIODIC_MONITORING_BY_EVENT,
                     ParamValues.PARAM_1_0)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.PHYSICAL_SENSOR_TOPOLOGY,
                     ParamValues.PARAM_PHYSICAL_SENSOR_TOPO_RANDOMSPARSE)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.PHYSICAL_NETWORK_TOPOLOGY,
                     ParamValues.PARAM_PHYSICAL_NETWORK_TOPO_STAR)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.LOCATION_AWARENESS,
                     ParamValues.PARAM_1_0)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.SINK_REPORTING_MODE,
                     ParamValues.PARAM_SINK_REPORTING_MANY_TO_ONE)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.NODE_MOBILITY, ParamValues.PARAM_1_0)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, WSNAttributes.BEST_TECHNIQUE,
                     ParamValues.PARAM_BEST_TECHNIQUE_LEACH)

    def loadEarthquake_Tracking(self, loadFunction):

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.NAME, ParamValues.NAME_EARTHQUAKE.MONITORING)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.EVENT_STATE, ParamValues.PARAM_EVENT_STATE_TRACKING)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.PRIMARY_FEATURE, ParamValues.PARAM_PRIMARY_FEATURE_VIBRATION)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.SPREAD_RATE, ParamValues.PARAM_0_1)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.PRIMARY_MEDIUM, ParamValues.PARAM_PRIMARY_MEDIUM_SOLID)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.AGGREG_ALGORITHM, ParamValues.PARAM_AGGREG_ALGORITHM_MAX)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.FIELD_SIZE_CHANGE, ParamValues.PARAM_0_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.AGGREGATION_TYPE, ParamValues.PARAM_AGGREGATION_TYPE_PUSH)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.OBJECTIVE_ENERGY, ParamValues.PARAM_0_2)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.OBJECTIVE_BANDWIDTH, ParamValues.PARAM_0_2)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.OBJECTIVE_LATENCY, ParamValues.PARAM_0_8)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.MIN_ENERGY_CONSUMPTION, ParamValues.PARAM_0_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.MAX_ENERGY_CONSUMPTION, ParamValues.PARAM_0_4)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.MIN_BANDWIDTH_CONSUMPTION, ParamValues.PARAM_0_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.MAX_BANDWIDTH_CONSUMPTION, ParamValues.PARAM_0_6)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.MIN_LATENCY, ParamValues.PARAM_0_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.MAX_LATENCY, ParamValues.PARAM_0_6)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.REQUIRED_CONNECTIVITY, ParamValues.PARAM_REQUIRED_CONN_PARTIAL)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.SAMPLING_RATE, ParamValues.PARAM_0_6)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.HOMOGENOUS_NODES, ParamValues.PARAM_1_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.COMMUNICATION_ALGORITHM, ParamValues.PARAM_COMMS_ALGO_HIERARCHICAL)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.SENSING_TRIGGER, ParamValues.PARAM_SENSING_TRIGGER_BY_EVENT)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.REPORTING_BY_PERIODIC, ParamValues.PARAM_1_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.REPORTING_BY_EVENT, ParamValues.PARAM_1_0)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.PERIODIC_MONITORING_BY_REALTIME, ParamValues.PARAM_1_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.PERIODIC_MONITORING_BY_QUERY, ParamValues.PARAM_0_0)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.PERIODIC_MONITORING_BY_EVENT, ParamValues.PARAM_1_0)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.PHYSICAL_SENSOR_TOPOLOGY, ParamValues.PARAM_PHYSICAL_SENSOR_TOPO_RANDOMSPARSE)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.PHYSICAL_NETWORK_TOPOLOGY, ParamValues.PARAM_PHYSICAL_NETWORK_TOPO_STAR)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.LOCATION_AWARENESS, ParamValues.PARAM_1_0)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.SINK_REPORTING_MODE, ParamValues.PARAM_SINK_REPORTING_MANY_TO_ONE)
        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.NODE_MOBILITY, ParamValues.PARAM_1_0)

        loadFunction(ParamValues.NAME_EARTHQUAKE.DETECTION, ScenarioStates.TRACKING, WSNAttributes.BEST_TECHNIQUE, ParamValues.PARAM_BEST_TECHNIQUE_HEED)


    def loadForestfire_Detection(self, loadFunction):
        loadFunction('earthquake', WSNAttributes.NAME, ParamValues.NAME_FORESTFIRE.DETECTION)
        loadFunction('earthquake', WSNAttributes.EVENT_STATE,
                     ParamValues.PARAM_EVENT_STATE_DETECTION)
        loadFunction('earthquake', WSNAttributes.PRIMARY_FEATURE,
                     ParamValues.PARAM_PRIMARY_FEATURE_VIBRATION)

        loadFunction('earthquake', WSNAttributes.SPREAD_RATE, ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.PRIMARY_MEDIUM,
                     ParamValues.PARAM_PRIMARY_MEDIUM_SOLID)

        loadFunction('earthquake', WSNAttributes.AGGREG_ALGORITHM,
                     ParamValues.PARAM_AGGREG_ALGORITHM_MAX)
        loadFunction('earthquake', WSNAttributes.FIELD_SIZE_CHANGE, ParamValues.PARAM_0_2)

        loadFunction('earthquake', WSNAttributes.AGGREGATION_TYPE,
                     ParamValues.PARAM_AGGREGATION_TYPE_PUSH)
        loadFunction('earthquake', WSNAttributes.OBJECTIVE_ENERGY, ParamValues.PARAM_0_8)
        loadFunction('earthquake', WSNAttributes.OBJECTIVE_BANDWIDTH, ParamValues.PARAM_0_2)
        loadFunction('earthquake', WSNAttributes.OBJECTIVE_LATENCY, ParamValues.PARAM_0_2)

        loadFunction('earthquake', WSNAttributes.MIN_ENERGY_CONSUMPTION,
                     ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.MAX_ENERGY_CONSUMPTION,
                     ParamValues.PARAM_0_6)
        loadFunction('earthquake', WSNAttributes.MIN_BANDWIDTH_CONSUMPTION,
                     ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.MAX_BANDWIDTH_CONSUMPTION,
                     ParamValues.PARAM_0_6)
        loadFunction('earthquake', WSNAttributes.MIN_LATENCY, ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.MAX_LATENCY, ParamValues.PARAM_0_6)

        loadFunction('earthquake', WSNAttributes.REQUIRED_CONNECTIVITY,
                     ParamValues.PARAM_REQUIRED_CONN_PARTIAL)
        loadFunction('earthquake', WSNAttributes.SAMPLING_RATE, ParamValues.PARAM_0_1)

        loadFunction('earthquake', WSNAttributes.HOMOGENOUS_NODES, ParamValues.PARAM_1_0)
        loadFunction('earthquake', WSNAttributes.COMMUNICATION_ALGORITHM,
                     ParamValues.PARAM_COMMS_ALGO_HIERARCHICAL)

        loadFunction('earthquake', WSNAttributes.SENSING_TRIGGER,
                     ParamValues.PARAM_SENSING_TRIGGER_BY_EVENT)
        loadFunction('earthquake', WSNAttributes.REPORTING_BY_PERIODIC, ParamValues.PARAM_1_0)
        loadFunction('earthquake', WSNAttributes.REPORTING_BY_EVENT, ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.PERIODIC_MONITORING_BY_REALTIME,
                     ParamValues.PARAM_1_0)
        loadFunction('earthquake', WSNAttributes.PERIODIC_MONITORING_BY_QUERY,
                     ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.PERIODIC_MONITORING_BY_EVENT,
                     ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.PHYSICAL_SENSOR_TOPOLOGY,
                     ParamValues.PARAM_PHYSICAL_SENSOR_TOPO_RANDOMSPARSE)
        loadFunction('earthquake', WSNAttributes.PHYSICAL_NETWORK_TOPOLOGY,
                     ParamValues.PARAM_PHYSICAL_NETWORK_TOPO_STAR)
        loadFunction('earthquake', WSNAttributes.LOCATION_AWARENESS, ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.SINK_REPORTING_MODE,
                     ParamValues.PARAM_SINK_REPORTING_MANY_TO_ONE)
        loadFunction('earthquake', WSNAttributes.NODE_MOBILITY, ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.BEST_TECHNIQUE,
                     ParamValues.PARAM_BEST_TECHNIQUE_LEACH)

    def loadForestfire_Tracking(self, loadFunction):

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.NAME, ParamValues.NAME_FORESTFIRE.MONITORING)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.EVENT_STATE, ParamValues.PARAM_EVENT_STATE_TRACKING)

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.PRIMARY_FEATURE, ParamValues.PARAM_PRIMARY_FEATURE_VIBRATION)

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.SPREAD_RATE, ParamValues.PARAM_0_1)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.PRIMARY_MEDIUM, ParamValues.PARAM_PRIMARY_MEDIUM_SOLID)

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.AGGREG_ALGORITHM, ParamValues.PARAM_AGGREG_ALGORITHM_MAX)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.FIELD_SIZE_CHANGE, ParamValues.PARAM_0_0)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.AGGREGATION_TYPE, ParamValues.PARAM_AGGREGATION_TYPE_PUSH)

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.OBJECTIVE_ENERGY, ParamValues.PARAM_0_2)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.OBJECTIVE_BANDWIDTH, ParamValues.PARAM_0_2)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.OBJECTIVE_LATENCY, ParamValues.PARAM_0_8)

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.MIN_ENERGY_CONSUMPTION, ParamValues.PARAM_0_0)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.MAX_ENERGY_CONSUMPTION, ParamValues.PARAM_0_4)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.MIN_BANDWIDTH_CONSUMPTION, ParamValues.PARAM_0_0)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.MAX_BANDWIDTH_CONSUMPTION, ParamValues.PARAM_0_6)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.MIN_LATENCY, ParamValues.PARAM_0_0)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.MAX_LATENCY, ParamValues.PARAM_0_6)

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.REQUIRED_CONNECTIVITY, ParamValues.PARAM_REQUIRED_CONN_PARTIAL)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.SAMPLING_RATE, ParamValues.PARAM_0_6)

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.HOMOGENOUS_NODES, ParamValues.PARAM_1_0)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.COMMUNICATION_ALGORITHM, ParamValues.PARAM_COMMS_ALGO_HIERARCHICAL)

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.SENSING_TRIGGER, ParamValues.PARAM_SENSING_TRIGGER_BY_EVENT)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.REPORTING_BY_PERIODIC, ParamValues.PARAM_1_0)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.REPORTING_BY_EVENT, ParamValues.PARAM_1_0)

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.PERIODIC_MONITORING_BY_REALTIME, ParamValues.PARAM_1_0)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.PERIODIC_MONITORING_BY_QUERY, ParamValues.PARAM_0_0)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.PERIODIC_MONITORING_BY_EVENT, ParamValues.PARAM_1_0)

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.PHYSICAL_SENSOR_TOPOLOGY, ParamValues.PARAM_PHYSICAL_SENSOR_TOPO_RANDOMSPARSE)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.PHYSICAL_NETWORK_TOPOLOGY, ParamValues.PARAM_PHYSICAL_NETWORK_TOPO_STAR)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.LOCATION_AWARENESS, ParamValues.PARAM_1_0)

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.SINK_REPORTING_MODE, ParamValues.PARAM_SINK_REPORTING_MANY_TO_ONE)
        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.NODE_MOBILITY, ParamValues.PARAM_1_0)

        loadFunction('forestfire', ScenarioStates.TRACKING, WSNAttributes.BEST_TECHNIQUE, ParamValues.PARAM_BEST_TECHNIQUE_HEED)


    def loadTsunami_Detection(self, loadFunction):
        loadFunction('earthquake', WSNAttributes.NAME, ParamValues.NAME_TSUNAMI.DETECTION)
        loadFunction('earthquake', WSNAttributes.EVENT_STATE,
                     ParamValues.PARAM_EVENT_STATE_DETECTION)
        loadFunction('earthquake', WSNAttributes.PRIMARY_FEATURE,
                     ParamValues.PARAM_PRIMARY_FEATURE_VIBRATION)

        loadFunction('earthquake', WSNAttributes.SPREAD_RATE, ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.PRIMARY_MEDIUM,
                     ParamValues.PARAM_PRIMARY_MEDIUM_SOLID)

        loadFunction('earthquake', WSNAttributes.AGGREG_ALGORITHM,
                     ParamValues.PARAM_AGGREG_ALGORITHM_MAX)
        loadFunction('earthquake', WSNAttributes.FIELD_SIZE_CHANGE, ParamValues.PARAM_0_2)

        loadFunction('earthquake', WSNAttributes.AGGREGATION_TYPE,
                     ParamValues.PARAM_AGGREGATION_TYPE_PUSH)
        loadFunction('earthquake', WSNAttributes.OBJECTIVE_ENERGY, ParamValues.PARAM_0_8)
        loadFunction('earthquake', WSNAttributes.OBJECTIVE_BANDWIDTH, ParamValues.PARAM_0_2)
        loadFunction('earthquake', WSNAttributes.OBJECTIVE_LATENCY, ParamValues.PARAM_0_2)

        loadFunction('earthquake', WSNAttributes.MIN_ENERGY_CONSUMPTION,
                     ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.MAX_ENERGY_CONSUMPTION,
                     ParamValues.PARAM_0_6)
        loadFunction('earthquake', WSNAttributes.MIN_BANDWIDTH_CONSUMPTION,
                     ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.MAX_BANDWIDTH_CONSUMPTION,
                     ParamValues.PARAM_0_6)
        loadFunction('earthquake', WSNAttributes.MIN_LATENCY, ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.MAX_LATENCY, ParamValues.PARAM_0_6)

        loadFunction('earthquake', WSNAttributes.REQUIRED_CONNECTIVITY,
                     ParamValues.PARAM_REQUIRED_CONN_PARTIAL)
        loadFunction('earthquake', WSNAttributes.SAMPLING_RATE, ParamValues.PARAM_0_1)

        loadFunction('earthquake', WSNAttributes.HOMOGENOUS_NODES, ParamValues.PARAM_1_0)
        loadFunction('earthquake', WSNAttributes.COMMUNICATION_ALGORITHM,
                     ParamValues.PARAM_COMMS_ALGO_HIERARCHICAL)

        loadFunction('earthquake', WSNAttributes.SENSING_TRIGGER,
                     ParamValues.PARAM_SENSING_TRIGGER_BY_EVENT)
        loadFunction('earthquake', WSNAttributes.REPORTING_BY_PERIODIC, ParamValues.PARAM_1_0)
        loadFunction('earthquake', WSNAttributes.REPORTING_BY_EVENT, ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.PERIODIC_MONITORING_BY_REALTIME,
                     ParamValues.PARAM_1_0)
        loadFunction('earthquake', WSNAttributes.PERIODIC_MONITORING_BY_QUERY,
                     ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.PERIODIC_MONITORING_BY_EVENT,
                     ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.PHYSICAL_SENSOR_TOPOLOGY,
                     ParamValues.PARAM_PHYSICAL_SENSOR_TOPO_RANDOMSPARSE)
        loadFunction('earthquake', WSNAttributes.PHYSICAL_NETWORK_TOPOLOGY,
                     ParamValues.PARAM_PHYSICAL_NETWORK_TOPO_STAR)
        loadFunction('earthquake', WSNAttributes.LOCATION_AWARENESS, ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.SINK_REPORTING_MODE,
                     ParamValues.PARAM_SINK_REPORTING_MANY_TO_ONE)
        loadFunction('earthquake', WSNAttributes.NODE_MOBILITY, ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.BEST_TECHNIQUE,
                     ParamValues.PARAM_BEST_TECHNIQUE_LEACH)

    def loadTsunami_Tracking(self, loadFunction):

        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.NAME, ParamValues.NAME_TSUNAMI.MONITORING)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.EVENT_STATE, ParamValues.PARAM_EVENT_STATE_TRACKING)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.PRIMARY_FEATURE, ParamValues.PARAM_PRIMARY_FEATURE_VIBRATION)

        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.SPREAD_RATE, ParamValues.PARAM_0_1)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.PRIMARY_MEDIUM, ParamValues.PARAM_PRIMARY_MEDIUM_SOLID)

        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.AGGREG_ALGORITHM, ParamValues.PARAM_AGGREG_ALGORITHM_MAX)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.FIELD_SIZE_CHANGE, ParamValues.PARAM_0_0)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.AGGREGATION_TYPE, ParamValues.PARAM_AGGREGATION_TYPE_PUSH)

        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.OBJECTIVE_ENERGY, ParamValues.PARAM_0_2)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.OBJECTIVE_BANDWIDTH, ParamValues.PARAM_0_2)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.OBJECTIVE_LATENCY, ParamValues.PARAM_0_8)

        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.MIN_ENERGY_CONSUMPTION, ParamValues.PARAM_0_0)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.MAX_ENERGY_CONSUMPTION, ParamValues.PARAM_0_4)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.MIN_BANDWIDTH_CONSUMPTION, ParamValues.PARAM_0_0)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.MAX_BANDWIDTH_CONSUMPTION, ParamValues.PARAM_0_6)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.MIN_LATENCY, ParamValues.PARAM_0_0)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.MAX_LATENCY, ParamValues.PARAM_0_6)

        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.REQUIRED_CONNECTIVITY, ParamValues.PARAM_REQUIRED_CONN_PARTIAL)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.SAMPLING_RATE, ParamValues.PARAM_0_6)

        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.HOMOGENOUS_NODES, ParamValues.PARAM_1_0)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.COMMUNICATION_ALGORITHM, ParamValues.PARAM_COMMS_ALGO_HIERARCHICAL)

        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.SENSING_TRIGGER, ParamValues.PARAM_SENSING_TRIGGER_BY_EVENT)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.REPORTING_BY_PERIODIC, ParamValues.PARAM_1_0)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.REPORTING_BY_EVENT, ParamValues.PARAM_1_0)

        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.PERIODIC_MONITORING_BY_REALTIME, ParamValues.PARAM_1_0)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.PERIODIC_MONITORING_BY_QUERY, ParamValues.PARAM_0_0)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.PERIODIC_MONITORING_BY_EVENT, ParamValues.PARAM_1_0)

        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.PHYSICAL_SENSOR_TOPOLOGY, ParamValues.PARAM_PHYSICAL_SENSOR_TOPO_RANDOMSPARSE)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.PHYSICAL_NETWORK_TOPOLOGY, ParamValues.PARAM_PHYSICAL_NETWORK_TOPO_STAR)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.LOCATION_AWARENESS, ParamValues.PARAM_1_0)

        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.SINK_REPORTING_MODE, ParamValues.PARAM_SINK_REPORTING_MANY_TO_ONE)
        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.NODE_MOBILITY, ParamValues.PARAM_1_0)

        loadFunction('tsunami', ScenarioStates.TRACKING, WSNAttributes.BEST_TECHNIQUE, ParamValues.PARAM_BEST_TECHNIQUE_HEED)


    def loadSurveillance_Detection(self, loadFunction):
        loadFunction('earthquake', WSNAttributes.NAME, ParamValues.NAME_SURVEILLANCE.DETECTION)
        loadFunction('earthquake', WSNAttributes.EVENT_STATE,
                     ParamValues.PARAM_EVENT_STATE_DETECTION)
        loadFunction('earthquake', WSNAttributes.PRIMARY_FEATURE,
                     ParamValues.PARAM_PRIMARY_FEATURE_VIBRATION)

        loadFunction('earthquake', WSNAttributes.SPREAD_RATE, ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.PRIMARY_MEDIUM,
                     ParamValues.PARAM_PRIMARY_MEDIUM_SOLID)

        loadFunction('earthquake', WSNAttributes.AGGREG_ALGORITHM,
                     ParamValues.PARAM_AGGREG_ALGORITHM_MAX)
        loadFunction('earthquake', WSNAttributes.FIELD_SIZE_CHANGE, ParamValues.PARAM_0_2)

        loadFunction('earthquake', WSNAttributes.AGGREGATION_TYPE,
                     ParamValues.PARAM_AGGREGATION_TYPE_PUSH)
        loadFunction('earthquake', WSNAttributes.OBJECTIVE_ENERGY, ParamValues.PARAM_0_8)
        loadFunction('earthquake', WSNAttributes.OBJECTIVE_BANDWIDTH, ParamValues.PARAM_0_2)
        loadFunction('earthquake', WSNAttributes.OBJECTIVE_LATENCY, ParamValues.PARAM_0_2)

        loadFunction('earthquake', WSNAttributes.MIN_ENERGY_CONSUMPTION,
                     ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.MAX_ENERGY_CONSUMPTION,
                     ParamValues.PARAM_0_6)
        loadFunction('earthquake', WSNAttributes.MIN_BANDWIDTH_CONSUMPTION,
                     ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.MAX_BANDWIDTH_CONSUMPTION,
                     ParamValues.PARAM_0_6)
        loadFunction('earthquake', WSNAttributes.MIN_LATENCY, ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.MAX_LATENCY, ParamValues.PARAM_0_6)

        loadFunction('earthquake', WSNAttributes.REQUIRED_CONNECTIVITY,
                     ParamValues.PARAM_REQUIRED_CONN_PARTIAL)
        loadFunction('earthquake', WSNAttributes.SAMPLING_RATE, ParamValues.PARAM_0_1)

        loadFunction('earthquake', WSNAttributes.HOMOGENOUS_NODES, ParamValues.PARAM_1_0)
        loadFunction('earthquake', WSNAttributes.COMMUNICATION_ALGORITHM,
                     ParamValues.PARAM_COMMS_ALGO_HIERARCHICAL)

        loadFunction('earthquake', WSNAttributes.SENSING_TRIGGER,
                     ParamValues.PARAM_SENSING_TRIGGER_BY_EVENT)
        loadFunction('earthquake', WSNAttributes.REPORTING_BY_PERIODIC, ParamValues.PARAM_1_0)
        loadFunction('earthquake', WSNAttributes.REPORTING_BY_EVENT, ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.PERIODIC_MONITORING_BY_REALTIME,
                     ParamValues.PARAM_1_0)
        loadFunction('earthquake', WSNAttributes.PERIODIC_MONITORING_BY_QUERY,
                     ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.PERIODIC_MONITORING_BY_EVENT,
                     ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.PHYSICAL_SENSOR_TOPOLOGY,
                     ParamValues.PARAM_PHYSICAL_SENSOR_TOPO_RANDOMSPARSE)
        loadFunction('earthquake', WSNAttributes.PHYSICAL_NETWORK_TOPOLOGY,
                     ParamValues.PARAM_PHYSICAL_NETWORK_TOPO_STAR)
        loadFunction('earthquake', WSNAttributes.LOCATION_AWARENESS, ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.SINK_REPORTING_MODE,
                     ParamValues.PARAM_SINK_REPORTING_MANY_TO_ONE)
        loadFunction('earthquake', WSNAttributes.NODE_MOBILITY, ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.BEST_TECHNIQUE,
                     ParamValues.PARAM_BEST_TECHNIQUE_LEACH)

    def loadSurveillance_Tracking(self, loadFunction):

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.NAME, ParamValues.NAME_SURVEILLANCE.MONITORING)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.EVENT_STATE, ParamValues.PARAM_EVENT_STATE_TRACKING)

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.PRIMARY_FEATURE, ParamValues.PARAM_PRIMARY_FEATURE_VIBRATION)

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.SPREAD_RATE, ParamValues.PARAM_0_1)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.PRIMARY_MEDIUM, ParamValues.PARAM_PRIMARY_MEDIUM_SOLID)

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.AGGREG_ALGORITHM, ParamValues.PARAM_AGGREG_ALGORITHM_MAX)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.FIELD_SIZE_CHANGE, ParamValues.PARAM_0_0)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.AGGREGATION_TYPE, ParamValues.PARAM_AGGREGATION_TYPE_PUSH)

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.OBJECTIVE_ENERGY, ParamValues.PARAM_0_2)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.OBJECTIVE_BANDWIDTH, ParamValues.PARAM_0_2)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.OBJECTIVE_LATENCY, ParamValues.PARAM_0_8)

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.MIN_ENERGY_CONSUMPTION, ParamValues.PARAM_0_0)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.MAX_ENERGY_CONSUMPTION, ParamValues.PARAM_0_4)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.MIN_BANDWIDTH_CONSUMPTION, ParamValues.PARAM_0_0)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.MAX_BANDWIDTH_CONSUMPTION, ParamValues.PARAM_0_6)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.MIN_LATENCY, ParamValues.PARAM_0_0)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.MAX_LATENCY, ParamValues.PARAM_0_6)

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.REQUIRED_CONNECTIVITY, ParamValues.PARAM_REQUIRED_CONN_PARTIAL)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.SAMPLING_RATE, ParamValues.PARAM_0_6)

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.HOMOGENOUS_NODES, ParamValues.PARAM_1_0)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.COMMUNICATION_ALGORITHM, ParamValues.PARAM_COMMS_ALGO_HIERARCHICAL)

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.SENSING_TRIGGER, ParamValues.PARAM_SENSING_TRIGGER_BY_EVENT)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.REPORTING_BY_PERIODIC, ParamValues.PARAM_1_0)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.REPORTING_BY_EVENT, ParamValues.PARAM_1_0)

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.PERIODIC_MONITORING_BY_REALTIME, ParamValues.PARAM_1_0)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.PERIODIC_MONITORING_BY_QUERY, ParamValues.PARAM_0_0)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.PERIODIC_MONITORING_BY_EVENT, ParamValues.PARAM_1_0)

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.PHYSICAL_SENSOR_TOPOLOGY, ParamValues.PARAM_PHYSICAL_SENSOR_TOPO_RANDOMSPARSE)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.PHYSICAL_NETWORK_TOPOLOGY, ParamValues.PARAM_PHYSICAL_NETWORK_TOPO_STAR)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.LOCATION_AWARENESS, ParamValues.PARAM_1_0)

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.SINK_REPORTING_MODE, ParamValues.PARAM_SINK_REPORTING_MANY_TO_ONE)
        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.NODE_MOBILITY, ParamValues.PARAM_1_0)

        loadFunction('surveillance', ScenarioStates.TRACKING, WSNAttributes.BEST_TECHNIQUE, ParamValues.PARAM_BEST_TECHNIQUE_HEED)


    def loadBodyhealth_Tracking(self, loadFunction):
        loadFunction('earthquake', WSNAttributes.NAME, ParamValues.NAME_BODYHEALTH.MONITORING)
        loadFunction('earthquake', WSNAttributes.EVENT_STATE,
                     ParamValues.PARAM_EVENT_STATE_DETECTION)
        loadFunction('earthquake', WSNAttributes.PRIMARY_FEATURE,
                     ParamValues.PARAM_PRIMARY_FEATURE_VIBRATION)

        loadFunction('earthquake', WSNAttributes.SPREAD_RATE, ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.PRIMARY_MEDIUM,
                     ParamValues.PARAM_PRIMARY_MEDIUM_SOLID)

        loadFunction('earthquake', WSNAttributes.AGGREG_ALGORITHM,
                     ParamValues.PARAM_AGGREG_ALGORITHM_MAX)
        loadFunction('earthquake', WSNAttributes.FIELD_SIZE_CHANGE, ParamValues.PARAM_0_2)

        loadFunction('earthquake', WSNAttributes.AGGREGATION_TYPE,
                     ParamValues.PARAM_AGGREGATION_TYPE_PUSH)
        loadFunction('earthquake', WSNAttributes.OBJECTIVE_ENERGY, ParamValues.PARAM_0_8)
        loadFunction('earthquake', WSNAttributes.OBJECTIVE_BANDWIDTH, ParamValues.PARAM_0_2)
        loadFunction('earthquake', WSNAttributes.OBJECTIVE_LATENCY, ParamValues.PARAM_0_2)

        loadFunction('earthquake', WSNAttributes.MIN_ENERGY_CONSUMPTION,
                     ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.MAX_ENERGY_CONSUMPTION,
                     ParamValues.PARAM_0_6)
        loadFunction('earthquake', WSNAttributes.MIN_BANDWIDTH_CONSUMPTION,
                     ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.MAX_BANDWIDTH_CONSUMPTION,
                     ParamValues.PARAM_0_6)
        loadFunction('earthquake', WSNAttributes.MIN_LATENCY, ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.MAX_LATENCY, ParamValues.PARAM_0_6)

        loadFunction('earthquake', WSNAttributes.REQUIRED_CONNECTIVITY,
                     ParamValues.PARAM_REQUIRED_CONN_PARTIAL)
        loadFunction('earthquake', WSNAttributes.SAMPLING_RATE, ParamValues.PARAM_0_1)

        loadFunction('earthquake', WSNAttributes.HOMOGENOUS_NODES, ParamValues.PARAM_1_0)
        loadFunction('earthquake', WSNAttributes.COMMUNICATION_ALGORITHM,
                     ParamValues.PARAM_COMMS_ALGO_HIERARCHICAL)

        loadFunction('earthquake', WSNAttributes.SENSING_TRIGGER,
                     ParamValues.PARAM_SENSING_TRIGGER_BY_EVENT)
        loadFunction('earthquake', WSNAttributes.REPORTING_BY_PERIODIC, ParamValues.PARAM_1_0)
        loadFunction('earthquake', WSNAttributes.REPORTING_BY_EVENT, ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.PERIODIC_MONITORING_BY_REALTIME,
                     ParamValues.PARAM_1_0)
        loadFunction('earthquake', WSNAttributes.PERIODIC_MONITORING_BY_QUERY,
                     ParamValues.PARAM_0_0)
        loadFunction('earthquake', WSNAttributes.PERIODIC_MONITORING_BY_EVENT,
                     ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.PHYSICAL_SENSOR_TOPOLOGY,
                     ParamValues.PARAM_PHYSICAL_SENSOR_TOPO_RANDOMSPARSE)
        loadFunction('earthquake', WSNAttributes.PHYSICAL_NETWORK_TOPOLOGY,
                     ParamValues.PARAM_PHYSICAL_NETWORK_TOPO_STAR)
        loadFunction('earthquake', WSNAttributes.LOCATION_AWARENESS, ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.SINK_REPORTING_MODE,
                     ParamValues.PARAM_SINK_REPORTING_MANY_TO_ONE)
        loadFunction('earthquake', WSNAttributes.NODE_MOBILITY, ParamValues.PARAM_1_0)

        loadFunction('earthquake', WSNAttributes.BEST_TECHNIQUE,
                     ParamValues.PARAM_BEST_TECHNIQUE_LEACH)

