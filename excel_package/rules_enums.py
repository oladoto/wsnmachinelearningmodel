

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

    class NumberOfNodes:

        One_Ten_Nodes = '10'
        One_Hundred_Nodes = '100'
        One_Thousand_Nodes = '1000'

    class DecimalValues:
        Two_Percent = '0.2'
        Six_Percent = '0.6'
        Eighty_Percent = '0.8'

    class SamplingRate:
        Ten = '10'
        Hundred = '50'
        Thousand = '100'

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
        STAR = 'Star'
        BUS = 'Bus'
        LINEAR = 'Linear'
        RING = 'Ring'
        MESH = 'Mesh'

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