from excel_package.rules_enums import RulesEnums


class RulesEngine:
    """
    Order of selectionsRoom
    ###################
    Field_Size
    Sink_Distance
    Atmosphere
    No of Nodes
    Physical Topology
    Logical Topology

    ['classification', 'atmosphere', 'field_size', 'number_of_nodes',
                             'physical_topology', 'sampling_rate',
                             'required_connectivity', 'obj_energy', 'obj_bandwidth', 'obj_latency',
                             'best_technique']

    """

    def __init__(self):
        self.rules_pack = [

            # Sink Distance == Field Size
            #######################################################################################
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

            # Physical Topology
            #######################################################################################
            {'conditions': {RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR},
             'settings': {RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER}
             },
            {'conditions': {RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.LINEAR},
             'settings': {RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CHAIN}
             },
            {'conditions': {RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.MESH},
             'settings': {RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.MESH}
             },
            {'conditions': {RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.BUS},
             'settings': {RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CHAIN}
             },

            # Field Size == Num of Nodes
            #######################################################################################
            {
                'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.OBJECT},
                'settings': {
                    RulesEnums.Attributes.Sink_Distance: RulesEnums.SinkDistance.OBJECT,
                    RulesEnums.Attributes.Sink_Distance: RulesEnums.Atmosphere.OBJECT
                }
            },
            {
                'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.ROOM,
                               RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Ten_Nodes},
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.LEACH,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },
            {
                'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.APARTMENT_COMPLEX,
                               RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Ten_Nodes},
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },
            {
                'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.APARTMENT_COMPLEX,
                               RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Hundred_Nodes},
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },
            {
                'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.ROOM,
                               RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Hundred_Nodes,
                               RulesEnums.Attributes.Latency_Objective: RulesEnums.DecimalValues.Eighty_Percent},
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.LEACH,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },
            {
                'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.ROOM,
                               RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Thousand_Nodes,
                               RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Eighty_Percent},
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },
            {
                'conditions': {RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.APARTMENT_COMPLEX,
                               RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Hundred_Nodes,
                               RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Eighty_Percent},
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },

            # Air, Vaccum Space - Gas,
            #######################################################################################

            # Field Size: Room, Nodes: 1-10, Objective: Energy
            #######################################################################################
            {'conditions':
                {
                    RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.ROOM,
                    RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.AIR,
                    RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Ten_Nodes,
                },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.LEACH,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },
            {'conditions':
                {
                    RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.ROOM,
                    RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.UNDERWATER_SURFACE,
                    RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Ten_Nodes,
                },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },
            {'conditions':
                {
                    RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.OCEAN,
                    RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.UNDERWATER_SURFACE,
                    RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Hundred_Nodes,
                },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },
            {'conditions':
                {
                    RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.OCEAN,
                    RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.UNDERWATER_SURFACE,
                    RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Thousand_Nodes,
                },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },
            {'conditions':
                {
                    RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.APARTMENT_COMPLEX,
                    RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.UNDERWATER_SURFACE,
                    RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Hundred_Nodes,
                    RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Eighty_Percent,
                    RulesEnums.Attributes.Bandwidth_Objective: RulesEnums.DecimalValues.Two_Percent,
                    RulesEnums.Attributes.Latency_Objective: RulesEnums.DecimalValues.Two_Percent
                },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },
            {'conditions':
                {
                    RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY,
                    RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.UNDERWATER_SURFACE,
                    RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Hundred_Nodes,
                    RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Eighty_Percent,
                },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },
            {'conditions':
                {
                    RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.REGION,
                    RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.UNDERWATER_SURFACE,
                    RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Hundred_Nodes
                },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.DBST,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.TREE
                }
            },
            {'conditions':
                {
                    RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY,
                    RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.VACUUM_SPACE,
                },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.DBST,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.BUS,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.TREE
                }
            },
            {'conditions':
                {
                    RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY_BLOCK,
                    RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.VACUUM_SPACE,
                },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.DBST,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.BUS,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.TREE
                }
            },
            {'conditions':
                {
                    RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.REGION,
                    RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.AIR,
                    RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Eighty_Percent,
                    RulesEnums.Attributes.Bandwidth_Objective: RulesEnums.DecimalValues.Two_Percent,
                    RulesEnums.Attributes.Latency_Objective: RulesEnums.DecimalValues.Six_Percent
                },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.DBST,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.BUS,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.TREE
                }
            },
            {'conditions':
                {
                    RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.OBJECT,
                    RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Ten_Nodes,
                    RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Eighty_Percent,
                    RulesEnums.Attributes.Bandwidth_Objective: RulesEnums.DecimalValues.Two_Percent,
                    RulesEnums.Attributes.Latency_Objective: RulesEnums.DecimalValues.Six_Percent
                },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.LEACH,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER,
                }
            },
            #######################################################################################
            # All underground should be Pegasis
            {'conditions': {
                    RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.OVERGROUND,
                    RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Thousand_Nodes,
                    RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Six_Percent,
                    RulesEnums.Attributes.Bandwidth_Objective: RulesEnums.DecimalValues.Two_Percent,
                    RulesEnums.Attributes.Latency_Objective: RulesEnums.DecimalValues.Eighty_Percent
            },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.PEGASIS,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.BUS,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CHAIN,
                }
            },
            #######################################################################################
            {'conditions': {
                RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.UNDERWATER_SURFACE,
                    RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Eighty_Percent,
                    RulesEnums.Attributes.Bandwidth_Objective: RulesEnums.DecimalValues.Two_Percent
            },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER
                }
            },
            {'conditions': {
                RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.UNDERWATER_DEEP,
                RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.OCEAN
            },
                'settings': {
                    RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.PEGASIS,
                    RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.BUS,
                    RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CHAIN
                }
            },
            #######################################################################################
            {'conditions': {RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Ten_Nodes,
                            RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Eighty_Percent,
                            RulesEnums.Attributes.Bandwidth_Objective: RulesEnums.DecimalValues.Two_Percent,
                            RulesEnums.Attributes.Latency_Objective: RulesEnums.DecimalValues.Two_Percent,
                            RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.UNDERWATER_DEEP,
                            },
             'settings': {
                 RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.PEGASIS,
                 RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.BUS,
                 RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CHAIN
             }
             },
            # Wild Fire - Detection
            {'conditions': {RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Hundred_Nodes,
                            RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Two_Percent,
                            RulesEnums.Attributes.Bandwidth_Objective: RulesEnums.DecimalValues.Two_Percent,
                            RulesEnums.Attributes.Latency_Objective: RulesEnums.DecimalValues.Eighty_Percent,
                            RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.OVERGROUND,
                            },
             'settings': {
                 RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.DIRECTED_DIFFUSION,
                 RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.MESH,
                 RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.MESH
             }
             },
            {'conditions': {RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Thousand_Nodes,
                            RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Two_Percent,
                            RulesEnums.Attributes.Latency_Objective: RulesEnums.DecimalValues.Eighty_Percent,
                            RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.OVERGROUND,
                            },
             'settings': {
                 RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.DIRECTED_DIFFUSION,
                 RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.MESH,
                 RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.MESH
             }
             },
            {'conditions': {RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Thousand_Nodes,
                            RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.APARTMENT_COMPLEX,
                            RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Two_Percent,
                            RulesEnums.Attributes.Latency_Objective: RulesEnums.DecimalValues.Eighty_Percent,
                            RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.OVERGROUND,
                            },
             'settings': {
                 RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.DIRECTED_DIFFUSION,
                 RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.MESH,
                 RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.MESH
             }
             },
            {'conditions': {RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Thousand_Nodes,
                            RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.APARTMENT_COMPLEX,
                            # RulesEnums.Attributes.Energy_Objective: RulesEnums.DecimalValues.Two_Percent,
                            # RulesEnums.Attributes.Latency_Objective: RulesEnums.DecimalValues.Eighty_Percent,
                            RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.AIR,
                            },
             'settings': {
                RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER
             }
             },
            {'conditions': {RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Ten_Nodes,
                            RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY_BLOCK,
                            RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY,
                            RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.AIR,
                            },
             'settings': {
                RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.DBST,
                RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.TREE
             }
             },
            {'conditions': {
                RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Hundred_Nodes,
                RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Thousand_Nodes,
                RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY,
                RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY_BLOCK,
                RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.AIR
            },
             'settings': {
                RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.DBST,
                RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.TREE
             }
             },
            {'conditions': {
                    RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Ten_Nodes,
                RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Ten_Nodes,
                RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Hundred_Nodes,
                RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY,
                RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY_BLOCK,
                RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.AIR
            },
             'settings': {
                RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.DBST,
                RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.TREE
             }
             },
            {'conditions': {RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Ten_Nodes,
                            RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY_BLOCK,
                            RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.AIR,
                            },
             'settings': {
                RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER
             }
             },
            {'conditions': {RulesEnums.Attributes.Number_of_Nodes: RulesEnums.NumberOfNodes.One_Hundred_Nodes,
                            RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY,
                            RulesEnums.Attributes.Field_Size: RulesEnums.FieldSize.CITY_BLOCK,
                            RulesEnums.Attributes.Atmosphere: RulesEnums.Atmosphere.AIR,
                            },
             'settings': {
                RulesEnums.Attributes.Best_Technique: RulesEnums.BestTechnique.HEED,
                RulesEnums.Attributes.Physical_Topology: RulesEnums.PhysicalTopology.STAR,
                RulesEnums.Attributes.Logical_Topology: RulesEnums.LogicalTopology.CLUSTER
             }
             }

        ]
