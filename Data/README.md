# Dataset for Heterogeneous Spatial-Temporal Graph Prediction

The dataset takes *(i)* a heterogeneous subgraph including 74 heterogeneous nodes, 554 edges and 312 routes; *(ii)* temporal data within 10 days. 




## Spatial Information

The heterogeneous graph includes 30 warehouses(Nodetype: Node1) and 44 sorting centers(Nodetype: Node2).

The spatial information is recorded in *Spatial_Graph_Edge.csv*, *Spatial_Delivery_Route.csv* and *Node_Type2_Rank_Score.csv*, the details of these files are shown below:

### Spatial_Graph_Edge.csv
This file record the 554 edges in the heterogeneous warehouse-center graph, and the heterogeneous graph is constructed by this file.
|  Column | Description  | 
|  ----  | ----  | 
|  source_id   |  ID of source node |
| target_id  | ID of target node |
| relation_type  | edge type (Node1_to_Node2, Node2_to_Node2)|


### Spatial_Delivery_Route.csv
This file record the 312 routes that a package is delivered from warehouse to customer. There is only one route from a warehouse node to a sorting center node. When the node is *Null*, it is marked as *-1*.

|  Column | Description  | 
|  ----  | ----  | 
|  Node_Type1_id   | ID of warhouse node  |
| 1_Node_Type2_id  | ID of first sorting center node |
| 2_Node_Type2_id  | ID of second sorting center node |
| 3_Node_Type2_id  | ID of third sorting center node |
| 4_Node_Type2_id  | ID of fourth sorting center node |
| 5_Node_Type2_id  | ID of fifth sorting center node |

### Node_Type2_Rank_Score.csv
The file record the sorting rank score of sorting center nodes.
|  Column | Description  | 
|  ----  | ----  | 
|  Node_Type2_ID   | ID of sorting center node  |
| Node_Type2_Rank_Score  | Soring rank score |


## Temporal Information
The temporal infomation of each heterogeneous node is recorded in *Temporal_Data.csv*, and the ground truth of each route is recorded in *Y_Delivery_Time.csv*.

The details of these files are shown below:

### Temporal_Data.csv
The file record the temporal data of warehouse node and sorting center node, including temporal data of heterogeneous nodes and temporal background data.

#### Node Flag
|  Column | Description  | 
|  ----  | ----  | 
|  time   | time index  |
| node_type  | node type (Node1: warehouse, Node2: sorting center) |
| node_id| ID of node |

#### Temporal Data for Node1


|  Column | Description  | 
|  ----  | ----  | 
|  xt_0   |  total amount of packages to be processed by warehouse node |
| xt_1  | hourly cumulative sales volume |
| xt_2| daily cumulative sales volume |
| xt_3| total amount of packages delivered in downstream routes |

#### Temporal Data for Node2
|  Column | Description  | 
|  ----  | ----  | 
|  xt_0   | total amount of packages delivered in current routes  |
| xt_1  | total amount of packages sent by upstream distribution units |
| xt_2| hourly cumulative sales volume |
| xt_3| daily cumulative sales volume |
#### Temporal Background Data
|  Column | Description  | 
|  ----  | ----  | 
| xb_0|working time:1, non-working time |
|xb_1|working day:1, non-working day:0|
|xb_2| sales promotion period:1, normal period:0|
### Y_Delivery_Time.csv
The file record the ground truth time of each routes within ten days, where *full_time = Node1_time + pack_time + Node2_time*.
|  Column | Description  | 
|  ----  | ----  | 
| time|Time index|
|route_id|ID of route (0-311)|
|full_time| Full-link delivery time in a route|
|Node1_time| Time in warhouse|
|pack_time| Package loading time |
|Node2_time| Total time in downstream sorting centers|