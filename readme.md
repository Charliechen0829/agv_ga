

## 2023-2024（二）计算智能课程设计

项目内容为2022 年第十二届MathorCup高校数学建模挑战赛题目问题一、二题解

### 数据说明

实验数据提供了无人仓模型相关的4个数据文件，其数据格式解释如下。

**仓库地图数据 (map.csv)**
此文件存放是地图信息，内容包括仓库内各类节点的具体分布情况，以及节点之间的连通性。给定的地图是32*22的一块矩形区域。数据字段含意如下：

TYPE: 节点类型

路径节点：AGV可以通行
储位节点：放置托盘或者普通货架，AGV可以到达。一般只有一个位置可以进出
保留节点：保留位置，暂不处理
柱子节点：障碍物，AGV不能到达
拣选工位节点：拣选机器人在这里把商品打包后从传送带出库，一般有多个托盘停靠位
补货位节点：从高密度区补货的商品放置点，一般通过传送带输送
空托集放工位节点：空托盘回收处
X: 节点所在位置的X轴坐标

Y: 节点所在位置的Y轴坐标

NEIGHBORS: AGV可到达的节点，格式为，表示可以从当前节点能到达的邻近节点的坐标集合。这里最多是4个邻近节点。

TYPE	X	Y	NEIGHBORS
1	3	0	3:1;2:0;4:0
1	4	0	4:1;3:0;5:0
2	5	5	5:4
2	6	5	6:4
...	...	...	...
**搬运机器人数据 (agv.csv)**
此文件存放的是搬运机器人的初始信息，内容包括每个AGV位置信息。数据字段含意如下：

AGV_ID: 搬运机器人AGV的唯一ID

X: AGV所在位置的X轴坐标

Y: AGV所在位置的Y轴坐标

AGV_ID	X	Y
1	31	14
2	7	11
3	21	13
...	...	...
**订单数据 (orders.csv)**
此文件存放是订单信息，为一段时间内累计的订单。这里订单已经根据商品号SKU拆分。需要分派AGV根据订单上的商品号SKU去寻找对应的托盘，并将托盘搬运到拣选工位上出库。数据字段含意如下：

ORDER_ID: 订单的唯一ID

SKU: SKU的唯一ID

AMOUNT: 所需该SKU的数量

ORDER_ID	SKU	AMOUNT
1	1579172	6
2	2609314	12
3	1335852	53
...	...	...
**库存数据 (pallets.csv)**
此文件存放的是仓库内托盘上商品信息，包括每个托盘的位置以及存放的各个商品SKU和数量。数据字段含意如下：

SKU_QUANTITY_LIST: 该托盘上存放的各个SKU的ID和数量，格式为""，表示存放了个，...，和个

X: 该托盘所在位置的X轴坐标

Y: 该托盘所在位置的Y轴坐标

PALLET_ID: 托盘的唯一ID

SKU_QUANTITY_LIST	X	Y	PALLET_ID
"104151:9,840211:35,1297235:1"	18	8	10000
"901897:13,1297235:18,2171945:14"	19	16	10001
"653296:69"	14	12	10002
"1473101:66"	18	15	10003
"961380:8,1187577:49"	14	6	10004
...	...	...	...

**无人仓地图节点邻接矩阵(adj_matrix.csv)**

**无人仓地图节点距离矩阵(distance_matrix.csv)**