import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
map = pd.read_csv("./data/map.csv")
agv = pd.read_csv("./data/agv.csv")
order = pd.read_csv("./data/orders.csv")
pallet = pd.read_csv("./data/pallets.csv")

cross_rate = 0.5  # 交叉率
individual_num = 80  # 每一代个体数
gen_num = 500  # 迭代次数

num_nodes = len(map[" NEIGHBORS"])
agv_num = len(agv["#AGV_ID"])


# 绘制栅格地图
def plot_map():
    # 定义不同类型节点的颜色
    colors = {
        1: "gray",  # Path nodes
        2: "green",  # Storage nodes
        3: "yellow",  # Reserved nodes
        4: "black",  # Obstacle nodes
        5: "blue",  # Picking station nodes
        6: "pink",  # Replenishment nodes
        7: "red"  # Empty pallet recycling nodes
    }
    # 创建散点图
    for node_type, color in colors.items():
        nodes = map[map["#TYPE"] == node_type]
        plt.scatter(nodes[" X"], nodes[" Y"], color=color, label=f"Type {node_type}", marker='s')

    # 在散点图上补偿agv信息
    for i in range(agv_num):
        plt.scatter(agv["X"][i], agv["Y"][i], color="orange", marker='s')
    # x=25,y=6的机器人不可用

    # 绘制可行路径
    for i in range(num_nodes):
        loc = map[" NEIGHBORS"][i]
        if loc == 'nan':
            continue
        neighbor = str(loc).split(";")
        for item in neighbor:
            if item == 'nan':
                continue
            x = int(item.split(":")[0])
            y = int(item.split(":")[1])
            plt.plot([map[" X"][i], x], [map[" Y"][i], y], color="grey")
    # 添加标签和图例
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.title("栅格地图")
    # 显示图形
    plt.show()


# 创建邻接矩阵
loc = map[" NEIGHBORS"]
adj_matrix = np.full((num_nodes, num_nodes), np.inf)


def find(x, y):
    for i in range(num_nodes):
        if map[" X"][i] == int(x) and map[" Y"][i] == int(y):
            return i
        else:
            continue
    return -1


# 构建邻接矩阵
storage_label = []  # 储位结点标签
agv_label = []  # agv结点标签
pick_label = []  # 拣选结点标签
for i in range(num_nodes):
    if map['#TYPE'][i] == 5:
        pick_label.append(i)
    if map["#TYPE"][i] == 2:
        storage_label.append(i)
    for j in range(agv_num):
        if agv["#AGV_ID"][j] == 11:
            continue
        if map[" X"][i] == agv["X"][j] and map[" Y"][i] == agv["Y"][j]:
            agv_label.append(i)
    # if loc[i] == 'nan':
    #     continue
    # neighbor = str(loc[i]).split(";")
    # for item in neighbor:
    #     if item == 'nan':
    #         continue
    #     x = item.split(":")[0]
    #     y = item.split(":")[1]
    #     neighbor_node = find(x, y)
    #     adj_matrix[i][i] = 0
    #     adj_matrix[i][neighbor_node] = 1
pick_num = len(pick_label)

adj_matrix[num_nodes - 1][num_nodes - 1] = 0
adj_matrix = pd.DataFrame(adj_matrix)
# adj_matrix.to_csv('adj_matrix.csv', sep=',', index=False)
adj_matrix = pd.read_csv('./data/adj_matrix.csv')
print(adj_matrix)
storage_num = len(storage_label)


# 利用floyd算法创建距离矩阵
def floyd(adj_matrix):
    num_vertices = len(adj_matrix)
    distance_matrix = np.copy(adj_matrix)
    route_matrix = np.full((num_vertices, num_vertices), -1)

    for k in range(num_vertices):
        for i in range(num_vertices):
            for j in range(num_vertices):
                if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                    route_matrix[i][j] = k

    return distance_matrix, route_matrix


# distance_matrix, route_matrix = floyd(adj_matrix)
# distance_matrix = pd.DataFrame(distance_matrix)
# distance_matrix.to_csv('distance_matrix.csv', sep=',', index=False)
# print(distance_matrix)

# route_matrix = pd.DataFrame(route_matrix)
# print(route_matrix)

distance_matrix = pd.read_csv("./data/distance_matrix.csv")
print(distance_matrix)

# 对托盘与订单数据进行对比
order_sum = 0  # 订单总量
pallet_dict = {}  # 货物ID与其数量构成的二维数组
for good in pallet["#SKU_QUANTITY_LIST"]:
    good_list = good.split(",")  # good_list = [90001:1 00129:2...]
    for item in good_list:
        sku = item.split(":")
        sku_id = sku[0]
        sku_num = int(sku[1])
        if pallet_dict.get(sku_id) is None:  # 未查询到该商品id，插入该商品信息
            pallet_dict.update({sku_id: sku_num})
        else:
            pallet_dict[sku_id] += sku_num
# pallet_dict = sorted(pallet_dict)

order_dict = {}
for i in range(len(order)):
    sku_id = str(order[" SKU"][i])
    sku_num = int(order[" AMOUNT"][i])
    if order_dict.get(sku_id) is None:
        order_dict.update({sku_id: sku_num})
    else:
        order_dict[sku_id] += sku_num

pd = {}
for key in sorted(pallet_dict):
    pd[key] = pallet_dict[key]
od = {}
for key in sorted(order_dict):
    order_sum += order_dict[key]
    od[key] = pallet_dict[key] - order_dict[key]
for (key, value) in od.items():
    if value == 0:
        continue
    else:
        print((key, value))

order_aver = order_sum / pick_num
print("order_aver : ", order_aver)


# 计算当前结点到最近拣选摊位的距离
def find_pick(pre_loc):
    min_pick_dist = distance_matrix.iloc[pre_loc, pick_label[0]]
    for i in range(pick_num):
        pick_dist = distance_matrix.iloc[pre_loc, pick_label[i]]
        if pick_dist < min_pick_dist:
            min_pick_dist = pick_dist
    return min_pick_dist


# 测试返回路径
def return_routine(genes):
    agv_id_list = []
    for i in range(0, agv_num):
        agv_id_list.append([])

    for i in range(storage_num):
        head = str(random.randint(0, 18))
        tail = str(storage_label[i])
        if i == storage_num - 1:
            genes += (head + ":" + tail)
        else:
            genes += (head + ":" + tail + '|')
    frag_list = genes.split("|")
    for frag in frag_list:
        agv_id_list[int(frag.split(":")[0])].append(frag.split(":")[1])
    for i in range(agv_num - 1):
        print("(", map[" X"][agv_label[i]], ",", map[" Y"][agv_label[i]], ")", agv_id_list[i])


# 基因：agv+储位点编号，每个储位点都需要被搬运
# 适应度：行走距离

# 改进遗传算法设计


# 个体类
class Individual:
    def __init__(self, genes=""):
        # 随机生成初始基因序列
        if genes == "":
            for i in range(storage_num):
                head = str(random.randint(0, 18))
                tail = str(storage_label[i])
                if i == storage_num - 1:
                    genes += (head + ":" + tail)
                else:
                    genes += (head + ":" + tail + '|')

        self.genes = genes
        self.fitness = self.evaluate_fitness()

    # 计算个体适应度
    def evaluate_fitness(self):
        agv_loc = np.zeros(19)
        fitness = 0
        genes = self.genes
        gene_list = genes.split("|")
        for frag in gene_list:
            pre = int(frag.split(":")[0])
            dest = int(frag.split(":")[1])
            if agv_loc[pre] == 0:
                fitness += distance_matrix.iloc[int(agv_label[pre]), dest]
                fitness += 2 * find_pick(int(agv_label[pre]))
                agv_loc[pre] = dest
            else:
                fitness += distance_matrix.iloc[int(agv_loc[pre]), dest]
                fitness += 2 * find_pick(int(agv_loc[pre]))
                agv_loc[pre] = dest
        return fitness


class Ga:
    def __init__(self):
        self.best = None  # 每一代的最佳个体
        self.individual_list = []  # 每一代的个体列表
        self.gene_list = []  # 每一代对应的基因
        self.fitness_list = []  # 每一代对应的适应度
        self.result_list = []

    # 单点交叉
    def cross(self, cross_rate):
        random.shuffle(self.individual_list)
        cross_num = int(cross_rate * individual_num)
        cross_index = random.sample(range(0, individual_num - 2), cross_num)
        index = random.randint(0, storage_num - 1)
        for i in cross_index:
            genes1 = self.individual_list[i].genes
            genes2 = self.individual_list[i + 1].genes
            frag1 = genes1.split("|")
            frag2 = genes2.split("|")
            frag1[index:], frag2[index:] = frag2[index:], frag1[index:]
            new_gene1 = ""
            new_gene2 = ""
            for j in range(len(frag1)):
                if j == len(frag1) - 1:
                    new_gene1 += frag1[j]
                    new_gene2 += frag2[j]
                else:
                    new_gene1 += (frag1[j] + "|")
                    new_gene2 += (frag2[j] + "|")
            self.individual_list[i].genes = new_gene1
            self.individual_list[i + 1].genes = new_gene2

    # 变异
    def mutate(self, mutate_rate):
        r = random.random()
        if r <= mutate_rate:
            mutate_label = random.randint(0, individual_num - 1)
            mutate_gene = self.individual_list[mutate_label].genes
            frag_list = mutate_gene.split("|")
            agv_list = []
            storage_list = []
            for frag in frag_list:
                agv_id = frag.split(":")[0]
                agv_list.append(agv_id)
                storage_id = frag.split(":")[1]
                storage_list.append(storage_id)
            random.shuffle(agv_list)  # 打乱顺序
            new_gene = ""
            for i in range(len(agv_list)):
                if i == len(agv_list) - 1:
                    new_gene += (agv_list[i] + ":" + storage_list[i])
                else:
                    new_gene += (agv_list[i] + ":" + storage_list[i] + "|")
            self.individual_list[mutate_label].genes = new_gene

    # 轮盘赌
    def select(self):
        total_fitness = sum([1 / individual.fitness for individual in self.individual_list])
        wheel = [1 / individual.fitness / total_fitness for individual in self.individual_list]

        for i in range(1, len(wheel)):
            wheel[i] += wheel[i - 1]

        new_individual_list = []
        for _ in range(individual_num):
            rand_num = random.random()
            for i in range(len(wheel)):
                if rand_num <= wheel[i]:
                    new_individual_list.append(Individual(self.individual_list[i].genes))
                    break
        self.individual_list = new_individual_list

    def next_gen(self):
        self.cross(cross_rate)
        self.mutate(0.05)
        self.select()
        best_fitness_in_gen = min([id.fitness for id in self.individual_list])
        print("best in gen: ", best_fitness_in_gen)
        self.fitness_list.append(best_fitness_in_gen)
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:
                self.best = individual

    def train(self):
        # 初始化第一代种群
        self.individual_list = [Individual() for _ in range(individual_num)]
        self.best = self.individual_list[0]
        # 迭代
        for i in range(gen_num):
            self.next_gen()
            self.result_list.append(self.best.fitness)
            # print(self.best.fitness)
        return self.result_list


# 测试
def generate_fitness():
    agv_loc = np.zeros(19)
    fitness = 0
    genes = ""
    for i in range(storage_num):
        head = str(random.randint(0, 18))
        tail = str(storage_label[i])
        if i == storage_num - 1:
            genes += (head + ":" + tail)
        else:
            genes += (head + ":" + tail + '|')
    print("genes:", genes)
    gene_list = genes.split("|")
    for frag in gene_list:
        pre = int(frag.split(":")[0])
        dest = int(frag.split(":")[1])
        if agv_loc[pre] == 0:
            fitness += distance_matrix.iloc[int(agv_label[pre]), dest]
            fitness += 2 * find_pick(dest)
            agv_loc[pre] = dest
        else:
            fitness += distance_matrix.iloc[int(agv_loc[pre]), dest]
            fitness += 2 * find_pick(dest)
            agv_loc[pre] = dest

    print("fitness:", fitness)
    return fitness


plot_map()
# generate_fitness()
ga = Ga()
result_list = ga.train()
# 每次迭代的全局最优
print(result_list)
# 每次迭代的单次最优
print(ga.fitness_list)
plt.plot(range(1, gen_num + 1), result_list, color='red', label='全局最优')
plt.plot(range(1, gen_num + 1), ga.fitness_list, color='green', label='单次迭代内最优')
plt.xlabel("迭代次数")
plt.ylabel("agv总行走距离")
plt.legend()
plt.show()

print("最短距离： ", ga.best.fitness)

# 输出最佳个体各机器人的路径
return_routine(ga.best.genes)
