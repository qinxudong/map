#coding=utf-8
from classAgent import Agent
import numpy as np
import pandas as pd

#输入搜索区域矩阵数据
df = pd.read_csv('/home/qin/github/multiagent_targetsearch@qin/test_input.csv', header=None)
matrix_sr = df.values
num_row = matrix_sr.shape[0]
num_column = matrix_sr.shape[1]
map_init = np.zeros([num_row, num_column])

#agent初始化
agent0 = Agent(0, [2, 3], map_init, 0.9, 0.3, 8, 10, 8, 3,  10000000000000000)
agent1 = Agent(1, [6, 9], map_init, 0.9, 0.3, 8, 10, 8, 3,  10000000000000000)
agent2 = Agent(2, [12, 7], map_init, 0.9, 0.3, 8, 10, 8, 3,  10000000000000000)
agent3 = Agent(3, [18, 22], map_init, 0.9, 0.3, 8, 10, 8, 3,  10000000000000000)
agent4 = Agent(4, [16, 5], map_init, 0.9, 0.3, 8, 10, 8, 3,  10000000000000000)
agent5 = Agent(5, [13, 8], map_init, 0.9, 0.3, 8, 10, 8, 3,  10000000000000000)

#初始化一些后面用到的列表
list_agent = [agent0, agent1, agent2, agent3, agent4, agent5]#agent列表
N = len(list_agent)#agent数量
coord_all = []#所有agent坐标的列表
list_map = []#所有agent概率图的列表
for agent in list_agent:
    coord_all.append(agent.coord)
    list_map.append(agent.map)
array_map = np.array(list_map)#转array

#主循环
while True:
    #单个agent概率图更新
    for agent in list_agent:
        cell_search = agent.cellsearch()
        agent.update(matrix_sr, cell_search)
    #概率图融合
    for agent in list_agent:
        agent_neighbor = agent.neighbor(coord_all)
        agent.fuse(agent_neighbor, N, array_map)
    #agent移动
    for agent in list_agent:
        cell_consider = agent.cellconsider()
        agent.move(cell_consider)

    for agent in list_agent:
            print(agent.map)
    #循环退出条件
    unsure = 0#概率图的不确定度
    for agent in list_agent:
        for i in range(num_row):
            for j in range(num_column):
                unsure += np.exp(-2*np.linalg.norm(agent.map[i][j]))
    unsure_final = unsure / (N * num_column * num_row)
    if unsure_final <= 0.01:
        for agent in list_agent:
            print(agent.map)
        break
