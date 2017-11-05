from classAgent import Agent
import numpy as np

import threading
from time import clock, sleep
import pandas as pd
cell_row = 25
cell_column = 25
N = 6
bound = 10000000000000000
area = 1
f = pd.read_csv('/Users/BuleSky/Desktop/K.csv')
surveillance_mat = f.values
agent1 = Agent('a1', [2, 3], np.zeros([25, 25]), 0.9, 0.3, 4, 3,10)
agent2 = Agent('a2', [6, 9], np.zeros([25, 25]), 0.9, 0.3, 4, 3,10)
agent3 = Agent('a3', [12, 7], np.zeros([25, 25]), 0.9, 0.3, 4, 3,10)
agent4 = Agent('a4', [18, 22], np.zeros([25, 25]), 0.9, 0.3, 4, 3,10)
agent5 = Agent('a5', [16, 5], np.zeros([25, 25]), 0.9, 0.3, 4, 3,10)
agent6 = Agent('a6', [13, 8], np.zeros([25, 25]), 0.9, 0.3, 4, 3,10)
while True:
    agent1.SingleUp(cell_row,cell_column,surveillance_mat,bound)
    agent2.SingleUp(cell_row,cell_column,surveillance_mat,bound)
    agent3.SingleUp(cell_row,cell_column,surveillance_mat,bound)
    agent4.SingleUp(cell_row,cell_column,surveillance_mat,bound)
    agent5.SingleUp(cell_row,cell_column,surveillance_mat,bound)
    agent6.SingleUp(cell_row,cell_column,surveillance_mat,bound)
    agent_list = {agent1.name:agent1.agent_local,agent2.name:agent2.agent_local,agent3.name:agent3.agent_local,agent4.name:agent4.agent_local,agent5.name:agent5.agent_local,agent6.name:agent6.agent_local}
    print(agent_list)
    allagent_list = [agent1.name,agent2.name,agent3.name,agent4.name,agent5.name,agent6.name]
    list_neighbor1 = agent1.Neighbor(agent_list,allagent_list)
    list_neighbor2 = agent2.Neighbor(agent_list,allagent_list)
    list_neighbor3 = agent3.Neighbor(agent_list,allagent_list)
    list_neighbor4 = agent4.Neighbor(agent_list,allagent_list)
    list_neighbor5 = agent5.Neighbor(agent_list,allagent_list)
    list_neighbor6 = agent6.Neighbor(agent_list,allagent_list)
    print(list_neighbor1)
    map_array = [agent1.agent_map,agent2.agent_map,agent3.agent_map,agent4.agent_map,agent5.agent_map,agent6.agent_map]
    agent1.Fusion(list_neighbor1, N, map_array, allagent_list)
    agent2.Fusion(list_neighbor2, N, map_array, allagent_list)
    agent3.Fusion(list_neighbor3, N, map_array, allagent_list)
    agent4.Fusion(list_neighbor4, N, map_array, allagent_list)
    agent5.Fusion(list_neighbor5, N, map_array, allagent_list)
    agent6.Fusion(list_neighbor6, N, map_array, allagent_list)
    print(agent1.agent_map,agent2.agent_map,agent3.agent_map,agent4.agent_map,agent5.agent_map,agent6.agent_map)
    voronoi_cell1 = agent1.Voronoi()
    voronoi_cell2 = agent2.Voronoi()
    voronoi_cell3 = agent3.Voronoi()
    voronoi_cell4 = agent4.Voronoi()
    voronoi_cell5 = agent5.Voronoi()
    voronoi_cell6 = agent6.Voronoi()
    agent1.Move(voronoi_cell1,area)
    agent2.Move(voronoi_cell2,area)
    agent3.Move(voronoi_cell3,area)
    agent4.Move(voronoi_cell4,area)
    agent5.Move(voronoi_cell5,area)
    agent6.Move(voronoi_cell6,area)
    agent_mapdict = {agent1.name:agent1.agent_map,agent2.name:agent2.agent_map,agent3.name:agent3.agent_map,agent4.name:agent4.agent_map,agent5.name:agent5.agent_map,agent6.name:agent6.agent_map}
    unsure = 0
    for value in agent_mapdict.values():
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                unsure += np.exp(-2*np.linalg.norm(value[i][j]))
    n = float(unsure)/float(N * cell_column * cell_row)
    if n <= 0.01:
        print(agent_mapdict)   # 所有key的值理论上应该都收敛成一个相同的矩阵,矩阵中每个点的值均为ln(1/p - 1),还要转换为P
        break



