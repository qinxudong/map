import numpy as np
class Agent:

    def __init__(self,name,agent_local,agent_map,p,q,search_radius,max_speed,communication_radius):
        self.name = name
        self.agent_local = agent_local
        self.agent_map = agent_map
        self.p = p
        self.q = q
        self.search_radius = search_radius
        self.max_speed = max_speed
        self.communication_radius = communication_radius


    def SingleUp(self,cell_row,cell_column,surveillance_mat,bound):  # 单图
        agent_cell = []
        value_array = np.array(self.agent_local)
        for i in range(cell_row):
            for j in range(cell_column):
                cell_coord = np.array([i, j])
                dist1 = np.linalg.norm(value_array - cell_coord)
                if dist1 <= self.search_radius:  # 4为搜索范围
                   agent_cell.append(cell_coord)
        for one in agent_cell:
            point = surveillance_mat[one[0]][one[1]]
            if point == 1:
                 Q = self.agent_map[one[0]][one[1]] + np.log(float(self.q)/float(self.p))
                 final_Q = max(min(Q,bound), -bound)
                 self.agent_map[one[0]][one[1]] = final_Q

            else:
                Q = self.agent_map[one[0]][one[1]] + np.log(float(1-self.q)/float(1- self.p))
                final_Q = max(min(Q, bound), -bound)
                self.agent_map[one[0]][one[1]] = final_Q
        print(self.agent_map)

    def Neighbor(self,agent_list,allagent_list):
        list_neighbor = []
        for key in agent_list:
            if np.linalg.norm(self.agent_local - np.array(agent_list[key])) <= self.communication_radius:
               list_neighbor.append(key)
        list_neighbor01 = []
        for one in allagent_list:
            if one in agent_list:
                list_neighbor01.append(1)
            else:
                list_neighbor01.append(0)
        return list_neighbor01


    def Fusion(self,list_neighbor01,N,map_array,allagent_list):  # 融合
        d = sum(list_neighbor01)
        list_w = [float(1/N), float(1/N),float(1/N),float(1/N),float(1/N),float(1/N)]

        k = allagent_list.index(self.name)
        list_w[k] = 1-(float(d -1)/N)
        w = np.array(list_w)
        mat_w = w * np.array(list_neighbor01)
        i = 0
        fusion_map = np.zeros([25,25])
        for one in mat_w:
            fusion_map += one * map_array[i]
            i += 1
        self.agent_map = fusion_map


    def Voronoi(self):
        self.agent_local
        voronoi_cell = [[], [], []]
        return voronoi_cell


    def Move(self, voronoi_cell,area):  # 移动
        mass = 0
        centroid_mole = 0
        for i in voronoi_cell:
            point = self.agent_map[i[0]][i[1]]
            density = np.exp(-2 * np.linalg.norm(point))
            mass += density * area
            centroid_mole += np.array(i) * density *area
        centroid = float(centroid_mole) /float(mass)
        speed = centroid - self.agent_local

        if np.linalg.norm(speed) <= 3:
            new_speed = speed
        else:
            new_speed = (3 /float(np.linalg.norm(speed))) * speed  # 先不考虑除法数值的类型等细节
        self.agent_local += new_speed



