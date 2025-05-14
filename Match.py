import numpy as np
from utils import *
from Estimate import *
from Calculate import *
import pygmtools as pygm
from PIL import Image
import os
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import pulp


def dual_threshing(source_graph, left_adj_M, left_weight, target_graph, right_adj_M, right_weight, index, low_threash, high_thresh):
    
    left_points = source_graph['pos']
    right_points = target_graph['pos']
    H = calculate_homography(left_points, right_points, index, cv2.RHO)

    if H is None:
        print("H is None")
        exit()
    error = calculate_error(H, left_points, right_points, index)
    error = iter(error)
    good_match = index
    good_match = []
    select_match = []
    for i, (d1, d2) in enumerate(zip(error, error)):
        if d1 < low_threash and d2 < low_threash:
            good_match.append((index[i][0], index[i][1]))
        elif d1 + d2 < high_thresh:
            select_match.append((index[i][0], index[i][1]))
            
    for i in range(len(select_match)):
        for j in range(len(good_match)):
            if left_adj_M[select_match[i][0]][good_match[j][0]] != 0 and right_adj_M[select_match[i][1]][good_match[j][1]] != 0:
                good_match.append(select_match[i])
                break      

    ## 特判处理没有匹配到的点：
    source_good_index = [x[0] for x in good_match]
    target_good_index = [x[1] for x in good_match]

    source_error_index = []
    target_error_index = []
    for index in source_graph['pos']:
        if index not in source_good_index:
            source_error_index.append(index)
    for index in target_graph['pos']:
        if index not in target_good_index:
            target_error_index.append(index)
    
    return H, good_match, source_error_index, target_error_index


def find_max_similarity(matrix):
    max_similarity = -1
    max_row = -1
    max_col = -1
    row = matrix.shape[0]
    col = matrix.shape[1]

    for i in range(row):
        for j in range(col):
            if matrix[i][j] > max_similarity:
                max_similarity = matrix[i][j]
                max_row = i
                max_col = j

    return max_row, max_col, max_similarity

def find_best_matches(simi_matrix,hypothesis):
    matches = []
    matrix = simi_matrix.copy()
    row = matrix.shape[0]
    col = matrix.shape[1]
    if hypothesis is not None:
        for item in hypothesis:
            matches.append(item)
            for i in range(row):
                matrix[i][item[1]] = -1
            for i in range(col):
                matrix[item[0]][i] = -1
    while len(matches) < min(row,col):
        max_row, max_col, max_similarity = find_max_similarity(matrix)
        matches.append((max_row, max_col))

        for i in range(row):
            matrix[i][max_col] = -1
        for i in range(col):
            matrix[max_row][i] = -1

    return matches

def find_best_matches_mwis(simi_matrix,hypothesis, threshold):
    matches = []
    matrix = simi_matrix.copy()
    row = matrix.shape[0]
    col = matrix.shape[1]

    if hypothesis is not None:
        for item in hypothesis:
            matches.append(item)
            for i in range(row):
                matrix[i][item[1]] = -1
            for i in range(col):
                matrix[item[0]][i] = -1
    
    matrix[matrix < threshold] = -1
    edge_set = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(matrix[i][j] != -1):
                edge_set.append((i,matrix.shape[0] + j))

    collision_set=[]
    for indexi, dege_seti in enumerate(edge_set): 
        for indexj, dege_setj in enumerate(edge_set): 
            if indexi != indexj:
                if dege_seti[0] == dege_setj[0]  or dege_seti[0] == dege_setj[1] or dege_seti[1] == dege_setj[0] or dege_seti[1] == dege_setj[1]:
                    if((indexi,indexj) not in collision_set):
                        if((indexj,indexi) not in collision_set):
                                collision_set.append((indexi,indexj))

    if collision_set == []:
        while len(matches) < min(row,col):
            max_row, max_col, max_similarity = find_max_similarity(matrix)
            matches.append((max_row, max_col))

            for i in range(row):
                matrix[i][max_col] = -1
            for i in range(col):
                matrix[max_row][i] = -1
        return matches

    prob = pulp.LpProblem("Maximum_Weighted_Independent_Set", pulp.LpMaximize)
    solver = pulp.CPLEX(
        msg = False, 
        options=[
            'writemps=0',     
            'writeprob=0',    
            'solnfile=',      
            'log=0'           
        ]
    )
    prob.setSolver(solver)
    pulp.LpSolverDefault.msg = 0
    
    x = pulp.LpVariable.dicts("x", [i for i in range(len(edge_set))], cat=pulp.LpBinary)

    prob += pulp.lpSum([matrix[edge_set[i][0]][edge_set[i][1] - matrix.shape[0]]* x[i] for i in range(len(edge_set))])

    for (u, v) in collision_set:
        prob += x[u] + x[v] <= 1

    solution_status = prob.solve()
    max_weight = pulp.value(prob.objective)

    filtered_match = [i for i in range(len(edge_set)) if x[i].varValue is not None and x[i].varValue > 0]
    SelectedEdge =[]
    for index in range(len(filtered_match)):
        SelectedEdge.append((edge_set[filtered_match[index]][0], edge_set[filtered_match[index]][1] - matrix.shape[0]))

    merged_list = [item for item in SelectedEdge] + [item for item in hypothesis]
    return merged_list


def find_friends(left_friends, left_adj_matrix, left_graph, right_friends, right_adj_matrix, right_graph):
    for i in range(len(left_adj_matrix)):
        for j in range(len(left_adj_matrix)):
            if left_adj_matrix[i][j] != 0:
                left_friends[i].append(left_graph['id'][j])
    for i in range(len(right_adj_matrix)):
        for j in range(len(right_adj_matrix)):
            if right_adj_matrix[i][j] != 0:
                right_friends[i].append(right_graph['id'][j])
    
def get_adjacent_tri(tri,index):
    adj_list = []
    for item in tri.simplices:
        if index in item:
            adj_list.append(list(item))
    return adj_list

def order_tri(tri,adj,index):
    item_edge = []
    ordered_edge = []
    for edge in adj:
        while edge[0] != index:
            item = edge.pop(0)
            edge.append(item)
        item_edge.append(edge)
    first = item_edge.pop(0)
    ordered_edge.append(first)
    while len(item_edge) != 0:
        item = item_edge.pop(0)
        flag = False
        for i in range(len(ordered_edge)):
            first = ordered_edge[i]
            if item[1] == first[2]:
                ordered_edge.insert(i+1,item)
                flag = True
                break
            elif item[2] == first[1]:
                ordered_edge.insert(i,item)
                flag = True
                break
        if flag == False:
            item_edge.append(item)
    return ordered_edge

def pygraph_match(left_origin_points,left_vanish_label,left_entropy,right_points,right_vanish_label,right_entropy):
    left_points = []
    for item in left_origin_points:
        if item[0] not in left_vanish_label:
            left_points.append(item)
    n1 = len(left_points)
    n2 = len(right_points)
    A1 = np.zeros((n1,n1))
    A2 = np.zeros((n2,n2))
    for i in range(n1):
        for j in range(n1):
            A1[i,j] = np.linalg.norm(left_points[i]-left_points[j])
    for i in range(n2):
        for j in range(n2):
            A2[i,j] = np.linalg.norm(right_points[i]-right_points[j])
    conn1, edge1 = pygm.utils.dense_to_sparse(A1)
    conn2, edge2 = pygm.utils.dense_to_sparse(A2)
    node1_feat = np.array([left_entropy[i] for i in range(n1)]).reshape(1,1,-1)
    node2_feat = np.array([right_entropy[i] for i in range(n2)]).reshape(1,1,-1)
    an1 = np.array([n1])
    an2 = np.array([n2])
    import functools
    gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1)
    K = pygm.utils.build_aff_mat(node1_feat, edge1, conn1, node2_feat, edge2, conn2, an1, None, an2, None, edge_aff_fn=gaussian_aff)
    # K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, an1, None, an2, None, edge_aff_fn=gaussian_aff)
    X = pygm.rrwm(K, n1, n2)
    X = pygm.hungarian(X)

    
    matches_index = []
    for i in range(n1):
        for j in range(n2):
            if X[i,j] != 0:
                matches_index.append((i,j))
    return matches_index

        


        
