import numpy as np
import networkx as nx
import math
import numpy as np
from utils import *
from Match import *


def calculate_grad(pts1,pts2):
    return (pts1[1]-pts2[1])/(pts1[0]-pts2[0])

def calculate_dist(pts1,pts2):
    return np.linalg.norm(pts1-pts2)

def calculate_edge_feat(pts1,pts2):
    return (calculate_dist(pts1,pts2),calculate_grad(pts1,pts2))

def calculate_adjacency_entropy(graph):
    num = graph.number_of_nodes()
    adj_matrix = nx.adjacency_matrix(graph).todense()
    # sum of degree
    k = np.zeros(num)
    for i in range(num):
        for j in range(num):
            k[i] += adj_matrix[i,j]
    # adjacency_degree
    A = np.zeros_like(k)
    for i in range(num):
        for j in range(num):
            if adj_matrix[i,j] != 0:
                A[i] += k[j]
    # selection_probability
    P = np.zeros_like(adj_matrix).astype(np.float32)
    for i in range(num):
        for j in range(num):
            if adj_matrix[i,j] != 0:
                P[i,j] = float(k[i] / A[j])
    # adjacency_information_entropy
    E = np.zeros_like(k)
    for i in range(num):
        for j in range(num):
            if adj_matrix[i,j] != 0:
                E[i] += -P[i,j] * math.log2(P[i,j])
    return E

def calculate_edge_value(img,p1,p2,num = 32):

    x1,y1 = p1
    x2,y2 = p2
    item = []
    for j in range(3):
        line_value = []
        for lambda_value in np.linspace(0, 1, num):  
            x = x1 + lambda_value * (x2 - x1)  
            y = y1 + lambda_value * (y2 - y1)  
            line_value.append(img[int(x),int(y),j])
        item.append(line_value)
    return np.array(item)

def calculate_cosine(a, b, c):  
    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)  
    C = math.acos(cos_C)  
    return math.degrees(C)

def calculate_angles(A, B, C):
    c = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)  
    a = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)      
    b = math.sqrt((C[0]-A[0])**2 + (C[1]-A[1])**2)  
    angle_A = calculate_cosine(b, c, a)  
    angle_B = calculate_cosine(a, c, b)  
    angle_C = calculate_cosine(a, b, c)  
  
    angles = np.array([angle_A, angle_B, angle_C])   
    return angles

def calculate_sequence(graph):
    sequence = dict()
    for i in range(graph['graph'].number_of_nodes()):
        edge_sets_tmp = get_adjacent_tri(graph['tri'],i)
        edge_sets = order_tri(graph['tri'],edge_sets_tmp,i)
        sequence[i] = edge_sets
    return sequence

def calculate_d(left_points,right_points,H,dic):
    nums = len(dic)
    d = 0
    for item in dic:
        trans_right_pts = H @ [left_points[item['left_index']][1], left_points[item['left_index']][2], 1]
        trans_right_pts = (trans_right_pts / trans_right_pts[2])[:2]
        trans_left_pts = np.linalg.inv(H) @ [right_points[item['right_index']][1], right_points[item['right_index']][2], 1]
        trans_left_pts = (trans_left_pts / trans_left_pts[2])[:2]
        d += np.linalg.norm(trans_right_pts - right_points[item['right_index']][1:])
    d /= nums
    return d

    
def calculate_diff_angle(source_tris, source_pts, target_tris, target_pts):
    iter = max(len(source_tris), len(target_tris))
    turn = min(len(source_tris), len(target_tris))
    left_top_angles = []
    right_top_angles = []
    # 计算顶角度数
    for left_tri in source_tris:
        x, y, z = left_tri
        left_angle = calculate_angles(source_pts[x], source_pts[y], source_pts[z])
        left_top_angles.append(left_angle[0])
    for right_tri in target_tris:
        x, y, z = right_tri
        right_angle = calculate_angles(target_pts[x], target_pts[y], target_pts[z])
        right_top_angles.append(right_angle[0])

    left_top_angles = np.array(left_top_angles)
    right_top_angles = np.array(right_top_angles)
    sum_top_angel = np.sum(left_top_angles) + np.sum(right_top_angles)

    lam = 2.1
    W_s = 0

    for t in range(iter):
        w_x = 0
        tri_simi = 0
        for i in range(turn):
            left_tri = source_tris[(i + t) % turn]
            right_tri = target_tris[i]
            x, y, z = left_tri
            left_angle = calculate_angles(source_pts[x], source_pts[y], source_pts[z])
            x, y, z = right_tri
            right_angle = calculate_angles(target_pts[x], target_pts[y], target_pts[z])
            ij_diff_angle = np.sum(np.abs(left_angle - right_angle))
            w_x = 1 - math.log(1 + lam * ij_diff_angle / 180)
            tri_simi += (left_angle[0] + right_angle[0]) * w_x / sum_top_angel
        if W_s < tri_simi:
            W_s = tri_simi

    return W_s

def calculate_tri_simi(source_graph, target_graph):
    source_sequence = calculate_sequence(source_graph)
    target_sequence = calculate_sequence(target_graph)
    tri_topo_simi = np.zeros((len(source_sequence),len(target_sequence)))
    for i in range(len(source_sequence)):
        for j in range(len(target_sequence)):
            source_tris = source_sequence[i]
            target_tris = target_sequence[j]
            wdiff = calculate_diff_angle(source_tris, source_graph['pos'], target_tris, target_graph['pos'])
            tri_topo_simi[i,j] = wdiff
    return tri_topo_simi

def calculate_support_simi(left_friends,right_friends):
    simi_matrix = np.zeros((len(left_friends),len(right_friends)))
    for i in range(len(left_friends)):
        for j in range(len(right_friends)):
            simi_matrix[i,j] = calculate_IoU(left_friends[i], right_friends[j])
    return simi_matrix

def calculate_IoU(left_list, right_list):
    n1 = len(set(left_list) & set(right_list))
    n2 = len(set(left_list) | set(right_list))
    return n1 / n2
    
def calculate_second_IoU(l_index, left_friends, left_graph, r_index, right_friends, right_graph, matches_set, alpha):
    f_order_iou = calculate_IoU(left_friends[l_index], right_friends[r_index])
    s_oder_iou = 0
    matched_nodes_left = []
    matched_nodes_right = []
    for node_pair in matches_set:
        if node_pair[0] in left_friends[l_index] and node_pair[1] in right_friends[r_index]:
            matched_nodes_left.append(node_pair[0])
            matched_nodes_right.append(node_pair[1])
    iou = 0
    n = 0
    for i,j in zip(matched_nodes_left,matched_nodes_right):
        l_tmp = left_graph['index'][i]
        r_tmp = right_graph['index'][j]
        iou += calculate_IoU(left_friends[l_tmp],right_friends[r_tmp])
        n += 1
    if n == 0:
        s_oder_iou = 0
    else:
        s_oder_iou = iou / n
    return alpha * f_order_iou + (1 - alpha) * s_oder_iou

def calculate_fundamental_matrix(left_pts,right_pts,matches):
    src_pts = []
    dst_pts = []
    for item in matches:
        src_pts.append(left_pts[item[0]][1:])
        dst_pts.append(right_pts[item[1]][1:])
    src_pts = np.array(src_pts).reshape(-1,2)
    dst_pts = np.array(dst_pts).reshape(-1,2)
    F, _ = cv2.findFundamentalMat(src_pts, dst_pts, method = cv2.FM_RANSAC,ransacReprojThreshold=0.9, confidence=0.99)
    return F


    
                            
