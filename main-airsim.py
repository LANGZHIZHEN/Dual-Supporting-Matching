import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt 
from utils import *
from Estimate import *
from Calculate import *
from scipy.optimize import linear_sum_assignment
from Match import *
from read_airsim import *
from PYGM import *
import os
import time

save_output = False
show_output = True
ENABLE_PCA_GM = True
ENABLE_IPCA_GM = True
ENABLE_NGM = True
ENABLE_RRWM = True
ENABLE_TTS = True
ENABLE_OURS = True
dir = '8'

xml_path = "./datasets/Airsim/"+ dir +"/"+ dir +".xml"
image_path = "./datasets/Airsim/" + dir + "/" + "output/"
source_images_path = read_images_path("./datasets/Airsim/"+ dir +"/1")
target_images_path = read_images_path("./datasets/Airsim/"+ dir +"/2")
os.makedirs(image_path, exist_ok=True)
data_parser = DualViewParser(xml_path)
image_parser = FrameTimestampParser(xml_path)
visualizer = DualViewVisualizer(
        parser=image_parser,
        left_img_dir="./datasets/Airsim/"+ dir +"/1",
        right_img_dir="./datasets/Airsim/"+ dir +"/2",
        img_extension=".png"
    )
output_ACC = []
qualified = data_parser.get_frames_with_min_objects(20)[:20]
for frame in qualified:
    print(frame)
    source_object, target_object = data_parser.get_views_by_frame(frame)
    prior_hypothesis = []
    source_graph = init_graph(source_object)
    target_graph = init_graph(target_object)
    num_nodes = min(source_graph['graph'].number_of_nodes(), target_graph['graph'].number_of_nodes())
    t1 = time.time()
    triPolo_simi_matrix = calculate_tri_simi(source_graph, target_graph)
    t2 = time.time()
    tts_fps = 1/(t2-t1)

    source_adj_matrix = normalize_pixels(nx.adjacency_matrix(source_graph['graph']).toarray())
    target_adj_matrix = normalize_pixels(nx.adjacency_matrix(target_graph['graph']).toarray())
    source_entropy = calculate_adjacency_entropy(source_graph['graph'])
    target_entropy = calculate_adjacency_entropy(target_graph['graph'])

    source_entropy = (len(source_entropy) * source_entropy) / np.sum(source_entropy)
    target_entropy = (len(target_entropy) * target_entropy) / np.sum(target_entropy)
    source_betw = nx.closeness_centrality(source_graph['graph'])
    target_betw = nx.closeness_centrality(target_graph['graph'])

    alpha = 0.5
    left_key = [alpha * source_entropy[i] + (1 - alpha) * source_betw[i] for i in range(len(source_entropy))]
    right_key = [alpha * target_entropy[i] + (1 - alpha) * target_betw[i] for i in range(len(target_entropy))]
    if ENABLE_PCA_GM or ENABLE_IPCA_GM:
        t1 = time.time()
        source_node_feat = get_feature_vector(source_images_path[frame], source_graph['pos'])
        target_node_feat = get_feature_vector(target_images_path[frame], target_graph['pos'])
        t2 = time.time()
        vgg_time = t2 - t1
    if ENABLE_PCA_GM:
        t1 = time.time()
        matches_pca_gm_matrix = PCA_GM(source_adj_matrix, source_node_feat, target_adj_matrix, target_node_feat)
        t2 = time.time()
        pca_fps = 1/(t2-t1)
        matches_pca_gm, matches_pca_gm_index = index2ids(matches_pca_gm_matrix, source_graph, target_graph)
        ACC_pca = count_matching_pairs(matches_pca_gm_matrix, source_graph, target_graph) / len(matches_pca_gm_matrix[0])

    if ENABLE_IPCA_GM:
        t1 = time.time()
        matches_ipca_gm_matrix = IPCA_GM(source_adj_matrix, source_node_feat, target_adj_matrix, target_node_feat)
        t2 = time.time()
        ipca_fps = 1/(t2-t1)
        matches_ipca_gm, matches_ipca_gm_index = index2ids(matches_ipca_gm_matrix, source_graph, target_graph)
        ACC_ipca = count_matching_pairs(matches_ipca_gm_matrix, source_graph, target_graph) / len(matches_ipca_gm_matrix[0])
    
    if ENABLE_NGM:
        t1 = time.time()
        matches_ngm_matrix = NGM(source_adj_matrix, target_adj_matrix)
        t2 = time.time()
        ngm_fps = 1/(vgg_time + t2-t1)
        matches_ngm, matches_ngm_index = index2ids(matches_ngm_matrix, source_graph, target_graph)
        ACC_ngm = count_matching_pairs(matches_ngm_matrix, source_graph, target_graph) / len(matches_ngm_matrix[0])

    if ENABLE_RRWM:
        t1 = time.time()
        matches_rrwm_matrix = RRWM(source_adj_matrix, target_adj_matrix)
        t2 = time.time()
        rrwm_fps = 1/(t2-t1)
        matches_rrwm, matches_rrwm_index = index2ids(matches_rrwm_matrix, source_graph, target_graph)
        ACC_rrwm = count_matching_pairs(matches_rrwm_matrix, source_graph, target_graph) / len(matches_rrwm_matrix[0])
    
    source_friends = {i: [] for i in range(len(source_graph['pos']))}
    target_friends = {i: [] for i in range(len(target_graph['pos']))}
    find_friends(source_friends, source_adj_matrix, source_graph, target_friends, target_adj_matrix, target_graph)

    support_simi_matrix = calculate_support_simi(source_friends, target_friends)

    n1, n2 = len(source_graph['pos']), len(target_graph['pos'])
    adjEntro_simi_matrix = np.zeros((n1,n2)).astype(float)

    simi_matrix = triPolo_simi_matrix
    
    tts_matches_index = find_best_matches(simi_matrix, prior_hypothesis)
    tts_matches = [(source_graph['id'][item[0]], target_graph['id'][item[1]]) for item in tts_matches_index]
    ACC_tts = sum(1 for a, b in tts_matches if a == b) / num_nodes
    # matches = find_best_matches_mwis(simi_matrix, prior_hypothesis, 4)
    one_stage_matches = None
    two_stage_matches = None
    t1 = time.time()
    H = None
    if len(tts_matches_index) >= 4:
        train_pts = [source_graph['pos'][int(tts_matches_index[i][0])] for i in range(len(tts_matches_index))]
        query_pts = [target_graph['pos'][int(tts_matches_index[i][1])] for i in range(len(tts_matches_index))]
        index_pts = [tts_matches_index[i] for i in range(len(tts_matches_index))]

        source_coords = np.array(list(source_graph['pos'].values())) 
        target_coords = np.array(list(target_graph['pos'].values()))  

        source_ax, source_ay = np.mean(source_coords, axis=0).astype(int)
        target_ax, target_ay = np.mean(target_coords, axis=0).astype(int)
        source_coords -= (source_ax, source_ay)
        target_coords -= (target_ax, target_ay)

        tri_matches_index = []
        tri_matches_labels = []
        for pair in tts_matches_index:
            i, j = int(pair[0]), int(pair[1])
            tri_matches_labels.append([
                source_graph['id'][i], 
                target_graph['id'][j]  
            ])
            tri_matches_index.append([i, j])
        H, one_stage_matches_index, source_bad_index, target_bad_index = dual_threshing(source_graph, source_adj_matrix, source_entropy,target_graph, target_adj_matrix, target_entropy, tri_matches_index, 5, 60)

    one_stage_matches = []
    one_stage_matches = [(source_graph['id'][item[0]], target_graph['id'][item[1]]) for item in one_stage_matches_index]
    one_stage_H = H

    two_stage_matches = []
    dist_error_matrix = np.zeros((len(source_bad_index), len(target_bad_index)),dtype=int)

    for i, source_index in enumerate(source_bad_index):
        for j, target_index in enumerate(target_bad_index):
            src_x, src_y = source_graph['pos'][source_index]
            dst_x, dst_y = target_graph['pos'][target_index]
            dist_error_matrix[i, j] = calculate_reprojection_error(H, (src_x, src_y), (dst_x, dst_y))

    row_ind, col_ind = linear_sum_assignment(dist_error_matrix, False)
    left_bad_support_index = []
    right_bad_support_index = []

    LOW_CONF = 0.4
    HIGH_CONF = 0.6
    LOW_DIST = 10
    HIGH_DIST = 60
    matches_set = set(one_stage_matches)
    for i, j in zip(row_ind, col_ind):
        l_index = source_bad_index[i]
        r_index = target_bad_index[j]
        iou = calculate_second_IoU(l_index, source_friends, source_graph, r_index, target_friends, target_graph, matches_set, LOW_CONF)
        if dist_error_matrix[i,j] < LOW_DIST or (dist_error_matrix[i,j] < HIGH_DIST and iou > HIGH_CONF): 
            two_stage_matches.append((source_graph['id'][l_index], target_graph['id'][r_index]))
        else:
            left_bad_support_index.append(l_index)
            right_bad_support_index.append(r_index)

    two_stage_matches += one_stage_matches

    support_matrix = np.zeros((len(left_bad_support_index), len(right_bad_support_index)), dtype=float)
    matches_set = set(two_stage_matches)
    for i, l_index in enumerate(left_bad_support_index):
        for j, r_index in enumerate(right_bad_support_index):
            iou = calculate_second_IoU(l_index, source_friends, source_graph, r_index, target_friends, target_graph, matches_set, LOW_CONF)
            support_matrix[i,j] = iou

    matches = find_best_matches(support_matrix, None)
    filtered_matches = []
    low_confidence_matches = []
    for item in matches:
        i,j = item
        if support_matrix[i, j] > HIGH_CONF:
            two_stage_matches.append((source_graph['id'][left_bad_support_index[i]], target_graph['id'][right_bad_support_index[j]]))
        else:
            low_confidence_matches.append((source_graph['id'][left_bad_support_index[i]], target_graph['id'][right_bad_support_index[j]]))

    two_stage_matches_index = [(source_graph['index'][item[0]], target_graph['index'][item[1]]) for item in two_stage_matches]
    t2 = time.time()
    if low_confidence_matches:
        low_confidence_matches_index = [(source_graph['index'][item[0]], target_graph['index'][item[1]]) for item in low_confidence_matches]
        H = calculate_homography(source_graph['pos'], target_graph['pos'], two_stage_matches_index, cv2.RHO)
        two_bad_pairs = []
        two_left_bad_index = []
        two_right_bad_index = []
        for i, item in enumerate(low_confidence_matches_index):
            src_x, src_y = source_graph['pos'][item[0]]
            dst_x, dst_y = target_graph['pos'][item[1]]
        
            gd = calculate_reprojection_error(H, (src_x, src_y), (dst_x, dst_y))
            if gd > HIGH_DIST:
                two_bad_pairs.append(low_confidence_matches_index[i])
                two_left_bad_index.append(low_confidence_matches_index[i][0])
                two_right_bad_index.append(low_confidence_matches_index[i][1])
            else:
                two_stage_matches_index.append(low_confidence_matches_index[i])
                
        re_dist = np.zeros((len(two_left_bad_index),len(two_right_bad_index)))
        for i, l_index in enumerate(two_left_bad_index):
            for j, r_index in enumerate(two_right_bad_index):
                src_x, src_y = source_graph['pos'][l_index] 
                dst_x, dst_y = target_graph['pos'][r_index] 

                re_dist[i, j] = calculate_reprojection_error(H, (src_x, src_y), (dst_x, dst_y))

        row_ind,col_ind = linear_sum_assignment(re_dist, False)
        for i, j in zip(row_ind, col_ind):
            l_index = two_left_bad_index[i]
            r_index = two_right_bad_index[j]
            two_stage_matches_index.append((l_index, r_index))
    
        two_stage_matches.clear()
        for item in two_stage_matches_index:
            two_stage_matches.append((source_graph['id'][item[0]], target_graph['id'][item[1]]))
    
    dsm_fps = 1/(t2-t1)
    ACC_Ours = 0
    for item in two_stage_matches:
        if item[0] == item[1]:
            ACC_Ours += 1
    ACC_Ours /= num_nodes

    if show_output:
        result = visualizer.visualize_frame(
            frame_number=frame,
            matches_index=two_stage_matches_index,
            matches_id=two_stage_matches,
            left_positions=source_graph['pos'],
            right_positions=target_graph['pos'],
            output_size=(1920, 1080)
        )
        cv2.namedWindow("Dual View Association", cv2.WINDOW_NORMAL)
        cv2.imshow("Dual View Association", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # print(f"RRWM_ACC:{ACC_rrwm}")
    # print(f"NGM_ACC:{ACC_ngm}")
    # print(f"PCA_ACC:{ACC_pca}")
    # print(f"IPCA_ACC:{ACC_ipca}")
    # print(f"TTS_ACC:{ACC_tts}")
    # print(f"Ours_ACC:{ACC_Ours}")

    


                    


                                
                

            




    




