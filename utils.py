import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay
from PIL import Image, ImageDraw
import random
from matplotlib.patches import ConnectionPatch, Rectangle


def init_graph(points):
    coords = np.array([[point['x'], point['y']] for point in points])
    tri = Delaunay(coords)

    pos = dict()
    labels = dict()
    index = dict()

    for i, point in enumerate(points):
        pos[i] = (point['x'], point['y']) 
        labels[i] = point['id']           
        index[point['id']] = i            

    graph = nx.Graph()
    for i in range(len(points)):
        graph.add_node(i, label=labels[i])

    for edge in tri.simplices:
        x, y, z = edge

        def _distance(u, v):
            dx = pos[u][0] - pos[v][0]
            dy = pos[u][1] - pos[v][1]
            return math.sqrt(dx**2 + dy**2)
        graph.add_edge(x, y, weight=_distance(x, y))
        graph.add_edge(y, z, weight=_distance(y, z))
        graph.add_edge(x, z, weight=_distance(x, z))

    return {
        'graph': graph,
        'pos': pos,
        'id': labels,
        'index': index,
        'tri': tri
    }

def visualize_graph(G):
    graph = G['graph']
    pos = G['pos']
    labels = G['labels']

    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_weight='bold')

    node_labels = {i: labels[i] for i in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels=node_labels)

    plt.title("Graph Visualization")
    plt.show()
    
def read_txt(url,vanish_labels,var):
    pts = []
    vanish_pts = []
    noise = np.random.normal(0, var, size=(100, 2))
    with open(url, 'r') as file:
        for i, line in enumerate(file, start=1):
            line = line.strip()
            parts = line.split(':')
            i_value = int(parts[0])
            bracket_content = parts[1][1:-1]
            nums = bracket_content.split(',')
            x = int(nums[0])
            y = int(nums[1])
            if i_value not in vanish_labels:
                pts.append((i_value,x,y))
            else:
                vanish_pts.append((i_value,x,y))
    return np.array(pts)


def normalize_2d(src_points):
    src_ax,src_ay = np.mean(src_points,axis=0)
    n = len(src_points)
    T1 = np.array([[1,0,-src_ax],
                [0,1,-src_ay],
                [0,0,1]])
    p1 = (T1 @ np.array([[x,y,1] for x,y in src_points]).T).T
    src_sum = 0
    for i in range(n):
        src_sum += math.sqrt(p1[i][0]**2+p1[i][1]**2)
    src_sum /= n
    alpha = math.sqrt(2) / src_sum
    S1 = np.array([[alpha,0,0],
                [0,alpha,0],
                [0,0,1]])
    u = (S1 @ p1.T).T
    return u[:,:2], S1 @ T1

def add_gaussian_noise(points, mean, std_dev):
    noise = np.random.normal(mean, std_dev, size=(len(points), 2))
    noisy_points = np.array(points) + noise
    return noisy_points

def generate_points(num_points, min_distance, max_coordinate, vanish_labels,angle,seed = None,var=1):
    if seed is not None:
        random.seed(seed)
    np.random.seed(seed)
    points = []
    vanish_points = []
    origin_points = []
    idx = 1
    noise = np.random.normal(0, var, size=(num_points, 2))
    while len(points) < num_points:
        x = random.randint(0, max_coordinate)
        y = random.randint(0, max_coordinate)
        new_point = (x, y)
        if check_distance(points, new_point, min_distance):
            x,y = (x,y) + noise[idx-1]
            if idx in vanish_labels:
                vanish_points.append([idx,x,y])
            points.append([idx,x,y])
            origin_points.append([idx,x,y])
            idx += 1
    for item in vanish_points:
        points.remove(item)
    

    rotation_matrix = np.array([[math.cos(math.radians(angle)), math.sin(math.radians(angle))],
                                [-math.sin(math.radians(angle)), math.cos(math.radians(angle))]])
    points = np.array(points)
    pts_tmp = points[:,1:].copy()
    rotated_points = np.dot(pts_tmp, rotation_matrix).astype(np.int32)
    points[:,1:] = rotated_points

    if len(vanish_points) != 0:
        vanish_points = np.array(vanish_points)
        van_pts_tmp = vanish_points[:,1:].copy()
        rotated_vanish_points = np.dot(van_pts_tmp, rotation_matrix).astype(np.int32)
        vanish_points[:,1:] = van_pts_tmp
    
    origin_points = np.array(origin_points)
    ori_pts_tmp = origin_points[:,1:].copy()
    rotated_ori_points = np.dot(ori_pts_tmp, rotation_matrix).astype(np.int32)
    origin_points[:,1:] = rotated_ori_points
    return points, vanish_points,origin_points

def check_distance(points, new_point, min_distance):
    for point in points:
        if calculate_distance(point, new_point) < min_distance:
            return False
    return True

def calculate_distance(point1, point2):
    x1, y1 = point1[1:]
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def plot_points(points, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for i, point in enumerate(points):
        plt.scatter(point[0], point[1], color='blue')
        plt.text(point[0], point[1], str(i+1), fontsize=10, ha='left', va='bottom')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Original Distribution')
    plt.grid(True)

    rotation_matrix = np.array([[math.cos(math.radians(45)), math.sin(math.radians(45))],
                                [-math.sin(math.radians(45)), math.cos(math.radians(45))]])

    points_matrix = np.array(points)
    rotated_points = np.dot(points_matrix, rotation_matrix)

    plt.subplot(1, 2, 2)
    for i, point in enumerate(rotated_points):
        plt.scatter(point[0], point[1], color='red')
        plt.text(point[0], point[1], str(i+1), fontsize=10, ha='left', va='bottom')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rotated Distribution')
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def draw_match_info(algo, rotate,rate, corr_num, total_num, err, matches_index, left_points, left_graph, right_points, right_graph):
    info = {  
        'rotate': rotate, 
        'rate': rate * 100,  
        'corr': corr_num * 100 / total_num,  
        'error': err,  
        'alg': algo  
    }  
    draw_match_graph(left_points, left_graph, right_points, right_graph, matches_index, info)  

def color_distance(color1, color2):  
    return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2))) 
def generate_unique_colors(num_colors, threshold=0.25):  
    unique_colors = []  
    while len(unique_colors) < num_colors:  
        new_color = (random.random(), random.random(), random.random())  
        if all(color_distance(new_color, c) > threshold for c in unique_colors):  
            unique_colors.append(new_color)  
    return unique_colors

def draw_match_graph(left_points,left_graph,right_points,right_graph,matched_index,info):

    rotation_matrix = np.array([[math.cos(math.radians(-info['rotate'])), math.sin(math.radians(-info['rotate']))],
                                [-math.sin(math.radians(-info['rotate'])), math.cos(math.radians(-info['rotate']))]])
    left_graph_pos_copy = left_graph['pos'].copy()
    right_graph_pos_copy = right_graph['pos'].copy()
    left_points_copy = left_points.copy()
    right_points_copy = right_points.copy()
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    left_color_list = generate_unique_colors(20)
    left_edges = left_graph['graph'].edges()
    left_matched_index = []
    right_matched_index = []
    left_unmatched_index = []
    right_unmatched_index = []
    for item in matched_index:
        left_matched_index.append(item[0])
        right_matched_index.append(item[1])
    for i in range(len(left_points)):
        if i not in left_matched_index:
            left_unmatched_index.append(i)
    for i in range(len(right_points)):
        if i not in right_matched_index:
            right_unmatched_index.append(i)
    
    nx.draw_networkx_nodes(left_graph['graph'], pos=left_graph_pos_copy, node_color=left_color_list, node_size=80, edgecolors='black', linewidths=2, ax=ax)
    nx.draw_networkx_edges(left_graph['graph'], pos=left_graph_pos_copy, edgelist=left_edges, width=1.5, style='dashed', ax=ax)
    for i in left_unmatched_index:
        x, y = left_graph_pos_copy[i]
        circle = plt.Circle((x, y), 60, fill=False, edgecolor='r', linestyle='solid')
        ax.add_patch(circle)

    right_color_list = []
    for i in range(len(right_points_copy)):
        r_label = right_graph['labels'][i]
        l_index = left_graph['index'][r_label]
        right_color_list.append(left_color_list[l_index])
    ax.set_aspect('equal', 'box') 
    ax2 = fig.add_subplot(1,2,2)
    for i in range(len(right_graph_pos_copy)):
        right_graph_pos_copy[i] = np.dot(right_graph_pos_copy[i], rotation_matrix).astype(np.int)
    right_edges = right_graph['graph'].edges()
    nx.draw_networkx_nodes(right_graph['graph'], pos=right_graph_pos_copy, node_color=right_color_list, node_size=80, edgecolors='black', linewidths=2, ax=ax2)
    nx.draw_networkx_edges(right_graph['graph'], pos=right_graph_pos_copy, edgelist=right_edges, width=1.5, style='dashed', ax=ax2)
    
    if info['alg'] == 'pygm':
        tmp_points = []
        right_labels = [item[0] for item in left_points_copy]
        for item in left_points_copy:
            if item[0] in right_labels:
                tmp_points.append(item)
        left_points_copy = tmp_points
    
    for i,pt in enumerate(left_points_copy):
        label = pt[0]
        left_points_copy[i][1:] = left_graph_pos_copy[left_graph['index'][label]]
    for i,pt in enumerate(right_points_copy):
        label = pt[0]
        right_points_copy[i][1:] = right_graph_pos_copy[right_graph['index'][label]]
    for item in matched_index:  
        left_index = item[0]
        right_index = item[1]
        left_pos = left_points_copy[left_index][1:]
        right_pos = right_points_copy[right_index][1:]   
        color = 'g' if left_points_copy[left_index][0] == right_points_copy[right_index][0] else 'r'  
        
        con = ConnectionPatch(left_pos,right_pos,coordsA="data", coordsB="data",axesA=ax, axesB=ax2,color=color)
        con.set_linewidth(0.7)
        ax.add_artist(con)
    ax2.set_aspect('equal', 'box') 
    
    ax.axis('off')
    ax2.axis('off')
    rect1 = Rectangle((0, 0), 1, 1, transform=ax.transAxes, edgecolor='black', linestyle='dashed', linewidth=4, fill=False)
    ax.add_patch(rect1)
    rect2 = Rectangle((0, 0), 1, 1, transform=ax2.transAxes, edgecolor='black', linestyle='dashed', linewidth=4, fill=False)
    ax2.add_patch(rect2)

    rotate = info['rotate']
    rate = info['rate']
    corr = info['corr']
    error = info['error']
    alg = info['alg']
    plt.suptitle("corr:{:.2f}%_err:{:.1f}_alg:{}".format(corr,error,alg))
    plt.savefig("./imgs/rot:{}alg:{}.png".format(rotate,alg))


def draw_match_physical(left_image_path, right_image_path, left_points, right_points, matches):
    left_image = Image.open(left_image_path)
    right_image = Image.open(right_image_path)

    left_width, left_height = left_image.size
    right_width, right_height = right_image.size

    interval = 200
    total_width = left_width + interval + right_width
    max_height = max(left_height, right_height)
    combined_image = Image.new('RGB', (total_width, max_height), color='white')
    combined_image.paste(left_image, (0, 0))
    combined_image.paste(right_image, (left_width + interval, 0))

    draw = ImageDraw.Draw(combined_image)

    left_coords = {index: (x, y) for index, x, y in left_points}
    right_coords = {index: (x, y) for index, x, y in right_points}

    for i, (left_index, right_index) in enumerate(matches):
        left_x, left_y = left_coords[left_index]
        right_x, right_y = right_coords[right_index]

        right_x += left_width + interval
        if left_index == right_index:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        r = 20
        draw.ellipse((left_x - r, left_y - r, left_x + r, left_y + r), fill=color, outline='blue')
        draw.ellipse((right_x - r, right_y - r, right_x + r, right_y + r), fill=color, outline='blue')
        draw.line((left_x, left_y, right_x, right_y), fill=color, width=6)

    combined_image.show()

import os
def read_images_path(folder_path):
    images_path = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        images_path.append(file_path)
    
    return images_path

def count_matching_pairs(matches, source_graph, target_graph):
    source_ids = np.array(list(source_graph['id'].values()))
    target_ids = np.array(list(target_graph['id'].values()))
    
    i_indices, j_indices = np.where(matches[0] == 1)
    
    selected_source_ids = source_ids[i_indices]
    selected_target_ids = target_ids[j_indices]
    
    return int(np.sum(selected_source_ids == selected_target_ids))

def normalize_pixels(matrix):
    matrix = np.array(matrix, dtype=np.float32)
    
    nonzero_mask = (matrix != 0)
    nonzero_min = np.min(matrix[nonzero_mask])
    global_max = np.max(matrix)
    normalized = np.where(
        matrix != 0,
        (matrix - nonzero_min) / (global_max - nonzero_min),
        0
    )
    return normalized

def index2ids(index, source_graph, target_graph):
    matching_indices = np.where(index[0] == 1)

    matches = [
        (source_graph['id'][source_idx], target_graph['id'][target_idx])
        for source_idx, target_idx in zip(matching_indices[0], matching_indices[1])
    ]
    matches_index = [(source_idx, target_idx)
        for source_idx, target_idx in zip(matching_indices[0], matching_indices[1])
    ]
    return matches, matches_index

