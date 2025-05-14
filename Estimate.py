import cv2
import numpy as np
from utils import *
from scipy.optimize import least_squares

def calculate_homography(source_points, target_points, match_index, flag):
    src_pts = []
    dst_pts = []
    H = None
    for item in match_index:
        src_pts.append(source_points[item[0]])
        dst_pts.append(target_points[item[1]])
    src_pts = np.array(src_pts).reshape(-1,1,2)
    dst_pts = np.array(dst_pts).reshape(-1,1,2)
    # cv2.RHO算法优于cv2.FM_RANSAC
    if len(match_index) >= 4:
        if flag == cv2.FM_RANSAC:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.FM_RANSAC, 4.0)
        elif flag == cv2.RHO:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO, 4.0)
        else:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, 4.0)
    return H
    


def calculate_ls(left_points,left_weight,right_points,right_weight,match_index):
    left_normal_pts, LT = normalize_2d(np.array(left_points[:,1:]))
    right_normal_pts, RT = normalize_2d(np.array(right_points[:,1:]))
    A = []
    for pt1, pt2 in zip(left_normal_pts,right_normal_pts):
        x1,y1 = pt1
        x2,y2 = pt2
        A.append((x1,y1,1,0,0,0,-x1*x2,-y1*x2,-x2))
        A.append((0,0,0,x1,y1,1,-x1*y2,-y1*y2,-y2))
    np.array(A)
    U,Sigma,VT = np.linalg.svd(A)
    H = VT.T[:,-1].reshape(-1,3)
    H_initial = np.linalg.inv(RT) @ H @ LT
    result = least_squares(homography_error, H_initial.flatten(), args = (left_points, left_weight, right_points, right_weight, match_index))
    H_optimized = result.x.reshape((3, 3))
    H = H_optimized / H_optimized[2,2]
    return H

def homography_error(params, left_points, left_weight, right_points, right_weight, index):
    H = params.reshape((3, 3))  # 重构H矩阵
    error = []
    for item in index:
        homo_src_pts = np.array([left_points[item[0]][1], left_points[item[0]][2], 1])
        homo_dst_pts = np.array([right_points[item[1]][1], right_points[item[1]][2], 1])
        
        # 直接变换
        hd_pts = H @ homo_src_pts
        hd_pts = (hd_pts / hd_pts[2])[:2]
        
        # 逆变换
        hs_pts = np.linalg.inv(H) @ homo_dst_pts
        hs_pts = (hs_pts / hs_pts[2])[:2]
        
        sd = np.linalg.norm(hs_pts - left_points[item[0]][1:])
        dd = np.linalg.norm(hd_pts - right_points[item[1]][1:])
        if left_weight is not None:
            error.append(sd * left_weight[item[0]])
            error.append(dd * right_weight[item[1]])
        else:
            error.append(sd)
            error.append(dd)
    return np.array(error)

def calculate_error(H, source_points, target_points, index):
    homo_src_pts = []
    homo_dst_pts = []
    error = []
    for item in index:
        src_x, src_y = source_points[item[0]]  # 从 source_graph['pos'] 获取坐标
        dst_x, dst_y = target_points[item[1]]  # 从 target_graph['pos'] 获取坐标
        homo_src_pts.append([src_x, src_y, 1])  # 齐次坐标
        homo_dst_pts.append([dst_x, dst_y, 1])  # 齐次坐标
    sd = 0
    dd = 0
    for i in range(len(index)):
        # 正向变换：source -> target
        hd_pts = H @ homo_src_pts[i]  # 单应性变换
        hd_pts = (hd_pts / hd_pts[2])[:2]  # 归一化并取前两个值
        dst_x, dst_y = target_points[index[i][1]]  # 目标点实际坐标
        forward_error = np.linalg.norm(hd_pts - [dst_x, dst_y])  # 计算误差

        # 反向变换：target -> source
        hs_pts = np.linalg.inv(H) @ homo_dst_pts[i]  # 单应性逆变换
        hs_pts = (hs_pts / hs_pts[2])[:2]  # 归一化并取前两个值
        src_x, src_y = source_points[index[i][0]]  # 源点实际坐标
        backward_error = np.linalg.norm(hs_pts - [src_x, src_y])  # 计算误差

        # 将误差添加到列表
        error.append(forward_error)
        error.append(backward_error)
    return error
    
def point_to_epiline_distance(point, line):
    return np.abs(line[0]*point[0] + line[1]*point[1] + line[2]) / np.sqrt(line[0]**2 + line[1]**2)

def estimate_epiline_error(F,left_vanish_pts,right_vanish_pts):
    distances = dict()
    for i,(pt1, pt2) in enumerate(zip(left_vanish_pts, right_vanish_pts)):
        epiline = np.dot(F, np.array([pt1[0], pt1[1], 1]))
        distance = point_to_epiline_distance(pt2, epiline)
        pair = (left_vanish_pts[i][0],right_vanish_pts[i][0])
        distances[pair] = distance
    return distances


def homography_transform(H, point):
    h_point = H @ [point[0], point[1], 1]
    return (h_point / h_point[2])[:2]

def calculate_reprojection_error(H, src_point, dst_point):
    forward_proj = homography_transform(H, src_point)
    backward_proj = homography_transform(np.linalg.inv(H), dst_point)
    forward_error = np.linalg.norm(forward_proj - dst_point)
    backward_error = np.linalg.norm(backward_proj - src_point)
    return forward_error + backward_error

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
def get_feature_vector(image_path, nodes, model=None, feature_size=1024, crop_size=40):
    image = Image.open(image_path).convert('RGB')
    # 如果没有传入模型，则使用 VGG16 作为默认模型
    if model is None:
        model = models.vgg16(pretrained=True)
        model.eval()  # 设置为评估模式

    # 定义图像预处理方法
    preprocess = transforms.Compose([
        transforms.Resize(512),  # 将图像调整为 256x256
        transforms.CenterCrop(crop_size),  # 裁剪图像为 224x224
        transforms.ToTensor(),  # 转为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    
    image_tensor = preprocess(image)
    
    # 用于存储所有节点的特征
    all_features = []

    # 处理每个节点
    for node in nodes.items():
        x, y = node[1]
        x, y = int(x), int(y)
        left = max(0, x - crop_size // 2)
        top = max(0, y - crop_size // 2)
        right = min(image_tensor.shape[2], x + crop_size // 2)
        bottom = min(image_tensor.shape[1], y + crop_size // 2)
        
        # 裁剪区域的宽度和高度
        crop_width = right - left
        crop_height = bottom - top

        # 根据节点坐标，在原图中裁剪一个 crop_size x crop_size 的区域
        crop_area = image_tensor[:, top:bottom, left:right]
        
        # 若裁剪区域尺寸不足，则进行填充
        if crop_area.shape[1] != crop_size or crop_area.shape[2] != crop_size:
            padding_left = (crop_size - crop_area.shape[2]) // 2
            padding_right = crop_size - crop_area.shape[2] - padding_left
            padding_top = (crop_size - crop_area.shape[1]) // 2
            padding_bottom = crop_size - crop_area.shape[1] - padding_top
            padding = (padding_left, padding_right, padding_top, padding_bottom)
            
            # 使用 padding 填充图像
            crop_area = torch.nn.functional.pad(crop_area, padding, mode='constant', value=0)
        
        # 扩展 batch 维度
        crop_area = crop_area.unsqueeze(0)
        
        # 提取特征
        with torch.no_grad():
            features = model.features(crop_area)
            features = model.avgpool(features)
            features = features.view(features.size(0), -1)
            features = features[:, :feature_size]
        
        all_features.append(features)
    
    all_features = torch.cat(all_features, dim=1)
    all_features = all_features.reshape(1, -1, feature_size)
    
    return all_features.numpy()

