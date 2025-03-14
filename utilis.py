import numpy as np
import torch
import random
import torch.nn as nn
from skimage.segmentation import mark_boundaries
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image

def set_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

def visiualize_seg(img, segments):
    # 使用 mark_boundaries 绘制超像素边界
    boundaries_img = mark_boundaries(img, segments, color=(1, 1, 0), mode='thick')
    # 保存边界图像
    Image.fromarray((boundaries_img * 255).astype(np.uint8)).save('./data/superpixel_boundaries.png')

def im_normalize(im):
    # 将图像转换为float32类型
    im = im.astype(np.float32)
    # 计算图像的最小值
    min_value = im[np.unravel_index(np.argmin(im), im.shape)]
    # 计算图像的最大值
    max_value = im[np.unravel_index(np.argmax(im), im.shape)]
    # 将图像的像素值归一化
    im = (im - min_value) / (max_value - min_value)
    # 返回归一化后的图像
    return im

def build_adjacency(segments):
    """
    构建超像素的邻接关系。

    参数：
    - segments (np.ndarray): 超像素标签数组，形状为 (H, W)。

    返回：
    - unique_pairs (np.ndarray): 形状为 (num_edges, 2) 的标签对数组。
    """
    # 计算右邻和下邻
    right_shift = np.roll(segments, -1, axis=1)
    down_shift = np.roll(segments, -1, axis=0)

    # 获取边缘位置（排除最后一列和最后一行）
    right_diff = segments[:, :-1] != right_shift[:, :-1]
    down_diff = segments[:-1, :] != down_shift[:-1, :]

    # 获取标签对
    right_pairs = np.stack((segments[:, :-1][right_diff], right_shift[:, :-1][right_diff]), axis=1)
    down_pairs = np.stack((segments[:-1, :][down_diff], down_shift[:-1, :][down_diff]), axis=1)

    # 合并所有边
    all_pairs = np.concatenate((right_pairs, down_pairs), axis=0)

    # 去重并确保 (min, max) 顺序
    all_pairs = np.sort(all_pairs, axis=1)
    unique_pairs = np.unique(all_pairs, axis=0)

    return unique_pairs
def map_superpixels_to_pixels(segments, features):
    """
    将超像素特征映射回像素空间。
    
    参数:
    - segments: 超像素分割图，每个像素的标签。
    - features: 超像素特征数组，形状为 (num_superpixels, feature_dim)。
    
    返回:
    - pixel_image: 映射后的像素图像，形状为 (height, width, feature_dim)。
    """
    # 确保标签从0开始
    min_label = segments.min()
    if min_label != 0:
        segments = segments - min_label
    
    # 检查特征数组的长度是否与超像素数量匹配
    num_superpixels = segments.max() + 1
    if features.shape[0] != num_superpixels:
        raise ValueError(f"特征数组的超像素数量 ({features.shape[0]}) 与分割图中的超像素数量 ({num_superpixels}) 不匹配。")
    
    # 使用高级索引将特征映射到像素
    pixel_image = features[segments]
    
    return pixel_image

def get_adj(segments):
    height, width = segments.shape
    # 定义八连通邻接偏移，用于寻找邻居
    offsets = [
        (-1, 0),  # 上
        (1, 0),   # 下
        (0, -1),  # 左
        (0, 1),   # 右
        (-1, -1), # 左上
        (-1, 1),  # 右上
        (1, -1),  # 左下
        (1, 1)    # 右下
    ]

    # 初始化邻接矩阵
    num_superpixels = segments.max() + 1
    adjacency_matrix = np.zeros((num_superpixels, num_superpixels), dtype=np.int32)

    # 遍历图像的每一个像素
    for y in range(height):
        for x in range(width):
            current_segment = segments[y, x]
            
            # 遍历八个邻居像素
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_segment = segments[ny, nx]
                    
                    # 如果邻居的超像素编号不同，表示两个超像素相邻
                    if neighbor_segment != current_segment:
                        adjacency_matrix[current_segment, neighbor_segment] = 1
                        adjacency_matrix[neighbor_segment, current_segment] = 1

    return adjacency_matrix

def get_knn(feat):

    dist_matrix = euclidean_distances(feat, feat)
    k = 50
    num_nodes = dist_matrix.shape[0]

    # 初始化邻接矩阵为零矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int32)

    # 遍历每个节点，找到 k 个最近邻节点
    for i in range(num_nodes):
        # 获取与节点 i 的所有其他节点的距离，升序排序后取前 k+1 个（包含自己）
        nearest_indices = np.argsort(dist_matrix[i])[:k + 1]
        for j in nearest_indices:
            if i != j:  # 排除自身
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1 
    return adjacency_matrix