import numpy as np
import torch
from sklearn.metrics import pairwise
import scipy
import scipy.sparse as sp
from torch_scatter import scatter_add
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix, degree, add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch.utils.data import Dataset,DataLoader
from skimage.segmentation import slic
from scipy import ndimage
from torch_geometric.data import Data
from glob import glob
import os
from PIL import Image
from utilis import *

from ipdb import set_trace


def build_adjacency(segments):

    right_shift = np.roll(segments, -1, axis=1)
    down_shift = np.roll(segments, -1, axis=0)

    right_diff = segments[:, :-1] != right_shift[:, :-1]
    down_diff = segments[:-1, :] != down_shift[:-1, :]

    right_pairs = np.stack((segments[:, :-1][right_diff], right_shift[:, :-1][right_diff]), axis=1)
    down_pairs = np.stack((segments[:-1, :][down_diff], down_shift[:-1, :][down_diff]), axis=1)


    all_pairs = np.concatenate((right_pairs, down_pairs), axis=0)

    all_pairs = np.sort(all_pairs, axis=1)
    unique_pairs = np.unique(all_pairs, axis=0)

    return unique_pairs



def get_rw_adj(edge_index, norm_dim=1, fill_value=0., num_nodes=None, type='sys'):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float32, device=edge_index.device)
    
    if not fill_value == 0:
        edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    indices = row if norm_dim == 0 else col
    deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)

    if type=='sys':   
       deg_inv_sqrt = deg.pow_(-0.5)
       edge_weight = deg_inv_sqrt[indices] * edge_weight * deg_inv_sqrt[indices]
    else: 
       deg_inv_sqrt = deg.pow_(-1)
       edge_weight = deg_inv_sqrt[indices] * edge_weight if norm_dim == 0 else edge_weight * deg_inv_sqrt[indices]
    return edge_index, edge_weight


def adj_normalized(adj, type='sys'):
    row_sum = torch.sum(adj, dim=1)
    row_sum = (row_sum==0)*1+row_sum
    if type=='sys':
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt.mm(adj).mm(d_mat_inv_sqrt)
    else: 
        d_inv = torch.pow(row_sum, -1).flatten()
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diag(d_inv)
        return d_mat_inv.mm(adj)

def FeatureNormalize(mx):
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum == 0) * 1 + rowsum              
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def compute_knn(args, features, distribution='t-distribution'):
    features = FeatureNormalize(features)
    Dis = pairwise.cosine_distances(features, features)
    Dis = Dis/np.max(np.max(Dis, 1))
    if distribution=='t-distribution':
        gamma = CalGamma(args.v_input)
        sim = gamma * np.sqrt(2 * np.pi) * np.power((1 + args.sigma*np.power(Dis,2) / args.v_input), -1 * (args.v_input + 1) / 2)
    else:
        sim = np.exp(-Dis/(args.sigma**2))

    K = args.knn
    if K>0:
        idx = sim.argsort()[:,::-1]
        sim_new = np.zeros_like(sim)
        for ii in range(0, len(sim_new)):
            sim_new[ii, idx[ii,0:K]] = sim[ii, idx[ii,0:K]]      
        Disknn = (sim_new + sim_new.T)/2
    else:
        Disknn = (sim + sim.T)/2
    
    Disknn = torch.from_numpy(Disknn).type(torch.FloatTensor)
    Disknn = torch.add(torch.eye(Disknn.shape[0]), Disknn)
    Disknn = adj_normalized(Disknn)

    return Disknn

def CalGamma(v):
    a = scipy.special.gamma((v + 1) / 2)
    b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
    out = a / b
    return out


# def cal_norm(edge_index0, args, feat=None, cut=False, num_nodes=None):
#     # calculate normalization factors: (2*D)^{-1/2} or (D)^{-1/2}
#     edge_index0 = sp.coo_matrix(edge_index0)
#     values = edge_index0.data  
#     indices = np.vstack((edge_index0.row, edge_index0.col))
#     edge_index0 = torch.LongTensor(indices).to(args.device) 
    
#     edge_weight = torch.ones((edge_index0.size(1),), dtype=torch.float32, device=args.device)
#     edge_index, _ = add_remaining_self_loops(edge_index0, edge_weight, 0, args.N)
    
#     if num_nodes is None:
#         num_nodes = edge_index.max()+1
#     D = degree(edge_index[0], num_nodes)  # 传入edge_index[0]计算节点出度, 该处为无向图，所以即计算节点度

#     if cut: 
#         D = torch.sqrt(1/D)
#         D[D == float("inf")] = 0.
#         edge_index = to_undirected(edge_index, num_nodes=num_nodes) 
#         row, col = edge_index
#         mask = row<col
#         edge_index = edge_index[:,mask]
#     else:
#         D = torch.sqrt(1/2/D)
#         D[D == float("inf")] = 0.
    
#     if D.dim() == 1:
#         D = D.unsqueeze(-1)
        
#     edge_index, edge_weight = get_rw_adj(edge_index, norm_dim=1, fill_value=1, num_nodes=args.N )
#     adj_norm = to_scipy_sparse_matrix(edge_index, edge_weight).todense()
#     adj_norm = torch.from_numpy(adj_norm).type(torch.FloatTensor).to(args.device)
#     Lap = 1./D - adj_norm
    
#     if feat == None:
#         return Lap
    
#     knn = compute_knn(args, feat).to(args.device)
#     feat = feat.to(args.device)

#     return D, edge_index, edge_weight, adj_norm, knn, Lap


def cal_norm(edge_index0, args, feat=None, cut=False, num_nodes=None):

    edge_index0 = sp.coo_matrix(edge_index0)
    values = edge_index0.data  
    indices = np.vstack((edge_index0.row, edge_index0.col))
    edge_index0 = torch.LongTensor(indices).to(args.device) 
    
    edge_weight = torch.ones((edge_index0.size(1),), dtype=torch.float32, device=args.device)
    edge_index, _ = add_remaining_self_loops(edge_index0, edge_weight, 0, args.N)
    
    if num_nodes is None:
        num_nodes = edge_index.max()+1
    D = degree(edge_index[0], num_nodes) 

    if cut: 
        D = torch.sqrt(1/D)
        D[D == float("inf")] = 0.
        edge_index = to_undirected(edge_index, num_nodes=num_nodes) 
        row, col = edge_index
        mask = row<col
        edge_index = edge_index[:,mask]
    else:
        D = torch.sqrt(1/2/D)
        D[D == float("inf")] = 0.
    
    if D.dim() == 1:
        D = D.unsqueeze(-1)
        
    edge_index, edge_weight = get_rw_adj(edge_index, norm_dim=1, fill_value=1, num_nodes=args.N, type=args.type)
    adj_norm = to_scipy_sparse_matrix(edge_index, edge_weight).todense()
    adj_norm = torch.from_numpy(adj_norm).type(torch.FloatTensor).to(args.device)
    Lap = 1./D - adj_norm
    
    if feat == None:
        return Lap
    
    knn = compute_knn(args, feat).to(args.device)

    return D, edge_index, edge_weight, adj_norm, knn, Lap


def cal_Neg(knn, adj_norm, args):

    ones = torch.ones((args.N,args.N), dtype=torch.float32, device=args.device)
    zero = torch.zeros((args.N,args.N), dtype=torch.float32, device=args.device)
    Neg = torch.where((knn + adj_norm)==0, ones, zero).cpu()
    
    Lap_Neg = cal_norm(Neg, args)
    return Lap_Neg
    
class train_dataset(Dataset):
    def __init__(self, folder, phase, args):

        self.folder1 = os.path.join(folder, phase, "time1")
        self.folder2 = os.path.join(folder, phase, "time2")
        self.label_folder = os.path.join(folder, phase, "label")

        self.path1 = sorted(glob(os.path.join(self.folder1, "*.png")))
        self.path2 = sorted(glob(os.path.join(self.folder2, "*.png")))
        self.path3 = sorted(glob(os.path.join(self.label_folder, "*.png")))

        if len(self.path1) != len(self.path2):
            raise ValueError("time1 和 time2 文件夹中的图像数量不一致")

        self.path = list(zip(self.path1, self.path2))
        self.args = args

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        try:
            path1, path2 = self.path[idx]
            img1 = Image.open(path1).convert("RGB")
            img2 = Image.open(path2).convert("RGB")
        except Exception as e:
            print(f"Error loading images at index {idx}: {e}")
            return None

        img1 = np.array(img1)
        img2 = np.array(img2)
        X1 = im_normalize(img1)
        X2 = im_normalize(img2)
        concatenated_img_np = np.concatenate((X1, X2), axis=2)

        segments = slic(
            concatenated_img_np,
            n_segments=1000,
            compactness=10,
            # sigma=5,
            start_label=0,
            enforce_connectivity=False,
        )

        labels = np.unique(segments)
        num_labels = len(labels)

        H, W, C = img1.shape
        flattened_img1 = X1.reshape(-1, C)
        flattened_img2 = X2.reshape(-1, C)
        flattened_segments = segments.reshape(-1) 
        superpixel_features1 = np.stack([
        ndimage.mean(flattened_img1[:, c], labels=flattened_segments, index=labels)
        for c in range(3)
        ], axis=1)
        superpixel_features2 = np.stack([
        ndimage.mean(flattened_img2[:, c], labels=flattened_segments, index=labels)
        for c in range(3)
        ], axis=1)

        feat1 = torch.tensor(superpixel_features1, dtype=torch.float32) 
        feat2 = torch.tensor(superpixel_features2, dtype=torch.float32) 
        feat = torch.cat((feat1, feat2), dim=1)
        adj = compute_knn(self.args, feat)
        A = adj
        np.savetxt('adj.txt', A, fmt='%d')
        in_dim = feat1.shape[1]
        N = feat1.shape[0]

        args.N = segments.shape[0]
        norm_factor1, edge_index1, edge_weight1, adj_norm1, knn1, Lap1 = cal_norm(A, args, feat1)
        norm_factor2, edge_index2, edge_weight2, adj_norm2, knn2, Lap2 = cal_norm(A, args, feat2)
        norm_factor, edge_index, edge_weight, adj_norm, knn, Lap = cal_norm(A, self.args, feat)
        Lap_Neg1 = cal_Neg(adj_norm1, knn1, args)
        Lap_Neg2 = cal_Neg(adj_norm2, knn2, args)
        Lap_Neg = cal_Neg(adj_norm, knn, self.args)
        return (norm_factor, edge_index, edge_weight, adj_norm, knn, Lap, Lap_Neg)

class test_dataset(Dataset):
    def __init__(self, folder, phase, args):

        self.folder1 = os.path.join(folder, phase, "time1")
        self.folder2 = os.path.join(folder, phase, "time2")
        self.label_folder = os.path.join(folder, phase, "label")

        self.path1 = sorted(glob(os.path.join(self.folder1, "*.png")))
        self.path2 = sorted(glob(os.path.join(self.folder2, "*.png")))
        self.path3 = sorted(glob(os.path.join(self.label_folder, "*.png")))

        if len(self.path1) != len(self.path2):
            raise ValueError("time1 和 time2 文件夹中的图像数量不一致")

        self.path = list(zip(self.path1, self.path2, self.path3))
        self.args = args

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        try:
            path1, path2, path3 = self.path[idx]
            img1 = Image.open(path1).convert("RGB")
            img2 = Image.open(path2).convert("RGB")
            label = Image.open(path3).convert("L")
        except Exception as e:
            print(f"Error loading images at index {idx}: {e}")
            return None

        img1 = np.array(img1)
        img2 = np.array(img2)
        label_np = np.array(label)
        X1 = im_normalize(img1)
        X2 = im_normalize(img2)
        label = im_normalize(label_np)
        concatenated_img_np = np.concatenate((X1, X2), axis=2)

        segments = slic(
            concatenated_img_np,
            n_segments=1000,
            compactness=10,
            # sigma=5,
            start_label=0,
            enforce_connectivity=False,
        )

        labels = np.unique(segments)
        num_labels = len(labels)

        H, W, C = img1.shape

        flattened_img1 = X1.reshape(-1, C)
        flattened_img2 = X2.reshape(-1, C)
        flattened_segments = segments.reshape(-1)             
        superpixel_features1 = np.stack([
        ndimage.mean(flattened_img1[:, c], labels=flattened_segments, index=labels)
        for c in range(3)
        ], axis=1)
        superpixel_features2 = np.stack([
        ndimage.mean(flattened_img2[:, c], labels=flattened_segments, index=labels)
        for c in range(3)
        ], axis=1)



        feat1 = torch.tensor(superpixel_features1, dtype=torch.float32)  
        feat2 = torch.tensor(superpixel_features2, dtype=torch.float32)  

        feat = torch.cat((feat1, feat2), dim=1)

        adj = compute_knn(self.args, feat)
        A = adj
        np.savetxt('adj.txt', A, fmt='%d')

        data1 = list(cal_norm(A, self.args, feat1))
        data2 = list(cal_norm(A, self.args, feat2))


        return data1, data2, label, segments






