import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import warnings
import numpy as np
from skimage import io, graph
from skimage.segmentation import slic
from scipy import ndimage
from model import SCDGN
from dataset import *
from task import *
from utilis import *

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='SCDGN Training')
    parser.add_argument('--dataset', type=str, default='guangzhou',
                      choices=['guangzhou', 'beijing', 'montreal'],
                      help='Dataset to use for training')
    parser.add_argument('--n_layers', type=int, default=12,
                      help='Number of layers in the network')
    parser.add_argument('--hid_dim', type=int, default=128,
                      help='Hidden dimension size')
    parser.add_argument('--dropout', type=float, default=0.0,
                      help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=300,
                      help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=20,
                      help='Early stopping patience')
    parser.add_argument('--knn', type=int, default=700,
                      help='Number of KNN neighbors')
    parser.add_argument('--imp_lr', type=float, default=1e-3,
                      help='Learning rate for implicit parameters')
    parser.add_argument('--exp_lr', type=float, default=1e-5,
                      help='Learning rate for explicit parameters')
    parser.add_argument('--imp_wd', type=float, default=1e-5,
                      help='Weight decay for implicit parameters')
    parser.add_argument('--exp_wd', type=float, default=1e-5,
                      help='Weight decay for explicit parameters')
    parser.add_argument('--beta', type=float, default=9,
                      help='Beta parameter for loss function')
    parser.add_argument('--gamma', type=float, default=1,
                      help='Gamma parameter for loss function')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    return parser.parse_args()

def load_data(dataset_name):
    data_path = os.path.join('data', dataset_name.capitalize())
    img1 = io.imread(os.path.join(data_path, 'im1.bmp'))
    img2 = io.imread(os.path.join(data_path, 'im2.bmp'))
    label = im_normalize(io.imread(os.path.join(data_path, 'ref.bmp'), as_gray=True))
    
    X1 = im_normalize(img1)
    X2 = im_normalize(img2)
    
    num_segments = 4000
    concatenated_img_np = np.concatenate((X1, X2), axis=2)
    segments = slic(concatenated_img_np, n_segments=num_segments, compactness=0.3, start_label=0)
    
    labels = np.unique(segments)
    num_labels = len(labels)
    print(f"Number of segments: {num_labels}")
    
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
    
    return feat, segments, num_labels, label

def train(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    feat, segments, num_labels, label = load_data(args.dataset)
    
    # Compute adjacency matrix
    adj = compute_knn(args, feat)
    norm_factor1, edge_index1, edge_weight1, adj_norm1, knn1, Lap1 = cal_norm(A, args, feat1)
    norm_factor2, edge_index2, edge_weight2, adj_norm2, knn2, Lap2 = cal_norm(A, args, feat2)
    Lap_Neg = cal_Neg(adj_norm, knn, args)
    
    # Initialize model
    model = Net(num_labels, edge_index, edge_weight, args).to(args.device)
    optimizer = optim.Adam([
        {'params': model.params_imp, 'weight_decay': args.imp_wd, 'lr': args.imp_lr},
        {'params': model.params_exp, 'weight_decay': args.exp_wd, 'lr': args.exp_lr}
    ])
    
    # Training loop
    checkpt_file = os.path.join('best', f'{args.dataset}.pt')
    cnt_wait = 0
    best_loss = 1e9
    best_epoch = 0
    EYE = torch.eye(num_labels).to(args.device)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        emb = model(knn, adj_norm, norm_factor)
        loss1 =( torch.trace(torch.mm(torch.mm(emb1.t(), Lap1), emb2)) \
                        - args.beta*(torch.trace(torch.mm(torch.mm(emb1.t(), Lap_Neg1), emb1))) \
                        + args.gamma*nn.MSELoss()(torch.mm(emb1,emb1.t()), EYE))/args.N 
        loss2 =( torch.trace(torch.mm(torch.mm(emb2.t(), Lap2), emb1)) \
                    - args.beta*(torch.trace(torch.mm(torch.mm(emb2.t(), Lap_Neg2), emb2))) \
                    + args.gamma*nn.MSELoss()(torch.mm(emb2,emb2.t()), EYE))/args.N 
        
        loss = -0.5*(loss1+loss2)

        print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')
        
        loss.backward()
        optimizer.step()
        
        if loss <= best_loss:
            best_loss = loss
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), checkpt_file)
        else:
            cnt_wait += 1
            
        if cnt_wait == args.patience or torch.isnan(loss):
            print('\nEarly stopping!')
            break
    
    print(f"Training completed! Best epoch: {best_epoch}, Best loss: {best_loss:.4f}")

if __name__ == '__main__':
    args = parse_args()
    train(args) 