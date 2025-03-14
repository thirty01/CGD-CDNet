import argparse
import os
import torch
import numpy as np
from skimage import io
from model import SCDGN
from dataset import *
from task import *
from utilis import *
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='SCDGN Testing')
    parser.add_argument('--dataset', type=str, default='guangzhou',
                      choices=['guangzhou', 'beijing', 'montreal'],
                      help='Dataset to use for testing')
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='Device to use for testing')
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to the trained model')
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
    
    return feat, segments, num_labels, label, img1, img2


def test(args):
    # Load data
    feat, segments, num_labels, label, img1, img2 = load_data(args.dataset)
    
    # Compute adjacency matrix
    adj = compute_knn(args, feat)
    norm_factor, edge_index, edge_weight, adj_norm, knn, Lap = cal_norm(adj, args, feat)
    
    # Initialize model
    model = SCDGN(num_labels, edge_index, edge_weight, args).to(args.device)
    
    # Load trained model
    if args.model_path is None:
        args.model_path = os.path.join('best', f'{args.dataset}.pt')
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        emb1 = model(knn1, knn2, norm_factor1)
        emb2 = model(knn2, knn1, norm_factor2)

        # # Clustering 
        emb1 = emb1.cpu().detach().numpy()
        emb2 = emb2.cpu().detach().numpy()
        # emb = emb.cpu().detach().numpy()
        diff = np.abs(emb1 - emb2)
        diff = np.mean(diff, axis=1)
        cm = map_superpixels_to_pixels(segments, diff)
        threshold = filters.threshold_otsu(cm)
        binary_map = ((cm >= threshold) * 255).astype(np.uint8)
        import imageio
        imageio.imwrite('output.png', binary_map)
        metric_train = Metric(init_metric={'f1': 0.0, 'iou': 0.0, 'pr': 0.0, 're': 0.0,'oa': 0.0,'kappa':0.0})
        metric_train(binary_map, label)
        r = metric_train.calculate(local=False)
        #print(ptsr)
        pstr = metric_train.print(local=False)
        print("result{}".format(r))


if __name__ == '__main__':
    args = parse_args()
    test(args) 