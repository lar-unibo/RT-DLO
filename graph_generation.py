import cv2
import numpy as np
import torch
from torch import nn
import torch_geometric


torch.manual_seed(0)
np.random.seed(0)


class AngleNet(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.conv_w1 = nn.Conv2d(1, hidden_dim//4, 3)
        self.bn_w1 = nn.BatchNorm2d(hidden_dim//4)
        self.pool_w1 = nn.MaxPool2d(3)
        self.conv_w2 = nn.Conv2d(hidden_dim//4, hidden_dim//4, 3)
        self.bn_w2 = nn.BatchNorm2d(hidden_dim//4)
        self.flatten = nn.Flatten()

        self.lin_reg = nn.Linear(hidden_dim, 180)

    def encode_w(self, w):
        y = self.bn_w1(self.conv_w1(w.unsqueeze(dim=1))).relu()
        y = self.pool_w1(y)
        y = self.bn_w2(self.conv_w2(y)).relu()
        return self.flatten(y)
    
    def forward(self, windows):
        y = self.encode_w(windows)
        return self.lin_reg(y)




class GraphGeneration():

    def __init__(self, n_knn, th_edges_similarity, th_mask=127, wsize=15, sampling_ratio=0.1, network=None):
        self.n_knn = n_knn
        self.th_edges_similarity = th_edges_similarity
        self.th_mask = th_mask
        self.wsize = wsize
        self.sampling_ratio = sampling_ratio
        
        if network is None:
            self.load_network_model()
        else:
            self.network = network

    def load_network_model(self):
        self.network = AngleNet(hidden_dim=128)
        self.network.load_state_dict(torch.load("checkpoints/CP_angle.pth"))
        self.network.eval()


    def exec(self, mask, edges_filtering=True, debug=True):
        mask[mask < self.th_mask] = 0
        mask[mask != 0] = 255

        # DISTANCE IMAGE
        dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dist_img = cv2.GaussianBlur(dist_img,(3,3),0)

        rv_graph = self.run_graph_generation(mask, dist_img)

        if rv_graph is None:
            return None, None
        else:
            nodes, edges, edges_knn, nodes_dir, windows, points_maxmask = rv_graph

        if edges.shape[0] == 0:
            return None, None

        if edges_filtering:
            edges = self.filter_edges_from_mask(nodes, edges, mask)

        # OUTPUT
        nodes_dict = {}
        max_r = np.max(dist_img)
        for it, n in enumerate(nodes):
            n_pos = tuple([int(n[0]), int(n[1])])
            n_dir = tuple([nodes_dir[it,0], nodes_dir[it,1]])
            r = dist_img[tuple(n_pos)]
            nodes_dict[it] = {"pos": n_pos, "dir": n_dir, "window": windows[it], "r": r/max_r, "r_raw": r}

        self.nodes_pos = nodes
        self.nodes_dir = nodes_dir
        self.edges_np = np.array(edges).T
        self.dist_img = dist_img
        self.points_maxmask = points_maxmask
        return nodes_dict, edges

    def edges_similarity_fast(self, matrix_scores, X, edges, edges_norm, adj_fake, th=0.25):
        
        edges_tuples = edges.T        
        E = np.repeat(edges_norm.reshape(1,-1), matrix_scores.shape[0], axis=0)
        X2 = np.repeat(X[edges].reshape(1,-1), matrix_scores.shape[0], axis=0)   

        M = matrix_scores * X2 * adj_fake
        pos_matrix_scores = M > th
        E_pos_zero = E.copy()
        E_pos_zero[pos_matrix_scores == False] = 0
        if E_pos_zero.shape[1] > 0:
            E_pos_max = np.repeat(np.max(E_pos_zero, axis=1).reshape(-1,1), matrix_scores.shape[1], axis=1)

            E_pos_inf = E.copy()
            E_pos_inf[pos_matrix_scores == False] = np.inf
            E_pos_min = np.repeat(np.min(E_pos_inf, axis=1).reshape(-1,1), matrix_scores.shape[1], axis=1)
            W_pos = 1 - (E - E_pos_min) / E_pos_max
            W_pos[pos_matrix_scores == False] = 0
            S_pos = M * W_pos
            S_pos[pos_matrix_scores == False] = 0

            pos_indeces = np.argmax(S_pos, axis=1)
            pos_edges = edges_tuples[pos_indeces]
        else:
            pos_edges = []

        neg_matrix_scores = M < -th
        E_neg_zero = E.copy()
        E_neg_zero[neg_matrix_scores == False] = 0
        if E_neg_zero.shape[1] > 0:
            E_neg_max = np.repeat(np.max(E_neg_zero, axis=1).reshape(-1,1), matrix_scores.shape[1], axis=1)

            E_neg_inf = E.copy()
            E_neg_inf[neg_matrix_scores == False] = np.inf
            E_neg_min = np.repeat(np.min(E_neg_inf, axis=1).reshape(-1,1), matrix_scores.shape[1], axis=1)
            W_neg = 1 - (E - E_neg_min) / E_neg_max
            W_neg[neg_matrix_scores == False] = 0
            S_neg = M * W_neg
            S_neg[neg_matrix_scores == False] = 0

            neg_indeces = np.argmin(S_neg, axis=1)
            neg_edges = edges_tuples[neg_indeces]
        else:
            neg_edges = []

        new_edges_fast = np.concatenate([pos_edges, neg_edges])       
        return new_edges_fast
      

    def cosine_sim(self, a, b, eps=1e-8):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        return torch.mm(a_norm, b_norm.transpose(0, 1))


    def similarity_score_nodes_edges(self, nodes, edges, nodes_angles, debug=False):
        edges_dirs, edges_norm, adj_fake = self.edges_prep(edges, nodes)
        edges_dirs = torch.Tensor(edges_dirs) 

        angles = torch.Tensor(np.array(nodes_angles)) 
        nodes_dir = torch.stack([torch.sin(angles), torch.cos(angles)]).T

        X = torch.abs(self.cosine_sim(nodes_dir, nodes_dir))
        similarity_score_edges = self.cosine_sim(nodes_dir, edges_dirs) * torch.abs(torch.Tensor(adj_fake))

        return similarity_score_edges, X, edges_norm, adj_fake



    def edges_prep(self, edges, nodes, debug=False):
        nodes_0 = nodes[edges[0,:]]
        nodes_1 = nodes[edges[1,:]]
        edge_dirs = (nodes_1 - nodes_0).astype(np.float32)
        edge_norms = np.linalg.norm(edge_dirs, axis=1)
        edge_dirs[:,0] = edge_dirs[:,0] / edge_norms
        edge_dirs[:,1] = edge_dirs[:,1] / edge_norms
        edge_dirs = edge_dirs.reshape(-1,2)


        edges_list = list(zip(edges[0], edges[1]))
        adj_fake = np.zeros((len(nodes), len(edges_list)))
        for it, (e0,e1) in enumerate(edges_list):
            adj_fake[e0,it] = 1
            adj_fake[e1,it] = -1


        return edge_dirs, edge_norms, adj_fake


    def orientations_from_network(self, windows):

        windows = torch.Tensor(np.array(windows))
        preds = self.network(windows).sigmoid().detach().cpu().numpy()

        angles = np.deg2rad(np.argmax(preds, axis=1))
        dirs = np.stack([np.sin(angles), np.cos(angles)]).T
        return dirs, angles


    def run_graph_generation(self, mask, dist_img):

        # local maximus
        max_image = cv2.dilate(dist_img, np.ones((3,3)))
        maxmask = (dist_img == max_image) & mask

        x,y = np.nonzero(maxmask)
        points_maxmask = np.stack([x,y]).T

        # fps
        points = torch.Tensor(points_maxmask)
        indices_fps = torch_geometric.nn.fps(points, ratio=self.sampling_ratio)
        reduced = points[indices_fps].detach().cpu().numpy().astype(int)

        # windows
        l = (self.wsize-1)//2
        windows, points = [], []
        for p in reduced:    
            w = dist_img[p[0]-l:p[0]+l+1, p[1]-l:p[1]+l+1]
            if w.shape == (self.wsize, self.wsize):
                windows.append(w)
                points.append(p)
        points = np.array(points)

        if points.shape[0] == 0:
            return None        
    
        # orientations
        W = np.array(windows) / np.max(windows)
        points_dirs, points_angles = self.orientations_from_network(W)

        # edges with knn
        edges_knn = torch_geometric.nn.knn_graph(torch.Tensor(points), self.n_knn).detach().cpu().numpy()

        # matrix score
        matrix_scores, X, edges_norm, adj_fake = self.similarity_score_nodes_edges(points, edges_knn, points_angles)

        # edges computation
        new_edges = self.edges_similarity_fast(matrix_scores, X, edges_knn, edges_norm, adj_fake, th=self.th_edges_similarity)

        return points, new_edges, edges_knn, points_dirs, W, points_maxmask



    def filter_edges_from_mask(self, nodes, edges, mask):

        mask_large = cv2.dilate(mask,np.ones((5,5),np.uint8),iterations = 2)

        nodes_0 = nodes[edges[:,0]]
        nodes_1 = nodes[edges[:,1]]
        
        distances = np.linalg.norm(nodes_0 - nodes_1, axis=1)
        indeces = np.where(distances > np.mean(distances))[0]

        mid_points_to_test = ((nodes_0[indeces] + nodes_1[indeces]) / 2).astype(int)
        indeces_zero = np.where(mask_large[mid_points_to_test[:,0], mid_points_to_test[:,1]] == 0)[0]
                
        mask = np.ones(edges.shape[0], np.bool)
        mask[indeces[indeces_zero]] = 0
        new_edges = edges[mask]
        return new_edges