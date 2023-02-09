import cv2, itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graph_generation import GraphGeneration
from utils import compute_splines, draw_color_maks, extension_points_maxmask, intersections_layout


class EdgeSimilaritySolver():

    def __init__(self, n_knn, th_edges_similarity, sampling_ratio, th_mask):
        self.th_edges_similarity = th_edges_similarity
        self.n_knn = n_knn
        self.th_mask = th_mask
        self.sampling_ratio = sampling_ratio

        self.nodes_dict = {}
        self.edges = []
        self.new_edges = []

    def clear(self):
        self.nodes_dict = {}
        self.edges = []
        self.new_edges = []


    def exec(self, mask, source_img):

        # STEP 1 : graph generation
        graph_gen = GraphGeneration(n_knn=self.n_knn, th_edges_similarity=self.th_edges_similarity, 
                                    sampling_ratio=self.sampling_ratio, th_mask=self.th_mask)

        self.nodes_dict, self.edges = graph_gen.exec(mask, debug=False)
        if self.nodes_dict is None:
            return None, None, None
            
        self.nodes_pos = graph_gen.nodes_pos
        self.nodes_dir = graph_gen.nodes_dir
        self.edges_np = graph_gen.edges_np

        # STEP 2 : graph analysis
        g = nx.Graph()
        g.add_nodes_from([(k,v) for k,v in self.nodes_dict.items()])
        g.add_edges_from(self.edges, label=0)


        subgraphs_dict = self.subgraphs(g)
    
        test_edges_dict = self.candidate_edges_test(subgraphs_dict) 
        if test_edges_dict:
            self.new_edges = self.solver(self.nodes_dict, test_edges_dict)
            for k,v in subgraphs_dict.items():
                g.remove_nodes_from(v["int_nodes"])
            g.add_edges_from(self.new_edges, label=1)
            self.edges = list(g.edges())

        # INSTANCES
        paths = self.extract_paths(g, graph_gen.nodes_pos)

        # CHECK PATHS CONSISTENCY
        paths = self.check_paths_consistency(paths, self.nodes_dict)

        # EXT
        paths = extension_points_maxmask(paths, graph_gen.points_maxmask)

        # SPLINE
        paths = compute_splines(paths)

        points = {}
        for k,v in paths.items():
            points[k] = np.array(v["points"])

        # COLOR MAKS
        mask_output = draw_color_maks(paths, self.nodes_dict, mask.shape, max_dist_img = np.max(graph_gen.dist_img))

        # INTERSECTIONS
        mask_output = intersections_layout(paths, dist_img=graph_gen.dist_img, image=source_img, colored_mask=mask_output)

        return paths, mask_output, points


    def cosine_sim(self, a, b):
        return np.dot(a, b)


    def solver(self, nodes_dict, edges_dict):

        new_edges = []
        for k,v in edges_dict.items():
            nodes = np.unique(np.array(v).flatten())
            num_conn = nodes.shape[0] // 2
            E = {}
            for e0,e1 in v:
                dire = np.array(nodes_dict[e1]["pos"]) - np.array(nodes_dict[e0]["pos"])
                dire = dire / np.linalg.norm(dire)

                dir0 = nodes_dict[e0]["dir"] / np.linalg.norm(nodes_dict[e0]["dir"])

                dir1 = nodes_dict[e1]["dir"] / np.linalg.norm(nodes_dict[e1]["dir"])

                E[(e0,e1)] = np.fabs(self.cosine_sim(dir0, dire) * self.cosine_sim(dir1, dire))

            E = {k: v for k, v in sorted(E.items(), key=lambda item: item[1], reverse=True)}
            
            nodes_done = []
            counter = 0
            for ke,_ in E.items():
                if counter > num_conn:
                    break

                if ke[0] not in nodes_done and ke[1] not in nodes_done:
                    new_edges.append(ke)
                    nodes_done.extend([ke[0], ke[1]])
                    counter += 1

        return new_edges



    def check_paths_consistency(self, paths, nodes_dict):
        

        if len(paths)==0:
            return {}


        curr_key = max(list(paths.keys()))+1
        new_paths = {}
        for k,v in paths.items():
            seq = v["sequence"]
            points_divide = []

            nodes_pos_1 = np.array([nodes_dict[seq[it-1]]["pos"] for it in range(1, len(seq)-1)]).astype(float)
            nodes_pos_2 = np.array([nodes_dict[seq[it]]["pos"] for it in range(1, len(seq)-1)]).astype(float)
            nodes_pos_3 = np.array([nodes_dict[seq[it+1]]["pos"] for it in range(1, len(seq)-1)]).astype(float)

            dirs12 = nodes_pos_2 - nodes_pos_1
            dirs12_norms = np.linalg.norm(dirs12, axis=1)
            dirs12[:,0] = dirs12[:,0] / dirs12_norms
            dirs12[:,1] = dirs12[:,1] / dirs12_norms

            dirs23 = nodes_pos_3 - nodes_pos_2
            dirs23_norms = np.linalg.norm(dirs23, axis=1)
            dirs23[:,0] = dirs23[:,0] / dirs23_norms
            dirs23[:,1] = dirs23[:,1] / dirs23_norms

            dirs = np.array([nodes_dict[seq[it]]["dir"] for it in range(1, len(seq)-1)])
            dirs_norms = np.linalg.norm(dirs, axis=1)
            dirs[:,0] = dirs[:,0] / dirs_norms
            dirs[:,1] = dirs[:,1] / dirs_norms

            for it in range(1, len(seq)-1):
                s12 = self.cosine_sim(dirs[it-1,:], dirs12[it-1,:])
                s23 = self.cosine_sim(dirs[it-1,:], dirs23[it-1,:])

                if s12*s23 < 0:
                    points_divide.append(it)
            
            if not points_divide:
                new_paths[k] = v
            else:
                for p in points_divide:
                    seq0 = seq[:p]
                    points0 = v["points"][:p]
                    if len(seq0) > 2:
                        new_paths[curr_key] = {"sequence": seq0, "points": points0}
                        curr_key += 1

                    seq1 = seq[p+1:]
                    points1 = v["points"][p+1:]
                    if len(seq1) > 2:
                        new_paths[curr_key] = {"sequence": seq1, "points": points1}
                        curr_key += 1

        return new_paths


    def candidate_edges_test(self, subgraphs_dict):
        return {k: list(map(list, itertools.combinations(v["neighs_0"], 2))) for k,v in subgraphs_dict.items()}


    def subgraphs(self, graph):

        # problematic nodes
        int_nodes = [node for (node, val) in graph.degree() if val > 2]

        # neighbors of each problematic node
        neighbors_list = []
        for n in int_nodes:
            subgraph = nx.ego_graph(graph, n, radius=1)   
            neighbors_list.append(list(subgraph.nodes()))

        # combine overlapping neighbors employing networkx connected components
        gtemp = nx.Graph()
        for sub_list in neighbors_list:
            edges = [(sub_list[i], sub_list[i+1]) for i in range(len(sub_list)-1)]
            gtemp.add_edges_from(edges)

        subgraphs_nodes = list(nx.connected_components(gtemp))

        # compute output
        out = {}
        for it, sub in enumerate(subgraphs_nodes):
            sub_int_nodes = [n for n in int_nodes if n in sub]       
            
            neighs_0 = []
            for n in sub_int_nodes:
                neighs_0.extend([n for n in graph.neighbors(n)])
            neighs_0 = [n for n in list(set(neighs_0)) if n not in sub_int_nodes]
            
            neighs_0_filt = neighs_0.copy()
            for n in neighs_0:
                n_neigh = [n for n in graph.neighbors(n)]
                n_neigh_not_int = [n for n in n_neigh if n not in sub_int_nodes]
                if len(n_neigh_not_int) == 0 and len(n_neigh) > 1:
                    neighs_0_filt.remove(n)
                    sub_int_nodes.append(n)

            out[it] = {"int_nodes": sub_int_nodes, "neighs_0": neighs_0_filt}

        return out


    def extract_paths(self, graph, nodes):
        sub_graphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        paths = {}
        for it, sg in enumerate(sub_graphs):
            endpoints = [n[0] for n in sg.degree() if n[1] == 1]
            if len(endpoints) == 2:
                path = nx.shortest_path(sg, endpoints[0], endpoints[1])
                if len(path) > 3:
                    paths[it] = {"sequence": path, "points": nodes[path]}

        return paths




    def viz(self, img):

        if isinstance(self.new_edges, tuple):
            edges_1 = self.new_edges[0]
            edges_2 = self.new_edges[1]
        else:
            edges_1 = self.new_edges
            edges_2 = []

        g = nx.Graph()
        g.add_nodes_from([(k,v) for k,v in self.nodes_dict.items()])
        g.add_edges_from(self.edges, label=0)
        g.add_edges_from(edges_1, label=1)
        g.add_edges_from(edges_2, label=2)
        g.remove_nodes_from(list(nx.isolates(g)))

        fig = plt.figure(figsize=(13,9))
        plt.imshow(img)

        mask_w = np.ones(img.shape[:2], dtype=np.uint8)*255
        mask_w = cv2.cvtColor(mask_w, cv2.COLOR_GRAY2RGB)
        plt.imshow(mask_w); plt.axis("off")
        options = {
            "node_size": 500,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 2,
            "width": 2,
        }

        nx.draw(g, pos={k: (n["pos"][1], n["pos"][0]) for k, n in g.nodes(data=True)}, **options)
        for k, n in g.nodes(data=True): plt.text(n["pos"][1], n["pos"][0], k, fontsize=14, ha='center', va='center', color="black")
        plt.imshow(img); plt.axis("off")    
        plt.show()


