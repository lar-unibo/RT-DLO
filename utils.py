import itertools, cv2
import networkx as nx
import numpy as np
from matplotlib import cm
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString
import shapely

COLORS = cm.get_cmap("Set1", 10)



def retrieve_points_close_to_line(point_start, line_dir, points_set, l, s, max_distance=5):

    ls = np.arange(0, l, s).reshape(-1, 1)
    ls = np.repeat(ls, 2, axis=1)
    D = np.repeat(line_dir.reshape(1,-1), ls.shape[0], axis=0)
    points_line = (point_start + D * ls).astype(int)

    C = cdist(points_line, points_set)
    if C.shape[1] == 0:
        return None, None

    indeces = np.argmin(C, axis=1)
    values = np.min(C, axis=1)
    indeces_checked = []
    for i, idx in enumerate(indeces):
        if values[i] < max_distance and idx not in indeces_checked:
            indeces_checked.append(idx)

    return points_set[indeces_checked], indeces_checked




def extension_points_maxmask(paths, points_maxmask, s=5, l=50, shape=(360,640)):

    mask = np.zeros(shape, dtype=np.uint8)
    for k, value in paths.items():
        sub_points = list(value["points"])
        points = np.array(sub_points)
        points[:, [1, 0]] = points[:, [0, 1]]
        pts = points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(mask,[pts],False, 255,5)
    
    indeces = np.where(mask[points_maxmask[:,0], points_maxmask[:,1]] != 255)[0]
    points_maxmask_masked = points_maxmask[indeces]

    for k, value in paths.items():
        sub_points = list(value["points"])
    
        start_point = sub_points[0]
        start_nn_point = sub_points[1]
        tail_point = sub_points[-1]
        tail_nn_point = sub_points[-2]

        # begin
        dir_begin = start_point - start_nn_point
        dir_begin = dir_begin / np.linalg.norm(dir_begin)
        nodes_begin, _ = retrieve_points_close_to_line(start_point, dir_begin, points_maxmask_masked, l=l, s=s)
        if nodes_begin is not None:
            nodes_begin = nodes_begin[1:,:]
            if nodes_begin.shape[0] > 0:
                sub_points.insert(0, nodes_begin[-1,:])

        # tail
        dir_end = tail_point - tail_nn_point
        dir_end = dir_end / np.linalg.norm(dir_end)

        nodes_end, _ = retrieve_points_close_to_line(tail_point, dir_end, points_maxmask_masked, l=l, s=s)
        if nodes_end is not None:
            nodes_end = nodes_end[1:,:]
            if nodes_end.shape[0] > 0:
                sub_points.append(nodes_end[-1,:])
        
        paths[k]["points"] = sub_points


    return paths



def draw_function(mask, points, color, radius):
    points[:, [1, 0]] = points[:, [0, 1]]
    pts = points.reshape((-1, 1, 2)).astype(np.int32)
    c = [int(color[0]*255), int(color[1]*255), int(color[2]*255)]
    cv2.polylines(mask, [pts], False, c, radius)   


def draw_color_maks(paths_dict, nodes_dict, img_shape, max_dist_img):
    mask = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    for k, value in paths_dict.items():      
        r = 0.0
        for n in value["sequence"]:
            r += nodes_dict[n]["r"]
        r /= len(value["sequence"])
        r = int(np.round(r * max_dist_img * 2))
        draw_function(mask, value["spline"], COLORS(k), r)

    return mask



def compute_spline(points, k=3, smoothing=0.0, periodic=0, num_points=100):
    tck, u = splprep(points.T, u=None, k=k, s=smoothing, per=periodic)
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.stack([x_new, y_new]).T
    


def compute_splines(paths):
    for k, v in paths.items():
        if len(v["points"]) > 3:
            paths[k]["spline"] = compute_spline(np.array(v["points"]))
        else:
            paths[k]["spline"] = np.array(v["points"])

    return paths



def candidate_edges_test(subgraphs_dict):
    test_edges_0 = []
    for _,v in subgraphs_dict.items():
        combi_0 = list(map(list, itertools.combinations(v["neighs_0"], 2)))
        test_edges_0.extend(combi_0)
    return test_edges_0


def subgraphs(graph):

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



###########################################################################################
###########################################################################################

def intersections_btw_splines(paths, dist_img):

    def get_spline_portion(s, int_point, int_point_size):
        u = np.argmin(np.linalg.norm(np.subtract(s, int_point), axis=1))
        if u > 1:
            p_b = s[u-1]
        else:
            p_b = s[u]

        if u < len(s)-2:
            p_f = s[u+1]
        else:
            p_f = s[u]
        
        dir = p_f - p_b
        dir = dir / np.linalg.norm(dir)

        point_f = s[u] + 2*int_point_size * dir
        point_b = s[u] - 2*int_point_size * dir
        uf = np.argmin(np.linalg.norm(np.subtract(s, point_f), axis=1))
        ub = np.argmin(np.linalg.norm(np.subtract(s, point_b), axis=1))

        return s[ub:uf+1]

    keys = list(paths.keys())
    combs = list(map(list, itertools.combinations(keys, 2)))

    out = []
    for k0, k1 in combs:
        s0 = paths[k0]["spline"]
        s1 = paths[k1]["spline"]

        line0 = LineString(s0)
        line1 = LineString(s1)
        intersect_point = line0.intersection(line1)

        if type(intersect_point) == shapely.geometry.multipoint.MultiPoint:
            intersect_points = [(point.x, point.y) for point in intersect_point]
        elif type(intersect_point) == shapely.geometry.point.Point:
            intersect_points = [(intersect_point.x, intersect_point.y)]
        else:
            intersect_points = None

        if intersect_points is None:
            continue
        
        for int_point in intersect_points:
            int_point = np.array(int_point)
            int_point_size = dist_img[int(int_point[1]), int(int_point[0])]
            
            # 0
            points0_draw = get_spline_portion(s0, int_point, int_point_size)
            if len(points0_draw)>0:
                p0_0 = points0_draw[0].astype(int)
                p0_1 = points0_draw[-1].astype(int)
                radius0_draw = (dist_img[p0_0[1], p0_0[0]] + dist_img[p0_1[1], p0_1[0]])/2
            else:
                radius0_draw = 0

            # 1
            points1_draw = get_spline_portion(s1, int_point, int_point_size)
            if len(points1_draw)>0:
                p1_0 = points1_draw[0].astype(int)
                p1_1 = points1_draw[-1].astype(int)
                radius1_draw = (dist_img[p1_0[1], p1_0[0]] + dist_img[p1_1[1], p1_1[0]])/2
            else:
                radius1_draw = 0

            out.append({"points0_draw": points0_draw, "radius0_draw": radius0_draw, "id0": k0,
                        "points1_draw": points1_draw, "radius1_draw": radius1_draw, "id1": k1})
 
    return out


def std_colors(points, image):
    locations = points.astype(np.int32)   
    locations[:, [1, 0]] = locations[:, [0, 1]]
    values = image[locations[:,0], locations[:,1]]
    return np.std(values, axis=0), locations 


def intersection_scores_from_color(data_list, image, colored_mask=None, debug=False):

    for v in data_list:
        std0, locs0 = std_colors(v["points0_draw"], image)
        std1, locs1 = std_colors(v["points1_draw"], image)

        if np.mean(std0) < np.mean(std1):
            score0 = 1000
            score1 = 0
        else:
            score0 = 0
            score1 = 1000        

        v["score0"] = score0
        v["score1"] = score1

        if colored_mask is not None:  
            if v["score0"] > v["score1"]:
                draw_function(colored_mask, locs1, radius=int(np.round(v["radius1_draw"])*2), color=COLORS(v["id1"]))
                draw_function(colored_mask, locs0, radius=int(np.round(v["radius0_draw"])*2), color=COLORS(v["id0"]))
            else:
                draw_function(colored_mask, locs0, radius=int(np.round(v["radius0_draw"])*2), color=COLORS(v["id0"]))
                draw_function(colored_mask, locs1, radius=int(np.round(v["radius1_draw"])*2), color=COLORS(v["id1"]))

    return data_list




def intersections_layout(paths, dist_img, image, colored_mask):
    int_splines = intersections_btw_splines(paths, dist_img)
    int_splines = intersection_scores_from_color(int_splines, image, colored_mask)
    return colored_mask