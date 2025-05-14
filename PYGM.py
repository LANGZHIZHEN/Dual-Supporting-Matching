import numpy as np
import pygmtools as pygm
import functools
pygm.set_backend('numpy')
np.random.seed(1)


def RRWM(source_adj_matrix, target_adj_matrix):
    A1 = np.array(source_adj_matrix).reshape(1, source_adj_matrix.shape[0], source_adj_matrix.shape[1])
    A2 = np.array(target_adj_matrix).reshape(1, target_adj_matrix.shape[0], target_adj_matrix.shape[1])
    n1 = np.array([A1.shape[1]])
    n2 = np.array([A2.shape[1]])


    conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
    conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
    import functools
    gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
    
    K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)
    X = pygm.rrwm(K, n1, n2)
    X = pygm.hungarian(X)
    
    return X

def NGM(source_adj_matrix, target_adj_matrix):
    A1 = np.array(source_adj_matrix).reshape(1, source_adj_matrix.shape[0], source_adj_matrix.shape[1])
    A2 = np.array(target_adj_matrix).reshape(1, target_adj_matrix.shape[0], target_adj_matrix.shape[1])
    n1 = np.array([A1.shape[1]])
    n2 = np.array([A2.shape[1]])

    conn1, edge1, ne1 = pygm.utils.dense_to_sparse(A1)
    conn2, edge2, ne2 = pygm.utils.dense_to_sparse(A2)
    import functools
    gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=1.) # set affinity function
    K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)
    X, net = pygm.ngm(K, n1, n2, return_network=True, pretrain='willow')
    X = pygm.hungarian(X)
    return X


def IPCA_GM(source_adj_matrix, source_node_feat, target_adj_matrix, target_node_feat):
    batch_size = 1
    A1 = np.array(source_adj_matrix).reshape(1, source_adj_matrix.shape[0], source_adj_matrix.shape[1])
    A2 = np.array(target_adj_matrix).reshape(1, target_adj_matrix.shape[0], target_adj_matrix.shape[1])
    n1 = np.array([A1.shape[1]])
    n2 = np.array([A2.shape[1]])
    X = pygm.pca_gm(source_node_feat, target_node_feat, A1, A2, n1, n2, return_network=False, pretrain='willow')
    matches = pygm.hungarian(X)
    return matches

def PCA_GM(source_adj_matrix, source_node_feat, target_adj_matrix, target_node_feat):
    batch_size = 1
    A1 = np.array(source_adj_matrix).reshape(1, source_adj_matrix.shape[0], source_adj_matrix.shape[1])
    A2 = np.array(target_adj_matrix).reshape(1, target_adj_matrix.shape[0], target_adj_matrix.shape[1])
    n1 = np.array([A1.shape[1]])
    n2 = np.array([A2.shape[1]])
    X = pygm.pca_gm(source_node_feat, target_node_feat, A1, A2, n1, n2, return_network=False, pretrain='willow')
    matches = pygm.hungarian(X)
    return matches

if __name__ == "__main__":
    IPCA_GM()
    # graph_num = 2
    # As, X_gt = pygm.utils.generate_isomorphic_graphs(node_num=4, graph_num=graph_num)
    # matches = CAO(As, graph_num)
    # print(matches)