# Module to construct the K closest neighbors (eigenvectors) of each amino_acids (local vectors included)
import numpy as np
from collections import defaultdict as dd
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

def Neighborhood_motion(file,indexes,m):
    """Compute the Neighborhood graph of the K closest vectors for each amino_acid.
       indexes is a tensor (shape ? * Naa * ? (K) * 1) containing labels of neighbors of each key-labeled amino_acid.)
       m is the number of motions."""

    # to start with, we compute each amino_acids and their bonded vectors
    f = open(file,'r')
    line = f.readline()
    indexes = tf.reshape(indexes[0], shape=(indexes.shape[1], indexes.shape[2])) # Naa * ? (K)
    while line != "":
        line = line.replace("\n","")
        line = line.split("  ")
        i = 0
        while line[i] == '':
            i += 1
        index = int(line[i]) # index of the current amino_acid
        i += 3 # jump the center of mass
        line_aa = tf.convert_to_tensor([list(map(float,[line[k] for k in range(i+1,i+7)]))]) # motions for a specific aa (m * 6)
        for j in range(1,m):
            line_aa = tf.concat([line_aa,tf.convert_to_tensor([list(map(float,[line[k] for k in range(i+1+6*j,i+7+6*j)]))])],axis=0)
        if index == 0:
            amino_acids = tf.convert_to_tensor([line_aa]) # creation of a line of neighbors
        else:
            amino_acids = tf.concat([amino_acids,tf.convert_to_tensor([line_aa])],axis=0) # filling
        line = f.readline()
    # now, we just create the graph matrix
    f.close()
    G = tf.gather(amino_acids,indexes) # create the graph
    return G

def Alignement(Graph,frames):
    """Compute the local alignment of eigenvectors
       Returns an aligned graph
       Graph is the Neighborhood Graph (Tensor of shape Naa * K * m * 6)
       Frames contains the local frames of amino_acids (Tensor of shape (Naa * 4 * 3)
    """

    # note for the future : if a batch dimension is added, some reshape might help to handle it.
    
    for i in range(Graph.shape[0]):

        neighborhood = Graph[i][1:] # we convert every neighbor vector in (v,w)
        neighborhood = tf.reshape(neighborhood, shape=(-1, 3, 1)) # (K*m*2) * 3

        R = frames[i][1:]# creation of the matrix
        neighborhood = tf.linalg.matmul(R, neighborhood)
        neighborhood = tf.squeeze(neighborhood, axis=-1) # retire the extra_dimension
        neighborhood = tf.reshape(neighborhood,shape = (Graph.shape[1]-1,Graph.shape[2],6)) # K * m * 6

        new_neighborhood = tf.concat([tf.expand_dims(Graph[i][0],axis=0),neighborhood],axis=0) # new neighborhood of aa

        Graph = tf.tensor_scatter_nd_update(Graph, indices=[[i]], updates=[new_neighborhood]) # update of the graph
    return Graph
