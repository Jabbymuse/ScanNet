import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np

def Alignement(Graph,frames):
    """Compute the local alignment of eigenvectors
       Graph is the Neighborhood Graph (Tensor of shape Naa * K * (M*6))
       Frames contains the local frames of amino_acids (Tensor of shape (Naa * 4 * 3))"""
    frames = frames.eval(session=tf.compat.v1.Session()) # attention : troubles if tensors are random generated (cf eval seems to relaunch the generation)
    Graph = Graph.eval(session=tf.compat.v1.Session())
    Naa = Graph.shape[0]
    K = Graph.shape[1]
    M = Graph.shape[2]
    for i in range(Naa): # iterate over each acid
        Rot_Mat = np.transpose(frames[i][1:]) # creation of the rotation matrix (we don't use center)
        for k in range(K): # iterate over each vector of the current aa
            for m in range(M):
                w = Graph[i][k][m][:3]
                v = Graph[i][k][m][3:]
                Graph[i][k][m] = np.concatenate((w @ Rot_Mat,v @ Rot_Mat))
    return tf.convert_to_tensor(Graph,dtype=tf.float64)

# tests
"""
Naa = 102
K = 16
M = 10
G = tf.ones(shape=[Naa,K,M,6])
frames = []


for _ in range(Naa):
    frames.append([[1,1,1],[1,0,0],[0,1,0],[0,0,1]])

frames = tf.convert_to_tensor(frames)

print(G.eval(session=tf.compat.v1.Session()) - Alignement(G,frames).eval(session=tf.compat.v1.Session()))
"""