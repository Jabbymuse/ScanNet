import tensorflow as tf
import numpy as np
from collections import defaultdict as dd
from Neighborhood_motion import Neighborhood_motion,Alignement
tf.compat.v1.enable_eager_execution() # only use this to print the output of signnet

Naa = 7
m = 10
K = 3
from random import uniform as ru
from random import randint as ri
"""

frames = []
for _ in range(Naa):
    x_c,y_c,z_c = ru(-2,2),ru(-2,2),ru(-2,2)
    frames.append([[x_c,y_c,z_c],[1,0,0],[0,1,0],[0,0,1]]) # Id test

frames = tf.convert_to_tensor(frames)

# test Neigh_motion

T = tf.convert_to_tensor([])
for j in range(5):
    T = tf.concat([T,tf.convert_to_tensor([1],dtype=tf.float32)],axis=0)

T = tf.convert_to_tensor([1])
#T = tf.gather(T,indices=tf.constant([0,0,0,0]))

#print(T.eval(session=tf.compat.v1.Session()))

amino_acids = tf.convert_to_tensor([[[ru(-2,2) for _ in range(6)] for _ in range(m)] for _ in range(Naa)])
#print(amino_acids.eval(session=tf.compat.v1.Session()))

indexes = tf.convert_to_tensor([[ri(0,Naa-1) for _ in range(K)] for _ in range(Naa)])
#print(indexes.eval(session=tf.compat.v1.Session()))

g = tf.gather(amino_acids,indexes) # gather concatenates with respect to indices => graph
#print(aa0)
#print(aa0.eval(session=tf.compat.v1.Session()))

file = "motion_data/pdb1b9e_rtb (copy).txt"
G = Neighborhood_motion(file,indexes,m)
#print(G)
#print(G.eval(session=tf.compat.v1.Session()))

G1 = Alignement(G,frames)
#print(G1)
#print("G1 = ",G1.eval(session=tf.compat.v1.Session()))
#print((G-G1).eval(session=tf.compat.v1.Session())) # test with Id matrices (res is null)

# tests eigenvectors selection
G_column = tf.convert_to_tensor([[[[ru(-2,2) for _ in range(3)] for _ in range(3)] for _ in range(2)] for _ in range(4)])
#print(G_column.eval(session=tf.compat.v1.Session()))
#print("column = ",G_column[:,0].eval(session=tf.compat.v1.Session()))

x = G1[:,0]
#print("x = ",x.eval(session=tf.compat.v1.Session()))


from network import signnet
net = signnet.GINDeepSigns(6,6,6,1,m)
out = net(G1,x)
out2 = net(-G1,-x)

#print(out - out2) # use tf.compat... at the beginning of the file to print result
"""
# tests implementation direct dans scannet

B = 3
Naa = 4
K = 5
m = 2

Batch_frames = []
for _ in range(B):
    frames = []
    for _ in range(Naa):
        x_c,y_c,z_c = ru(-2,2),ru(-2,2),ru(-2,2)
        frames.append([[x_c,y_c,z_c],[1,0,0],[0,1,0],[0,0,1]]) # Id test
    Batch_frames.append(frames)

Batch_frames = tf.convert_to_tensor(Batch_frames)
#print(Batch_frames.shape)
#print("B = ",Batch_frames.eval(session=tf.compat.v1.Session()))

Batch_motion_vectors = []
for _ in range(B):
    motion_vectors = []
    for _ in range(Naa):
        line = []
        for _ in range(m):
            motion = []
            for _ in range(6):
                motion.append(ru(-2,2))
            line.append(motion)
        motion_vectors.append(line)
    Batch_motion_vectors.append(motion_vectors)

Batch_motion_vectors = tf.convert_to_tensor(Batch_motion_vectors)
#print(Batch_motion_vectors.shape)

Batch_neighbors = []
for _ in range(B):
    line = []
    for _ in range(Naa):
        neighbor = []
        for _ in range(K):
            neighbor.append([ri(0,2)])
        line.append(neighbor)
    Batch_neighbors.append(line)

Batch_neighbors = tf.convert_to_tensor(Batch_neighbors)
#print(Batch_neighbors.shape)

Batch_motion_vectors = tf.gather_nd(Batch_motion_vectors,Batch_neighbors,batch_dims=1)
#print(Batch_motion_vectors.shape)

v_Batch_motion_vectors = tf.expand_dims(Batch_motion_vectors[:,:,:,:,:3],axis=-1)
#print(v_Batch_motion_vectors.shape)


Batch_rot = tf.expand_dims(tf.tile(tf.expand_dims(Batch_frames[:,:,1:4],axis=-3),multiples=[1,1,m,1,1]),axis=-4)
#print(Batch_rot.shape)

#print(tf.reduce_sum(v_Batch_motion_vectors*Batch_rot,axis=-2).shape)
#v_Batch_motion_vectors = tf.matmul(v_Batch_motion_vectors,Batch_rot)
#print(v_Batch_motion_vectors.shape)

t1 = tf.tile(tf.expand_dims(tf.constant([[[4],[5],[6]],[[1],[2],[3]]]),axis=-4),multiples=[5,1,1,1])
mat = tf.tile(tf.expand_dims(tf.constant([[1,0,1],[0,2,2],[3,4,0]]),axis=-3),multiples=[2,1,1])

print(t1)
print(mat)
print(t1*mat)
print(tf.reduce_sum(t1 * mat,axis=-2))
