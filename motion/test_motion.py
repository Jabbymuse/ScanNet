import tensorflow as tf
import numpy as np
from collections import defaultdict as dd
from Neighborhood_motion import Neighborhood_motion,Alignement
Naa = 7
m = 10
K = 3
from random import uniform as ru
from random import randint as ri

frames = []
for _ in range(Naa):
    x_c,y_c,z_c = ru(-2,2),ru(-2,2),ru(-2,2)
    frames.append([[x_c,y_c,z_c],[1,0,0],[0,1,0],[0,0,1]])

frames = tf.convert_to_tensor(frames)


# test Neig_motion

T = tf.convert_to_tensor([])
for j in range(5):
    T = tf.concat([T,tf.convert_to_tensor([1],dtype=tf.float32)],axis=0)

T = tf.convert_to_tensor([1])
#T = tf.gather(T,indices=tf.constant([0,0,0,0]))

#print(T.eval(session=tf.compat.v1.Session()))

amino_acids = tf.convert_to_tensor([[[ru(-2,2) for _ in range(6)] for _ in range(m)] for _ in range(Naa)])
#print(amino_acids.eval(session=tf.compat.v1.Session()))

indexes = tf.convert_to_tensor([[ri(0,Naa-1) for _ in range(K)] for _ in range(Naa)])
indexes = tf.reshape(indexes,shape=(1,Naa,K,1))
#print(indexes.eval(session=tf.compat.v1.Session()))

g = tf.gather(amino_acids,indexes) # gather concatenates with respect to indices => graph
#print(aa0)
#print(aa0.eval(session=tf.compat.v1.Session()))

file = "motion_data/pdb1b9e_rtb (copy).txt"
G = Neighborhood_motion(file,indexes,m)
#print(G)
#print(G.eval(session=tf.compat.v1.Session()))

G1 = Alignement(G,frames)
#print("G1 = ",G1.eval(session=tf.compat.v1.Session()))
#print((G-G1).eval(session=tf.compat.v1.Session()))
