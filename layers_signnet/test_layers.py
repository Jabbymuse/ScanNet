# a module to test our programs together


# Tensorflow stuff
from gin import GINConv
import tensorflow as tf
tf.compat.v1.enable_eager_execution() # to print tensors
"""
# Torch stuff
import torch as th
import dgl

Naa = 4 # nb of amino_acids
m = 2 # nb of motions per aa
K = 2 # nb of closest neighbors

# test of GINConv
print("test de GINConv =========================")
# creation of input
M = [[[1 for _ in range(6)] for _ in range(m)] for _ in range(Naa)]
M[0][1][3] *= 2

input = tf.convert_to_tensor(M,dtype=tf.float64) # conversion to tensor
#print("input = ",input)

# creation of graph (closest neighbors)
G = [[[[1 for _ in range(6)] for _ in range(m)] for _ in range(K)] for _ in range(Naa)]
G[0][0] = M[0]
G[1][1] = M[0]

g = tf.convert_to_tensor(G,dtype=tf.float64)
#print("neighborhood =",g)

# apply filter
conv = GINConv(None,'sum',init_eps=0.5)
res = conv(g,input)
#print("res_Tens = ",res)
"""
"""
# Check with pytorch
from dgl.nn.pytorch import GINConv as GINConv_t
input_torch = th.FloatTensor(M)
g_torch = dgl.graph(([0,1,2,3],[1,0,3,2]))
conv_torch = GINConv_t(None,'sum',init_eps=0.5)
#print(conv_torch(g_torch,input_torch))
"""

"""
# test of MLP
# Tensorflow
from mlp import MLP

M = [[[1 for _ in range(6)] for _ in range(m)] for _ in range(Naa)]
input = tf.convert_to_tensor(M,dtype=tf.float32) # conversion to tensor
#print("input = ", input)
mlp = MLP(6,6,6,1) # choosing the same number of units per layer seems to work well
res = mlp(input)
#print("res = ",res)

# Torch
#from gin_torch import MLP as MLP_t

weights = []
for _ in range(6):
    weights += [[1 for _ in range(6)]]

input_torch = th.FloatTensor(M)
#mlp1 = MLP_t(6,6,6,1)
#print(mlp1(input_torch))

# obj : initialize values (seed) !!!!
# tests of dense layer

print("TEST DENSE LAYER ================================")
import numpy as np
from keras import initializers

# Tensorflow
# Comparison between pytorch and tensorflow version with same weights and bias

dense_layer = tf.keras.layers.Dense(units=6,kernel_initializer=initializers.glorot_uniform(seed=0)) # why expecting 0 weights ?????
dense_layer.build(tf.TensorShape([4,2,6])) # because not build ...
#print(dense_layer.get_weights())
"""
"""
class CustomDense(tf.keras.layers.Dense):
    def build(self, input_shapes:tf.TensorShape, param_values:np.array):
        super().build(input_shapes)
        super().weights = tf.convert_to_tensor(param_values)

layer = CustomDense(units=6)
layer.build(tf.TensorShape([4,2,6]),custom_weights)
print("lw= ",layer.get_weights())
"""
"""
import torch.nn as nn
th.manual_seed(0)
in_channels = 6
out_channels = 6
l = nn.Linear(in_channels,out_channels)
w = [[-0.56446546,  0.6649162 ,  0.49322706, -0.63882667, -0.6384848 ,
         0.39240843],
       [ 0.48715132, -0.11727375,  0.01408327,  0.21948951,  0.6903494 ,
        -0.1881054 ],
       [-0.17267853,  0.27037942,  0.7006635 , -0.04778272,  0.69466716,
         0.27904773],
       [ 0.38769346,  0.5489499 , -0.6526006 ,  0.6723098 , -0.38149068,
         0.47858185],
       [ 0.41758603,  0.6175992 ,  0.32128042,  0.6347577 , -0.30923697,
         0.57425195],
       [-0.1905973 ,  0.48086447,  0.25321722, -0.41863993, -0.4793329 ,
        -0.02794641]]
bias = [0,0,0,0,0,0]

#w = th.empty(6, 6) # init of kernel with xavier_uniform distrib
#nn.init.xavier_uniform_(w)
with th.no_grad():
    l.weight.copy_(th.FloatTensor(w))
    l.bias.copy_(th.FloatTensor(bias))
    #print(l.weight)
    #print(l.bias)

# Comparison
#print(input)
#print(dense_layer(input))
#print(input_torch)
#print(l(input_torch))

# results have similar behaviours but remain diff
"""

# Tests of Siggnet architecture
print("Test of Signnet Arch #####################################################################")

#print(tf.__version__)
import numpy as np
from random import uniform as ru
from random import choice as ch
from network.signnet import GINDeepSigns as GIND

Naa = 3 # nb of aa
m = 5 # nb of motions (cf eigen vectors)
K = 4
B = 2 # batch dim

# tests with simple values
# we consider the K closest neighbors
input = tf.convert_to_tensor([[[[1 for _ in range(6)] for _ in range(m)] for _ in range(Naa)] for _ in range(B)],dtype=tf.float32) # conversion to tensor
G = [[[[[1 for _ in range(6)] for _ in range(m)] for _ in range(K)] for i in range(Naa)] for _ in range(B)]
g = tf.convert_to_tensor(G,dtype=tf.float32)
net = GIND(6,6,6,1,m)
out = net(g,input)
out2 = net(-g,-input)

#print(out == out2) # returns False interesting, while out - out2 is null Tensor...
#print(out - out2)


# tests with more specific values (cf real graph)
M_qq = [[[[ru(5,-5) for _ in range(6)] for _ in range(m)] for _ in range(Naa)] for _ in range(B)]
input = tf.convert_to_tensor(M_qq,dtype=tf.float32)
#print(input)

G = []
for b in range(B):
    g = []
    for i in range(Naa):
        l = [M_qq[b][i]] # each vector is its own neighbor
        seq = list(range(Naa))
        for _ in range(K-1):
            index_v = ch(seq)
            l.append(M_qq[b][index_v])
        g.append(l)
    G.append(g)

g = tf.convert_to_tensor(G,dtype=tf.float32)
#print("g=",g)

out = net(g,input)
out2 = net(-g,-input)

#print(out)
#print(out - out2)

# tests minus for phi
# sign flip on a specific ith motion
g_minus = tf.expand_dims(tf.multiply(g[:,:,:,1,:], -1),axis=3)
g_minus = tf.concat([g[:, :, :, :1, :], g_minus, g[:, :, :, 2:, :]], axis=3)

#print(g+g_minus)

input_minus = tf.expand_dims(tf.multiply(input[:,:,1,:], -1),axis=2)
input_minus = tf.concat([input[:, :, :1, :], input_minus, input[:, :, 2:, :]], axis=2)
out3 = net(g_minus,input_minus)

#print(out-out3)

# Test of rot equivariance
"""
Rot = np.matrix([[1,0,0],
               [0,np.cos(0.3),-np.sin(0.3)],
               [0,np.sin(0.3),np.cos(0.3)]])

v = np.array([1,2,3])
v1 = np.dot(v,Rot)

# check if norms are equal
#print(np.linalg.norm(v1) == np.linalg.norm(v))

# Rotate each rot part of eigen_vectors of a particular motion, for example the first one
for i in range(Naa):
    M = np.dot(np.array(M_qq[i][0][:3]),Rot)
    for j in range(3):
        M_qq[i][0][j] = M[j]

print(M_qq)
"""







