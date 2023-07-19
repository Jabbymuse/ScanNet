from layers_signnet.mlp import MLP
from layers_signnet.gin import GIN
import tensorflow as tf

class GINDeepSigns(tf.keras.layers.Layer):
    """Sign invariant neural network
       f(v1, ..., vm) = rho(enc(v1) + enc(-v1), ..., enc(vm) + enc(-vm))
       x = (v1,...,vm)
       g is the adjacency graph (cf neighborhood)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, k, use_bn=False, use_ln=False, dropout=0.5, activation='relu'):
        super(GINDeepSigns, self).__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers, use_bn=use_bn, dropout=dropout, activation=activation)
        rho_dim = out_channels * k
        self.rho = MLP(rho_dim, hidden_channels, out_channels, num_layers, use_bn=use_bn, dropout=dropout, activation=activation)
        self.k = k

    def call(self, inputs, g = None): # x are the eigenvectors and g the neigbhours of the amino_acid (both tensors)
        vectors = g[:,:,0]
        m = g.shape[3] # g has shape B * Naa * K * m * 6
        list_x = [] # list of modified columns for m motions
        for i in range(m):
            # creation of g_minus and x_minus at index i
            g_minus = tf.expand_dims(tf.multiply(g[:,:,:,i,:], -1),axis=3)
            g_minus = tf.concat([g[:, :, :, :i, :], g_minus, g[:, :, :, i+1:, :]], axis=3)
            vectors_minus = g_minus[:,:,0]
            list_x.append(tf.expand_dims((self.enc(None,g=g,x=vectors)+self.enc(None,g=g_minus,x=vectors_minus))[:,:,i],axis=2)) # list of [B,Naa,1,6]
        return self.rho(tf.concat(list_x,axis=2)) # [B,Naa,m,6] and then MLP