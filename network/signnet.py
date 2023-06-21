from layers.mlp import MLP
from layers.gin import GIN
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

class GINDeepSigns(tf.keras.Model):
    """Sign invariant neural network
       f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, k, use_bn=False, use_ln=False, dropout=0.5, activation='relu'):
        super(GINDeepSigns, self).__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers, use_bn=use_bn, dropout=dropout, activation=activation)
        rho_dim = out_channels * k
        self.rho = MLP(rho_dim, hidden_channels, out_channels, num_layers, use_bn=use_bn, dropout=dropout, activation=activation)
        #self.k = k

    def call(self, g, x): # x are the eigenvectors and g the neigbhours of the amino_acid (both tensors)
        x = self.enc(g, x) + self.enc(-g, -x) # cf the sign of all the vectors in the graph is changing when x -> -x ... so g -> -g (neighbors are all in input files)
        #orig_shape = tf.shape(x)
        #x = tf.reshape(x, (orig_shape[0], -1))
        x = self.rho(x)
        #x = tf.reshape(x, (orig_shape[0], self.k, 1))
        return x