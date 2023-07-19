from layers_signnet.mlp import MLP
from layers_signnet.gin import GIN
import tensorflow as tf
from keras.engine.base_layer import Layer

from keras.layers import Input, Dense, Masking, TimeDistributed, Concatenate, Activation, Embedding, Dropout, Lambda,Add
from keras.initializers import Zeros, Ones, RandomUniform
from . import embeddings

def GINDeepSigns(input_graph, in_channels, hidden_channels, out_channels, num_layers, k, use_bn=False, use_ln=False, dropout=0.5, activation='relu'):

    def init_MLP(layer_sizes=[64,32,16], use_bn=True, activation=None ):
        if activation == 'tanh':
            center = True
            scale = True
        elif 'multitanh' in activation:  # e.g. 'multitanh5'
            center = False
            scale = False
        elif activation == 'relu':
            center = True
            scale = False
        elif activation == 'elu':
            center = True
            scale = True
        elif activation in [None, 'linear']:
            center = False
            scale = False
        else:
            center = True
            scale = True
        list_layers = []
        for k,layer_size in enumerate(layer_sizes):
            list_layers.append(TimeDistributed(Dense(layer_size, use_bias=False, activation=None),
                                                name='GIN_MLP_projection_%s'%k) )
            if use_bn:
                list_layers.append( embeddings.MaskedBatchNormalization(
                epsilon=1e-3, axis=-1, center=center, scale=scale, name='GIN_MLP_normalization_%s'%k) )
            list_layers.append( TimeDistributed(Activation(activation), name='GIN_MLP_activation_%s'%k) )
        return list_layers

    def apply_list_layers(input , list_layers,ndim_input=3):
        intermediate_output = input
        for layer in list_layers:
            if ndim_input >3:
                layer_ = layer
                for _ in range(ndim_input-3):
                    layer_ = TimeDistributed(layer_)
            intermediate_output = layer_(intermediate_output)
        return intermediate_output

    layer_sizes = [hidden_channels] * num_layers
    list_encoder_layers = init_MLP(layer_sizes=layer_sizes, use_bn=use_bn, activation=activation)
    list_mlp_layers = init_MLP(layer_sizes=layer_sizes, use_bn=use_bn, activation=activation)

    enc_vectors = apply_list_layers(input_graph, list_encoder_layers,ndim_input=5)

    neg_input_graph = Lambda(lambda x: -x)(input_graph)
    enc_neg_vectors = apply_list_layers(neg_input_graph, list_encoder_layers, ndim_input=5)

    enc_sign_invariant_vectors = Add()([enc_vectors,enc_neg_vectors]) # [B, Naa, K, m, dim_enc]
    enc_sign_invariant_vectors = Lambda(lambda x: tf.reshape(x,[-1,x.shape[1], x.shape[2], x.shape[3]*x.shape[4]]) )(enc_sign_invariant_vectors) # [B,Naa,K,m*dim_enc])
    intermediate_output =  apply_list_layers(enc_sign_invariant_vectors,  list_mlp_layers, ndim_input = 4)
    output_embedding = Lambda(lambda x: tf.reduce_mean(x, axis=2))(intermediate_output) # [B,Naa,m*dim_enc]
    return output_embedding

'''
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
'''
