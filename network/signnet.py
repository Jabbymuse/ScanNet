from layers_signnet.mlp import MLP
from layers_signnet.gin import GIN
import tensorflow as tf
from keras.engine.base_layer import Layer

from keras.layers import Input, Dense, Masking, TimeDistributed, Concatenate, Activation, Embedding, Dropout, Lambda,Add
from keras.initializers import Zeros, Ones, RandomUniform
from . import embeddings

def GINDeepSigns(input_graph, in_channels, hidden_channels, out_channels, num_layers, k, use_bn=False, use_ln=False, dropout=0.5, activation='relu',epsilon=0.5):

    def init_MLP(layer_sizes=[64,32,16], use_bn=True, activation=None):
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
                list_layers.append(embeddings.MaskedBatchNormalization(
                epsilon=1e-3, axis=-1, center=center, scale=scale, name='GIN_MLP_normalization_%s'%k) )
            list_layers.append(TimeDistributed(Activation(activation), name='GIN_MLP_activation_%s'%k) )
        return list_layers

    def update_graph(g,epsilon):
        """ compute the aggregation and the update of the input graph g """
        return tf.concat([tf.expand_dims(tf.reduce_sum(g,axis=2) + epsilon * g[:,:,0],axis=2),g[:,:,1:]],axis=2)

    def init_GIN(epsilon,n_layers,layer_sizes=None,use_bn=True,activation=None,dropout=None):
        """ first version without bn"""
        Gin_layers = []
        # input layer
        update_layer = Lambda(update_graph,arguments={'epsilon':epsilon})
        Gin_layers.append(update_layer)
        MLP = init_MLP(layer_sizes=layer_sizes,use_bn=use_bn,activation=activation)
        Gin_layers += MLP
        #hidden layers
        for k in range(n_layers-2):
            Gin_layers.append(Dropout(rate=dropout))
            update_layer = Lambda(update_graph, arguments={'epsilon': epsilon})
            Gin_layers.append(update_layer)
            MLP = init_MLP(layer_sizes=layer_sizes, use_bn=use_bn, activation=activation)
            Gin_layers += MLP
        #output layer
        Gin_layers.append(Dropout(rate=dropout))
        update_layer = Lambda(update_graph, arguments={'epsilon': epsilon})
        Gin_layers.append(update_layer)
        MLP = init_MLP(layer_sizes=layer_sizes, use_bn=use_bn, activation=activation)
        Gin_layers += MLP
        Gin_layers.append(Dropout(rate=dropout))
        return Gin_layers

    def apply_list_layers(input , list_layers,ndim_input=3):
        intermediate_output = input
        for layer in list_layers:
            if ndim_input >3:
                layer_ = layer
                for _ in range(ndim_input-3):
                    layer_ = TimeDistributed(layer_)
            intermediate_output = layer_(intermediate_output)
        return intermediate_output
    """
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
    """

    def minus(g,i):
        """compute g_minus """
        g_minus_i = tf.expand_dims(tf.multiply(g[:,:,:,i,:], -1),axis=3)
        return tf.concat([g[:, :, :, :i, :], g_minus_i, g[:, :, :, i+1:, :]], axis=3)
    # version finale

    m = input_graph.shape[3]
    layer_sizes = [hidden_channels] * num_layers
    # init of enc (GIN)
    GIN_layers = init_GIN(epsilon=epsilon,n_layers=num_layers,layer_sizes=layer_sizes,use_bn=use_bn,activation=activation,dropout=dropout)
    list_x = []
    enc_vectors = apply_list_layers(input_graph, GIN_layers, ndim_input=5)  # [B,Naa,m,6]
    for i in range(m):
        g_minus = Lambda(minus,arguments={'i':i})(input_graph) # multiply motion i by -1
        enc_neg_vectors = apply_list_layers(g_minus,GIN_layers,ndim_input=5) # [B,Naa,m,6]
        enc_sign_invariant_vectors = Add()([enc_vectors, enc_neg_vectors]) # enc(vi) + enc(-vi)
        list_x.append(Lambda(lambda x: tf.expand_dims(x[:,:,0,i],axis=2))(enc_sign_invariant_vectors))
    sign_invariant_vectors = Concatenate(axis=2)(list_x)  # [B,Naa,m,6]
    MLP = init_MLP(layer_sizes, use_bn, activation)
    sign_invariant_vectors = Lambda(lambda x: apply_list_layers(x,MLP,ndim_input=4))(sign_invariant_vectors)
    return sign_invariant_vectors

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
