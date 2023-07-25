from layers_signnet.mlp import MLP
from layers_signnet.gin import GIN
import tensorflow as tf
from keras.engine.base_layer import Layer

from keras.layers import Input, Dense, Masking, TimeDistributed, Concatenate, Activation, Embedding, Dropout, Lambda,Add
from keras.initializers import Zeros, Ones, RandomUniform
from . import embeddings

def GINDeepSigns(input_graph, hidden_channels, num_layers, use_bn=False, dropout=0.5, activation='relu',epsilon=0.5):

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

    def minus(g,i):
        Naa = g.shape[1]
        """compute g_minus """
        # Motion to multiply
        g_sliced = tf.transpose(g,[4,1,3,2,0]) # [6,Naa,m,K,B]
        g_sliced_i = tf.strided_slice(g_sliced,begin=[0,0,i],end=[6,Naa,i+1]) # [6,Naa,K,1,B]
        g_minus_i = tf.multiply(g_sliced_i,-1)
        g_minus_i = tf.transpose(g_minus_i,[4,1,3,2,0]) # [B,Naa,K,1,6]
        # left leftover
        g_sliced_left = tf.strided_slice(g_sliced,begin=[0,0,0],end=[6,Naa,i]) # [B,Naa,K,i,6]
        g_sliced_left = tf.transpose(g_sliced_left,[4,1,3,2,0]) # [B,Naa,K,i,6]
        # right leftover
        g_sliced_right = tf.strided_slice(g_sliced, begin=[0,0,i+1], end=[6,Naa,m])  # [B,Naa,K,m-i-1,6]
        g_sliced_right = tf.transpose(g_sliced_right, [4,1,3,2,0])  # [B,Naa,K,m-i-1,6]
        return Concatenate(axis=3)([g_sliced_left,g_minus_i,g_sliced_right])


    m = input_graph.shape[3]
    Naa = input_graph.shape[1]
    layer_sizes = [hidden_channels] * num_layers
    # init of enc (GIN)
    GIN_layers = init_GIN(epsilon=epsilon,n_layers=num_layers,layer_sizes=layer_sizes,use_bn=use_bn,activation=activation,dropout=dropout)
    list_x = []
    enc_vectors = apply_list_layers(input_graph, GIN_layers, ndim_input=5)  # [B,Naa,m,6]
    for i in range(m):
        g_minus = Lambda(minus,arguments={'i':i},name='G_minus')(input_graph) # multiply ith motion by -1
        enc_neg_vectors = apply_list_layers(g_minus,GIN_layers,ndim_input=5) # [B,Naa,K,m,6]
        enc_sign_invariant_vectors = Add()([enc_vectors, enc_neg_vectors])# enc(vi) + enc(-vi)
        enc_sign_invariant_vectors = Lambda(lambda x : tf.transpose(x,[4,1,2,3,0]),name='First_transpose_GIN')(enc_sign_invariant_vectors) # [6,Naa,K,m,B]
        enc_sign_invariant_vectors = Lambda(lambda x: tf.strided_slice(x,begin=[0,0,0,i],end=[6,Naa,1,i+1]),name='Strided_Slice_GIN')(enc_sign_invariant_vectors) # [6,Naa,1,1,B]
        enc_sign_invariant_vectors = Lambda(lambda x : tf.transpose(x,[4,1,2,3,0]),name='Second_transpose_GIN')(enc_sign_invariant_vectors) # [B,Naa,1,1,6]
        list_x.append(Lambda(lambda x: tf.reshape(x,shape=[-1,Naa,1,6]),name='Reshape_GIN')(enc_sign_invariant_vectors)) # [B,Naa,1,6]
    if len(list_x) > 1: # to treat the case where there is just one motion vector
        sign_invariant_vectors = Concatenate(axis=2)(list_x)  # [B,Naa,m,6]
    else:
        sign_invariant_vectors = list_x[0]
    MLP = init_MLP(layer_sizes, use_bn, activation)
    sign_invariant_vectors = Lambda(lambda x: apply_list_layers(x,MLP,ndim_input=4),name='Last_MLP_GIN')(sign_invariant_vectors)
    return sign_invariant_vectors
