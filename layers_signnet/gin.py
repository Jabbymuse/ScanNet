"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
# Tensorflow version
import tensorflow as tf
from layers_signnet.mlp import MLP

class GINConv(tf.keras.layers.Layer):
    def __init__(self,apply_func=None,aggregator_type='sum',init_eps=0,learn_eps=False,activation=None):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        self.aggregator_type = aggregator_type
        self.activation = activation
        if aggregator_type != 'sum':
            raise KeyError(
                'Aggregator type {} not recognized (cf not sum).'.format(aggregator_type))
        if learn_eps:
            self.eps = tf.Variable([init_eps], dtype=tf.float32) # if trainable
        else:
            self.eps = tf.constant([init_eps], dtype=tf.float32)

    def call(self,inputs,g=None, x=None):
        if self.aggregator_type == 'sum':
            aggregated = tf.reduce_sum(g, axis=2) # sum aggregation (center included) of neighbours, watch out axis value (fix it on the neighbors axis)
        else:
            raise ValueError('Aggregator type {} not recognized.'.format(self.aggregator_type))
        aggregated += self.eps * x
        outputs = aggregated
        if self.apply_func is not None:
            outputs = self.apply_func(aggregated)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


class GIN(tf.keras.layers.Layer):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, use_bn=True, dropout=0.5,activation='relu'):
        super(GIN, self).__init__()
        self.Layers = []
        if use_bn: self.bns = []
        self.use_bn = use_bn
        self.activation = tf.keras.activations.get(activation) # init activation function

        # Input layer
        update_net = MLP(in_channels, hidden_channels, hidden_channels, 2, use_bn=use_bn, dropout=dropout,
                         activation=activation)
        self.Layers.append(GINConv(update_net, 'sum'))

        # Hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(hidden_channels, hidden_channels, hidden_channels, 2, use_bn=use_bn, dropout=dropout,
                             activation=activation)
            self.Layers.append(GINConv(update_net, 'sum'))
            if use_bn: self.bns.append(tf.keras.layers.BatchNormalization())

        # Output layer
        update_net = MLP(hidden_channels, hidden_channels, out_channels, 2, use_bn=use_bn, dropout=dropout,
                         activation=activation)
        self.Layers.append(GINConv(update_net, 'sum'))
        if use_bn: self.bns.append(tf.keras.layers.BatchNormalization())

        self.dropout = tf.keras.layers.Dropout(rate=dropout) # prevents overfitting

    def call(self,inputs,g=None, x=None):
        for i, layer in enumerate(self.Layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if len(x.shape) == 2:
                        x = self.bns[i - 1](x)
                    elif len(x.shape) == 3:
                        x = tf.transpose(x, [0, 2, 1])
                        x = self.bns[i - 1](x)
                        x = tf.transpose(x, [0, 2, 1])
                    else:
                        raise ValueError('Invalid dimension of x')
            x = layer(None,g=g,x=x)
        return x