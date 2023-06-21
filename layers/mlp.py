import tensorflow as tf
from keras import initializers
tf.compat.v1.enable_eager_execution()
class MLP(tf.keras.Model):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, use_bn=False, use_ln=False, dropout=0.5, activation='relu', residual=False):
        super(MLP, self).__init__()
        self.lins = []
        if use_bn: self.bns = []
        if use_ln: self.lns = []
        if num_layers == 1:
            layer = tf.keras.layers.Dense(out_channels)
            self.lins.append(layer)
        else:
            self.lins.append(tf.keras.layers.Dense(hidden_channels))
            if use_bn: self.bns.append(tf.keras.layers.BatchNormalization()) # batch norm
            if use_ln: self.lns.append(tf.keras.layers.LayerNormalization()) # layer norm
            for layer in range(num_layers-2):
                self.lins.append(tf.keras.layers.Dense(hidden_channels))
                if use_bn: self.bns.append(tf.keras.layers.BatchNormalization())
                if use_ln: self.lns.append(tf.keras.layers.LayerNormalization())
            self.lins.append(tf.keras.layers.Dense(out_channels))
        if activation == 'relu':
            self.activation = tf.keras.layers.ReLU()
        elif activation == 'elu':
            self.activation = tf.keras.layers.ELU()
        elif activation == 'tanh':
            self.activation = tf.keras.layers.Activation('tanh')
        else:
            raise ValueError('Invalid activation')

        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual

    def call(self, x, training=False):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if len(x.shape) == 2:
                    x = self.bns[i](x, training=training)
                elif len(x.shape) == 3:
                    x = tf.transpose(x, [0, 2, 1])
                    x = self.bns[i](x, training=training)
                    x = tf.transpose(x, [0, 2, 1])
                else:
                    raise ValueError('Invalid dimension of x')
            if self.use_ln:
                x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape:
                x = x + x_prev
            x = tf.keras.layers.Dropout(rate=self.dropout)(x, training=training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x