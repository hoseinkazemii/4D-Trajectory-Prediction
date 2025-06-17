from tensorflow.keras.layers import Layer, Conv1D, SpatialDropout1D, Activation, add, Lambda
from tensorflow.keras.regularizers import l2

class ResidualBlock(Layer):
    def __init__(self, dilation_rate, nb_filters, kernel_size, padding, dropout_rate, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout_rate = dropout_rate
        
        self.conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                            dilation_rate=dilation_rate, padding=padding, kernel_regularizer=l2(1e-4))
        self.conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                            dilation_rate=dilation_rate, padding=padding, kernel_regularizer=l2(1e-4))
        self.dropout1 = SpatialDropout1D(dropout_rate)
        self.dropout2 = SpatialDropout1D(dropout_rate)
        self.activation1 = Activation('relu')
        self.activation2 = Activation('relu')
        
        self.downsample = Lambda(lambda x: x)
        
        self.add = add
    
    def build(self, input_shape):
        if input_shape[-1] != self.nb_filters:
            self.downsample = Conv1D(self.nb_filters, kernel_size=1, padding=self.padding)
        super(ResidualBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        residual = self.downsample(inputs)
        
        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.dropout2(x, training=training)
        
        x = self.add([x, residual])
        return x
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'dilation_rate': self.dilation_rate,
            'nb_filters': self.nb_filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'dropout_rate': self.dropout_rate
        })
        return config

class TCNLayer(Layer):
    def __init__(self, nb_filters, kernel_size, nb_stacks, dilations, 
                 padding, use_skip_connections, dropout_rate, 
                 return_sequences, **kwargs):
        super(TCNLayer, self).__init__(**kwargs)

        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.dilations = dilations
        self.padding = padding
        self.use_skip_connections = use_skip_connections
        self.dropout_rate = dropout_rate
        self.return_sequences = return_sequences
        
        self.residual_blocks = []
        
        for s in range(nb_stacks):
            for d in dilations:
                self.residual_blocks.append(
                    ResidualBlock(
                        dilation_rate=d,
                        nb_filters=nb_filters,
                        kernel_size=kernel_size,
                        padding=padding,
                        dropout_rate=dropout_rate,
                        name=f'residual_block_{s}_{d}'  
                    )
                )
        
        self.activation = Activation('relu')
    
    def call(self, inputs, training=None):
        x = inputs
        
        if self.use_skip_connections and not self.return_sequences:
            skip_connections = []
        
        for block in self.residual_blocks:
            x = block(x, training=training)
            if self.use_skip_connections and not self.return_sequences:
                skip_connections.append(x)

        if self.use_skip_connections and not self.return_sequences:
            x = add(skip_connections)
            x = self.activation(x)

        if not self.return_sequences:
            x = Lambda(lambda tt: tt[:, -1, :])(x)
        
        return x
    
    def get_config(self):
        config = super(TCNLayer, self).get_config()
        config.update({
            'nb_filters': self.nb_filters,
            'kernel_size': self.kernel_size,
            'nb_stacks': self.nb_stacks,
            'dilations': self.dilations,
            'padding': self.padding,
            'use_skip_connections': self.use_skip_connections,
            'dropout_rate': self.dropout_rate,
            'return_sequences': self.return_sequences
        })
        return config