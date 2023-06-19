import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#%% Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(scale,
                                             mode='fan_avg',
                                             distribution='uniform')


#%% Attention block

class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.BatchNormalization()
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj
    
    
#%% Time embedding

class TimeEmbedding(layers.Layer):
    
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim 
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)
    
    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb
    
#%% Residual Block

def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    
    def apply(inputs):
        x, t=inputs
        input_width = x.shape[3]
        
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(filters=width,
                                    kernel_size=1,
                                    kernel_initializer=kernel_init(1.0))(x)
            
        temb = activation_fn(t)
        temb = layers.Dense(width, 
                            kernel_initializer=kernel_init(1.0))(temb)[:, None, None, :]
        
        x = layers.BatchNormalization()(x)
        x = activation_fn(x)
        x = layers.Conv2D(filters=width,
                         kernel_size=3,
                         padding='same',
                         kernel_initializer=kernel_init(1.0))(x)
        
        x = layers.Add()([x, temb])
        x = layers.BatchNormalization()(x)
        x = activation_fn(x)
        
        x = layers.Conv2D(filters=width,
                         kernel_size=3,
                         padding='same',
                         kernel_initializer=kernel_init(1.0))(x)
        x=layers.Add()([x, residual])
        
        return x
    
    return apply

#%%
def DownSample(width):
    def apply(x):
        x = layers.Conv2D(filters=width,
                         kernel_size=3,
                         strides=2,
                         padding='same',
                         kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply

#%%
def Upsample(width, interpolation='nearest'):
    def apply(x):
        x = layers.UpSampling2D(size=2, 
                                interpolation=interpolation)(x)
        x = layers.Conv2D(filters=width,
                         kernel_size=3,
                         strides=1,
                         padding='same',
                         kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply

#%%
def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply

#%%
def build_model(img_size,
               img_channels,
               widths,
               first_conv_channel,
               has_attention,
               num_residual_blocks=2,
               norm_groups=8,
               interpolation='nearest',
               activation_fn=keras.activations.swish):
    
    # Image x_t and time t as input
    image_input = layers.Input(shape=(img_size, img_size, img_channels), 
                               name='image_input')
    time_input = layers.Input(shape=(),
                             dtype=tf.int64,
                             name='time_input')
    
    # First convolution of the image
    x = layers.Conv2D(filters=first_conv_channel,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_initializer=kernel_init(1.0))(image_input)
    
    # Time embedding
    temb = TimeEmbedding(dim=first_conv_channel * 4)(time_input)
    temb = TimeMLP(units=first_conv_channel * 4,
                  activation_fn=activation_fn)(temb)
    
    skips = [x]
    
    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_residual_blocks):
            x = ResidualBlock(width=widths[i],
                             groups=norm_groups,
                             activation_fn=activation_fn)([x, temb])
            
            if has_attention[i] == True:
                x = AttentionBlock(widths[i], 
                                   groups=norm_groups)(x)
            
            skips.append(x)
                
        if widths[i] != widths[-1]:
            x = DownSample(width=widths[i])(x)
            skips.append(x)
    
    # MiddleBlock
    x = ResidualBlock(width=widths[-1],
                     groups=norm_groups,
                     activation_fn=activation_fn)([x, temb])
    
    x = AttentionBlock(widths[-1],
                      groups=norm_groups)(x)
     
    x = ResidualBlock(widths[-1], 
                      groups=norm_groups, 
                      activation_fn=activation_fn)([x, temb])
    
    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_residual_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            
            x = ResidualBlock(width=widths[i],
                             groups=norm_groups,
                             activation_fn=activation_fn)([x, temb])
            
            if has_attention[i] == True:
                x = AttentionBlock(widths[i],
                                  groups=norm_groups)(x)
                
        if i != 0:
            x = Upsample(width=widths[i],
                         interpolation=interpolation)(x)
            
    # EndBlock
    x = layers.BatchNormalization()(x)
    x = activation_fn(x)
    x = layers.Conv2D(filters=3,
                     kernel_size=(3, 3),
                     padding='same',
                     kernel_initializer=kernel_init(0.0))(x)
    
    return keras.Model([image_input, time_input], x, name='unet')
    
    
    