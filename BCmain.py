import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"


from bildingModelUtils import build_model
from gaussianDiffusionUtils import GaussianDiffusion
from DDPM import DiffusionModel

#%% HyperParameters

IMG_SIZE = 48
IMG_CHANNELS = 3
CLIP_MIN = -1.0
CLIP_MAX = 1.0

BATCH_SIZE = 32
NUM_EPOCHS = 800
LEARNING_RATE = 2e-4

T = 1000 # total timesteps
NORM_GROUPS = 8 # number of groups used in GroupNormalization layer

# Set the number of convolutional filters
FIRST_CONV_CHANNELS = 64
CHANNEL_MULTIPLIER = [1, 2, 4, 8]
WIDTHS = [FIRST_CONV_CHANNELS * ch for ch in CHANNEL_MULTIPLIER]
HAS_ATTENTION = [False, False, True, True]
NUM_RESIDUAL_BLOCKS = 2


#%% Image dataset preparation    
directory = './BreastCancerImages'   
    
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode=None,
    class_names=None,
    color_mode="rgb",
    batch_size=None,
    image_size=(50, 50),
    shuffle=True,
    seed=123,
    validation_split=0.9,
    subset='training',
    interpolation="bilinear"
)

def crop_and_normalize(image):
    img = tf.image.crop_to_bounding_box(image, 
                                        offset_height=1, 
                                        offset_width=1, 
                                        target_height=48, 
                                        target_width=48)
    img = img / 127.5 - 1.0
    return img

train_ds = train_ds.map(crop_and_normalize).batch(32)

#%%



network = build_model(img_size=IMG_SIZE,
                     img_channels=IMG_CHANNELS,
                     widths=WIDTHS,
                     first_conv_channel=FIRST_CONV_CHANNELS,
                     has_attention=HAS_ATTENTION, 
                     num_residual_blocks=NUM_RESIDUAL_BLOCKS,
                     norm_groups=NORM_GROUPS,
                     activation_fn=keras.activations.swish)

ema_network = build_model(img_size=IMG_SIZE,
                          img_channels=IMG_CHANNELS,
                          widths=WIDTHS,
                          first_conv_channel=FIRST_CONV_CHANNELS,
                          has_attention=HAS_ATTENTION, 
                          num_residual_blocks=NUM_RESIDUAL_BLOCKS,
                          norm_groups=NORM_GROUPS,
                          activation_fn=keras.activations.swish)

# Initially the weights are the same
ema_network.set_weights(network.get_weights()) 

gdf_util = GaussianDiffusion(timesteps=T)

model = DiffusionModel(network=network,
                      ema_network=ema_network,
                      gdf_util=gdf_util,
                      timesteps=T)

model.compile(loss=keras.losses.MeanSquaredError(),
             optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

#callbacks = [keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)]

model.fit(train_ds,
         epochs=NUM_EPOCHS,
         batch_size=BATCH_SIZE,
         callbacks=None)


num_images_to_gen = 30
generated_images = model.generate_images(num_images_to_gen)
generated_images = tf.clip_by_value(generated_images*127.5 + 127.5, clip_value_min=0.0, clip_value_max=255.0).numpy().astype(np.uint8)


for i in range(30):
    img = Image.fromarray(generated_images[i], 'RGB')
    img.save(f"./Generated/gen{i}.png")



