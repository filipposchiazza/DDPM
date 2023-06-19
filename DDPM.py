import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from bildingModelUtils import build_model
from gaussianDiffusionUtils import GaussianDiffusion


class DiffusionModel(keras.Model):
    
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999, **kwargs):
        super().__init__(**kwargs)
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema
        
    
    
    def train_step(self, images):
        # 1. Get the batch size
        batch_size = tf.shape(images)[0]
        
        # 2. Sample timesteps uniformly
        t = tf.random.uniform(minval=0,
                             maxval=self.timesteps,
                             shape=(batch_size,),
                             dtype=tf.int64)
        
        with tf.GradientTape() as tape:
            # 3. Sample random noise to add to the images
            noise = tf.random.normal(shape=tf.shape(images),
                                    dtype=images.dtype)
            
            # 4. Diffuse the images with noise
            images_t = self.gdf_util.q_sample(images, t, noise)
        
            # 5. Pass the diffused images togheter with time steps to the network
            pred_noise = self.network([images_t, t], training=True)
        
            # 6. Loss evaluation
            loss = self.loss(noise, pred_noise)
        
        # 7. Get the gradients
        gradients = tape.gradient(loss, self.network.trainable_weights)
        
        # 8. Network weight's update
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        
        # 9. Updates the weight values for the network with EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # 10. Return loss values
        return {"loss":loss}
        
        
    
    def generate_images(self, num_images=16):
        
        # 1. Randomly sample noise (starting point for reverse process)
        img_shape = self.network.input_shape[0][1:]
        shape = np.concatenate(((num_images,), img_shape), axis=0)
        samples = tf.random.normal(shape=shape,
                                   dtype=tf.float32
        )       
        # 2. Sample from the model iteratively
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict([samples, tt], 
                                                  verbose=0, 
                                                  batch_size=num_images)
            samples = self.gdf_util.p_sample(pred_noise, 
                                             samples, 
                                             tt, 
                                             clip_denoised=True)
            
        # 3. Return generated samples
        return samples
    
    
    def plot_images(self, 
                    epoch=None, 
                    logs=None, 
                    num_rows=2, 
                    num_cols=8, 
                    figsize=(12, 5)):
        """Utility to plot images using the diffusion model during training."""
        generated_samples = self.generate_images(num_images=num_rows * num_cols)
        generated_samples = (tf.clip_by_value(generated_samples * 127.5 + 127.5, 0.0, 255.0).numpy().astype(np.uint8))

        _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        for i, image in enumerate(generated_samples):
            if num_rows == 1:
                ax[i].imshow(image)
                ax[i].axis("off")
            else:
                ax[i // num_cols, i % num_cols].imshow(image)
                ax[i // num_cols, i % num_cols].axis("off")

        plt.tight_layout()
        plt.show()
        
        
    def save_model(self, 
                   path,
                   widths,
                   first_conv_channel,
                   has_attention,
                   num_residual_blocks,
                   save_history=True):
        # Create folder
        dir_path = os.path.join('models', path)
        os.mkdir(dir_path)
        
        # save model hyperparameters
        img_size = self.network.input_shape[0][1]
        img_channels = self.network.input_shape[0][-1]
        timesteps = self.timesteps
        
        parameters = {'img_size': img_size,
                      'img_channels': img_channels,
                      'widths': widths,
                      'first_conv_channel': first_conv_channel,
                      'has_attention': has_attention,
                      'num_residual_blocks': num_residual_blocks,
                      'timesteps': timesteps}
        
        param_file_path = os.path.join(dir_path, 'parameters.pkl')
        with open(param_file_path, 'wb') as f:
            pickle.dump(parameters, f)
        
        # save weights
        network_weights_dir = os.path.join(dir_path, 'network_weights/network_weights')
        self.network.save_weights(network_weights_dir)
        
        ema_weights_dir = os.path.join(dir_path, 'ema_weights/ema_weights')
        self.ema_network.save_weights(ema_weights_dir)
        
        # save history
        if save_history == True:
            history = self.history.history
            hist_file_path = os.path.join(dir_path, 'history.pkl')
            with open(hist_file_path, 'wb') as f:
                pickle.dump(history, f)
            
    
    @classmethod
    def load_model(cls, save_folder):
        
        # Load and unpack parameters
        param_file = os.path.join(save_folder, 'parameters.pkl')
        with open(param_file, 'rb') as f:
            parameters = pickle.load(f)
        img_size = parameters['img_size']
        img_channels = parameters['img_channels']
        widths = parameters['widths']
        first_conv_channel = parameters['first_conv_channel']
        has_attention = parameters['has_attention']
        num_residual_blocks = parameters['num_residual_blocks']
        timesteps = parameters['timesteps']
        
        # Build networks and load weights
        network = build_model(img_size=img_size, 
                              img_channels=img_channels, 
                              widths=widths, 
                              first_conv_channel=first_conv_channel, 
                              has_attention=has_attention,
                              num_residual_blocks=num_residual_blocks,
                              activation_fn=keras.activations.swish)
        
        network_weights_path = os.path.join(save_folder, 'network_weights/network_weights')
        network.load_weights(network_weights_path)
        
        ema_network = build_model(img_size=img_size, 
                              img_channels=img_channels, 
                              widths=widths, 
                              first_conv_channel=first_conv_channel, 
                              has_attention=has_attention,
                              num_residual_blocks=num_residual_blocks,
                              activation_fn=keras.activations.swish)
        
        ema_weights_path = os.path.join(save_folder, 'ema_weights/ema_weights')
        ema_network.load_weights(ema_weights_path)
        
        # Build gdf util
        gdf_util = GaussianDiffusion(timesteps=timesteps)
        
        # Build model
        model = cls(network=network,
                    ema_network=ema_network,
                    gdf_util=gdf_util,
                    timesteps=timesteps)
        
        model.compile(loss=keras.losses.MeanSquaredError())
        
        return model
    
    @classmethod
    def load_history(cls, save_folder):
        
        history_file = os.path.join(save_folder, 'history.pkl')
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
        
        return history
        

    
    

