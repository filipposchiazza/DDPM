import numpy as np
import tensorflow as tf



class GaussianDiffusion:
    """Gaussian diffusion utilities.
    
    Args:
        beta_start : start value of the scheduled variance
        beta_end : end value of the scheduled variance
        timesteps : number of time steps in the forward process
    """
    
    def __init__(self,
                beta_start=1e-4,
                beta_end=0.02,
                timesteps=1000,
                clip_min=-1.0,
                clip_max=1.0):
        
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = int(timesteps)
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        # Variance linear schedule definition
        betas = np.linspace(start=beta_start,
                           stop=beta_end,
                           num=timesteps,
                           dtype=np.float64)
        
        alphas = 1 - betas
        alpha_cumprod = np.cumprod(alphas, axis=0)
        alpha_cumprod_prev = np.append(1.0, alpha_cumprod[:-1])
        
        # Conversion to tf Tensors
        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alpha_cumprod = tf.constant(alpha_cumprod)
        self.alpha_cumprod_prev = tf.constant(alpha_cumprod_prev)
        
        # Calculation for diffusion q(x_t | x_{t-1})
        sqrt_alphas_cumprod = np.sqrt(alpha_cumprod)
        self.sqrt_alphas_cumprod = tf.constant(sqrt_alphas_cumprod,
                                              dtype = tf.float32)
        
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0-alpha_cumprod)
        self.sqrt_one_minus_alphas_cumprod = tf.constant(sqrt_one_minus_alphas_cumprod,
                                                        dtype = tf.float32)
        
        log_one_minus_alphas_cumprod = np.log(1.0-alpha_cumprod)
        self.log_one_minus_alphas_cumprod = tf.constant(log_one_minus_alphas_cumprod,
                                                       dtype=tf.float32)
        
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alpha_cumprod)
        self.sqrt_recip_alphas_cumprod = tf.constant(sqrt_recip_alphas_cumprod,
                                                    dtype = tf.float32)
        
        sqrt_recip_m1_alphas_cumprod = np.sqrt(1.0 / self.alpha_cumprod - 1)
        self.sqrt_recip_m1_alphas_cumprod = tf.constant(sqrt_recip_m1_alphas_cumprod,
                                                       dtype=tf.float32)
        
        
        # Calculation for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.posterior_variance = tf.constant(posterior_variance,
                                             dtype=tf.float32)
        
        posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
        self.posterior_log_variance_clipped = tf.constant(posterior_log_variance_clipped,
                                                         dtype=tf.float32)
        
        posterior_mean_coef1 = betas * np.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.posterior_mean_coef1 = tf.constant(posterior_mean_coef1, 
                                               dtype = tf.float32)
        
        
        posterior_mean_coef2 = (1.0 - alpha_cumprod_prev) * np.sqrt(alphas) / (1.0 - alpha_cumprod)
        self.posterior_mean_coef2 = tf.constant(posterior_mean_coef2,
                                               dtype=tf.float32)
        
        
    def _extract(self, a, t, x_shape):
        """Extract some coefficients at specified timestep,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
            
        Args:
            a: Tensor to extract from
            t: timestep for which the coefficients are to be extracted
            x_shape: shape of the current batched samples
        """
        batch_size = x_shape[0]
        out = tf.gather(a, t)
        return tf.reshape(out, [batch_size, 1, 1, 1])
        
    # Extract mean and variance from N(x_t; c1*x_0, c2*eps)
    def q_mean_variance(self, x_start, t):
        """Extract the mean and the variance at the current timestep.
            
        Args:
            x_start: initial sample (before the first diffusion step)
            t: current timestep
                
        Returns:
            mean, variance and log variance of the current timestep
        """
        x_start_shape = tf.shape(x_start)
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(self.sqrt_one_minus_alphas_cumprod ** 2, t, x_start_shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start_shape)
        return mean, variance, log_variance
        
        
    # Evaluate x_t = c1 * x_0 + c2 * eps
    def q_sample(self, x_start, t, noise):
        """Diffuse the data.
        
        Args:
            x_start: initial sample (before the first diffusion step)
            t: current timestep
            noise: gaussian noise to be added at the current timestep
        Returns:
            diffused samples at timestep 't'
        """
        x_start_shape = tf.shape(x_start)
        c1 = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape)
        c2 = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape)
        x_t = c1 * x_start + c2 * noise
        return x_t 
        
        
    # Evaluate x_0 = c3 * x_t - c4 * eps
    def predict_start_from_noise(self, x_t, t, noise):
        """Evaluate x_0 = c3 * x_t - c4 * eps"""
        x_t_shape = tf.shape(x_t)
        c3 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape)
        c4 = self._extract(self.sqrt_recip_m1_alphas_cumprod, t, x_t_shape)
        x_0 = c3 * x_t - c4 * noise
        return x_0
        
        
    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and the variance of the diffusion 
            posterior q(x_{t-1} | x_t, x_0).
            
        Args:
            x_start: initial sample (before the first diffusion step)
            x_t: sample at timestep 't'
            t: current timestep
        Returns: 
            Posterior mean and variance at current timestep
        """
        x_t_shape = tf.shape(x_t)
        c1 = self._extract(self.posterior_mean_coef1, t, x_t_shape)
        c2 = self._extract(self.posterior_mean_coef2, t, x_t_shape)
        posterior_mean = c1 * x_start + c2 * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t_shape)
            
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        
        
        
        
    def p_mean_variance(self, pred_noise, x_t , t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x_t=x_t, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)
            
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon,
                                                                                     x_t = x_t,
                                                                                     t=t)
        return model_mean, posterior_variance, posterior_log_variance
            
            
    def p_sample(self, pred_noise, x_t, t, clip_denoised=True):
        """Sample from the diffusion model.
            
        Args:
            pred_noise: noise predicted by the diffusion model
            x_t: samples at a given timestep for which the noise was predicted
            t: current timestep
            clip_denoised(bool): whether to clip the predicted noise within the
                                     specified range or not.
        """
            
        model_mean, _, model_log_variance = self.p_mean_variance(pred_noise,
                                                                    x_t=x_t,
                                                                    t=t,
                                                                    clip_denoised=clip_denoised)
        noise = tf.random.normal(shape=x_t.shape, dtype=x_t.dtype)
        # No noise when t==0
        nonzero_mask = tf.reshape(1. - tf.cast(tf.equal(t, 0), tf.float32),
                                     [tf.shape(x_t)[0], 1, 1, 1])
            
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
