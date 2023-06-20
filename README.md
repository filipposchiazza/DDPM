1) **Denoising Diffusion Probabilistic Models** (Dec 2020):

 https://arxiv.org/pdf/2006.11239.pdf 
 
 https://github.com/hojonathanho/diffusion
 
4) **Improved Denoising Diffusion Probabilistic Models** (Feb 2021):

 https://arxiv.org/pdf/2102.09672.pdf

 https://github.com/openai/improved-diffusion
 
* Reverse process variances are learnt rather than kept fixed
* A cosine schedule is introduced for $\bar{\alpha}_t$ rather than a linear schedule for $\beta_t$
* Importance sampling is added for the loss terms $L_t$
* The number of time steps is reduced for the generation process
  

