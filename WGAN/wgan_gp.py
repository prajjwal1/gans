from tensorop import *  # For basic functionality
from tensorop.vision.gan import * # Contains GAN specifc modules

path = 'data' # Should contain a directory which contains images
bs,img_size=32,64             # Batch Size, Image Size    
tfms = get_tfms_gan(img_size) # Transformations 
data = get_img_dl(path,tfms,bs)  # Returns a dataloader

#Loads up Standard generator and critic arch, custom can also be used
generator = std_generator(img_size=img_size,num_channels=num_channels,sz_latent_vec=sz_latent_vec,num_fmap_gen=num_fmap_gen).to(device)
critic = std_critic(num_channels=num_channels,num_fmap_crit=num_fmap_crit).to(device)

WGAN = WGAN_GP(generator,critic,data) 
WGAN.train()
img_list,gan_loss,critic_loss = WGAN.get_loss() 


