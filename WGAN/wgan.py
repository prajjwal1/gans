from tensorop import *
from tensorop.vision import * 
from tensorop.vision.gan import * 
path,bs,img_size = 'data',32,64                             # Should contain a directory which contains images, # Batch Size, Image Size                                         
tfms = get_tfms_gan(img_size)                               # Transformations 
data = get_img_dl(path,tfms,bs)                             # Returns a dataloader
generator,critic = GAN_Model('standard') #Uses default parameters : num_channels,sz_latent_vec,num_fmap_gen,num_fmap_crit = 3,100,64,64 
WGAN = WGAN(generator,critic,data) 
WGAN.train()


