
# coding: utf-8

# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')


# In[17]:


from fastai.conv_learner import *
from fastai.dataset import *
import gzip
torch.cuda.set_device(0)


# In[18]:


PATH = Path('data/lsun/')
IMG_PATH = PATH/'bedroom'
CSV_PATH = PATH/'files.csv'
TMP_PATH = PATH/'tmp'
TMP_PATH.mkdir(exist_ok=True)


# In[19]:


files = PATH.glob('bedroom/**/*.jpg')

with CSV_PATH.open('w') as fo:
    for f in files: fo.write(f'{f.relative_to(IMG_PATH)},0\n')


# In[20]:


CSV_PATH = PATH/'files_sample.csv'


# In[21]:


files = PATH.glob('bedroom/**/*.jpg')

with CSV_PATH.open('w') as fo:
    for f in files:
        if random.random()<0.1: fo.write(f'{f.relative_to(IMG_PATH)},0\n')


# In[68]:


class ConvBlock(nn.Module):
    def __init__(self,ni,no,kernel_size,stride,batch_norm=True,padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size//2//stride
        self.conv = nn.Conv2d(ni,no,kernel_size,stride,padding=padding,bias=False)
        self.batch_norm = nn.BatchNorm2d(no) if batch_norm else None
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        
    def forward(self,x):
        x = self.relu(self.conv(x))
        return self.batch_norm(x) if self.batch_norm else x
        


# In[84]:


class dcgan_discriminator(nn.Module):
    def __init__(self,in_size,num_channels,ndf,extra_layers=0):
        super().__init__()
        assert in_size % 16 == 0
        
        self.initial = ConvBlock(num_channels,ndf,4,2,batch_norm=False)
        csize,cndf = in_size/2,ndf
        self.extra = nn.Sequential(*[ConvBlock(cndf,cndf,3,1)
                                     for t in range(extra_layers)])
        pyr_layers = []
        
        while csize > 4:
            pyr_layers.append(ConvBlock(cndf,cndf*2,4,2))
            cndf*=2
            csize/=2
        self.pyramid = nn.Sequential(*pyr_layers)
        
        self.final = nn.Conv2d(cndf,1,4,padding=0,bias=False)
        
    def forward(self,input):
            x = self.initial(input)
            x = self.extra(x)
            x = self.pyramid(x)
            return self.final(x).mean(0).view(1)          


# In[85]:


class deconvolution_block(nn.Module):
    def __init__(self,ni,no,kernel_size,stride,padding,batch_norm=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ni,no,kernel_size,stride,padding=padding,bias=False)
        self.batch_norm = nn.BatchNorm2d(no)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        x = self.relu(self.conv(x))
        return self.batch_norm(x) if self.batch_norm else x


# In[86]:


class dcgan_generator(nn.Module):
    def __init__(self,in_size,nz,nc,ngf,extra_layers=0):
        super().__init__()
        assert in_size % 16 == 0
        
        cngf,tisize = ngf//2, 4
        while tisize!=in_size:
            cngf*=2
            tisize*=2
        layers = [deconvolution_block(nz,cngf,4,1,0)]
        csize,cndf=4,cngf
        while csize<in_size//2:
            layers.append(deconvolution_block(cngf,cngf//2,4,2,1))
            cngf//=2
            csize*=2
        
        layers+= [deconvolution_block(cngf,cngf,3,1,1) for t in range(extra_layers)]
        layers.append(nn.ConvTranspose2d(cngf,nc,4,2,1,bias=False))
        self.features = nn.Sequential(*layers)
        
    def forward(self,input):
        return F.tanh(self.features(input))


# In[100]:


batch_size,sz,nz = 64,64,100


# In[88]:


tfms = tfms_from_stats(inception_stats,sz)
md = ImageClassifierData.from_csv(PATH,'bedroom',CSV_PATH,tfms=tfms,bs=128,
                                 skip_header=False,continuous=True)


# In[89]:


md = md.resize(128)


# In[90]:


x,_ = next(iter(md.val_dl))


# In[91]:


plt.imshow(md.trn_ds.denorm(x)[2])


# In[123]:


generator = dcgan_generator(sz,nz,3,64,1).cuda()
discriminator = dcgan_discriminator(sz,3,64,1).cuda()


# In[124]:


def create_noise(b):
    return V(torch.zeros(b,nz,1,1).normal_(0,1))   # nz-> size of noise vector, Using normal distribution


# In[125]:


prediction = generator(create_noise(4))
prediction_img = md.trn_ds.denorm(prediction)

fig,axes = plt.subplots(2,2,figsize=(6,6))
for i,ax in enumerate(axes.flat):
    ax.imshow(prediction_img[i])


# In[126]:


def gallery(x,nc=3):
    n,h,w,c = x.shape
    nr = n//nc
    assert n == nr*nc
    return (x.reshape(nr,nc,h,w,c)
            .swapaxes(1,2)
            .reshape(h*nr,w*nc,c))


# In[127]:


optimizer_discriminator = optim.RMSprop(discriminator.parameters(),lr=1e-4)
optimizer_generator = optim.RMSprop(generator.parameters(),lr=1e-4)


# In[128]:


def train(num_iterations,first=True):
    gen_iterations=0
    for epoch in trange(num_iterations):
        generator.train(); discriminator.train()
        data_iterator = iter(md.trn_dl)
        i,n = 0,len(md.trn_dl)
        with tqdm(total=n) as progress_bar:
            while i<n:
                set_trainable(generator,False)
                set_trainable(discriminator,True)
                d_iters=100 if (first and (gen_iterations < 25) or (gen_iterations % 500 ==0 )) else 5
                j=0
                while ( (j<d_iters) and (i<n) ):
                    j+=1
                    i+=1
                    for p in discriminator.parameters():
                        p.data.clamp_(-0.01,0.01)
                    
                    real = V(next(data_iterator)[0])
                    real_loss = discriminator(real)
                    fake = generator(create_noise(real.size(0)))
                    fake_loss = discriminator(V(fake.data))
                    
                    discriminator.zero_grad()
                    discriminator_loss = real_loss-fake_loss
                    discriminator_loss.backward()
                    optimizer_discriminator.step()
                    
                    progress_bar.update()
                    
                set_trainable(generator,True)
                set_trainable(discriminator,False)
                generator.zero_grad()
                generator_loss = discriminator(generator(create_noise(batch_size))).mean(0).view(1)
                generator_loss.backward()
                optimizer_generator.step()
                gen_iterations+=1
                
            print(f'discriminator_loss {to_np(discriminator_loss)}; generator_loss {to_np(generator_loss)}; '
              f'D_real {to_np(real_loss)}; Loss_D_fake {to_np(fake_loss)}')


# In[129]:


torch.backends.cudnn.benchmark = True


# In[121]:


train(1,False)


# In[130]:


fixed_noise = create_noise(batch_size)


# In[133]:


set_trainable(discriminator,True)
set_trainable(generator,True)

optimizer_discriminator = optim.RMSprop(discriminator.parameters(),lr=1e-5)
optimizer_generator = optim.RMSprop(generator.parameters(),lr=1e-5)


# In[134]:


train(250,False)


# In[137]:


discriminator.eval(); generator.eval();
fake = generator(fixed_noise).data.cpu()
faked = np.clip(md.trn_ds.denorm(fake),0,1)

plt.figure(figsize=(9,9))
plt.imshow(gallery(faked, 8));


# In[139]:


torch.save(generator.state_dict(), TMP_PATH/'netG_2.h5')
torch.save(discriminator.state_dict(), TMP_PATH/'netD_2.h5')

