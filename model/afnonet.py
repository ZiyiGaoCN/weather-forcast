from functools import partial
from collections import OrderedDict
import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.utils.checkpoint import checkpoint_sequential

from loguru import logger 


class Maxmin_variance(nn.Module):
    def __init__(self, dim=(1,68,32,64)):
        super().__init__()
        self.max_logvar = nn.Parameter(torch.ones(dim)*2)
        self.min_logvar = nn.Parameter(torch.ones(dim)*-10)
        
        # varmax_ = self.varmax * torch.ones_like(logvar) 
        # varmin_ = self.varmin * torch.ones_like(logvar)
        # varmax_ = varmax_.to(logvar.device)
        # varmin_ = varmin_.to(logvar.device)        


            # TODO: to utils
            
    def compute_uncertain_loss(self,logvar,varmin,varmax):
        logvar = varmax - nn.functional.softplus(varmax - logvar)
        logvar = varmin + nn.functional.softplus(logvar - varmin)
        return logvar,varmin,varmax

    def forward(self, x):
        return self.compute_uncertain_loss(x,self.min_logvar,self.max_logvar)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = nn.AdaptiveAvgPool1d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaptiveFourierNeuralOperator(nn.Module):
    def __init__(self, dim, h=14, w=8,
                 fno_blocks = 4,
                 fno_bias = True,
                 fno_softshrink = 0.0
                 
                 ):
        super().__init__()
        # args = get_args()
        self.hidden_size = dim
        self.h = h
        self.w = w

        self.num_blocks = fno_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()

        if fno_bias:
            self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        else:
            self.bias = None

        self.softshrink = fno_softshrink

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape

        if self.bias:
            bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            bias = torch.zeros(x.shape, device=x.device)

        x = x.reshape(B, self.h, self.w, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0], inplace=True)
        x_imag = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1], inplace=True)
        x_real = self.multiply(x_real, self.w2[0]) - self.multiply(x_imag, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(x_real, self.w2[1]) + self.multiply(x_imag, self.w2[0]) + self.b2[1]

        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        return x + bias


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AdaptiveFourierNeuralOperator(dim, h=h, w=w)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # FIXME: self.double_skip = args.double_skip
        self.double_skip = True


    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x += residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x += residual
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_chans=13, embed_dim=768):
        super().__init__()

        if img_size is None:
            raise KeyError('img is None')

        patch_size = to_2tuple(patch_size)

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class AFNONet(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_chans=20, out_chans=20, embed_dim=768, depth=12, mlp_ratio=4.,
                 uniform_drop=False, drop_rate=0., drop_path_rate=0., norm_layer=None, dropcls=0,
                 uncertainty_loss=False,time_embed=False):
        super().__init__()

        if img_size is None:
            img_size = [720, 1440]

        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size

        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.blocks = nn.ModuleList([Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=self.h, w=self.w) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.uncertainty_loss = uncertainty_loss
        
        
        self.time_embed = time_embed
        if self.time_embed:
            raise NotImplementedError
    
        # Representation layer
        # self.num_features = out_chans * img_size[0] * img_size[1]
        # self.representation_size = self.num_features * 8
        # self.pre_logits = nn.Sequential(OrderedDict([
        #     ('fc', nn.Linear(embed_dim, self.representation_size)),
        #     ('act', nn.Tanh())
        # ]))
        
        # TODO: more elegant way
        # self.pre_logits = nn.Sequential(OrderedDict([
        #     ('conv1', nn.ConvTranspose2d(embed_dim, out_chans*16, kernel_size=(2, 2), stride=(2, 2))),
        #     ('act1', nn.Tanh()),
        #     ('conv2', nn.ConvTranspose2d(out_chans*16, out_chans*4, kernel_size=(2, 2), stride=(2, 2))),
        #     ('act2', nn.Tanh())
        # ]))

        # Generator head
        # self.head = nn.Linear(self.representation_size, self.num_features)
        self.head = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=(1, 1), stride=(1, 1))
        
        # self.final_tanh = nn.Tanh()
        
        # self.final_go = nn.nn.ConvTranspose2d(out_chans, out_chans, kernel_size=(1, 1), stride=(1, 1))
        
        if self.uncertainty_loss:
            self.final_sigma = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=(1, 1), stride=(1, 1))
            self.variance_control = Maxmin_variance(dim=(1,out_chans,self.h,self.w))
            # raise NotImplementedError

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x += self.pos_embed
        x = self.pos_drop(x)

        #FIXME: 
        # if not get_args().checkpoint_activations:
        for blk in self.blocks:
            x = blk(x)
        # else:
        #     x = checkpoint_sequential(self.blocks, 4, x)

        x = self.norm(x).transpose(1, 2)
        x = torch.reshape(x, [-1, self.embed_dim, self.h, self.w])
        return x

    def forward(self, x, step=None):
        
        x = self.forward_features(x)
        # logger.info(f"forward_features:{x.shape}")
        x = self.final_dropout(x)
        # logger.info(f"final_dropout:{x.shape}")
        
        # x = self.pre_logits(x)
        # logger.info(f"pre_logits:{x.shape}")
        
        if self.uncertainty_loss:
            mean , sigma = self.head(x) , self.final_sigma(x)
            # logger.info(f"final_sigma:{x.shape}")
            sigma,varmin,varmax = self.variance_control(sigma)
            # logger.info(f"variance_control:{x.shape}")
            return mean, sigma,varmin,varmax
        x = self.head(x)
        
        # logger.info(f"head:{x.shape}")
        return x

if __name__ == '__main__':
    model = AFNONet(img_size=[32,64],patch_size=1,in_chans=68,out_chans=68,embed_dim=768,mlp_ratio=4)
    print(model)
    x = torch.zeros((1,68,32,64))
    y = model(x)
    print(y.shape)