import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
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
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention . Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim ** -0.5
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # q = q * self.scale

        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(self.logit_scale.device) )).exp()
        attn = attn * logit_scale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0]*self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # print(B,H,W,C)
    # print(window_size)
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=2, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size[0] > 0 and self.shift_size[1] > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn_mask = None
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H0, W0 = self.input_resolution
        B, H, W, C = x.shape
        assert H == H0 and  W == W0, "input feature has wrong size"
        
        x = x.view(B, H* W, C)

        shortcut = x
        
        
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size[0] or self.shift_size[1] > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.view(B, H, W, C)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops



class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm,
                 patch_size=(2,2)):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, int(dim_scale*dim), bias=False)
        self.norm = norm_layer(dim // dim_scale)

        if patch_size != (2,2):
            raise NotImplementedError("PatchExpand only support patch_size = (2,2) now!")

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H0, W0 = self.input_resolution
        x = self.expand(x)
        B, H, W, C = x.shape
        assert H == H0 and W == W0, "input feature has wrong size"


        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)
        x = x.view(B, H * 2 , W * 2, C // 4)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm,extra_dim=1):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        # self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.conv2d = nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=dim_scale+extra_dim, stride=dim_scale, output_padding=0,bias=False) 
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        # x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = x.permute(0,3,1,2)
        x = self.conv2d(x)
        x = x.permute(0,2,3,1)
        x = x.view(B,-1,self.output_dim)
        # x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        # x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, 
                 lat_dim=161,
                 lon_dim=161,
                #  img_size=224, 
                 patch_size=4, 
                 in_chans=3, 
                 embed_dim=96, 
                 norm_layer=None , 
                 extra_dim=0):
        super().__init__()
        img_size = (lat_dim,lon_dim)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size[0]+extra_dim,patch_size[1]+extra_dim), stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B,Ph*Pw,C)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class PatchEmbed_3D(nn.Module):
    def __init__(self, 
                 lat_dim=32,
                 lon_dim=64,
                 z_dim=13, 
                 patch_size=(2,2,1), 
                 in_chans=3, 
                 embed_dim=96, 
                 norm_layer=None , 
                 extra_dim=0):
        super().__init__()
        img_size = (lat_dim,lon_dim,z_dim)
        patch_size = patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, H, W, Z, C = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1] and Z == self.img_size[2], \
            f"Input image size ({H}*{W}*{Z}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        
        x = x.permute(0,4,1,2,3)
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B,Ph*Pw*Pz,C)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops



def window_partition_3D(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, Z, C = x.shape
    # print(B,H,W,C)
    # print(window_size)
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], Z // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse_3D(windows, window_size, H, W, Z):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W *Z / window_size[0] / window_size[1]/window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], Z// window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, Z, -1)
    return x

def padding_3D(x):
    B,H,W,Z,C = x.shape
    if x.shape[3] % 2 == 1:
        x = torch.cat([x,torch.zeros(B,H,W,1,C).to(x.device)], dim=3)
    return x

class PatchMerging_2D(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm,
                 patch_size = (2,2)):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

        if patch_size != (2,2):
            raise NotImplementedError("PatchMerging_2D only support patch_size = (2,2) now!")

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H0, W0 = self.input_resolution
        B, H, W, C = x.shape
        assert H == H0 and W == W0, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        x = x.view(B, H // 2, W // 2, -1)  # B H/2 W/2 2*C

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchMerging_3D(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, patch_size =(2,2,2) , norm_layer=nn.LayerNorm,padding_z=(0,1)):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        self.patch_size=patch_size
        self.hidden_scale = patch_size[0]*patch_size[1]*patch_size[2]

        self.reduction = nn.Linear(self.hidden_scale * dim, 2 * dim, bias=False)
        self.norm = norm_layer(self.hidden_scale * dim)
        self.padding_z = padding_z

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H0, W0, Z0 = self.input_resolution
        B, H, W, Z, C = x.shape
        assert H == H0 and W == W0 and Z==Z0, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0 and Z%2 == 0, f"x size ({H} or {W} or {Z}) are not even."

        # x = x.view(B, H, W, Z, C)

        if self.padding_z[1]>0:
            x = padding_3D(x)
     
        
        variables = []
        for d1 in range(self.patch_size[0]):
            for d2 in range(self.patch_size[1]):
                for d3 in range(self.patch_size[2]):
                    variables.append(x[:, d1::self.patch_size[0], d2::self.patch_size[1], d3::self.patch_size[2], :])

        x = torch.cat(variables, -1)  # B H/2 W/2 Z/2 8*C
        x = x.view(B, -1, self.hidden_scale * C)  # B H/2*W/2 8*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(B, H // self.patch_size[0], W // self.patch_size[1], (Z+self.padding_z[1]) // self.patch_size[2], -1)  # B H/2 W/2 Z/2 2*C

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class WindowAttention_3D(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size ,num_heads,  
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 no_relative_z =False,
                 variable_order=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.variable_order = variable_order

        if window_size[2] > 1:
            raise NotImplementedError

        # define a parameter table of relative position bias
        
        # self.relative_position_bias_table = nn.ModuleDict({
        #     k: nn.Parameter(
        #         torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) , num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        #     for k in range(self.variable_order)})

        # # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # # coords_z = torch.arange(self.window_size[2])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 3, Wh, Ww, Wz
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 3
        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # # relative_coords[:, :, 2] += self.window_size[2] - 1

        # relative_coords[:, :, 0] *= 1
        # relative_coords[:, :, 1] *= (2 * self.window_size[0] - 1)
        # if no_relative_z == False:
        #     raise NotImplementedError
        #     relative_coords[:, :, 2] *= (2 * self.window_size[0] - 1)*(2 * self.window_size[1] - 1)
        # else:
        #     relative_coords[:, :, 2] *= 0
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.ModuleDict({
            k: nn.Linear(dim, dim * 3, bias=qkv_bias) for k in self.variable_order
        })

        self.attn_drop = nn.Dropout(attn_drop)
        # self.attn_drop = nn.ModuleDict({
        #     k: nn.Dropout(attn_drop) for k in self.variable_order
        # })

        # self.proj = nn.Linear(dim, dim)
        self.proj = nn.ModuleDict({
            k: nn.Linear(dim, dim) for k in self.variable_order
        })

        # self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop = nn.ModuleDict({
            k: nn.Dropout(proj_drop) for k in self.variable_order
        })

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x[self.variable_order[0]].shape

        big_N = N * len(self.variable_order)

        # assert N == self.window_size[0] * self.window_size[1] and V == self.window_size[2], \
        #     "input feature has wrong size"

        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = {
            k: self.qkv[k](x[k]).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) for k in self.variable_order
        }

        # Q, K, V = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        Q = { k: qkv[k][0] for k in self.variable_order}
        K = { k: qkv[k][1] for k in self.variable_order}
        V = { k: qkv[k][2] for k in self.variable_order}

        Q = torch.cat([Q[k] for k in self.variable_order],dim=1)
        K = torch.cat([K[k] for k in self.variable_order],dim=1)
        V = torch.cat([V[k] for k in self.variable_order],dim=1)

        Q = Q * self.scale
        attn = (Q @ K.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1] * self.window_size[2], self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        # if mask is not None:
        #     nW = mask.shape[0]
        #     attn = attn.view(B_ // nW, nW, self.num_heads, big_N, big_N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, big_N, big_N)
        #     attn = self.softmax(attn)
        # else:
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ V).transpose(1, 2).reshape(B_, big_N, C)

        x = { k: x[:, i*N:(i+1)*N, :] for i,k in enumerate(self.variable_order)}

        # x = self.proj(x)
        x = { k: self.proj[k](x[k]) for k in self.variable_order}

        # x = self.proj_drop(x)
        x = { k: self.proj_drop[k](x[k]) for k in self.variable_order}
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

def timestep_embedding(timestep,dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    )
    args = timestep * freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

class SwinTransformerBlock_3D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads,
                 window_size=(4,4,4), shift_size=(1,1,1),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 padding_z=(0,1),no_relative_z=False,
                 time_embed=False,
                 variable_order=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.variable_order = variable_order

        self.norm1 = nn.ModuleDict({
            k: norm_layer(dim) for k in self.variable_order
        })
        self.attn = WindowAttention_3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            no_relative_z = no_relative_z,
            variable_order=self.variable_order)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        
        self.norm2 = nn.ModuleDict({
            k: norm_layer(dim) for k in self.variable_order
        })
        
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp =nn.ModuleDict({
            k:Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) for k in self.variable_order
        }) 
        
        self.padding_z = padding_z    
        self.time_embed = time_embed
        if self.time_embed:
            self.time_embed_layer = nn.ModuleDict({
                nn.Sequential(
                nn.GELU(),
                nn.Linear(dim, dim),) for _ in range(len(self.variable_order))
            } for k in self.variable_order)

        H, W, Z = self.input_resolution
        # img_mask = torch.zeros((1, H, W, Z+padding_z[1], 1))  # 1 H W 1
        
        # def get_slices(window_size,shift_size):
        #     if shift_size ==0 :
        #         return (slice(0, None),)
        #     else :
        #         return (slice(0, shift_size), slice(shift_size, window_size), slice(window_size, None),)

        # h_slices = get_slices(self.window_size[0],self.shift_size[0])
        # w_slices = get_slices(self.window_size[1],0) #经度不需要mask
        # z_slices = get_slices(self.window_size[2],self.shift_size[2])

        # cnt = 0
        # for h in h_slices:
        #     for w in w_slices:
        #         for z in z_slices:
        #             img_mask[:, h, w, z, :] = cnt
        #             cnt += 1
        # assert padding_z[0] == 0 , "padding_z[0] must be 0"
        # for z in range(self.input_resolution[2],self.input_resolution[2]+padding_z[1]):
        #     for x in range(self.input_resolution[0]):
        #         for y in range(self.input_resolution[1]):
        #             img_mask[:,x,y,z,:] = cnt
        #             cnt += 1

        # mask_windows = window_partition_3D(img_mask, self.window_size)  # nW, window_size, window_size, 1
        # mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1]* self.window_size[2])
        # attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # self.register_buffer("attn_mask", attn_mask)
    def forward(self, x , timestep_embedding=None):
        H0, W0, Z0 = self.input_resolution
        B, H, W, Z, C = x[self.variable_order[0]].shape
        assert H == H0 and W == W0 and Z == Z0, f"input feature has wrong size,{H,W,Z},{H0,W0,Z0}"

        # x = x.view(B, H*W*Z, C)
        x = { k: v.view(B, H*W*Z, C) for k, v in x.items() }
        if self.time_embed:
            # x = x + self.time_embed_layer(timestep_embedding)[None, None, :]
            x = { k: v + self.time_embed_layer[i](timestep_embedding)[None, None, :] for i,(k,v) in enumerate(x.items()) }
        
        shortcut = x
        # x = self.norm1(x)
        x = { k: self.norm1[k](v) for k, v in x.items() }
        # x = x.view(B, H, W, Z, C)
        x = { k: v.view(B, H, W, Z, C) for k, v in x.items() }

        # cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0 or self.shift_size[2] > 0:
            # shifted_x = torch.roll(x, shifts=self.shift_size, dims=(1, 2, 3)) 
            shifted_x = { k: torch.roll(v, shifts=self.shift_size, dims=(1, 2, 3)) for k, v in x.items() } 
        else:
            shifted_x = x

        if self.padding_z[1] > 0:
            raise NotImplementedError
            shifted_x = torch.cat([shifted_x,torch.zeros((B,H,W,self.padding_z[1],C)).to(shifted_x.device)],dim=-2)        

        # partition windows
        # x_windows = window_partition_3D(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = { k: window_partition_3D(v, self.window_size) for k, v in shifted_x.items() }  # nW*B, window_size, window_size, C
        # x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size, C
        x_windows = { k: v.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C) for k, v in x_windows.items() }  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        # attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        attn_windows = { k: v.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C) for k, v in attn_windows.items() }
        

        # shifted_x = window_reverse_3D(attn_windows, self.window_size, H, W,Z+self.padding_z[1])  # B H' W' C
        shifted_x = { k: window_reverse_3D(v, self.window_size, H, W,Z+self.padding_z[1]) for k, v in attn_windows.items() }  # B H' W' C

        if self.padding_z[1] > 0:
            raise NotImplementedError
            shifted_x = shifted_x[:,:,:,:-self.padding_z[1],:]

        # reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0 or self.shift_size[2] > 0:
            # x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2, 3))
            x = { k: torch.roll(v, shifts=self.shift_size, dims=(1, 2, 3)) for k, v in shifted_x.items() }
        else:
            x = shifted_x
        # x = x.reshape(B, H * W * Z, C)
        x = { k: v.reshape(B, H * W * Z, C) for k, v in x.items() }

        # FFN
        # x = shortcut + self.drop_path(x)
        x = { k: shortcut[k] + self.drop_path(v) for k, v in x.items() }
        
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = { k: v + self.drop_path(self.mlp[k](self.norm2[k](v))) for k, v in x.items() }

        # x = x.view(B, H, W, Z, C)
        x = { k: v.view(B, H, W, Z, C) for k, v in x.items() }

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W, Z = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class BasicLayer_2D(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # self.padding = (0,input_resolution[2]%2) if input_resolution[2] > 1 else (0,0)
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0,0) if (i % 2 == 0) else (window_size[0] // 2,window_size[1] // 2),
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                #  padding_z = self.padding,
                                #  no_relative_z=True,
                                #  variable_order=variable_order
                                 )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer,
                                        #  padding_z = (0,0),
                                         patch_size = (2,2))
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_3D(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 variable_order=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.padding = (0,input_resolution[2]%2) if input_resolution[2] > 1 else (0,0)
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_3D(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0,0,0) if (i % 2 == 0) else (window_size[0] // 2,window_size[1] // 2,window_size[2] // 2),
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 padding_z = self.padding,
                                 no_relative_z=True,
                                 variable_order=variable_order
                                 )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer,
                                         padding_z = (0,1) if input_resolution[2] > 1 else (0,0),
                                         patch_size = (2,2,2) if input_resolution[2] > 1 else (2,2,1))
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class Encoder(nn.Module):
    def __init__(self,
                    lat_dim=32,
                    lon_dim=64,
                    z_dim = 13,
                    C_dim = 1,
                    hidden_dim = 96,
                    norm_layer = nn.LayerNorm,
                    patch_size = (2,2,1),
                    ape = False,
                    depths = [2,2],
                    num_heads = [3,6],
                    window_size = [4,4,2],
                    qk_scale = None,
                    drop_rate = 0.0,
                    attn_drop_rate = 0.0,
                    drop_path_rate = 0.1,
                    mlp_ratio = 4,
                    **kwargs):
        super().__init__()
        self.ape = ape
        self.patch_embed = PatchEmbed(
            lat_dim=lat_dim,lon_dim=lon_dim, patch_size=patch_size, embed_dim=hidden_dim,
            in_chans=C_dim,
            norm_layer=norm_layer,
            extra_dim=0)
        self.hidden_dim = hidden_dim

        num_patches = self.patch_embed.num_patches # 32*64/2/2 = 32*64/4 = 512 = 16*32
        patches_resolution = self.patch_embed.patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=0.0)
        self.num_layers = len(depths)

        self.layers = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule


        for i_layer in range(self.num_layers):
            layer = BasicLayer_2D(dim=int(self.hidden_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer), # 16, 8, 4, 2
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer], # 2, 2, 2, 2
                               num_heads=num_heads[i_layer], # 3, 6, 12, 24
                               window_size=tuple(window_size), # 7
                               mlp_ratio=mlp_ratio, # 4
                               qkv_bias=True, qk_scale=qk_scale, # True, None
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], 
                               norm_layer=norm_layer, # LayerNorm
                               downsample=PatchMerging_2D,
                               use_checkpoint=None)
            self.layers.append(layer)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = x.view(x.shape[0],self.patch_embed.patches_resolution[0],self.patch_embed.patches_resolution[1],self.hidden_dim)

        x_downsample = [x]
        for layer in self.layers:
            x = layer(x)
            x_downsample.append(x)
        return x_downsample

class PatchExpand_3D(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=(2,2,2), norm_layer=nn.LayerNorm,padding_z=(0,1)):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.scale = patch_size[0]*patch_size[1]*patch_size[2] // 2
        self.expand = nn.Linear(dim, self.scale*dim, bias=False)
        self.norm = norm_layer(dim // 2)
        self.padding=padding_z

    def forward(self, x):
        """
        x: B, H*W*Z, C
        """
        H0, W0, Z0 = self.input_resolution
        x = self.expand(x)
        B, H,W,Z, C = x.shape
        assert H == H0 and W == W0 and Z == Z0, "input feature has wrong size"

        x = x.view(B, H, W, Z, C)
        x = rearrange(x, 'b h w z (p1 p2 p3 c)-> b (h p1) (w p2) (z p3) c', 
                      p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2],
                      c=C//(self.patch_size[0]*self.patch_size[1]*self.patch_size[2]))
        # x = x.view(B,-1,C//8)
        x= self.norm(x)
        if self.padding[1] > 0:
            x = x[:,:,:,:-self.padding[1],:]

        return x

class Decoder(nn.Module):
    def __init__(self,
                    lat_dim=4,
                    lon_dim=8,
                    z_dim = (4,7,13),
                    C_dim = 1,
                    hidden_dim = 96,
                    out_dim = 13,
                    norm_layer = nn.LayerNorm,
                    patch_size = (2,2,1),
                    depths = [2,2],
                    num_heads = [3,6],
                    window_size = [4,4,2],
                    qk_scale = None,
                    drop_rate = 0.0,
                    attn_drop_rate = 0.0,
                    drop_path_rate = 0.1,
                    mlp_ratio = 4,
                    uncertainty_loss = False,
                    # varmin = -5,
                    # varmax = 5, 
                    **kwargs):
        super().__init__()

        patches_resolution =[
            (int(lat_dim*(2**i)),
             int(lon_dim*(2**i)))
             for i in range(len(depths))]
        self.hidden_dim = hidden_dim

        self.pos_drop = nn.Dropout(p=0.0)
        self.num_layers = len(depths)

        self.layers = nn.ModuleList()
        self.uncertainty_loss = uncertainty_loss

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.final_up = nn.ConvTranspose2d(hidden_dim, out_dim, kernel_size=patch_size,
                                            stride=patch_size, padding=0, output_padding=0, groups=1, bias=False, dilation=1, padding_mode='zeros')
        self.final_out = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)

        if self.uncertainty_loss:
            self.final_up_uncertainty = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
            # TODO: variable
            self.variance_control = Maxmin_variance(dim=(1,out_dim,32,64))
            # trunc_normal_(self.final_up_uncertainty.weight, std=0.0)
            torch.nn.init.normal_(self.final_up_uncertainty.weight,mean=0,std=0.01)
            # torch.nn.init.zeros_(self.final_up_uncertainty.weight)
            
        self.concat_back_dim = nn.ModuleList()
        self.norm = norm_layer(hidden_dim)

        for i_layer in range(self.num_layers):
            layer = BasicLayer_2D(dim=int(self.hidden_dim * 2 ** (self.num_layers-i_layer)),
                               input_resolution=patches_resolution[i_layer],
                               depth=depths[i_layer], # 2, 2, 2, 2
                               num_heads=num_heads[i_layer], # 3, 6, 12, 24
                               window_size=tuple(window_size), # 7
                               mlp_ratio=mlp_ratio, # 4
                               qkv_bias=True, qk_scale=qk_scale, # True, None
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], 
                               norm_layer=norm_layer, # LayerNorm
                               downsample=PatchExpand,
                               use_checkpoint=None)
            self.layers.append(layer)
            self.concat_back_dim.append(nn.Linear(int(self.hidden_dim * 2 ** (self.num_layers-i_layer),),int(self.hidden_dim * 2 ** (self.num_layers-i_layer-1)),bias=False))
    def forward(self, x_downsample):
        total = len(x_downsample)
        for inx, layer_up in enumerate(self.layers):
            if inx == 0:
                x = layer_up(x_downsample[total-1-inx])
            else:
                x = layer_up(x)
            x = torch.cat([x,x_downsample[total-1-inx-1]],-1)
            x = self.concat_back_dim[inx](x)

        x = self.norm(x)  # B L C

        B, H ,W ,C = x.shape

        x = self.norm(x)
        x = x.view(B, H*W, C).permute(0,2,1).contiguous().view(B,C,H,W)
        x = self.final_up(x)
        if self.uncertainty_loss:
            mean = self.final_out(x)
            
            logvar = self.final_up_uncertainty(x)
            logvar, logvar_min , logvar_max = self.variance_control(logvar)
          
                

            return mean , logvar , logvar_min , logvar_max
        
        x = self.final_out(x)
        
        return x
    def flops(self,x):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class Fusion(nn.Module):
    def __init__(self, 
                 lat_dim=32,
                 lon_dim=64,
                 z_dim=70, 
                 window_size=(4,4,2), 
                 in_chans=96, 
                 embed_dim=96, 
                 out_chans=96,
                 num_layers = 4,
                 norm_layer=nn.LayerNorm, 
                 num_heads=[4,8],
                 mlp_ratio=4,
                 qk_scale=None,
                 qkv_bias=True,
                 drop_rate=0.0,
                 attn_drop=0.0,
                 extra_dim=0,
                 time_embed=False,
                 variable_fusion=True,
                 variable_order=None):
        super().__init__()
        img_size = (lat_dim,lon_dim,z_dim)
        self.img_size = img_size
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.time_embed = time_embed
        self.variable_fusion = variable_fusion
        self.variable_order = variable_order

        self.proj = nn.ModuleDict({
            k : nn.Linear(in_chans, embed_dim) for k in variable_order
        })
        
        if norm_layer is not None:
            self.norm = nn.ModuleDict({
                k : norm_layer(embed_dim) for _ in range(len(variable_order)) for k in variable_order
            }) 
            # norm_layer(embed_dim)
        else:
            self.norm = None
        
        self.proj_out = nn.ModuleDict({
            k : nn.Linear(embed_dim, out_chans) for _ in range(len(variable_order)) for k in variable_order
        }) 

        if self.time_embed:
            self.time_embed_layer = nn.ModuleDict({
                k : nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim),
                ) for k in variable_order
            })
        
        if self.variable_fusion:
            self.layers = nn.ModuleList([
                SwinTransformerBlock_3D(dim=embed_dim, input_resolution=img_size,
                                    num_heads=num_heads, window_size=(window_size[0],window_size[1],1),
                                    shift_size=(0,0,0) if (i % 2 == 0) else (window_size[0] // 2,window_size[1] // 2,0),
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop,
                                    norm_layer=norm_layer,
                                    padding_z=(0,0),
                                    no_relative_z=True,
                                    variable_order=variable_order)
                # if i % 2 ==0 else
                # SwinTransformerBlock_3D(dim=embed_dim, input_resolution=img_size,
                #                         num_heads=num_heads, window_size=(1,1,window_size[2]),
                #                         shift_size=(0,0,0),
                #                         mlp_ratio=mlp_ratio,
                #                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                #                         drop=drop_rate, attn_drop=attn_drop,
                #                         norm_layer=norm_layer,
                #                         padding_z=(0,0),
                #                         no_relative_z=True,
                #                        variable_order=variable_order)
                for i in range(num_layers)])
        else:
            self.layers = nn.ModuleList([
                SwinTransformerBlock_3D(dim=embed_dim, input_resolution=img_size,
                                    num_heads=num_heads, window_size=(window_size[0],window_size[1],1),
                                    shift_size=(0,0,0) if (i % 2 == 0) else (window_size[0] // 2,window_size[1] // 2,0),
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop,
                                    norm_layer=norm_layer,
                                    padding_z=(0,0),
                                    no_relative_z=True,
                                    variable_order=variable_order) 
                                    for i in range(num_layers)])
            
        

    def forward(self, x, timestep = None):
        
        B, H, W, Z, C = x[self.variable_order[0]].shape
        # x = self.proj(x)
        
        x = {
            k: self.proj[k](x[k])
            for k in self.variable_order
        }


        if self.time_embed:
            # t = [timestep_embedding(timestep,dim=self.embed_dim).to(x.device) for _ in range(len(self.variable_order))]
            # t = [self.time_embed_layer[i](t[i]) for i in range(len(self.variable_order))]
            t = {
                k: self.time_embed_layer[k](timestep_embedding(timestep,dim=self.embed_dim).to(x[k].device)) for k in self.variable_order
            }
        else:
            t = None
        
        for layer in self.layers:
            x = layer(x,t)
        x = {
            k: self.proj_out[k](x[k]) for k in self.variable_order
        }
        return x

def up_sampling_z(z):
    if z==13:
        return (4,7,13)
    elif z == 1:
        return (1,1,1)
    else:
        raise NotImplementedError
    
class Model(nn.Module):
    def __init__(self,
                    lat_dim=32,
                    lon_dim=64,
                    z_dim = 70,
                    variable_dict=None,
                    variable_down=None,
                    variable_order=None,
                    variable_fusion=True,
                    C_dim = 1,
                    hidden_dim = 96,
                    norm_layer = nn.LayerNorm,
                    patch_size = (2,2,1),
                    ape = False,
                    time_embed = False,
                    depths = [2,2],
                    num_heads = [3,6],
                    window_size = [4,4,1],
                    fusion_layers = [4,4,4],
                    fusion_window_size = [2,2,6],
                    fusion_hidden_size = [96,192,384],
                    qk_scale = None,
                    drop_rate = 0.0,
                    attn_drop_rate = 0.0,
                    drop_path_rate = 0.1,
                    mlp_ratio = 4,
                    uncertainty_loss = False,
                    # varmin = -5,
                    # varmax = 5,
                    **kwargs):
        super().__init__()
        self.variable_dict = variable_dict
        # self.variable_down = list(variable_down)
        self.variable_order = list(variable_order)
        self.time_embed = time_embed
        
        self.uncertainty_loss = uncertainty_loss 
        self.scale_lens = len(depths)+1
        # if self.uncertainty_loss:
            # #TODO
            # raise NotImplementedError

        self.encoders = nn.ModuleDict({
            k: Encoder(
                lat_dim=lat_dim,
                lon_dim=lon_dim,
                z_dim = 1,
                C_dim = v,
                hidden_dim = hidden_dim,
                norm_layer = norm_layer,
                patch_size = patch_size[:2],
                ape = ape,
                depths = depths,
                num_heads = num_heads,
                window_size = window_size[:2],
                qk_scale = qk_scale,
                drop_rate = drop_rate,
                attn_drop_rate = attn_drop_rate,
                drop_path_rate = drop_path_rate,
                mlp_ratio = mlp_ratio,
            )
            for k,v in variable_dict.items()
        })
        
        self.patch_resolutions = (
            lat_dim//patch_size[0],
            lon_dim//patch_size[1],
        )
        
        self.fusions = nn.ModuleList([
            Fusion(
                lat_dim=self.patch_resolutions[0]//int(2**(i)),
                lon_dim=self.patch_resolutions[1]//int(2**(i)),
                z_dim = 1,
                window_size=fusion_window_size, 
                in_chans=hidden_dim*int(2**i), 
                embed_dim=fusion_hidden_size[i],
                out_chans=hidden_dim*int(2**(i)), 
                num_layers = fusion_layers[i],
                num_heads=num_heads[i],
                mlp_ratio=4,
                qk_scale=None,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop=0.0,
                time_embed = time_embed,
                variable_fusion=variable_fusion,
                variable_order = self.variable_order,
            ) for i in range(2)
        ])
        self.decoders = nn.ModuleDict({
            k: Decoder(
                lat_dim=self.patch_resolutions[0]//int(2**len(depths)) ,
                lon_dim=self.patch_resolutions[1]//int(2**len(depths)) ,
                z_dim = 1,
                C_dim = hidden_dim,
                hidden_dim = hidden_dim,
                out_dim = v,
                norm_layer = norm_layer,
                patch_size = patch_size,
                depths = depths,
                num_heads = num_heads,
                window_size = window_size[:2]+[1],
                qk_scale = qk_scale,
                drop_rate = drop_rate,
                attn_drop_rate = attn_drop_rate,
                drop_path_rate = drop_path_rate,
                mlp_ratio = mlp_ratio,
                uncertainty_loss=uncertainty_loss,
                variable_order = self.variable_order,
                # varmin = varmin,
                # varmax = varmax 
            )
            for k,v in variable_dict.items()
        })
    def forward(self, x, timestep = None):
        x_varibles = {}
        acc = 0
        
        indices = list(range(68))
        indices.remove(65)
        indices.remove(66)
        indices.remove(67)

        # 在指定位置插入索引67, 66, 65
        indices.insert(12 + 1, 67) # 在索引12后插入67
        indices.insert(25 + 2, 66) # 在索引25后插入66，注意索引偏移
        indices.insert(64 + 3, 65) # 在索引64后插入65，注意索引偏移
        
        x = x[:,indices,:,:]
        
        # 创建用于恢复原始顺序的索引数组
        restore_indices = [0] * 68
        for original_index, reordered_index in enumerate(indices):
            restore_indices[reordered_index] = original_index
        
        for k in self.variable_order:
            v = self.variable_dict[k]
            
            input = x[:,slice(acc,acc+v),:,:]
            
            x_varibles[k] = self.encoders[k]()
            
            for i in range(len(x_varibles[k])):
                x_varibles[k][i] = x_varibles[k][i].unsqueeze(-2)
 
            acc+=v
        logger.info(x_varibles[k][0].shape)
        x_downsample = [
            { k : x_varibles[k][i] for k in self.variable_order }
            for i in range(self.scale_lens)
        ]



        for i,fusion in enumerate(self.fusions):
            x_downsample[i]=fusion(x_downsample[i],timestep)
        
        # for i in range(2):
        #     x_downsample[i] = torch.split(x_downsample[i],[1 for _ in self.variable_order],dim=-2)
        x_upsample_varibles = {
            k: [x_downsample[i][k].contiguous().squeeze(-2) for i in range(self.scale_lens)]
            for k in self.variable_order
        }
        final = {}
        for k,decoder in self.decoders.items():
            final[k]=decoder(x_upsample_varibles[k])
        
        if self.uncertainty_loss:
            final_tensor = torch.cat([final[k][0] for k in self.variable_order],dim=1)
            sigma  = torch.cat([final[k][1] for k in self.variable_order],dim=1)       
            logvar_min  = torch.cat([final[k][2] for k in self.variable_order],dim=1)       
            logvar_max  = torch.cat([final[k][3] for k in self.variable_order],dim=1)            
            final_tensor.squeeze_(-1)
            sigma.squeeze_(-1)
            logvar_min.squeeze_(-1)
            logvar_max.squeeze_(-1)
            final_tensor = final_tensor[:,restore_indices,:,:]
            sigma = sigma[:,restore_indices,:,:]
            logvar_min = logvar_min[:,restore_indices,:,:]
            logvar_max = logvar_max[:,restore_indices,:,:]
            return final_tensor, sigma, logvar_min, logvar_max
        else:
            final = torch.cat([final[k] for k in self.variable_order],dim=1)
        # final = final.squeeze(1).permute(0,3,1,2).contiguous()
        final.squeeze_(-1)
        final = final[:,restore_indices,:,:]
        return final 
        

        

if __name__ == '__main__':
    pass
    import torch
    input = torch.randn(1,73,32,64)

    model = Model(
        ape=True,
        patch_size=(1,1),
        time_embed=True,
        window_size=[8,16,1],
        hidden_dim=512,
        fusion_window_size=[4,8,6],    
        fusion_hidden_size=[512,512],
        variable_fusion=True,
        norm_layer=nn.LayerNorm,
        num_heads=[4,8],
        depths=[2],
        variable_dict={
        "v1":13,
        "v2":13,
        "v3":13,
        "v4":13,
        "v5":13,
        "v6":8,   
        # "v1":(0,13),
        # "v2":(13,26),
        # "v3":(26,39),
        # "v4":(39,52),
        # "v5":(52,65),
        # "v6":(65,66),
        # "v7":(66,67),
        # "v8":(67,68),
        # "v9":(68,69),
        # "v10":(69,70),
        # "v11":(70,71),
        # "v12":(71,72),
        # "v13":(72,73),
    },
    # variable_down=[
    #     73,43,28
    # ],
    variable_order=[
        "v1" ,"v2","v3","v4","v5","v6"
    ])

    output = model(input,1)
    print(output.shape)