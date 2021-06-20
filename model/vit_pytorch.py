"""
credict: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py 
at Latest commit 85314cf 
"""
from loguru import logger
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from utils.checker import *

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            # logger.info('mask: {}', mask.shape)
            # mask = F.pad(mask, (1, 0), value = True)
            # logger.info('mask: {}', mask.shape)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions : {}, {}'.format(
                mask.shape, dots.shape)
            # mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            # logger.info('mask: {}, mask_value: {}, dot: {}, ', mask.shape, mask_value, dots.shape)
            # print(mask[0]) 
            del mask
        # logger.info('dots: {}', dots.shape) 
        # dots: B,nlayer,(Len+1,Len+1)
        attn = dots.softmax(dim=-1) 

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # logger.info('after softmax, and sum {} h{},b{},n{}', out.shape, h, b, n)
        out = rearrange(out, 'b h n d -> b n (h d)') ## B,Len,K 
        # B,Len+1,1024
        # logger.info('rearrange: {}', out.shape)
        out =  self.to_out(out)
        # B,Len+1,64 
        # logger.info('final: {}', out.shape)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:

            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, # num_classes, 
            dim, depth, heads, mlp_dim, pool = 'cls', 
            channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = patch_dim = channels * patch_size ** 2

        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        #self.mlp_head = nn.Sequential(
        #    nn.LayerNorm(dim),
        #    nn.Linear(dim, num_classes)
        #)

    def forward(self, img, mask = None):
        """
        Args: 
            img: shape (B,length,H*W*C)
        """
        p = self.patch_size
        nloc = img.shape[1] 
        B = img.shape[0]
        CHECKSIZE(img, (B,nloc,self.patch_dim))
        # x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(img)
        b, n, _ = x.shape # B,npatches,emb_dim
        
        # cls_tokens: B,1,emb_dim
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)[:,1:] 
        # logger.info('x output: {}, img: {}', x.shape, img.shape)
        return x # remove last mlp head 
        # return self.mlp_head(x)
if __name__ == '__main__':
    nheads = 9 
    enc = ViT(image_size=28*16, 
        patch_size=28, 
        dim=64, 
        depth=8, 
        mlp_dim=2048,
        channels=1,
        heads=nheads, dropout=0.1, emb_dropout=0.1)
    ## mask = torch.zeros(1,16,16)
    sz = 17
    mask = torch.tril(torch.ones(sz, sz)) 
    for i in range(17):
        print( mask[i,:i+1]) 
    mask = mask.unsqueeze(0).expand(nheads, -1, -1)
    input = torch.zeros(10,16,28*28*1)
    out = enc(input) #, mask)
    print(out.shape) 

"""
usage: 
    import torch
    from vit_pytorch import ViT

    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1)

    img = torch.randn(1, 3, 256, 256)
    mask = torch.ones(1, 8, 8).bool() # optional mask, designating which patch to attend to
    preds = v(img, mask = mask) # (1, 1000)
"""
