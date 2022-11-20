import sys
import math
import torch
import torch.nn as nn
import numpy as np
#from thop import profile
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath

import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor

from .pos_enc import ImgPosEnc

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(w, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output
    
    def relative_embedding(self):
        cord = cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)])).type(torch.long)
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0], relation[:,:,1]]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class SwinTransformer(nn.Module):
    """ Implementation of Swin Transformer https://arxiv.org/abs/2103.14030
    In this Implementation, the standard shape of data is (b h w c), which is a similar protocal as cnn.
    """
    #TODO make layers using configs
    # num_class is unnecessary
    def __init__(self, num_classes, config=[2,2,6,2], dim=96, drop_path_rate=0.2, input_resolution=224):
        super(SwinTransformer, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 7
        # self.patch_partition = Rearrange('b c (h1 sub_h) (w1 sub_w) -> b h1 w1 (c sub_h sub_w)', sub_h=4, sub_w=4)

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        begin = 0
        self.stage1 = [nn.Conv2d(3, dim, kernel_size=4, stride=4),
                       Rearrange('b c h w -> b h w c'),
                       nn.LayerNorm(dim),] + \
                      [Block(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//4) 
                      for i in range(config[0])]
        begin += config[0]
        self.stage2 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2), 
                       nn.LayerNorm(4*dim), nn.Linear(4*dim, 2*dim, bias=False),] + \
                      [Block(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                      for i in range(config[1])]
        begin += config[1]
        self.stage3 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2), 
                       nn.LayerNorm(8*dim), nn.Linear(8*dim, 4*dim, bias=False),] + \
                      [Block(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//16)
                      for i in range(config[2])]
        begin += config[2]
        self.stage4 = [Rearrange('b (h neih) (w neiw) c -> b h w (neiw neih c)', neih=2, neiw=2), 
                       nn.LayerNorm(16*dim), nn.Linear(16*dim, 8*dim, bias=False),] + \
                      [Block(8*dim, 8*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//32)
                      for i in range(config[3])]
        
        self.stage1 = nn.Sequential(*self.stage1)
        self.stage2 = nn.Sequential(*self.stage2)
        self.stage3 = nn.Sequential(*self.stage3)
        self.stage4 = nn.Sequential(*self.stage4)

        self.norm_last = nn.LayerNorm(dim * 8)
        self.mean_pool = Reduce('b h w c -> b c', reduction='mean')
        self.classifier = nn.Linear(8*dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x,x_mask):
        x = self.stage1(x)
        out_mask = x_mask[:, 0::2, 0::2]
        x = self.stage2(x)
        out_mask = out_mask[:, 0::2, 0::2]
        x = self.stage3(x)
        out_mask = x_mask[:, 0::2, 0::2]
        x = self.stage4(x)
        out_mask = x_mask[:, 0::2, 0::2]
        x = self.norm_last(x)

        # x = self.mean_pool(x)
        # x = self.classifier(x)
        return x,out_mask
    # def forward(self, x):
    # '''
    #    this function is the initial forword function
    # '''
    #     x = self.stage1(x)
    #     x = self.stage2(x)
    #     x = self.stage3(x)
    #     x = self.stage4(x)
    #     x = self.norm_last(x)

    #     # x = self.mean_pool(x)
    #     # x = self.classifier(x)
    #     return x




class Encoder(pl.LightningModule):
    def __init__(self, d_model:int, numclass,config=[2,2,26],dim=96,**kwargs):
        super().__init__()

        self.model = SwinTransformer(num_classes=numclass, config=config, dim=dim, **kwargs)

        self.feature_proj = nn.Conv2d(self.model.dim * 8,d_model,kernel_size=1)

        self.pos_enc_2d = ImgPosEnc(d_model,normalize=True)

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, img: FloatTensor, img_mask: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
        """encode image to feature

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']

        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, h, w, d], [b, h, w]
        """
        # extract feature
        feature, mask = self.model(img, img_mask)
        feature = self.feature_proj(feature)

        # proj
        feature = rearrange(feature, "b d h w -> b h w d")

        # positional encoding
        feature = self.pos_enc_2d(feature, mask)
        feature = self.norm(feature)

        # flat to 1-D
        return feature, mask


# The code here are used to test the swin transformer
# def Swin_T(num_classes, config=[2,2,6,2], dim=96, **kwargs):
#     return SwinTransformer(num_classes, config=config, dim=dim, **kwargs)

# def Swin_S(num_classes, config=[2,2,18,2], dim=96, **kwargs):
#     return SwinTransformer(num_classes, config=config, dim=dim, **kwargs)

# def Swin_B(num_classes, config=[2,2,18,2], dim=128, **kwargs):
#     return SwinTransformer(num_classes, config=config, dim=dim, **kwargs)

# def Swin_L(num_classes, config=[2,2,18,2], dim=192, **kwargs):
#     return SwinTransformer(num_classes, config=config, dim=dim, **kwargs)

# if __name__ == '__main__':
#     test_model = Swin_T(1000).cuda()
#     n_parameters = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
#     f=open('test_swin_transformer.log', mode='w', encoding='utf-8')
#     print(test_model,file=f)
#     dummy_input = torch.rand(3,3,224,224).cuda()
#     output = test_model(dummy_input)
#     print(output.size(),file=f)
#     # flops, params = profile(test_model, inputs=(dummy_input, ))
#     # print(params)
#     # print(flops)
#     print(n_parameters,file=f)
#     f.close()