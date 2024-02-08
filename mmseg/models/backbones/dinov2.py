from typing import Sequence, Tuple, Union, Callable
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmpretrain.models import VisionTransformer as MMPretrain_VisionTransformer

from mmseg.registry import MODELS

@MODELS.register_module()
class DINOv2VisionTransformerPretrained(MMPretrain_VisionTransformer):
    def __init__(self,
                 n_layers_to_take=4,
                 reshape_to_patched_size=False,
                 return_class_token=False,
                 norm_output_feats=True,
                 interpolate_offset=0.1,
                 interpolate_antialias=False,
                 num_register_tokens=0,
                 **kwargs):
        super(DINOv2VisionTransformerPretrained, self).__init__(**kwargs)
        self.n_layers_to_take = n_layers_to_take
        self.reshape_to_patched_size = reshape_to_patched_size
        self.return_class_token = return_class_token
        self.norm_output_feats = norm_output_feats
        self.interpolate_offset = interpolate_offset
        self.interpolate_antialias = interpolate_antialias
        self.num_register_tokens = num_register_tokens
        self.patch_size = kwargs.get('patch_size', 14)
        # base class not call init_weight after init
        self.init_weights()

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)[0]

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        return x
    
    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        # DINOv2 with register modify the interpolate_offset from 0.1 to 0.0
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        # w0, h0 = w0 + 0.1, h0 + 0.1
        
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            # (int(w0), int(h0)), # to solve the upsampling shape issue
            # mode="bicubic",
            antialias=self.interpolate_antialias
        )
        
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.layers)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.layers):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output
    
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)
        
    def forward(self, x):
        outputs = self._get_intermediate_layers_not_chunked(x, self.n_layers_to_take)
        if self.norm_output_feats:
            outputs = [self.ln1(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]

        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        if self.reshape_to_patched_size:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if self.return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)