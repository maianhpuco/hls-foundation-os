# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.runner import load_checkpoint
from mmseg.models.builder import BACKBONES, NECKS, HEADS
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Block
from typing import List


def _convTranspose2dOutput(
    input_size: int,
    stride: int,
    padding: int,
    dilation: int,
    kernel_size: int,
    output_padding: int,
):
    """
    Calculate the output size of a ConvTranspose2d.
    Taken from: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    """
    return (
        (input_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_3d_sincos_pos_embed(embed_dim: int, grid_size: tuple, cls_token: bool = False):
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    # All rights reserved.

    # This source code is licensed under the license found in the
    # LICENSE file in the root directory of this source tree.
    # --------------------------------------------------------
    # Position embedding utils
    # --------------------------------------------------------
    """
    grid_size: 3d tuple of grid size: t, h, w
    return:
    pos_embed: L, D
    """

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):
    """Frames of 2D Images to Patch Embedding
    The 3D version of timm.models.vision_transformer.PatchEmbed
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 3,
        tubelet_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: nn.Module = None,
        flatten: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert (
            H == self.img_size[0]
        ), f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert (
            W == self.img_size[1]
        ), f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[3], x.shape[4]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x, Hp, Wp


class Norm2d(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

@NECKS.register_module()
class GeospatialNeck(nn.Module):
    """
    Neck that transforms the token-based output of transformer into a single embedding suitable for processing with standard layers.
    Performs 4 ConvTranspose2d operations on the rearranged input with kernel_size=2 and stride=2
    """

    def __init__(
        self,
        embed_dim: int,
        first_conv_channels: int,
        Hp: int = 14,
        Wp: int = 14,
        channel_reduction_factor: int = 2,
        num_convs: int = 4,
        num_convs_per_upscale: int = 1,
        dropout: bool = False,
        drop_cls_token: bool = True,
    ):
        """

        Args:
            embed_dim (int): Input embedding dimension
            first_conv_channel (int): Number of channels for first dimension
            Hp (int, optional): Height (in patches) of embedding to be upscaled. Defaults to 14.
            Wp (int, optional): Width (in patches) of embedding to be upscaled. Defaults to 14.
            channel_reduction_factor (int): Factor that each convolutional block reduces number of channels by.
            num_convs (int): Number of convolutional upscaling blocks. Each upscales 2x.
            drop_cls_token (bool, optional): Whether there is a cls_token, which should be dropped. This assumes the cls token is the first token. Defaults to True.
        """
        super().__init__()
        self.drop_cls_token = drop_cls_token
        self.Hp = Hp
        self.Wp = Wp
        self.H_out = Hp
        self.W_out = Wp
        self.dropout = dropout

        conv_kernel_size = 3
        conv_padding = 1

        kernel_size = 2
        stride = 2
        dilation = 1
        padding = 0
        output_padding = 0

        self.embed_dim = embed_dim
        self.channels = [first_conv_channels // (channel_reduction_factor ** i) for i in range(num_convs)]
        self.channels = [embed_dim] + self.channels

        for _ in range(len(self.channels) - 1):
            self.H_out = _convTranspose2dOutput(
                self.H_out, stride, padding, dilation, kernel_size, output_padding
            )
            self.W_out = _convTranspose2dOutput(
                self.W_out, stride, padding, dilation, kernel_size, output_padding
            )
        
        def _build_upscale_block(channels_in, channels_out):
            layers = []
            layers.append(nn.ConvTranspose2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ))

            layers += [nn.Sequential(
                      nn.Conv2d(channels_out,
                      channels_out,
                      kernel_size=conv_kernel_size,
                      padding=conv_padding),
                      nn.BatchNorm2d(channels_out),
                      nn.Dropout() if self.dropout else nn.Identity(),
                      nn.ReLU()) for _ in range(num_convs_per_upscale)]

            return nn.Sequential(*layers)

        self.layers = nn.ModuleList([
            _build_upscale_block(self.channels[i], self.channels[i+1])
            for i in range(len(self.channels) - 1)
        ])

    def forward(self, x):
        x = x[0]
        if self.drop_cls_token:
            x = x[:, 1:, :]
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, self.Hp, self.Wp)

        for layer in self.layers:
            x = layer(x)

        x = x.reshape((x.shape[0], self.channels[-1], self.H_out, self.W_out))

        out = tuple([x])

        return out

@NECKS.register_module()
class ConvTransformerTokensToEmbeddingNeck(nn.Module):
    """
    Neck that transforms the token-based output of transformer into a single embedding suitable for processing with standard layers.
    Performs 4 ConvTranspose2d operations on the rearranged input with kernel_size=2 and stride=2
    """

    def __init__(
        self,
        embed_dim: int,
        output_embed_dim: int,
        # num_frames: int = 1,
        Hp: int = 14,
        Wp: int = 14,
        drop_cls_token: bool = True,
    ):
        """

        Args:
            embed_dim (int): Input embedding dimension
            output_embed_dim (int): Output embedding dimension
            Hp (int, optional): Height (in patches) of embedding to be upscaled. Defaults to 14.
            Wp (int, optional): Width (in patches) of embedding to be upscaled. Defaults to 14.
            drop_cls_token (bool, optional): Whether there is a cls_token, which should be dropped. This assumes the cls token is the first token. Defaults to True.
        """
        super().__init__()
        self.drop_cls_token = drop_cls_token
        self.Hp = Hp
        self.Wp = Wp
        self.H_out = Hp
        self.W_out = Wp
        # self.num_frames = num_frames

        kernel_size = 2
        stride = 2
        dilation = 1
        padding = 0
        output_padding = 0
        for _ in range(4):
            self.H_out = _convTranspose2dOutput(
                self.H_out, stride, padding, dilation, kernel_size, output_padding
            )
            self.W_out = _convTranspose2dOutput(
                self.W_out, stride, padding, dilation, kernel_size, output_padding
            )

        self.embed_dim = embed_dim
        self.output_embed_dim = output_embed_dim
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )

    def forward(self, x):
        x = x[0]
        if self.drop_cls_token:
            x = x[:, 1:, :]
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, self.Hp, self.Wp)

        x = self.fpn1(x)
        x = self.fpn2(x)

        x = x.reshape((-1, self.output_embed_dim, self.H_out, self.W_out))

        out = tuple([x])

        return out
    
@NECKS.register_module()
class ConvTransformerTokensToEmbeddingNeckPromptTuning(nn.Module):
    """
    Neck that transforms the token-based output of transformer into a single embedding suitable for processing with standard layers.
    Performs 4 ConvTranspose2d operations on the rearranged input with kernel_size=2 and stride=2
    """

    def __init__(
        self,
        embed_dim: int,
        output_embed_dim: int,
        Hp: int = 14,
        Wp: int = 14,
        drop_cls_token: bool = True,
        num_prompts: int = 0,  # Add num_prompts parameter
    ):
        super().__init__()
        self.drop_cls_token = drop_cls_token
        self.num_prompts = num_prompts  # Store num_prompts
        self.Hp = Hp
        self.Wp = Wp
        self.H_out = Hp
        self.W_out = Wp

        kernel_size = 2
        stride = 2
        dilation = 1
        padding = 0
        output_padding = 0
        for _ in range(4):
            self.H_out = _convTranspose2dOutput(
                self.H_out, stride, padding, dilation, kernel_size, output_padding
            )
            self.W_out = _convTranspose2dOutput(
                self.W_out, stride, padding, dilation, kernel_size, output_padding
            )

        self.embed_dim = embed_dim
        self.output_embed_dim = output_embed_dim
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )

    def forward(self, x):
        x = x[0]  # Input is a tuple, take the first element
        if self.drop_cls_token:
            x = x[:, 1:]  # Drop CLS token
        # Select only the patch tokens, skipping prompt tokens
        num_patch_tokens = self.Hp * self.Wp
        x = x[:, -num_patch_tokens:]  # Take the last Hp*Wp tokens
        x = x.permute(0, 2, 1).reshape(x.shape[0], self.embed_dim, self.Hp, self.Wp)

        x = self.fpn1(x)
        x = self.fpn2(x)

        x = x.reshape((-1, self.output_embed_dim, self.H_out, self.W_out))

        out = tuple([x])
        return out

@BACKBONES.register_module()
class TemporalViTEncoder(nn.Module):
    """Encoder from an ViT with capability to take in temporal input.

    This class defines an encoder taken from a ViT architecture.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_pix_loss: bool = False,
        pretrained: str = None
    ):
        """

        Args:
            img_size (int, optional): Input image size. Defaults to 224.
            patch_size (int, optional): Patch size to be used by the transformer. Defaults to 16.
            num_frames (int, optional): Number of frames (temporal dimension) to be input to the encoder. Defaults to 1.
            tubelet_size (int, optional): Tubelet size used in patch embedding. Defaults to 1.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Embedding dimension. Defaults to 1024.
            depth (int, optional): Encoder depth. Defaults to 24.
            num_heads (int, optional): Number of heads used in the encoder blocks. Defaults to 16.
            mlp_ratio (float, optional): Ratio to be used for the size of the MLP in encoder blocks. Defaults to 4.0.
            norm_layer (nn.Module, optional): Norm layer to be used. Defaults to nn.LayerNorm.
            norm_pix_loss (bool, optional): Whether to use Norm Pix Loss. Defaults to False.
            pretrained (str, optional): Path to pretrained encoder weights. Defaults to None.
        """
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.num_frames = num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.norm_pix_loss = norm_pix_loss
        self.pretrained = pretrained

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if isinstance(self.pretrained, str):
            self.apply(self._init_weights)
            print(f"load from {self.pretrained}")
            load_checkpoint(self, self.pretrained, strict=False, map_location="cpu")
        elif self.pretrained is None:
            # # initialize nn.Linear and nn.LayerNorm
            self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # embed patches
        x, _, _ = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return tuple([x])

def get_3d_sincos_pos_embed_prefix(embed_dim: int, grid_size: tuple, cls_token: bool = False, num_prompts: int = 0):
    """
    grid_size: 3d tuple of grid size: t, h, w
    num_prompts: number of prompt tokens to include in positional embeddings
    return:
    pos_embed: L, D
    """
    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    # Add positional embeddings for CLS token and prompt tokens
    prefix_embed = np.zeros([1 + num_prompts, embed_dim])
    if cls_token:
        pos_embed = np.concatenate([prefix_embed, pos_embed], axis=0)
    return pos_embed
 
 
@BACKBONES.register_module()
class TemporalViTEncoderPromptTuning(nn.Module):
    """Encoder from a ViT with capability to take in temporal input and support prompt tuning.

    This class defines an encoder taken from a ViT architecture.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_pix_loss: bool = False,
        pretrained: str = None,
        num_prompts: int = 0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.num_frames = num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1 + num_prompts, embed_dim), requires_grad=False
        )

        self.num_prompts = num_prompts
        if num_prompts > 0:
            self.prompt_tokens = nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
            nn.init.normal_(self.prompt_tokens, std=0.02)
        else:
            self.prompt_tokens = None

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.norm_pix_loss = norm_pix_loss
        self.pretrained = pretrained

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_3d_sincos_pos_embed_prefix(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True, num_prompts=self.num_prompts
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if isinstance(self.pretrained, str):
            self.apply(self._init_weights)
            print(f"load from {self.pretrained}")
            load_checkpoint(self, self.pretrained, strict=False, map_location="cpu")
        elif self.pretrained is None:
            self.apply(self._init_weights)

        if self.num_prompts > 0:
            for param in self.parameters():
                param.requires_grad = False
            if self.prompt_tokens is not None:
                self.prompt_tokens.requires_grad = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, _, _ = self.patch_embed(x)
        x = x + self.pos_embed[:, 1 + self.num_prompts :, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        if self.num_prompts > 0:
            prompt_tokens = self.prompt_tokens.expand(x.shape[0], -1, -1)
            prompt_tokens = prompt_tokens + self.pos_embed[:, 1 : 1 + self.num_prompts, :]
            x = torch.cat((cls_tokens, prompt_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return tuple([x])
    
 
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
@HEADS.register_module()
class UNetHead(BaseDecodeHead):
    """UNet-style segmentation head for MMSegmentation.

    This head implements a UNet decoder with skip connections, designed to process
    feature maps from the neck and produce segmentation maps.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: list = [512, 256, 128, 64],  # Channels for decoder blocks
        dropout_ratio: float = 0.1,
        norm_cfg: dict = dict(type="BN", requires_grad=True),
        align_corners: bool = False,
        ignore_index: int = 2,
        loss_decode: dict = dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels[0],  # Assuming channels[0] is the desired value
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            norm_cfg=norm_cfg,
            align_corners=align_corners,
            ignore_index=ignore_index,
            loss_decode=loss_decode,
        ) 
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.out_channels = num_classes  # Required by MMSegmentation
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.norm_cfg = norm_cfg
        self.align_corners = align_corners
        self.ignore_index = ignore_index
        self.loss_decode = loss_decode

        # Initial convolution to reduce input channels
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder (contracting path)
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = nn.Sequential(
                nn.Conv2d(channels[i], channels[i], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels[i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.encoder_blocks.append(block)

        # Bottom block (bottleneck)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(inplace=True),
        )

        # Decoder (expansive path) with skip connections
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    channels[i], channels[i - 1], kernel_size=2, stride=2, bias=False
                ),
                nn.BatchNorm2d(channels[i - 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    channels[i - 1] * 2,  # Fix: Expect 2 * channels[i-1] due to concatenation
                    channels[i - 1],
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(channels[i - 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    channels[i - 1],
                    channels[i - 1],
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(channels[i - 1]),
                nn.ReLU(inplace=True),
            )
            self.decoder_blocks.append(block)

        # Final convolution to produce segmentation map
        self.final_conv = nn.Sequential(
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
            nn.Conv2d(channels[0], num_classes, kernel_size=1),
        )

        self.init_weights()

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """Forward pass of the UNet head."""
        x = inputs[0]  # Input from neck (shape: [B, in_channels, H, W])

        # Initial convolution
        x = self.conv_in(x)

        # Encoder path (save skip connections)
        skip_connections = []
        for block in self.encoder_blocks:
            skip_connections.append(x)
            x = block(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        for i, block in enumerate(self.decoder_blocks):
            x = block[0](x)  # Upsample
            skip = skip_connections[-(i + 1)]
            x = torch.cat([x, skip], dim=1)  # Concatenate skip connection
            x = block[1:](x)  # Rest of the block

        # Final convolution
        x = self.final_conv(x)

        return x