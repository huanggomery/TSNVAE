import torch
import torch.nn as nn
import warnings
import math


class VTT(nn.Module):
    def __init__(self, img_size=[240], img_patch_size=80,
                 tactile_size=[120], tactile_patch_size=60,
                 sequence=8, in_chans=3, embed_dim=512, depth=6,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], tactile_size=tactile_size[0],
            img_patch_size=img_patch_size, tactile_patch_size=tactile_patch_size,
            in_chan=in_chans, embeded_dim=embed_dim
        )
        img_patches = self.patch_embed.img_patches
        tactile_patches = self.patch_embed.tactile_patches

        # success embedding, position embedding
        self.success_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_patches + tactile_patches)*sequence + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.compress_patches = nn.Sequential(nn.Linear(embed_dim, embed_dim//4),
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(embed_dim//4, embed_dim//16))

        self.compress_layer = nn.Sequential(nn.Linear(((img_patches + tactile_patches)*sequence + 1)*embed_dim//16, 1024),
                                          nn.LeakyReLU(0.2, inplace=True),
                                          nn.Linear(1024, 256))

        self.pos_err_perdiction = nn.Linear(256, 2)
        self.success_recognition = nn.Sequential(nn.Linear(embed_dim, 1),
                                               nn.Sigmoid())

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.success_embed, std=.02)

    def interpolate_pos_encoding(self, x, w: int, h: int):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        else:
            raise ValueError('Position Encoder does not match dimension')

    def prepare_tokens(self, x, tactile):
        B, S, nc, w, h = x.shape
        x, patched_tactile = self.patch_embed(x, tactile)
        x = torch.cat((x, patched_tactile),dim=1)
        success_embed = self.success_embed.expand(B, -1, -1)
        # introduce success embedding
        x = torch.cat((success_embed, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def forward(self, x, tactile, return_attention: bool = False):
        attn_layers = []
        x = self.prepare_tokens(x, tactile)
        for blk in self.blocks:
            x, attn = blk(x, return_attention=True)
            if return_attention:
                attn_layers.append(attn)
        attentions = torch.stack(attn_layers, dim=1)
        x = self.norm(x)
        img_tactile = self.compress_patches(x)
        B, patches, dim = img_tactile.size()
        img_tactile = img_tactile.view(B, -1)
        img_tactile = self.compress_layer(img_tactile)

        if return_attention:
            return self.pos_err_perdiction(img_tactile), self.success_recognition(x[:, 0]), attentions
        return self.pos_err_perdiction(img_tactile), self.success_recognition(x[:, 0])


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        attn = attn.view(B, -1, N, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, return_attention: bool = False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attn
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=240, tactile_size = 120,
                 img_patch_size=80, tactile_patch_size=60, in_chan=3, embeded_dim=512):
        super().__init__()
        self.img_patches = int((img_size/img_patch_size)*(img_size/img_patch_size))
        self.tactile_patches = int((tactile_size/tactile_patch_size)*(tactile_size/tactile_patch_size))
        self.img_size = img_size
        self.tactile_size = tactile_size
        self.embeded_dim = embeded_dim
        self.img_proj = nn.Conv2d(in_chan, embeded_dim, kernel_size=img_patch_size, stride=img_patch_size)
        self.tactile_proj = nn.Conv2d(in_chan, embeded_dim, kernel_size=tactile_patch_size, stride=tactile_patch_size)

    def forward(self, image, tactile):
        # Input shape: batch, Sequence, in_Channels, H, W
        # Output shape: batch, patches, embeded_dim
        B, S, C, H, W = image.shape
        image = image.view(B * S, C, H, W)
        pached_image = self.img_proj(image).flatten(2).transpose(1, 2).reshape(B, -1, self.embeded_dim)

        B, S, C, H, W = tactile.shape
        tactile = tactile.view(B * S, C, H, W)
        pached_tactile = self.tactile_proj(tactile).flatten(2).transpose(1, 2).reshape(B, -1, self.embeded_dim)

        return pached_image, pached_tactile


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,)*(x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.MLP = nn.Sequential(nn.Linear(in_features, hidden_features),
                            act_layer(),
                            nn.Linear(hidden_features, out_features))
    def forward(self, x):
        x = self.MLP(x)
        return x


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


if __name__ == "__main__":
    img = torch.zeros((32, 9, 3, 240, 240))
    tac = torch.zeros((32, 9, 3, 120, 120))
    vtt = VTT(sequence=9)
    y, success = vtt(img, tac)
    print(y.shape)
    print(success.shape)