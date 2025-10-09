import torch
from torch import nn as nn

class PatchEmbed(nn.Module):
    
    def __init__(self, im_size, patch_size, in_chs, emb_dim):
        super().__init__()
        
        self.im_size = im_size
        self.patch_size = patch_size
        self.n_patches = (im_size // patch_size) ** 2 # (224 // 8) ** 2 = 784 
        
        self.projection = nn.Conv2d(in_channels = in_chs, out_channels = emb_dim,
                                    kernel_size = patch_size, stride = patch_size)
        
    def forward(self, inp):
        
        proj = self.projection(inp) # (bs, emb_dim, self.n_patches ** 0.5, self.n_patches ** 0.5)
        proj = proj.flatten(start_dim = 2) # (bs, emb_dim, self.n_patches)
         
        return proj.transpose(1, 2) # (bs, self.n_patches, emb_dim) 

# inp = torch.rand(1, 3, 224, 224)
# m = PatchEmbed(im_size = 224, patch_size = 8, in_chs = 3, emb_dim = 32) 
# print(m(inp).shape)

class MLP(nn.Module):
    
    def __init__(self, in_fs, hid_fs, out_fs, p = 0):
        super().__init__()
        
        self.lin_1 = nn.Linear(in_features = in_fs, out_features = hid_fs)
        self.act = nn.GELU()
        self.lin_2 = nn.Linear(in_features = hid_fs, out_features = out_fs)
        self.drop = nn.Dropout(p = p) # to overcome overfitting - fits only for training data -> cannot generalize on test data
        
    def forward(self, inp):
        
        out_1 = self.drop(self.act(self.lin_1(inp)))
        out_2 = self.drop(self.lin_2(out_1))
        
        return out_2
    
# inp = torch.rand(1, 784)
# m = MLP(in_fs = 784, hid_fs = 256, out_fs = 100, p = 0)
# print(m(inp).shape)

class Attention(nn.Module):
    
    def __init__(self, dim, n_heads, qkv_bias = True, attn_p = 0, proj_p = 0):
        super().__init__()
        
        self.n_heads, self.dim, self.head_dim = n_heads, dim, dim // n_heads
        self.scale = self.head_dim ** -0.5 # from paper
        self.qkv = nn.Linear(in_features = dim, out_features = dim * 3, bias = qkv_bias)
        self.proj = nn.Linear(in_features = dim, out_features = dim)
        self.attn_drop, self.proj_drop = nn.Dropout(attn_p), nn.Dropout(proj_p)
        
    def forward(self, inp):
        
        batch, n_tokens, dim = inp.shape # (bs, self.n_patches, emb_dim) 
        
        qkv = self.qkv(inp) # (batch, n_ps + 1, dim) -> (batch, n_ps + 1, 3 * dim)
        qkv = qkv.reshape(batch, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch, n_heads, n_ps + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # q, k, v shapes are the same -> (batch, n_heads, n_ps + 1, head_dim)
        k_t = k.transpose(-2, -1) # (batch, n_heads, head_dim, n_ps + 1)
        qk = (q @ k_t) * self.scale # scaling
        qk = qk.softmax(dim = -1) # (batch, n_heads, n_ps + 1, n_ps + 1)
        qk = self.attn_drop(qk) # (batch, n_heads, n_ps + 1, n_ps + 1)
        
        attn = qk @ v # # (batch, n_heads, n_ps + 1, n_ps + 1) @ (batch, n_heads, n_ps + 1, head_dim) -> (batch, n_heads, n_ps + 1, head_dim)
        attn = attn.transpose(1, 2) # (batch, n_heads, n_ps + 1, head_dim) -> (batch, n_ps + 1, n_heads, head_dim)
        attn = attn.flatten(2) # (batch, n_ps + 1, n_heads, head_dim) -> (batch, n_ps + 1, n_heads * head_dim = dim)
        
        out = self.proj(attn) # (batch, n_ps + 1, dim)
        out = self.proj_drop(out)
        
        return out        # (batch, n_ps + 1, dim)
        

class Block(nn.Module):
    
    def __init__(self, dim, n_heads, mlp_ratio = 4., qkv_bias = True, attn_p = 0, proj_p = 0):
        super().__init__()
        
        self.norm_1 = nn.LayerNorm(normalized_shape = dim)
        self.attn = Attention(dim = dim, n_heads = n_heads, qkv_bias = qkv_bias, attn_p = attn_p, proj_p = proj_p) # 
        self.norm_2 = nn.LayerNorm(normalized_shape = dim)
        
        hid_fs = int(dim * mlp_ratio)
        self.mlp = MLP(in_fs = dim, hid_fs = hid_fs, out_fs = dim, p = 0)
        
    def forward(self, inp):
        
        out_1 = inp + self.attn(self.norm_1(inp))
        out_2 = out_1 + self.mlp(self.norm_2(out_1))
        
        return out_2 
        
        # out_2 = self.norm_2(out_1)
        # out_2 = self.mlp(out_2)
        # out_2 = out_2 + out_1
        
class VIT(nn.Module):
    
    def __init__(self, im_size, patch_size, in_chs, emb_dim, p, n_heads, mlp_ratio, qkv_bias, attn_p, proj_p, depth, n_cls):
        super().__init__()
        
        self.patch_embed = PatchEmbed(im_size = im_size, patch_size = patch_size, in_chs = in_chs, emb_dim = emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim)) # 0 token trainable
        self.pos_emb = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, emb_dim))
        self.pos_drop = nn.Dropout(p = p)
        
        self.blocks = nn.ModuleList(
        [Block(dim = emb_dim, n_heads = n_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, attn_p = attn_p, proj_p = proj_p) for _ in range(depth)]
        ) # torch doesnt recognize python list () 
        
        self.norm = nn.LayerNorm(emb_dim, eps = 1e-6)
        self.head = nn.Linear(in_features = emb_dim, out_features = n_cls)
        
    def forward(self, inp):
        
        bs = inp.shape[0]
        inp = self.patch_embed(inp) # (bs, n_ps, emb_dim)
        
        cls_token = self.cls_token.expand(bs, -1, -1) # (bs, 1, emb_dim)
        inp = torch.cat((cls_token, inp), dim = 1) # (bs, n_ps + 1, emb_dim)
        
        inp = inp + self.pos_emb
        inp = self.pos_drop(inp)
        
        for block in self.blocks: inp = block(inp)
        
        inp = self.norm(inp)
        cls_token_out = inp[:, 0] # only class token
        
        out = self.head(cls_token_out)
        
        return out
    
inp = torch.rand(1, 3, 384, 384)
m = VIT(im_size = 384, patch_size = 16, in_chs = 3, emb_dim = 768, p = 0, n_heads = 12, mlp_ratio = 4., qkv_bias = True, attn_p = 0, proj_p = 0, depth = 10, n_cls = 10)
print(m(inp).shape)