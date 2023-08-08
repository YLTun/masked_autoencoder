import torch
import torch.nn as nn

from einops import repeat, rearrange
from einops.layers.torch import Reduce, Rearrange

from timm.models.vision_transformer import Block


class PatchShuffle(nn.Module):
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio
        
    def forward(self, x):
        b, n, d = x.shape
        len_keep = int(n * (1 - self.mask_ratio))
        noise = torch.rand(b, n, device=x.device)
        
        # Sort noise for each sample.
        ids_shuffle = torch.argsort(noise, dim=1)           # Ascend: small is keep, large is remove.
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset. 
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))
        
        # Generate the binary mask.
        loss_mask = torch.ones([b, n], device=x.device)
        loss_mask[:, :len_keep] = 0
        
        # Unshuffle to get the binary mask.
        loss_mask = torch.gather(loss_mask, dim=1, index=ids_restore)
        
        return x_masked, loss_mask, ids_restore
        
        
class Encoder(nn.Module):
    def __init__(self, emb_dim=192, img_size=32, in_channels=3, patch_size=2, mask_ratio=0.75, num_heads=3, depth=12):
        super().__init__()
        
        if emb_dim == None:
            emb_dim = in_channels * patch_size * patch_size
            
        self.patchify = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, (img_size // patch_size) **2 + 1, emb_dim))

        self.patch_shuffle = PatchShuffle(mask_ratio)
        
        self.transformer = nn.Sequential(
            *[Block(emb_dim, num_heads) for _ in range(depth)]
        )
        self.layer_norm = nn.LayerNorm(emb_dim)
        
        self.init_weight()
        
    def init_weight(self):
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_emb, std=.02)
        
    def forward(self, x):
        b, _, _, _ = x.shape
        
        x = self.patchify(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x += self.pos_emb[:, 1:, :]                           # Add position embedding.
        
        # Mask patches.
        x, loss_mask, ids_restore  = self.patch_shuffle(x)         
        
        # Append cls token.
        cls_token = self.cls_token + self.pos_emb[:, :1, :]
        cls_token = repeat(cls_token, '() n e -> b n e', b=b)
        x = torch.cat((cls_token, x), dim=1)
        
        # Apply transformer blocks.
        x = self.transformer(x)
        x = self.layer_norm(x)
        
        return x, loss_mask, ids_restore
    
    
class Decoder(nn.Module):
    def __init__(self, emb_dim=192, img_size=32, in_channels=3, patch_size=2, decoder_emb_dim=None, num_heads=3, depth=4):
        super().__init__()
        
        if decoder_emb_dim == None: decoder_emb_dim = emb_dim
        self.decoder_embed = nn.Linear(emb_dim, decoder_emb_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_emb_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, (img_size // patch_size) **2 + 1, emb_dim))
        
        self.transformer = nn.Sequential(
            *[Block(decoder_emb_dim, num_heads) for _ in range(depth)]
        )
        
        self.layer_norm = nn.LayerNorm(decoder_emb_dim)
        self.head = torch.nn.Linear(decoder_emb_dim, patch_size ** 2 * in_channels)
        self.unpatchify = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)' , p1=patch_size, p2=patch_size, c=in_channels, h=img_size//patch_size, w=img_size//patch_size)
        
    def init_weight(self):
        nn.init.trunc_normal_(self.mask_token, std=.02)
        nn.init.trunc_normal_(self.pos_emb, std=.02)
        
    def forward(self, x, ids_restore):
        # Embed tokens.
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence.
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)            # No cls token.
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # Unshuffle.
        x = torch.cat([x[:, :1, :], x_], dim=1)    # Append cls token.

        x += self.pos_emb                          # Add pos embed.
        x = self.transformer(x)
        x = self.layer_norm(x)
        x = self.head(x)
        
        x = x [:, 1:, :]                          # Remove cls token
        x = self.unpatchify(x)
        return x
    
        
class MaskedAutoencoder(nn.Module):
    def __init__(self, emb_dim=192, img_size=32, in_channels=3, patch_size=2, 
                 mask_ratio=0.75, en_num_heads=3, en_depth=12, 
                 de_emb_dim=None, de_num_heads=3, de_depth=4, 
                 use_norm_pix_loss=True):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        self.use_norm_pix_loss = use_norm_pix_loss
        self.flatten_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2) ' , p1=patch_size, p2=patch_size, c=in_channels, h=img_size//patch_size, w=img_size//patch_size)
        
        self.encoder = Encoder(
            emb_dim=emb_dim, img_size=img_size, in_channels=in_channels, patch_size=patch_size, 
            mask_ratio=mask_ratio, num_heads=en_num_heads, depth=en_depth
        )
        
        self.decoder = Decoder(
            emb_dim=emb_dim, img_size=img_size, in_channels=in_channels, patch_size=patch_size, 
            decoder_emb_dim=de_emb_dim, num_heads=de_num_heads, depth=de_depth
        )
        
    def masked_loss(self, x, x_pred, loss_mask):
        """
        x_org: [b, c, h, w]
        x_pred: [b, c, h, w]
        loss_mask: [b, c, h, w]
        """
        
        if self.use_norm_pix_loss:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            x = (x - mean) / (var + 1.e-6)**.5
            
        loss = torch.mean((x_pred - x) ** 2 * loss_mask) / self.mask_ratio
        return loss
    
    def forward(self, x):
        x, loss_mask, ids_restore = self.encoder(x)
        x_pred = self.decoder(x, ids_restore)
        
        loss_mask = loss_mask.unsqueeze(-1).repeat(1, 1, self.patch_size ** 2 * self.in_channels)
        loss_mask = rearrange(loss_mask, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', 
                              p1=self.patch_size, p2=self.patch_size, c=self.in_channels, h=self.img_size//self.patch_size, w=self.img_size//self.patch_size)
        
        return x_pred, loss_mask


class ClassificationModel(nn.Module):
    def __init__(self,  emb_dim=192, img_size=32, in_channels=3, patch_size=2, mask_ratio=0.0, en_num_heads=3, en_depth=12, num_classes=10):
        super().__init__()
        
        self.encoder = Encoder(
            emb_dim=emb_dim, img_size=img_size, in_channels=in_channels, patch_size=patch_size, 
            mask_ratio=mask_ratio, num_heads=en_num_heads, depth=en_depth
        )
        
        self.classification_head = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.Linear(emb_dim, num_classes)
        )
        
    def forward(self, x):
        x, _, _, = self.encoder(x)
        x = self.classification_head(x)
        return x
        
        
        
        
        
        
        
        
        
        
        