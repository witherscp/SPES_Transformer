import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.multi_scale_ori import MSResNet


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1, attn_dropout=0.1, resid_scale=0.5):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=True, dropout=attn_dropout
        )
        self.attn_drop = nn.Dropout(attn_dropout)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.pre_norm = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Residual scaling (helps prevent blow-ups; small, learnable gain)
        self.resid_scale1 = nn.Parameter(torch.tensor(resid_scale))
        self.resid_scale2 = nn.Parameter(torch.tensor(resid_scale))

    def forward(self, x, key_padding_mask=None):
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(-1)
            x = x.masked_fill(mask, 0.0)

        # pre-norm + bound for stability
        x_norm = torch.tanh(self.pre_norm(x))

        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, need_weights=False
        )
        attn_out = self.attn_drop(torch.nan_to_num(attn_out))

        if key_padding_mask is not None:
            attn_out = torch.where(mask, torch.zeros_like(attn_out), attn_out)

        x = self.norm1(x_norm + self.resid_scale1 * attn_out)

        ff_out = self.ff(x)
        if key_padding_mask is not None:
            ff_out = ff_out.masked_fill(mask, 0.0)

        x = self.norm2(x + self.resid_scale2 * ff_out)

        if key_padding_mask is not None:
            x = x.masked_fill(mask, 0.0)

        x = torch.nan_to_num(x)

        return x


# ---------- Cross-Attention Fusion block ----------
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, n_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query, context):
        """
        query: [B, 1, dim]  (cls token or summary from one modality)
        context: [B, N, dim] (sequence from another modality)
        """
        attn_out, _ = self.cross_attn(query, context, context)
        out = self.norm(query + attn_out)
        return out


# ---------- Core SEEG Transformer for one paradigm ----------
class SEEGTransformer(nn.Module):
    def __init__(self, embed_dim=128, n_heads=4, device="cuda"):
        super().__init__()

        # Cross-trial self-attention
        self.cross_trial = TransformerBlock(embed_dim, n_heads=n_heads)

        # Cross-channel self-attention
        self.cross_channel = TransformerBlock(embed_dim, n_heads=n_heads)

        # CLS tokens for summarization
        self.cls_token1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, 1, embed_dim))).to(
            device
        )
        self.cls_token2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, 1, embed_dim))).to(
            device
        )

        # # Position embeddings (optional)
        # self.pos_embed_trials = nn.Parameter(torch.randn(1, n_trials + 1, embed_dim))
        # self.pos_embed_channels = nn.Parameter(torch.randn(1, n_electrodes + 1, embed_dim))

    def forward(self, x, key_padding_mask=None):
        """
        x: shape [B, n_electrodes, n_trials, n_features (input_dim)]
        """

        B, n_electrodes, n_trials, n_features = x.shape

        # ---- Step 1: Cross-trial attention (within each electrode) ----
        # x = x + self.pos_embed_trials[:, 1 : n_trials + 1]

        x = x.view(B * n_electrodes, n_trials, n_features)

        # include CLS token
        cls_token1 = self.cls_token1.expand(B * n_electrodes, -1, -1)
        trial_seq = torch.cat([cls_token1, x], dim=1)

        # update padding mask to account for cls token and shape change
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(B * n_electrodes, -1)
            # the CLS token is masked for electrodes with all trials masked
            key_padding_mask = torch.cat(
                [torch.unsqueeze(key_padding_mask[:, 0], -1), key_padding_mask], dim=1
            )

        trial_seq = self.cross_trial(trial_seq, key_padding_mask=key_padding_mask)
        electrode_emb = trial_seq[:, 0]  # take cls token per electrode

        # ---- Step 2: Cross-channel attention (across electrodes) ----
        electrode_emb = electrode_emb.view(B, n_electrodes, -1)
        cls_token2 = self.cls_token2.expand(B, -1, -1)
        channel_seq = torch.cat([cls_token2, electrode_emb], dim=1)

        # channel_seq = channel_seq + self.pos_embed_channels[:, : channel_seq.size(1)]

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, 0].view(B, n_electrodes)
            # the CLS token is never masked
            key_padding_mask = torch.cat(
                [
                    torch.full(
                        (key_padding_mask.shape[0], 1),
                        False,
                        dtype=torch.bool,
                        device=channel_seq.device,
                    ),
                    key_padding_mask,
                ],
                dim=1,
            )

        channel_seq = self.cross_channel(channel_seq, key_padding_mask=key_padding_mask)

        summary = channel_seq[:, 0]  # CLS token output summary for the entire paradigm
        return summary


# ---------- Full Fusion Model ----------
class SEEGFusionModel(nn.Module):
    def __init__(self, embed_dim=128, n_classes=2, device="cuda"):
        super().__init__()

        self.conv_msresnet = MSResNet(input_channel=1, num_classes=embed_dim, dropout_rate=0.2)
        self.div_msresnet = MSResNet(input_channel=1, num_classes=embed_dim, dropout_rate=0.2)
        self.conv_block = SEEGTransformer(embed_dim=embed_dim, n_heads=4, device=device)
        self.div_block = SEEGTransformer(embed_dim=embed_dim, n_heads=4, device=device)

        self.fusion = CrossAttentionFusion(embed_dim=embed_dim, n_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, inputs):

        # unpack inputs from dictionary
        x_conv = inputs["convergent"]  # [B, n_stims, n_trials, n_timepoints]
        x_div = inputs["divergent"]  # [B, n_responses, n_trials, n_timepoints]
        conv_padding_mask = inputs["convergent_mask"]  # [B, n_stims, n_trials]
        div_padding_mask = inputs["divergent_mask"]  # [B, n_responses, n_trials]

        # get dimensions from arrays
        B, n_stims, n_trials, n_timepoints = x_conv.shape
        n_responses = x_div.shape[1]

        # extract non-NaN trials for MSResNet input
        resnet_conv_input = x_conv.reshape(B * n_stims * n_trials, 1, n_timepoints)
        resnet_conv_input = resnet_conv_input[~conv_padding_mask.reshape(B * n_stims * n_trials)]
        resnet_div_input = x_div.reshape(B * n_responses * n_trials, 1, n_timepoints)
        resnet_div_input = resnet_div_input[~div_padding_mask.reshape(B * n_responses * n_trials)]

        # Create embeddings through MSResNet
        resnet_conv_output = self.conv_msresnet(resnet_conv_input)
        resnet_div_output = self.div_msresnet(resnet_div_input)

        # TODO: remove debugging once I fix large min/max issue
        print(
            "resnet_conv_output: mean {:.3e}, std {:.3e}, min {:.3e}, max {:.3e}".format(
                resnet_conv_output.mean().item(),
                resnet_conv_output.std().item(),
                resnet_conv_output.min().item(),
                resnet_conv_output.max().item(),
            )
        )

        # Reshape back to [B, n_electrodes, n_trials, input_dim]
        conv_embeddings = torch.zeros(
            (B * n_stims * n_trials, resnet_conv_output.shape[1]), device=resnet_conv_output.device
        )
        conv_embeddings[~conv_padding_mask.reshape(B * n_stims * n_trials)] = resnet_conv_output
        conv_embeddings = conv_embeddings.view(B, n_stims, n_trials, -1)
        div_embeddings = torch.zeros(
            (B * n_responses * n_trials, resnet_div_output.shape[1]),
            device=resnet_div_output.device,
        )
        div_embeddings[~div_padding_mask.reshape(B * n_responses * n_trials)] = resnet_div_output
        div_embeddings = div_embeddings.view(B, n_responses, n_trials, -1)

        # Create convergent and divergent embeddings through respective transformer blocks
        # output will have shape [B, embed_dim]
        conv_emb = self.conv_block(conv_embeddings, key_padding_mask=conv_padding_mask)
        div_emb = self.div_block(div_embeddings, key_padding_mask=div_padding_mask)

        # Add singleton dimension for cross-attention
        # Now shape: [B, 1, embed_dim]
        conv_emb = conv_emb.unsqueeze(1)
        div_emb = div_emb.unsqueeze(1)

        # Fuse: CLS from one attends to the other
        # TODO: remove asymmetry by doing self-attention?
        fused = self.fusion(conv_emb, div_emb)
        fused = fused.squeeze(1)

        logits = self.classifier(fused)
        return logits
