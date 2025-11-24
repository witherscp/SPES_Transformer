from loguru import logger
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


# ---------- Core SEEG Transformer for one paradigm ----------
class SEEGTransformer(nn.Module):
    def __init__(self, embed_dim=128, n_heads=4, dropout=0.1, num_layers=1, device="cuda"):
        super().__init__()

        # self.cross_trial = TransformerBlock(embed_dim, n_heads=n_heads)
        # self.cross_channel = TransformerBlock(embed_dim, n_heads=n_heads)

        # Cross-trial self-attention
        trial_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            device=device,
            activation="gelu",
        )
        self.trial_encoder = nn.TransformerEncoder(trial_encoder_layer, num_layers=num_layers)

        # Cross-channel self-attention
        channel_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            device=device,
            activation="gelu",
        )
        self.channel_encoder = nn.TransformerEncoder(channel_encoder_layer, num_layers=num_layers)

        # # Enable gradient checkpointing for memory efficiency
        # for m in self.trial_encoder.layers:
        #     m.gradient_checkpointing = True
        # for m in self.channel_encoder.layers:
        #     m.gradient_checkpointing = True

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

        trial_seq = self.trial_encoder(trial_seq, src_key_padding_mask=key_padding_mask)
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
                        device=key_padding_mask.device,
                    ),
                    key_padding_mask,
                ],
                dim=1,
            )

        channel_seq = self.channel_encoder(channel_seq, src_key_padding_mask=key_padding_mask)

        summary = channel_seq[:, 0]  # CLS token output summary for the entire paradigm
        return summary


# ---------- Full Fusion Model ----------
class SEEGFusionModel(nn.Module):
    def __init__(self, embed_dim=128, n_classes=2, device="cuda"):
        super().__init__()

        self.conv_msresnet = MSResNet(input_channel=1, num_classes=embed_dim, dropout_rate=0.2)
        self.div_msresnet = MSResNet(input_channel=1, num_classes=embed_dim, dropout_rate=0.2)
        self.conv_transformer = SEEGTransformer(embed_dim=embed_dim, n_heads=4, device=device)
        self.div_transformer = SEEGTransformer(embed_dim=embed_dim, n_heads=4, device=device)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
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

        # extract non-padded trials for MSResNet input
        resnet_conv_input = x_conv.reshape(B * n_stims * n_trials, 1, n_timepoints)
        resnet_conv_input = resnet_conv_input[~conv_padding_mask.reshape(B * n_stims * n_trials)]
        resnet_div_input = x_div.reshape(B * n_responses * n_trials, 1, n_timepoints)
        resnet_div_input = resnet_div_input[~div_padding_mask.reshape(B * n_responses * n_trials)]

        # Create embeddings through MSResNet
        resnet_conv_output = self.conv_msresnet(resnet_conv_input)
        resnet_div_output = self.div_msresnet(resnet_div_input)

        # Reshape back to [B, n_electrodes, n_trials, input_dim]
        conv_embeddings = torch.zeros(
            (B * n_stims * n_trials, resnet_conv_output.shape[1]),
            device=resnet_conv_output.device,
            dtype=resnet_conv_output.dtype,  # match bf16 under autocast
        )
        conv_embeddings[~conv_padding_mask.reshape(B * n_stims * n_trials)] = resnet_conv_output
        conv_embeddings = conv_embeddings.view(B, n_stims, n_trials, -1)
        div_embeddings = torch.zeros(
            (B * n_responses * n_trials, resnet_div_output.shape[1]),
            device=resnet_div_output.device,
            dtype=resnet_div_output.dtype,  # match bf16 under autocast
        )
        div_embeddings[~div_padding_mask.reshape(B * n_responses * n_trials)] = resnet_div_output
        div_embeddings = div_embeddings.view(B, n_responses, n_trials, -1)

        # Create convergent and divergent embeddings through respective transformer blocks
        # output will have shape [B, embed_dim]
        conv_emb = self.conv_transformer(conv_embeddings, key_padding_mask=conv_padding_mask)
        div_emb = self.div_transformer(div_embeddings, key_padding_mask=div_padding_mask)

        # Join embeddings
        joint_emb = torch.cat([conv_emb, div_emb], dim=1)

        logits = self.classifier(joint_emb)
        return logits


class BaselineModel(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        n_classes=2,
        stim_model="convergent",
        n_elecs=25,
        generator=None,
    ):
        super().__init__()

        assert stim_model in ["convergent", "divergent"]
        self.stim_model = stim_model
        self.g = generator
        self.n_elecs = n_elecs

        self.msresnet = MSResNet(input_channel=self.elecs, num_classes=embed_dim, dropout_rate=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, n_classes),
        )

    def forward(self, inputs):

        # unpack inputs from dictionary
        x = inputs[self.stim_model]  # [B, n_electrodes, n_trials, n_timepoints]
        padding_mask = inputs[f"{self.stim_model}_mask"]  # [B, n_electrodes, n_trials]

        # get dimensions from arrays
        B, _, _, n_timepoints = x.shape

        # default array to fill
        resnet_input = torch.zeros(B, self.n_elecs, n_timepoints)

        # select random trials for each target (max 1 trial per 1 electrode)
        target_inds, elec_inds, trial_inds = torch.where(~padding_mask)
        for target_ind in range(B):
            target_mask = target_inds == target_ind
            possible_elecs = elec_inds[target_mask].unique()
            for i in range(self.n_elecs):
                if len(possible_elecs) == 0:
                    logger.warning(
                        f"Ran out of electrodes to select for target {target_ind} at elec index {i}. Results will be invalid. Reduce n_elecs to {i} and re-run."
                    )
                    break

                # randomly select an electrode index
                rand_ind = torch.randint(0, len(possible_elecs), (1,), generator=self.g)
                elec_ind = possible_elecs[rand_ind]
                elec_mask = elec_inds[target_mask] == elec_ind

                # randomly select a trial for that electrode
                rand_ind = torch.randint(0, len(trial_inds[target_mask][elec_mask]), (1,))
                trial_ind = trial_inds[target_mask][elec_mask][rand_ind]
                resnet_input[target_ind, i] = x[target_ind, elec_ind, trial_ind, :]

                # remove elec from possible selections
                possible_elecs = possible_elecs[possible_elecs != elec_ind]

        # Create embeddings through MSResNet
        resnet_output = self.msresnet(resnet_input)

        logits = self.classifier(resnet_output)
        return logits
