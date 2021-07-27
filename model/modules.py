import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

from .blocks import (
    SwishBlock,
    LinearNorm,
    LConvBlock,
    ConvBlock,
    ConvNorm,
    FFTBlock,
    VariableLengthAttention,
    MultiHeadAttention,
)
from text.symbols import symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class TextEncoder(nn.Module):
    """ Text Encoder """

    def __init__(self, config):
        super(TextEncoder, self).__init__()
        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["text_encoder"]["encoder_hidden"]
        n_layers_conv = config["text_encoder"]["conv_layer"]
        n_layers_trans = config["text_encoder"]["trans_layer"]
        n_head = config["text_encoder"]["trans_head"]
        d_k = d_v = (
            config["text_encoder"]["encoder_hidden"]
            // config["text_encoder"]["trans_head"]
        )
        d_encoder = config["text_encoder"]["encoder_hidden"]
        d_inner = config["text_encoder"]["trans_filter_size"]
        kernel_size_conv = config["text_encoder"]["conv_kernel_size"]
        kernel_size_trans = config["text_encoder"]["trans_kernel_size"]
        dropout = config["text_encoder"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_encoder = d_encoder

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=0
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.convolution_stack = nn.ModuleList(
            [
                ConvBlock(
                    d_encoder, d_encoder, kernel_size_conv, dropout=dropout 
                )
                for _ in range(n_layers_conv)
            ]
        )

        self.transformer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_encoder, n_head, d_k, d_v, d_inner, kernel_size_trans, dropout=dropout
                )
                for _ in range(n_layers_trans)
            ]
        )

    def forward(self, src_seq, mask):

        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        enc_output = self.src_word_emb(src_seq)

        for enc_layer in self.convolution_stack:
            enc_output = enc_layer(
                enc_output, mask=mask
            )

        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = enc_output + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_encoder
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = enc_output + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.transformer_stack:
            enc_output, _ = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )

        return enc_output


class ResidualEncoder(nn.Module):
    """ Residual Encoder """

    def __init__(self, config):
        super(ResidualEncoder, self).__init__()
        n_position = config["max_seq_len"] + 1
        n_mel_channels = config["n_mel_channels"]
        d_encoder_t = config["text_encoder"]["encoder_hidden"]
        d_encoder = config["residual_encoder"]["encoder_hidden"]
        d_latent = config["residual_encoder"]["latent_dim"]
        d_neck = config["residual_encoder"]["bottleneck_size"]
        n_layers = config["residual_encoder"]["conv_layer"]
        n_head = config["residual_encoder"]["conv_head"]
        kernel_size = config["residual_encoder"]["conv_kernel_size"]
        dropout = config["residual_encoder"]["conv_dropout"]
        speaker_embed_size = config["speaker_embed_size"]

        self.max_seq_len = config["max_seq_len"]
        self.d_encoder = d_encoder
        self.d_latent = d_latent
        self.text_linear = LinearNorm(d_encoder_t, d_encoder)
        self.layer_norm1 = nn.LayerNorm(d_encoder)

        self.input_linear = nn.Sequential(
            LinearNorm(
                d_encoder + speaker_embed_size + n_mel_channels, d_encoder
            ),
            nn.ReLU()
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_encoder).unsqueeze(0),
            requires_grad=False,
        )

        self.convolution_stack = nn.ModuleList(
            [
                LConvBlock(
                    d_encoder, kernel_size, n_head, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.attention = VariableLengthAttention(
            d_encoder, d_encoder, d_encoder_t, dropout=dropout
        )

        self.fc_mu = LinearNorm(d_encoder, d_latent)
        self.fc_var = LinearNorm(d_encoder, d_latent)

        self.output_linear = nn.Sequential(
            LinearNorm(
                d_latent + speaker_embed_size + d_encoder, d_neck
            ),
            nn.Tanh()
        )
        self.layer_norm2 = nn.LayerNorm(d_neck)
    
    def reparameterize(self, mu, logvar):
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, mel, text_encoding, mel_mask, src_mask, max_mel_len, max_src_len, speaker_embedding):

        batch_size = text_encoding.shape[0]
        text_encoding = self.layer_norm1(self.text_linear(text_encoding))

        if not self.training and mel is None:
            mu = log_var = torch.zeros([batch_size, max_src_len, self.d_latent], device=device)
            attn = None
        else:
            speaker_embedding_m = speaker_embedding.unsqueeze(1).expand(
                -1, max_mel_len, -1
            )

            position_enc = self.position_enc[
                :, :max_mel_len, :
            ].expand(batch_size, -1, -1)

            enc_input = torch.cat([position_enc, speaker_embedding_m, mel], dim=-1)
            enc_output = self.input_linear(enc_input)

            for enc_layer in self.convolution_stack:
                enc_output = enc_layer(
                    enc_output, mask=mel_mask
                )

            enc_output, attn = self.attention(enc_output, text_encoding, mel_mask, src_mask)

            mu = self.fc_mu(enc_output).masked_fill(src_mask.unsqueeze(-1), 0)
            log_var = self.fc_var(enc_output).masked_fill(src_mask.unsqueeze(-1), 0)

        # Phoneme-Level Fine-Grained VAE
        z = self.reparameterize(mu, log_var)
        z = z.masked_fill(src_mask.unsqueeze(-1), 0)

        speaker_embedding_t = speaker_embedding.unsqueeze(1).expand(
            -1, max_src_len, -1
        )
        enc_output = torch.cat([z, speaker_embedding_t, text_encoding], dim=-1)
        enc_output = self.layer_norm2(self.output_linear(enc_output))
        enc_output = enc_output.masked_fill(src_mask.unsqueeze(-1), 0)

        return enc_output, attn, mu, log_var


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, config):
        super(DurationPredictor, self).__init__()
        d_predictor = config["duration_predictor"]["predictor_hidden"]
        d_encoder_t = config["text_encoder"]["encoder_hidden"]
        d_encoder_r = config["residual_encoder"]["bottleneck_size"]
        n_layers = config["duration_predictor"]["conv_layer"]
        n_head = config["duration_predictor"]["conv_head"]
        kernel_size = config["duration_predictor"]["conv_kernel_size"]
        dropout = config["duration_predictor"]["conv_dropout"]
        speaker_embed_size = config["speaker_embed_size"]

        self.input_linear = LinearNorm(
            d_encoder_t + d_encoder_r + speaker_embed_size, d_predictor
        )

        self.convolution_stack = nn.ModuleList(
            [
                LConvBlock(
                    d_predictor, kernel_size, n_head, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.projection = LinearNorm(d_predictor, 1)
    
    def forward(self, x, mask=None):
        
        V = self.input_linear(x)
        for enc_layer in self.convolution_stack:
            V = enc_layer(
                V, mask=mask
            )

        duration = F.softplus(self.projection(V))
        if mask is not None:
            duration = duration.masked_fill(mask.unsqueeze(-1), 0)
        duration = duration.squeeze(-1)

        return duration, V


class LearnedUpsampling(nn.Module):
    """ Learned Upsampling """

    def __init__(self, config):
        super(LearnedUpsampling, self).__init__()
        d_predictor = config["duration_predictor"]["predictor_hidden"]
        kernel_size = config["learned_upsampling"]["conv_kernel_size"]
        dropout = config["learned_upsampling"]["conv_dropout"]
        conv_output_size = config["learned_upsampling"]["conv_output_size"]
        dim_w = config["learned_upsampling"]["linear_output_size_w"]
        dim_c = config["learned_upsampling"]["linear_output_size_c"]

        self.max_seq_len = config["max_seq_len"]

        # Attention (W)
        self.conv_w = ConvBlock(
            d_predictor, conv_output_size, kernel_size, dropout=dropout, activation=nn.SiLU()
        )
        self.swish_w = SwishBlock(
            conv_output_size+2, dim_w, dim_w
        )
        self.linear_w = LinearNorm(dim_w, 1, bias=True)
        self.softmax_w = nn.Softmax(dim=2)

        # Auxiliary Attention Context (C)
        self.conv_c = ConvBlock(
            d_predictor, conv_output_size, kernel_size, dropout=dropout, activation=nn.SiLU()
        )
        self.swish_c = SwishBlock(
            conv_output_size+2, dim_c, dim_c
        )

        # Upsampled Representation (O)
        self.linear_einsum = LinearNorm(dim_c, d_predictor) # A
        self.layer_norm = nn.LayerNorm(d_predictor)

    def forward(self, duration, V, src_len, src_mask, max_src_len):

        batch_size = duration.shape[0]

        # Duration Interpretation
        mel_len = torch.round(duration.sum(-1)).type(torch.LongTensor).to(device)
        mel_len = torch.clamp(mel_len, max=self.max_seq_len)
        max_mel_len = mel_len.max().item()
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len)

        # Prepare Attention Mask
        src_mask_ = src_mask.unsqueeze(1).expand(-1, mel_mask.shape[1], -1) # [B, tat_len, src_len]
        mel_mask_ = mel_mask.unsqueeze(-1).expand(-1, -1, src_mask.shape[1]) # [B, tgt_len, src_len]
        attn_mask = torch.zeros((src_mask.shape[0], mel_mask.shape[1], src_mask.shape[1])).to(device)
        attn_mask = attn_mask.masked_fill(src_mask_, 1.)
        attn_mask = attn_mask.masked_fill(mel_mask_, 1.)
        attn_mask = attn_mask.bool()

        # Token Boundary Grid
        e_k = torch.cumsum(duration, dim=1)
        s_k = e_k - duration
        e_k = e_k.unsqueeze(1).expand(batch_size, max_mel_len, -1)
        s_k = s_k.unsqueeze(1).expand(batch_size, max_mel_len, -1)
        t_arange = torch.arange(1, max_mel_len+1, device=device).unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, max_src_len
        )
        S, E = (t_arange - s_k).masked_fill(attn_mask, 0), (e_k - t_arange).masked_fill(attn_mask, 0)

        # Attention (W)
        W = self.swish_w(S, E, self.conv_w(V)) # [B, T, K, dim_w]
        W = self.linear_w(W).squeeze(-1).masked_fill(src_mask_, -np.inf) #[B, T, K]
        W = self.softmax_w(W) #[B, T, K]
        W = W.masked_fill(mel_mask_, 0.)
        
        # Auxiliary Attention Context (C)
        C = self.swish_c(S, E, self.conv_c(V)) # [B, T, K, dim_c]

        # Upsampled Representation (O)
        upsampled_rep = torch.matmul(W, V) + self.linear_einsum(torch.einsum('btk,btkp->btp', W, C)) # [B, T, M]
        upsampled_rep = self.layer_norm(upsampled_rep)
        upsampled_rep = upsampled_rep.masked_fill(mel_mask.unsqueeze(-1), 0)

        return upsampled_rep, mel_mask, mel_len, W


class Decoder(nn.Module):
    """ Spectrogram Decoder With Iterative Mel Prediction """

    def __init__(self, config):
        super(Decoder, self).__init__()
        n_position = config["max_seq_len"] + 1
        n_mel_channels = config["n_mel_channels"]
        d_decoder = config["decoder"]["decoder_hidden"]
        n_layers = config["decoder"]["decoder_layer"]
        n_head_conv = config["decoder"]["trans_head"]
        n_head_trans = config["decoder"]["trans_head"]
        d_k = d_v = (
            config["decoder"]["decoder_hidden"]
            // config["decoder"]["trans_head"]
        )
        kernel_size = config["decoder"]["conv_kernel_size"]
        dropout = config["decoder"]["decoder_dropout"]

        self.n_layers = n_layers
        self.max_seq_len = config["max_seq_len"]

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_decoder).unsqueeze(0),
            requires_grad=False,
        )

        self.convolution_stack = nn.ModuleList(
            [
                LConvBlock(
                    d_decoder, kernel_size, n_head_conv, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.mel_projection = nn.ModuleList(
            [
                LinearNorm(
                    d_decoder, n_mel_channels
                )
                for _ in range(n_layers)
            ]
        )
    
    def forward(self, x, mask):

        mel_iters = list()
        batch_size, max_len = x.shape[0], x.shape[1]

        dec_output = x

        if not self.training and max_len > self.max_seq_len:
            dec_output = dec_output + get_sinusoid_encoding_table(
                max_len, self.d_decoder
            )[: max_len, :].unsqueeze(0).expand(batch_size, -1, -1).to(
                x.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)
            dec_output = dec_output + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]

        for i, (conv, linear) in enumerate(zip(self.convolution_stack, self.mel_projection)):
            dec_output = dec_output.masked_fill(mask.unsqueeze(-1), 0)
            dec_output = torch.tanh(conv(
                dec_output, mask=mask
            ))
            if self.training or not self.training and i == self.n_layers-1:
                mel_iters.append(
                    linear(dec_output).masked_fill(mask.unsqueeze(-1), 0)
                )

        return mel_iters, mask
