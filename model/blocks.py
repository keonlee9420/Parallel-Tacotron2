import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from fairseq.modules import LightweightConv


class SwishBlock(nn.Module):
    """ Swish Block """

    def __init__(self, in_channels, hidden_dim, out_channels):
        super(SwishBlock, self).__init__()
        self.layer = nn.Sequential(
            LinearNorm(in_channels, hidden_dim, bias=True),
            nn.SiLU(),
            LinearNorm(hidden_dim, out_channels, bias=True),
            nn.SiLU(),
        )

    def forward(self, S, E, V):

        out = torch.cat([
            S.unsqueeze(-1),
            E.unsqueeze(-1),
            V.unsqueeze(1).expand(-1, E.size(1), -1, -1),
        ], dim=-1)
        out = self.layer(out)

        return out


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        return x


class LConvBlock(nn.Module):
    """ Lightweight Convolutional Block """

    def __init__(self, d_model, kernel_size, num_heads, dropout, weight_softmax=True):
        super(LConvBlock, self).__init__()
        self.embed_dim = d_model
        padding_l = (
            kernel_size // 2
            if kernel_size % 2 == 1
            else ((kernel_size - 1) // 2, kernel_size // 2)
        )

        self.act_linear = LinearNorm(self.embed_dim, 2 * self.embed_dim, bias=True)
        self.act = nn.GLU()

        self.conv_layer = LightweightConv(
            self.embed_dim,
            kernel_size,
            padding_l=padding_l,
            weight_softmax=weight_softmax,
            num_heads=num_heads,
            weight_dropout=dropout,
        )

        self.fc1= LinearNorm(self.embed_dim, 4 * self.embed_dim, bias=True)
        self.fc2= LinearNorm(4 * self.embed_dim, self.embed_dim, bias=True)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, mask=None):

        x = x.contiguous().transpose(0, 1)

        residual = x
        x = self.act_linear(x)
        x = self.act(x)
        if mask is not None:
            x = x.masked_fill(mask.transpose(0, 1).unsqueeze(2), 0)
        x = self.conv_layer(x)
        x = residual + x

        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        x = x.contiguous().transpose(0, 1)
        x = self.layer_norm(x)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(2), 0)

        return x


class ConvBlock(nn.Module):
    """ Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, dropout, activation=nn.ReLU()):
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
            ),
            nn.BatchNorm1d(out_channels),
            activation
        )
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, enc_input, mask=None):
        enc_output = enc_input.contiguous().transpose(1, 2)
        enc_output = F.dropout(self.conv_layer(enc_output), self.dropout, self.training)

        enc_output = self.layer_norm(enc_output.contiguous().transpose(1, 2))
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class FFTBlock(nn.Module):
    """ FFT Block """

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = LinearNorm(d_model, n_head * d_k)
        self.w_ks = LinearNorm(d_model, n_head * d_k)
        self.w_vs = LinearNorm(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = LinearNorm(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class VariableLengthAttention(nn.Module):
    """ Variable-Length Attention """

    def __init__(self, d_model, d_encoder_m,  d_encoder_t, dropout=0.1):
        super(VariableLengthAttention, self).__init__()

        self.d_encoder_m = d_encoder_m
        self.d_encoder_t = d_encoder_t

        self.w_qs = LinearNorm(d_encoder_t, d_model)
        self.w_ks = LinearNorm(d_encoder_m, d_model)
        self.w_vs = LinearNorm(d_encoder_m, d_model)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_encoder_m, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, mel_encoding, text_encoding, mel_mask, src_mask):

        q = self.w_qs(text_encoding)
        k = self.w_ks(mel_encoding)
        v = self.w_vs(mel_encoding)

        src_len, mel_len = src_mask.shape[1], mel_mask.shape[1]
        src_mask_ = src_mask.unsqueeze(-1).expand(-1, -1, mel_len) # [batch, seq_len, mel_len]
        mel_mask_ = mel_mask.unsqueeze(1).expand(-1, src_len, -1) # [batch, seq_len, mel_len]

        output, attn = self.attention(q, k, v, src_mask=src_mask_, mel_mask=mel_mask_)

        output = self.layer_norm(self.dropout(output))

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None, src_mask=None, mel_mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
            attn = self.softmax(attn)
        elif src_mask is not None and mel_mask is not None:
            attn = attn.masked_fill(mel_mask, -np.inf)
            attn = self.softmax(attn)
            attn = attn.masked_fill(src_mask, 0.)
        else:
            attn = self.softmax(attn)

        output = torch.bmm(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output
