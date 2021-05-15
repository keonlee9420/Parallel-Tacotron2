import torch
import torch.nn as nn
from .soft_dtw_cuda import SoftDTW


class ParallelTacotron2Loss(nn.Module):
    """ Parallel Tacotron 2 Loss """

    def __init__(self, model_config, train_config):
        super(ParallelTacotron2Loss, self).__init__()
        self.L = model_config["decoder"]["decoder_layer"]
        self.lambda_ = train_config["loss"]["lambda"]
        self.start = train_config["loss"]["kl_start"]
        self.end = train_config["loss"]["kl_end"]
        self.upper = train_config["loss"]["kl_upper"]

        self.sdtw_loss = SoftDTW(
            use_cuda=True,
            gamma=train_config["loss"]["gamma"],
            warp=train_config["loss"]["warp"],
        )
        self.mae_loss = nn.L1Loss(reduction='none')
        self.guided_loss = GuidedAttentionLoss()

    def kl_anneal(self, step):
        if step < self.start:
            return .0
        elif step >= self.end:
            return self.upper
        else:
            return self.upper*((step - self.start) / (self.end - self.start))

    def forward(self, inputs, predictions, step):
        (
            src_lens_targets,
            _,
            mel_targets,
            mel_lens_targets,
            _,
        ) = inputs[4:]
        (
            mel_iters,
            mel_masks,
            mel_lens,
            src_masks,
            src_lens,
            log_durations,
            mus,
            log_vars,
            attns
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        src_lens_targets.requires_grad = False
        mel_targets.requires_grad = False
        mel_lens_targets.requires_grad = False

        # Iterative Loss Using Soft-DTW
        mel_iter_loss = 0
        mel_targets_comp = torch.sigmoid(mel_targets)
        for mel_iter in mel_iters:
            mel_iter_comp = torch.sigmoid(mel_iter)
            mel_iter_loss += self.sdtw_loss(mel_iter_comp, mel_targets_comp).mean()
        mel_loss = (mel_iter_loss / (self.L * mel_lens_targets)).mean()

        # Duration Loss
        duration_loss = self.lambda_ * (self.mae_loss((torch.exp(log_durations) - 1).sum(-1), mel_lens_targets) / src_lens_targets).mean()

        # KL Divergence Loss
        beta = torch.tensor(self.kl_anneal(step))
        log_vars, mus = log_vars.masked_select(src_masks.unsqueeze(-1)), mus.masked_select(src_masks.unsqueeze(-1))
        kl_loss = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())

        # Residual Attention Loss
        attn_loss = self.guided_loss(attns.transpose(-2, -1), src_lens_targets, mel_lens_targets)

        total_loss = (
            mel_loss + duration_loss + beta * kl_loss + attn_loss
        )

        return (
            total_loss,
            mel_loss,
            duration_loss,
            kl_loss,
            attn_loss,
            beta,
        )


class GuidedAttentionLoss(nn.Module):
    """Guided attention loss function module.
    See https://github.com/espnet/espnet/blob/e962a3c609ad535cd7fb9649f9f9e9e0a2a27291/espnet/nets/pytorch_backend/e2e_tts_tacotron2.py#L25
    This module calculates the guided attention loss described
    in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_,
    which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System
        Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control
                how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(
                att_ws.device
            )
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """
        Make guided attention mask.
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2))
        )

    @staticmethod
    def _make_masks(ilens, olens):
        """
        Make masks indicating non-padded part.
        """
        in_masks = make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = make_non_pad_mask(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)


def make_pad_mask(lengths, xs=None, length_dim=-1):
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    return ~make_pad_mask(lengths, xs, length_dim)