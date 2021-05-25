import torch
import torch.nn as nn
from .soft_dtw_cuda import SoftDTW


class ParallelTacotron2Loss(nn.Module):
    """ Parallel Tacotron 2 Loss """

    def __init__(self, model_config, train_config):
        super(ParallelTacotron2Loss, self).__init__()
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
            durations,
            mus,
            log_vars,
            _,
            _,
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
            # mel_iter_loss += self.sdtw_loss(mel_iter, mel_targets).mean()
        mel_loss = (mel_iter_loss / (len(mel_iters) * mel_lens_targets)).mean()

        # Duration Loss
        duration_loss = self.lambda_ * (self.mae_loss(durations.sum(-1), mel_lens_targets) / src_lens_targets).mean()

        # KL Divergence Loss
        beta = torch.tensor(self.kl_anneal(step))
        kl_loss = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())

        total_loss = (
            mel_loss + duration_loss + beta * kl_loss
        )

        return (
            total_loss,
            mel_loss,
            duration_loss,
            kl_loss,
            beta,
        )
