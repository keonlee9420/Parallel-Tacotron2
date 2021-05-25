import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import TextEncoder, ResidualEncoder, DurationPredictor, LearnedUpsampling, Decoder
from utils.tools import get_mask_from_lengths


class ParallelTacotron2(nn.Module):
    """ Parallel Tacotron 2 """

    def __init__(self, preprocess_config, model_config):
        super(ParallelTacotron2, self).__init__()
        self.model_config = model_config

        self.text_encoder = TextEncoder(model_config)
        self.residual_encoder = ResidualEncoder(model_config)
        self.duration_predictor = DurationPredictor(model_config)
        self.learned_upsampling = LearnedUpsampling(model_config)
        self.decoder = Decoder(model_config)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["speaker_embed_size"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        # Encoder
        text_encoding = self.text_encoder(texts, src_masks)
        if self.speaker_emb is not None:
            speaker_embedding = self.speaker_emb(speakers)

        residual_encoding, attns, mus, log_vars = self.residual_encoder(
            mels, text_encoding, mel_masks, src_masks, max_mel_len, max_src_len, speaker_embedding
        )
        
        speaker_embedding = speaker_embedding.unsqueeze(1).expand(
            -1, max_src_len, -1
        )
        encodings = torch.cat([text_encoding, residual_encoding, speaker_embedding], dim=-1)

        # Duration Modeling
        durations, V = self.duration_predictor(encodings, src_masks)

        upsampled_rep, mel_masks, mel_lens, Ws = \
            self.learned_upsampling(durations, V, src_lens, src_masks, max_src_len)

        # Decoder
        mel_iters, mel_masks = self.decoder(upsampled_rep, mel_masks)

        return (
            mel_iters,
            mel_masks,
            mel_lens,
            src_masks,
            src_lens,
            durations,
            mus,
            log_vars,
            attns,
            Ws,
        )