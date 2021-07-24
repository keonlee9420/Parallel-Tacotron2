import os
import random
import json

import tgt
import librosa
import numpy as np
from tqdm import tqdm

import audio as Audio
from text import grapheme_to_phoneme


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.skip_len = config["preprocessing"]["audio"]["skip_len"]
        self.trim_top_db = config["preprocessing"]["audio"]["trim_top_db"]
        self.filter_length = config["preprocessing"]["stft"]["filter_length"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        mel_min = float('inf')
        mel_max = -float('inf')
        n_frames = 0

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]

                ret = self.process_utterance(speaker, basename)
                if ret is None:
                    continue
                else:
                    info, n, m_min, m_max = ret
                out.append(info)

                if mel_min > m_min:
                    mel_min = m_min
                if mel_max < m_max:
                    mel_max = m_max

                n_frames += n

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "mel": [
                    float(mel_min),
                    float(mel_max),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav.astype(np.float32)

        if len(wav) < self.skip_len:
            return None

        wav = librosa.effects.trim(wav, top_db=self.trim_top_db, frame_length=self.filter_length, hop_length=self.hop_length)[0]

        # Compute mel-scale spectrogram
        mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, self.STFT)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Get phoneme
        phone = grapheme_to_phoneme(raw_text)
        text = "{" + " ".join(phone) + "}"

        # Save files
        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            mel_spectrogram.shape[1],
            np.min(mel_spectrogram),
            np.max(mel_spectrogram),
        )
