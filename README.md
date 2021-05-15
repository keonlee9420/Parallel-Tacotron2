# Parallel Tacotron2

Pytorch Implementation of Google's [Parallel Tacotron 2: A Non-Autoregressive Neural TTS Model with Differentiable Duration Modeling](https://arxiv.org/abs/2103.14574)

<p align="center">
    <img src="img/parallel_tacotron.png" width="80%">
</p>

<p align="center">
    <img src="img/parallel_tacotron2.png" width="40%">
</p>

# Updates

- 2021.05.15: Implementation done. Sanity checks on training and inference. But still the model cannot converge.

    `I'm waiting for your contribution!` Please inform me if you find any mistakes in my implementation or any valuable advice to train the model successfully. See the Implementation Issues section.

# Training

## Requirements

- You can install the Python dependencies with

    ```bash
    pip3 install -r requirements.txt
    ```

- In addition to that, install fairseq ([official document](https://fairseq.readthedocs.io/en/latest/index.html), [github](https://github.com/pytorch/fairseq)) to utilize `LConvBlock`.

## Datasets

The supported datasets:

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): a single-speaker English dataset consists of 13100 short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.
- (will be added more)

## Preprocessing

After downloading the datasets, set the `corpus_path` in `preprocess.yaml` and run the preparation script:

```
python3 prepare_data.py config/LJSpeech/preprocess.yaml
```

Then, run the preprocessing script:

```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

## Training

Train your model with

```
python3 train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

The model cannot converge yet. I'm debugging but it would be boosted if your awesome contribution is ready!

# TensorBoard

Use

```
tensorboard --logdir output/log/LJSpeech
```

to serve TensorBoard on your localhost.

# Implementation Issues

Overall, normalization or activation, which is not suggested in the original paper, is adequately arranged to prevent nan value (gradient) on forward and backward calculations.

## Text Encoder

1. Use the `FFTBlock` of FastSpeech2 for the transformer block of the text encoder.
2. Use dropout `0.2` for the `ConvBlock` of the text encoder.
3. To restore "proprietary normalization engine",
    - Apply the same text normalization as in FastSpeech2.
    - Implement `grapheme_to_phoneme` function. (See ./text/__init__).

## Residual Encoder

1. Use `80 channels` mel-spectrogrom instead of `128-bin`.
2. Regular sinusoidal positional embedding is used in frame-level instead of combinations of three positional embeddings in Parallel Tacotron. As the model depends entirely on unsupervised learning for the position, this choice can be a reason for the fails on model converge.

## Duration Predictor & Learned Upsampling (The most important but ambiguous part)

1. Use log durations with the prior: there should be at least one frame in total per sequence. 
2. Use `nn.SiLU()` for the swish activation.
3. When obtaining `W` and `C`, concatenation operation is applied among `S`, `E`, and `V` after frame-domain (T domain) broadcasting of `V`. As the detailed process is not described in the original paper, this choice can be a reason for the fails on model converge.

## Decoder

1. Use (Multi-head) `Self-attention` and `LConvBlock`.
2. Iterative mel-spectrogram is projected by a linear layer.
3. Apply `nn.Tanh()` to each `LConvBLock` output (following activation pattern of decoder part in FastSpeech2).

## Loss

1. Use optimization & scheduler of FastSpeech2 (which is from [Attention is all you need](https://arxiv.org/abs/1706.03762) as described in the original paper).
2. Base on [pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda) ([post](https://www.codefull.net/2020/05/fast-differentiable-soft-dtw-for-pytorch-using-cuda/)) for the soft-DTW.
    1. Implement customized soft-DTW in `model/soft_dtw_cuda.py`, reflecting the recursion suggested in the original paper.
    2. In the original soft-DTW, the final loss is not assumed and therefore only `E` is computed. But employed as a loss function, jacobian product is added to return target derivetive of `R` w.r.t. input `X`.
    3. Currently, the maximum batch size is `6` in 24GiB GPU (TITAN RTX) due to space complexity problem in soft-DTW Loss.
        - In the original paper, a custom differentiable diagonal band operation was implemented and used to solve the complexity of O(T^2), but this part has not been explored in the current implementation yet.
3. For the stability, mel-spectrogroms are compressed by a sigmoid function before the soft-DTW. If the sigmoid is eliminated, the soft-DTW value is too large, producing nan in the backward.
4. Guided attention loss is applied for fast convergence of the attention module in residual encoder.

# Citation

```
@misc{lee2021parallel_tacotron2,
  author = {Lee, Keon},
  title = {Parallel-Tacotron2},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/keonlee9420/Parallel-Tacotron2}}
}
```

# References

- [ming024's FastSpeech2](https://github.com/ming024/FastSpeech2) (Later than 2021.02.26 ver.)
- [Parallel Tacotron: Non-Autoregressive and Controllable TTS](https://arxiv.org/abs/2010.11439)
- [Parallel Tacotron 2: A Non-Autoregressive Neural TTS Model with Differentiable Duration Modeling](https://arxiv.org/abs/2103.14574)