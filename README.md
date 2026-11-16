# LSTM Chinese Text Generation — Journey to the West

A character-level LSTM that learns the language of *Journey to the West* and continues a short Chinese prompt in a similar classical style.

English | [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Research%20only-lightgrey)](#license)

## Introduction

The whole project lives in one notebook. After whitespace normalization the training text is 734,950 characters long and contains 4,198 unique characters. Every character is treated as one token, so the model predicts the next character rather than the next word. This keeps the vocabulary small and lets generation stay close to the original phrasing, including names, verse fragments and dialogue punctuation.

The pipeline is straightforward: load and clean the text, build a character vocabulary, cut the text into overlapping windows of 100 characters, train a three-layer LSTM with next-character cross-entropy, then sample new text character by character at a chosen temperature.

```mermaid
flowchart LR
    A[data/xyj.txt] --> B[Clean whitespace]
    B --> C[Character vocabulary<br/>4,198 tokens]
    C --> D[Windows of 100 characters]
    D --> E[Embedding 256]
    E --> F[3-layer LSTM<br/>hidden 512]
    F --> G[Linear projection]
    G --> H[Cross-entropy training]
    H --> I[models/lstm_xiyouji.pth]
    I --> J[Temperature sampling]
    J --> K[Generated text]
```

| Component | Configuration |
| --- | --- |
| Embedding | 4,198 × 256 |
| LSTM | 3 layers, hidden size 512, batch first |
| Dropout | 0.3 between LSTM layers and before the output layer |
| Output | Linear 512 → 4,198 |
| Parameters | 9,001,574 (about 34.3 MB in FP32) |
| Optimizer | Adam, learning rate 1e-3 |
| Schedule | ReduceLROnPlateau, factor 0.5, patience 3 |
| Precision | Automatic mixed precision on CUDA |
| Gradient clipping | Max norm 5 |

## Dependencies

- Python 3.12
- PyTorch with CUDA (the notebook was run on a Kaggle Tesla T4; CPU works but is much slower)
- NumPy
- Jupyter Notebook or JupyterLab

```bash
pip install torch numpy jupyter;
```

There is no requirements file. The notebook imports only `torch`, `numpy`, `pickle`, `re`, `os`, `random` and `collections`, so the list above is enough.

## Project Structure

```
.
├── data/
│   └── xyj.txt                  # full text of Journey to the West
├── models/
│   └── vocab.pkl                # character-to-index and index-to-character maps
├── lstm-text-generated.ipynb    # data prep, model, training and generation
├── README.md                    # this file
└── README.zh-CN.md              # Chinese version
```

The trained checkpoint `models/lstm_xiyouji.pth` is not committed because of its size. It is created the first time you train, and the notebook expects it at that exact path.

## Usage

### 1. Train

Open `lstm-text-generated.ipynb` and run the cells from top to bottom. The notebook is split into constants, data preparation, model definition, training and generation, and the cells depend on each other in that order.

Before running, check the data path in the constants cell. The active line points to a Kaggle dataset path; for a local run, switch it to the file in this repository:

```python
DATA_PATH = "./data/xyj.txt";
```

Key settings:

| Setting | Value | Meaning |
| --- | --- | --- |
| `SEQ_LENGTH` | 100 | Characters per training window |
| `BATCH_SIZE` | 1024 | Samples per batch |
| `EMBED_DIM` | 256 | Embedding size |
| `HIDDEN_DIM` | 512 | LSTM hidden size |
| `NUM_LAYERS` | 3 | Stacked LSTM layers |
| `DROPOUT` | 0.3 | Dropout probability |
| `EPOCHS` | 10 | Training epochs |
| `LEARNING_RATE` | 0.001 | Initial Adam learning rate |
| `CONTINUE_TRAIN` | 1 | Resume from `models/lstm_xiyouji.pth` if it exists |
| `GENERATE_LENGTH` | 500 | Characters produced during sampling |
| `TEMPERATURE` | 0.1 | Default sampling temperature |

`CONTINUE_TRAIN = 1` only resumes when a checkpoint is present; otherwise training starts from scratch. A checkpoint is saved after every epoch together with the optimizer state and the epoch loss, so training can be interrupted and resumed. Each save is followed by a 200-character sample starting from 话说, which gives a quick read on the current quality.

Training uses `DataParallel` automatically when more than one GPU is visible.

### 2. Generate

Run the notebook through the generation cell first, since `generate_text` is defined there. Then load the checkpoint and call it:

```python
model = load_model_for_inference("./models/lstm_xiyouji.pth", "./models/vocab.pkl", device);
text = generate_text(model, start_text="话说", length=500, device=device, temperature=0.8);
print(text);
```

`load_model_for_inference` reloads the vocabulary saved in `models/vocab.pkl`, so the character maps always match the checkpoint. It also strips the `module.` prefix from weights saved under `DataParallel`, which means a multi-GPU checkpoint can be loaded on a single GPU or on CPU.

The generation cell at the end of the notebook loops over four prompts and three temperatures and prints the results side by side, which is a convenient way to compare samples.

### 3. Temperature

- Below 0.5: conservative, repetitive, closest to the training text.
- Around 0.8: a reasonable balance between coherence and variation.
- Above 1.0: more surprising phrasing, but broken sentences become common.

The model samples with `torch.multinomial` instead of taking the argmax, so the same prompt rarely produces the same output twice.

## Notes

- Training on a GPU is strongly recommended. On CPU, one epoch over roughly 735,000 characters with batch size 1024 is slow.
- Lowering the temperature is the first thing to try when the output drifts into nonsense.
- The notebook uses `torch.cuda.amp` for mixed precision. On recent PyTorch versions this still runs but may print a deprecation warning; the non-CUDA branch trains in full precision.
- The training text is the public-domain novel, so the dataset can be redistributed with the code.

## Contributing

Issues and pull requests are welcome. If you change the model or the training loop, run the notebook end to end and include the resulting loss curve or sample text in the pull request. If you change the dataset or the vocabulary, mention whether the existing checkpoint is still compatible, since a different character set changes the output layer shape.

## License

Released for learning and research use. *Journey to the West* is in the public domain.

## Contact

谙弆悕博士（Ailan Anjuxi）

📫 Email: anjuxi.ME@outlook.com

📞 SIP: sip:anjuxi@sip.linphone.org
