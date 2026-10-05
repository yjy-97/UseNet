# UseNet: Unified Spatial EEG Decoding Network for Heterogeneous EEG Montages

This repository provides the official implementation of **UseNet**, a unified spatial decoding framework for heterogeneous EEG recordings.

## Data Preparation

Organize the EEG datasets according to the following structure:

```text
datasets/
├── APAVA/
├── ADFTD/
├── MCICN/
└── TDBRAIN/
```

The dataset files should be prepared before training.

## Training and Testing

Run the following command:

```bash
python run.py
```

To specify the dataset:

```bash
python run.py --dataset APAVA
```

Available datasets:

* APAVA
* ADFTD
* MCICN
* TDBRAIN

