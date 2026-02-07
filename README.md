# GasLeakage-MultiModal-MTF

Implementation of the deep learning pipeline for **gas leakage detection and identification** via multimodal fusion of electronic nose (E-nose) and infrared thermal imaging, as proposed in:

**"Multitask Deep Learning-Based Pipeline for Gas Leakage Detection via E-Nose and Thermal Imaging Multimodal Fusion"**  
Omneya Attallah  
Department of Electronics and Communications Engineering, College of Engineering and Technology, Arab Academy for Science, Technology and Maritime Transport, Alexandria, Egypt  
*Chemosensors* 2023, 11(7), 364 — https://doi.org/10.3390/chemosensors11070364

This repository contains our code that implements and extends the pipeline described in the paper. We also wrote a follow-up paper on this work and presented it at the **Korea Information Science Society (KIISE) Undergraduate/Junior Paper Competition**, where we placed **31st**.

---

## Overview

The pipeline combines:

- **E-nose**: array of seven metal oxide semiconductor (MOX) gas sensors (MQ2, MQ3, MQ5, MQ6, MQ7, MQ8, MQ135).
- **Thermal imaging**: infrared (IR) thermal camera data.

Gas sensor readings are converted to heatmap images and, together with thermal images, fed into **three CNN backbones** (ResNet-50, Inception, MobileNet) for spatial feature extraction. Two fusion strategies are used:

1. **Intermediate fusion**: spatial features from each modality are fused using the **Discrete Wavelet Transform (DWT)** to obtain a spatial–spectral–temporal representation, then reduced and fed to a **Bidirectional LSTM (Bi-LSTM)** for classification.
2. **Multitask fusion**: features from all three CNNs trained on both modalities are merged using the **Discrete Cosine Transform (DCT)** for dimension reduction, then passed to Bi-LSTM for gas detection and identification.

The paper reports accuracies of **98.47%** (intermediate fusion) and **99.25%** (multitask fusion) on the four-class gas dataset (NoGas, Smoke, Perfume, Mixture).

---

## Project Structure

```
GasLeakage-MultiModal-MTF/
├── main.py              # Entry point (runs test/inference)
├── train.py             # Training script
├── test.py              # Evaluation / inference
├── requirements.txt
├── src/
│   ├── config.py        # Configuration and paths
│   ├── dataset.py       # Dataset loaders
│   ├── GasDataSet.py    # Gas dataset utilities
│   ├── heat_transform.py # Convert 7 MOX sensor values to heatmap images
│   ├── transforms.py    # Image augmentations and preprocessing
│   └── models/
│       ├── extractors.py   # CNN feature extractors (ResNet, Inception, MobileNet)
│       ├── extractors2.py # Alternate CNN extractors
│       ├── bi_lstm.py     # Bi-LSTM classifier
│       ├── multitask_fusion_model.py  # Full pipeline: CNNs + DWT/DCT + Bi-LSTM
│       └── utils.py       # DWT, DCT and other helpers
└── data/                # Data and generation scripts (not in repo)
    ├── data_g.py
    └── Generating_train_test.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place your multimodal dataset (thermal images and E-nose sensor readings, or pre-generated heatmaps) in the paths expected by `src/config.py` and the dataset modules. Sensor data should be convertible to heatmaps (e.g. 7-dimensional MOX readings per sample) as in the paper.

---

## Usage

- **Training**: run `train.py` with the desired fusion setting (intermediate vs multitask) and data paths.
- **Testing / inference**: run `main.py` (which calls `test()` from `test.py`) to evaluate the trained model.

---

## Citation

If you use this implementation or build on the pipeline, please cite the original paper:

```bibtex
@article{attallah2023multitask,
  title={Multitask Deep Learning-Based Pipeline for Gas Leakage Detection via E-Nose and Thermal Imaging Multimodal Fusion},
  author={Attallah, Omneya},
  journal={Chemosensors},
  volume={11},
  number={7},
  pages={364},
  year={2023},
  publisher={MDPI},
  doi={10.3390/chemosensors11070364}
}
```

---

## License

See the repository license file.
