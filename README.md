# ConvLSTM-DynAE

## ConvLSTM-Based Dynamic Deep Embedded Clustering for Smart Meter Load Profiling

This repository contains the implementation of the **ConvLSTM-DynAE** framework, a deep clustering approach designed for electricity smart-meter load profiling.

The framework combines

- ConvLSTM Autoencoder
- ACAI pretraining
- Dynamic Deep Embedded Clustering (DynAE)
- Student's t-distribution cluster assignment
- Automatic K selection
- Cluster validity analysis

using the following evaluation metrics

- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Index

---

## Repository Status

🚧 **Research project under active development**

The accompanying manuscript is currently **under peer review**.

The repository will continue to be updated until publication.

---

## Repository Structure

```
ConvLSTMDYNAE
│
├── code/
│   ├── main.py
│   ├── clustering.py
│   ├── pretrain.py
│   ├── model.py
│   ├── generate_figures.py
│   └── *.ipynb
│
├── figures/
│
├── data/
│   └── README.md
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

## Features

✔ ConvLSTM Autoencoder

✔ ACAI latent regularization

✔ DynAE clustering

✔ Automatic K search

✔ London Smart Meter Dataset

✔ Irish CER Dataset

✔ Daily

✔ Weekly

✔ Monthly

✔ Seasonal

load profiling

---

## Datasets

The original datasets are **NOT included**.

Please download

- London Smart Meter Dataset

- Irish CER Smart Meter Dataset

and place them inside the data directory.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running

```bash
python code/main.py
```

or open one of the notebooks inside

```
code/
```

---

## Figures

The repository includes representative figures generated from the experiments.

---

## License

The license will be added after publication.

---

## Contact

**Dr.Hasna AlSaeed**

University of Bahrain

Email:

drhasnaalsaeed@gmail.com
