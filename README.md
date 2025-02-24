<!-- Credits to vienna paper and links
ref implementation of learning chern number
contains:
(each file)
how to use:

how to produce similar results in the paper:
[inline code]

add: link to the paper 
add: bibtex for paper

acknowledgements
Person A, Person B (for the foundational work, since this is a public version of the code)

Install 
    requirements python package -->

# Gauge Equivariant Network for Predicting Topological Invariants

This repository contains the code accompanying our paper:

**"Learning Chern Numbers of Topological Insulators with Gauge Equivariant Neural Networks"**

In this repository, we provide a Python package implementing gauge equivariant networks for predicting topological invariants (e.g., Chern numbers) in multiband topological insulators. The code features:

 - A gauge equivariant normalization layer (**"TrNorm"**) for to stabilize training of lattice gauge equivariant networks.
 - Scripts for data generation, model building, training, and evaluation.
 - Configurations for ablation studies to explore performance under different settings.

Everything is designed to ensure smooth and reproducible experimentation.

Our model architecture extends the LGE-CNNs introduced in [Favoni et al.](link_to_paper) and our implementation is based on the [LGE-CNN reference implementation](https://gitlab.com/openpixi/lge-cnn/-/tree/master).

## Repository Structure

Below is a simplified view of the package layout:

```
src/ 
├── gauge_net/ 
│   ├── __init__.py 
│   ├── __main__.py 
│   ├── data.py 
│   ├── model.py 
│   ├── train.py 
│   ├── eval.py 
│   ├── config.json
├── setup.py
└── pyproject.toml
```

Inside the package directory:

- **`data.py`**: Generates random datas.
- **`model.py`**: Contains the implementation of the gauge equivariant network, including our novel **'TrNorm'** layer.
- **`train.py`**: Script to train the model.
- **`eval.py`**: Script to evaluate the trained model and predict Chern numbers.
- **`utils.py`**: Utility functions used throughout the project.
- **`config.json`**:Logging configurations, including WandB settings.

## How to Use

### 1. Environment Setup

Clone the repository, move into the `src` directory, and install the package:

```bash
git clone https://github.com/sitronsea/GEN.git
cd src
pip install .
```

### 2. Model Training

#### Baseline Training

In **"setup.py"**, we introduce entry points ``gauge_net_train`` and ``gauge_net_eval``, which are equivalent to running ``python -m gauge_net``. To start a baseline training session, simply run:

```bash
gauge_net_train
```

#### Reproducing Our Results

- **General GEBLNet Model**  
  To train a GEBLNet model for 4-band insulators on a 5×5 grid and with **TrNorm** layers enabled:

  ```bash
  gauge_net_train --dims 5 5 --n_bands 4 --trnorm
  ```

  Specify layer channels with:

  ```bash
  --layer_channels 32 16 8
  ```

- **Training on Trivial Samples Only**  
  To train using only trivial samples:

  ```bash
  gauge_net_train --dims 5 5 --n_bands 4 --trnorm --keep_only_trivial_samples
  ```

- **Setting Diagonal Sample Ratio**  
  To specify the percentage of diagonal samples (e.g., 50%):

  ```bash
  --diag_ratio 0.5
  ```

- **Higher-Dimensional Data**  
  For training for 3-band insulators on a 3×3×3×3 grid:

  ```bash
  gauge_net_train --dims 3 3 3 3 --n_bands 3 --trnorm
  ```

- **Alternative Model Structure**  
  To train a model with a different architecture (e.g., GEConvNet with specified layer channels):

  ```bash
  gauge_net_train --model_type GEConvNet --layer_channels 32 16 16 8 --trnorm
  ```

### 3. Model Evaluation

- **Default Evaluation**  
  To evaluate the default trained model:

  ```bash
  gauge_net_eval --save_model_name <<filename>>
  ```

- **Evaluation with Specific Configurations**  
  When evaluating, ensure the model configuration, number of bands and grid dimension match the training setup, while the grid size can be different. For example:

  ```bash
  gauge_net_eval --save_model_name <<filename>> --n_bands 4 --layer_channels 32 16 8 --trnorm --dims 10 10
  ```

- **Rescaling for Trivial Samples**  
  If your model was trained on trivial samples, include a rescaling mechanism during evaluation:

  ```bash
  --rescale_eval
  ```

## Paper and Citation

For more details, please refer to the paper:

[arXiv:2502.15376](https://arxiv.org/abs/2502.15376).

If you use this code in your research, please cite our work :

```latex
@misc{huang2025learningchernnumberstopological,
      title={Learning Chern Numbers of Topological Insulators with Gauge Equivariant Neural Networks}, 
      author={Longde Huang and Oleksandr Balabanov and Hampus Linander and Mats Granath and Daniel Persson and Jan E. Gerken},
      year={2025},
      eprint={2502.15376},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.15376}, 
}
```

## Acknowledgements

We would also like to express our gratitude to Oleksandr Balabanov, Hampus Linander and Jan Gerken for their contributions to the initial version of this code.

## Installation Requirements

### Prerequisites
Ensure you have **Python 3.8** installed.

### Dependencies
This project requires the following dependencies:
- `torch==1.9.0`
- `numpy==1.22.0`
- `wandb`

### Install Dependencies

#### Using `pip` (with `setup.py`)
If you are installing via `setup.py`, you can run:

```bash
pip install -r requirements.txt
```

or install the package directly:

```bash
pip install .
```


