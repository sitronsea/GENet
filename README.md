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

 - A gauge equivariant normalization layer (**"TrNorm"**) to stabilize training.
 - Scripts for data generation, model building, training, and evaluation.
 - Configurations for ablation studies to explore performance under different settings.

Everything is designed to ensure smooth and reproducible experimentation.

We acknowledge the foundational work of the [Favoni et al.](link), whose foundational works in gauge equivariant network architectures inspired this research.

## Repository Structure


- **`data_loader.py`**: Loads and preprocesses datasets.
- **`model.py`**: Contains the implementation of the gauge equivariant network, including our novel normalization layer.
- **`train.py`**: Script to train the model.
- **`evaluate.py`**: Script to evaluate the trained model and predict Chern numbers.
- **`utils.py`**: Utility functions used throughout the project.

## How to Use

### 1. Set Up Your Environment

    git clone https://github.com/yourusername/your-repository-name.git
    cd your-repository-name
    pip install -r requirements.txt

### 2. Training the Model

    python train.py --config config.json

### 3. Evaluating the Model

    python evaluate.py --model_path /path/to/saved/model

## Reproducing Our Results

To reproduce the results presented in our paper, ensure that you use the provided configuration file (`config.json`) and follow these steps:

    python train.py --config config.json
    python evaluate.py --model_path /path/to/saved/model

## Paper and Citation

For more details, please refer to the paper:

[Link to the paper](link-to-your-paper)

If you use this code in your research, please cite our work :

    @inproceedings{yourpaper2023,
      title={Gauge Equivariant Networks for Predicting Chern Numbers of Topological Insulators},
      author={Your Name and Collaborator Name},
      booktitle={Conference/Journal Name},
      pages={123--130},
      year={2023},
      organization={Organization or Publisher}
    }

We also acknowledge the Vienna group for their foundational work on gauge equivariant network architectures.

## Acknowledgements

Special thanks to the Vienna group for their pioneering contributions to gauge equivariant network design. We are also grateful to Person A and Person B for their valuable insights and support in refining the model and preparing the public release of this code.

## Installation Requirements

Ensure you have Python 3.x installed. Then, install the required packages:

    pip install -r requirements.txt

The project depends on:

- torch
- numpy
- matplotlib
- scipy
