# Causal Clusters: Representing Explainable Features Learned by Deep Neural Networks as Causal Graphs

This repository implements the methodology described in the research paper "Causal Clusters: Representing Explainable Feature Learned by Deep Neural Networks as Causal Graphs". The implementation provides tools to analyze deep neural networks (DNNs) by identifying clustering patterns in the latent space and representing these relationships using causal graphs.

## Overview

Deep neural networks (DNNs) have become increasingly complex, making their decision-making processes opaque. This project introduces a novel approach to understanding DNNs by:

1. Training neural networks on diverse data types (images and tabular data)
2. Extracting activations from intermediate layers
3. Performing clustering on these activations to identify similar patterns
4. Explaining these clusters using SHAP values
5. Discovering causal relationships between clusters across layers
6. Generating explainable causal graphs that represent the model's hierarchical decision-making process

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.10 or higher

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/causal-clusters.git
cd causal-clusters
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Datasets

The implementation works with both image and tabular data:

### Image Datasets
- **Lung Cancer Images (IQ-OTHNCCD)**: Medical imaging dataset for lung cancer detection

### Tabular Datasets
- **Insurance Dataset**: Insurance cost prediction based on demographic and health factors
- **Aging Clock Dataset**: Simulated dataset for biological age prediction from methylation sites

## Usage

### Basic Usage

To run the complete Causal Clusters pipeline on a dataset:

```bash
python main.py --dataset lung_cancer --epochs 10 --batch_size 32 --num_clusters 10 --save_model
```

### Command-line Arguments

- `--dataset`: Dataset to use (choices: 'lung_cancer', 'insurance', 'aging_clock', default: 'lung_cancer')
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--num_clusters`: Number of clusters per layer (default: 10)
- `--save_model`: Flag to save the trained model
- `--model_path`: Path to pre-trained model (default: None)
- `--device`: Device to use (cuda or cpu, default: cuda if available)

### Using Pre-trained Models

To use a pre-trained model:

```bash
python main.py --dataset insurance --model_path models/insurance_model.pth
```

## Methodology

### 1. Network Training and Activation Extraction

The implementation trains neural networks (CNNs for images, MLPs for tabular data) and extracts activations from intermediate layers.

### 2. Clustering of Activations

K-means clustering is applied to the activations of each layer to identify similar patterns. PCA is used for dimensionality reduction when needed.

### 3. Cluster Explanation with SHAP

SHAP (SHapley Additive exPlanations) values are computed to explain what input features contribute to each cluster, providing interpretability.

### 4. Causal Discovery

Causal discovery algorithms from the causallearn library are applied to determine causal relationships between clusters across layers.

### 5. Visualization

The implementation generates comprehensive visualizations:
- Cluster visualizations in the latent space using t-SNE
- SHAP explanation visualizations for each cluster
- Causal graphs showing relationships between clusters across layers
- A complete HTML report summarizing all findings

## Project Structure

- `CC.py`: Core implementation containing datasets, models, and analysis functions
- `main.py`: Command-line interface to run the complete pipeline
- `create_aging_dataset.py`: Script to generate the simulated aging clock dataset
- `Data/`: Directory containing datasets
  - `Image/`: Image datasets
  - `Tabular/`: Tabular datasets

## Citation

If you use this implementation in your research, please cite:

```
@article{causalclusters2023,
  title={Causal Clusters: Representing Explainable Feature Learned by Deep Neural Networks as Causal Graphs},
  author={Your Name},
  journal={Arxiv},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SHAP: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
- causallearn: [https://github.com/cmu-phil/causallearn](https://github.com/cmu-phil/causallearn)
- PyTorch: [https://pytorch.org/](https://pytorch.org/) 