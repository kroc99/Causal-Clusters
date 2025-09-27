# Standard Imports
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import re
import argparse

# Additional imports for training, clustering, visualization, and causal discovery
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# For causal discovery (if installed)
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    from causallearn.utils.cit import chisq, gsq  # discrete CI tests
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    print("Warning: causallearn is not installed. Causal discovery functions will be limited.")

import networkx as nx

# For SHAP explanations (make sure to install shap via pip)
import shap

import sys
from functools import lru_cache



def seed_everything(seed=42):
    """
    Set seed for reproducibility.
    
    Args:
        seed: Seed value for random number generators
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")


def write_html_page(output_file="gcfn_report.html"):
    """
    Write an HTML page with all the generated visualizations.
    
    Args:
        output_file: Output HTML file path
    """
    print("Generating HTML report...")
    
    # Create figures directory if it doesn't exist
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # Get all PNG files in the figures directory
    figure_files = sorted([f for f in os.listdir(figures_dir) if f.endswith('.png')])
    
    # Check for integrated GCFN visualization
    integrated_gcfn_dir = os.path.join(figures_dir, 'integrated_gcfn')
    integrated_gcfn_html = os.path.join(integrated_gcfn_dir, 'integrated_gcfn.html')
    has_integrated_gcfn = os.path.exists(integrated_gcfn_html)
    
    # Group figures by type
    clusters = [f for f in figure_files if 'clusters' in f]
    features = [f for f in figure_files if 'features' in f]
    causal = [f for f in figure_files if 'causal' in f or 'graph' in f]
    shap = [f for f in figure_files if 'shap' in f or 'explanation' in f]
    other = [f for f in figure_files if f not in clusters + features + causal + shap]
    
    # HTML Header
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GCFN Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            color: #333;
            line-height: 1.6;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            text-align: center;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
            margin-bottom: 30px;
        }}
        h2 {{
            margin-top: 40px;
            padding-bottom: 5px;
            border-bottom: 1px solid #eee;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .figure-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
        .figure-item {{
            flex: 0 1 calc(50% - 20px);
            margin-bottom: 20px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .figure-item h3 {{
            margin-top: 0;
            font-size: 1.1em;
        }}
        .figure-item img {{
            max-width: 100%;
            display: block;
            margin: 0 auto;
        }}
        .integrated-link {{
            display: block;
            text-align: center;
            margin: 30px auto;
            padding: 15px 25px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-weight: bold;
            max-width: 500px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: background-color 0.2s;
        }}
        .integrated-link:hover {{
            background-color: #2980b9;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 0.9em;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Global Causal Feature Network Analysis Report</h1>
        
        <p>This report presents the results of causal analysis on neural network features,
        showing how different clusters of activations in the network relate to each other and
        what patterns they represent.</p>
"""

    # Add link to Integrated GCFN visualization if it exists
    if has_integrated_gcfn:
        integrated_gcfn_rel_path = os.path.join('figures', 'integrated_gcfn', 'integrated_gcfn.html')
        html_content += f"""
        <a href="{integrated_gcfn_rel_path}" class="integrated-link">
            View Integrated GCFN Visualization
        </a>
        <p style="text-align: center;">
            The integrated visualization provides a comprehensive view of all clusters, 
            their feature distributions, and causal relationships in an interactive format.
        </p>
"""

    # Causal Relationships Section
    if causal:
        html_content += """
        <div class="section">
            <h2>Causal Relationships Between Features</h2>
            <p>These visualizations show how features in different layers causally influence each other.</p>
            <div class="figure-container">
"""
        for fig in causal:
            fig_name = fig.replace('.png', '').replace('_', ' ').title()
            html_content += f"""
                <div class="figure-item">
                    <h3>{fig_name}</h3>
                    <img src="{os.path.join(figures_dir, fig)}" alt="{fig_name}">
                </div>
"""
        html_content += """
            </div>
        </div>
"""

    # Cluster Visualizations Section
    if clusters:
        html_content += """
        <div class="section">
            <h2>Cluster Visualizations</h2>
            <p>These visualizations show how the activations form clusters in each layer.</p>
            <div class="figure-container">
"""
        for fig in clusters:
            fig_name = fig.replace('.png', '').replace('_', ' ').title()
            html_content += f"""
                <div class="figure-item">
                    <h3>{fig_name}</h3>
                    <img src="{os.path.join(figures_dir, fig)}" alt="{fig_name}">
                </div>
"""
        html_content += """
            </div>
        </div>
"""

    # Feature Visualizations Section
    if features:
        html_content += """
        <div class="section">
            <h2>Feature Visualizations</h2>
            <p>These visualizations show the distribution of learned features in each layer.</p>
            <div class="figure-container">
"""
        for fig in features:
            fig_name = fig.replace('.png', '').replace('_', ' ').title()
            html_content += f"""
                <div class="figure-item">
                    <h3>{fig_name}</h3>
                    <img src="{os.path.join(figures_dir, fig)}" alt="{fig_name}">
                </div>
"""
        html_content += """
            </div>
        </div>
"""

    # SHAP Explanations Section
    if shap:
        html_content += """
        <div class="section">
            <h2>SHAP Explanations</h2>
            <p>These visualizations show the contributions of different features to model predictions.</p>
            <div class="figure-container">
"""
        for fig in shap:
            fig_name = fig.replace('.png', '').replace('_', ' ').title()
            html_content += f"""
                <div class="figure-item">
                    <h3>{fig_name}</h3>
                    <img src="{os.path.join(figures_dir, fig)}" alt="{fig_name}">
                </div>
"""
        html_content += """
            </div>
        </div>
"""

    # Other Visualizations Section
    if other:
        html_content += """
        <div class="section">
            <h2>Other Visualizations</h2>
            <div class="figure-container">
"""
        for fig in other:
            fig_name = fig.replace('.png', '').replace('_', ' ').title()
            html_content += f"""
                <div class="figure-item">
                    <h3>{fig_name}</h3>
                    <img src="{os.path.join(figures_dir, fig)}" alt="{fig_name}">
                </div>
"""
        html_content += """
            </div>
        </div>
"""

    # Footer
    html_content += """
        <div class="footer">
            <p>Generated by Causal Clusters for Deep Learning Models</p>
            <p>All visualizations are saved in the 'figures' directory</p>
        </div>
    </div>
</body>
</html>
"""

    # Write the HTML file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"HTML report saved to {output_file}")
    return output_file


figure_count = 1
figure_files = []

def save_current_figure(title, dpi=150):
    """
    Save the current matplotlib figure to the figures directory.
    
    Args:
        title: Title for the figure (used for filename)
        dpi: Resolution (dots per inch) for the saved figure
    """
    # Create figures directory if it doesn't exist
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # Convert title to a valid filename
    filename = re.sub(r'[^\w\s-]', '', title).strip().lower()
    filename = re.sub(r'[-\s]+', '_', filename)
    
    # Save figure
    filepath = os.path.join(figures_dir, f"{filename}.png")
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved: {filepath}")
    return filepath




# ---------------------------
# Dataset Definitions

class LungCancerDataset(Dataset):
    def __init__(self, image_dir, transform=None, metric_learning=False):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.metric_learning = metric_learning

        print(f"Loading images from directory: {image_dir}")
        print(f"Available subdirectories: {[f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]}")
        
        # Map folder names to labels (more flexible with folder names)
        # Allow for variations like "benign", "Bengin cases", etc.
        self.class_map = {}
        idx = 0
        for folder_name in os.listdir(image_dir):
            folder_path = os.path.join(image_dir, folder_name)
            if os.path.isdir(folder_path):
                # Normalize folder names for mapping
                normalized_name = folder_name.lower()
                if "benign" in normalized_name or "bengin" in normalized_name:
                    class_label = 0  # Benign
                elif "malignant" in normalized_name:
                    class_label = 1  # Malignant
                elif "normal" in normalized_name:
                    class_label = 2  # Normal
                else:
                    # Default to incremental labeling if class can't be determined
                    class_label = idx
                
                self.class_map[folder_name] = class_label
                idx += 1
        
        print(f"Class mapping: {self.class_map}")

        self.image_paths = []
        self.labels = []
        self.class_to_images = {lbl: [] for lbl in set(self.class_map.values())}

        for class_name, label in self.class_map.items():
            class_dir = os.path.join(image_dir, class_name)
            print(f"Loading class {class_name} (label {label}) from {class_dir}")
            
            image_count = 0
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, file_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    self.class_to_images[label].append(img_path)
                    image_count += 1
            
            print(f"Found {image_count} images in class {class_name}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        if not self.metric_learning:
            return image, label
        else:
            # Triplet for metric learning
            anchor_image = image
            anchor_label = label

            # Positive: same class
            positive_image_path = random.choice(self.class_to_images[anchor_label])
            positive_image = Image.open(positive_image_path).convert("RGB")
            if self.transform:
                positive_image = self.transform(positive_image)

            # Negative: different class
            negative_label = random.choice([
                l for l in self.class_to_images.keys() if l != anchor_label
            ])
            negative_image_path = random.choice(self.class_to_images[negative_label])
            negative_image = Image.open(negative_image_path).convert("RGB")
            if self.transform:
                negative_image = self.transform(negative_image)

            return anchor_image, positive_image, negative_image

def get_image_dataloader(image_dir, batch_size=32, shuffle=True, metric_learning=False):
    """
    Get dataloader for image data
    
    Args:
        image_dir (str): Path to the image directory
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        metric_learning (bool): Whether to use metric learning
    
    Returns:
        tuple: (dataloader, dataset)
    """
    # Check if the directory exists
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}\nPlease provide a valid path to your image dataset with folder structure: /path/to/dataset/[benign/malignant/normal]")
    
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = LungCancerDataset(
        image_dir=image_dir,
        transform=image_transform,
        metric_learning=metric_learning
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=0
    )
    return dataloader, dataset

# (The InsuranceDataset and get_tabular_dataloader remain unchanged.)
class InsuranceDataset(Dataset):
    def __init__(self, csv_path, classification=True, transform=None):
        super().__init__()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.classification = classification

        self.data["sex"] = self.data["sex"].map({"female": 0, "male": 1})
        self.data["smoker"] = self.data["smoker"].map({"no": 0, "yes": 1})
        region_map = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
        self.data["region"] = self.data["region"].map(region_map)

        if self.classification:
            self.data["target"] = pd.cut(
                self.data["charges"],
                bins=[0, 10000, 20000, float("inf")],
                labels=[0, 1, 2]
            )
            self.data.dropna(subset=["target"], inplace=True)
            self.data["target"] = self.data["target"].astype(int)
        else:
            self.data["target"] = self.data["charges"].astype(float)

        self.feature_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
        self.X = self.data[self.feature_cols].values
        self.y = self.data["target"].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.transform:
            X = self.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        if self.classification:
            y_tensor = torch.tensor(y, dtype=torch.long)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)

        return X_tensor, y_tensor

class AgingClockDataset(Dataset):
    def __init__(self, csv_path, classification=True, transform=None):
        super().__init__()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.classification = classification

        # Encode categorical variables
        if 'Gender' in self.data.columns:
            self.data['Gender'] = self.data['Gender'].map({'Male': 0, 'Female': 1})
        if 'Smoking' in self.data.columns:
            self.data['Smoking'] = self.data['Smoking'].map({'No': 0, 'Yes': 1})
        if 'Alcohol' in self.data.columns:
            alcohol_map = {'Never': 0, 'Occasional': 1, 'Regular': 2}
            self.data['Alcohol'] = self.data['Alcohol'].map(alcohol_map)
        if 'Exercise' in self.data.columns:
            exercise_map = {'Low': 0, 'Medium': 1, 'High': 2}
            self.data['Exercise'] = self.data['Exercise'].map(exercise_map)

        # Get all methylation sites
        self.methylation_cols = [col for col in self.data.columns if col.startswith('Methylation_Site_')]
        
        # Create feature columns (everything except Age and AgeRange)
        self.feature_cols = self.methylation_cols + ['Gender', 'Smoking', 'Alcohol', 'Exercise', 'BMI']
        
        if self.classification:
            # Use AgeRange as target for classification
            self.X = self.data[self.feature_cols].values
            self.y = self.data['AgeRange'].values
        else:
            # Use Age as target for regression
            self.X = self.data[self.feature_cols].values
            self.y = self.data['Age'].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.transform:
            X = self.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        if self.classification:
            y_tensor = torch.tensor(y, dtype=torch.long)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)

        return X_tensor, y_tensor

def get_tabular_dataloader(csv_path, batch_size=32, shuffle=True, classification=True, dataset_type='insurance'):
    """
    Get dataloader for tabular data
    
    Args:
        csv_path (str): Path to the CSV file
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        classification (bool): Whether to perform classification or regression
        dataset_type (str): Type of dataset ('insurance' or 'aging')
    
    Returns:
        tuple: (dataloader, dataset)
    """
    if dataset_type.lower() == 'insurance':
        dataset = InsuranceDataset(
            csv_path=csv_path,
            classification=classification,
            transform=None
        )
    elif dataset_type.lower() == 'aging':
        dataset = AgingClockDataset(
            csv_path=csv_path,
            classification=classification,
            transform=None
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'insurance' or 'aging'.")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    return dataloader, dataset


# ---------------------------
# Model Definitions

class SimpleCNN(nn.Module):
    """
    CNN for lung cancer image classification.
    """
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.activations = {}

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        
        # Use adaptive pooling to handle different input sizes
        # This will always output a 128x7x7 tensor regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        flattened_size = 128 * 7 * 7  # Fixed size after adaptive pooling
        
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Register forward hooks for latent space analysis
        self.conv1.register_forward_hook(self.save_activation('conv1'))
        self.conv2.register_forward_hook(self.save_activation('conv2'))
        self.conv3.register_forward_hook(self.save_activation('conv3'))
        self.fc1.register_forward_hook(self.save_activation('fc1'))

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def forward(self, x, return_embedding=False):
        # Save intermediate outputs explicitly
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Use adaptive pooling to get a fixed size output
        x = self.adaptive_pool(x)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        embedding = self.fc2(x)
        embedding = F.relu(embedding)
        
        logits = self.fc3(embedding)
        
        return embedding if return_embedding else logits

class SimpleMLP(nn.Module):
    """
    MLP for tabular data (Insurance or Aging Clock).
    """
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=3):
        super(SimpleMLP, self).__init__()
        self.activations = {}

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Register forward hooks for latent space analysis
        self.fc1.register_forward_hook(self.save_activation('fc1'))
        self.fc2.register_forward_hook(self.save_activation('fc2'))
        self.fc3.register_forward_hook(self.save_activation('fc3'))

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def forward(self, x, return_embedding=False):
        x = F.relu(self.fc1(x))
        embedding = F.relu(self.fc2(x))
        logits = self.fc3(embedding)
        return embedding if return_embedding else logits


# ---------------------------
# Training, Clustering & Visualization Helper Functions

def train_network(network, dataloader, device, num_epochs=5, lr=1e-3, use_metric_learning=False):
    criterion = nn.CrossEntropyLoss()
    metric_criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(network.parameters(), lr=lr)

    for epoch in range(num_epochs):
        network.train()
        running_loss = 0.0

        for batch in dataloader:
            if use_metric_learning:
                anchor, positive, negative = batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                optimizer.zero_grad()
                anchor_out = network(anchor, return_embedding=True)
                positive_out = network(positive, return_embedding=True)
                negative_out = network(negative, return_embedding=True)
                loss = metric_criterion(anchor_out, positive_out, negative_out)
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    print("Training completed.")


def extract_activations(model, dataloader, device, metric_learning=False):
    model.eval()
    # Define the expected layer keys
    expected_keys = ['conv1', 'conv2', 'conv3', 'fc1']
    # If the model's activations dict is empty (which may occur after loading),
    # initialize it with the expected keys.
    if not model.activations or len(model.activations) == 0:
        model.activations = {key: None for key in expected_keys}
    
    # Clear previous activations
    for layer in model.activations.keys():
        model.activations[layer] = None

    layer_outputs = {layer: [] for layer in model.activations.keys()}
    label_list = []

    with torch.no_grad():
        for batch in dataloader:
            # First check the batch structure instead of relying on metric_learning flag
            # This is more robust against mismatches in dataset initialization
            if isinstance(batch, tuple) and len(batch) == 3 and isinstance(batch[0], torch.Tensor):
                # This is a metric learning batch (anchor, positive, negative)
                anchor = batch[0].to(device)
                _ = model(anchor)
                label_list.extend([0] * anchor.size(0))
            elif isinstance(batch, tuple) and len(batch) == 2:
                # This is a standard (input, label) batch
                inputs, labels = batch[0].to(device), batch[1]
                _ = model(inputs)
                label_list.extend(labels.cpu().numpy())
            else:
                raise ValueError(f"Unexpected batch structure: {type(batch)}, {len(batch) if isinstance(batch, tuple) else 'not tuple'}")

            for layer_name, act_tensor in model.activations.items():
                if act_tensor is None:
                    continue
                if act_tensor.dim() == 4:
                    pooled = F.adaptive_avg_pool2d(act_tensor, (1, 1))
                    pooled = pooled.view(pooled.size(0), -1)
                    layer_outputs[layer_name].append(pooled.cpu().numpy())
                else:
                    layer_outputs[layer_name].append(act_tensor.cpu().numpy())

    for layer_name in layer_outputs.keys():
        if len(layer_outputs[layer_name]) > 0:
            layer_outputs[layer_name] = np.concatenate(layer_outputs[layer_name], axis=0)
        else:
            layer_outputs[layer_name] = np.array([])

    # Always return both activations and labels for consistency
    return layer_outputs, np.array(label_list)


def perform_clustering_optimized(activations, max_clusters=10, pca_components=50, min_clusters=2, max_clusters_override=None, force_clusters=None):
    """
    Perform clustering on activations with an optimized number of clusters.
    Uses silhouette score to determine the optimal number of clusters.
    
    Args:
        activations: Dictionary of layer activations or a single array
        max_clusters: Maximum number of clusters to consider (default: 10)
        pca_components: Number of PCA components to use for dimensionality reduction
        min_clusters: Minimum number of clusters to consider (default: 2)
        max_clusters_override: Override max_clusters if provided
        force_clusters: If provided, use this exact number of clusters
        
    Returns:
        Dictionary of clustering results or (clusters, centers) tuple
    """
    # Helper: compute clustering quality metrics
    def _metrics_for_labels(data, labels):
        try:
            unique = np.unique(labels)
            if len(unique) <= 1:
                return (False, float('-inf'), float('-inf'), float('-inf'))
            sil = silhouette_score(data, labels)
        except Exception:
            sil = float('-inf')
        try:
            ch = calinski_harabasz_score(data, labels)
        except Exception:
            ch = float('-inf')
        try:
            db = davies_bouldin_score(data, labels)
            db_term = -db  # lower DB is better
        except Exception:
            db_term = float('-inf')
        # The first boolean indicates validity
        return (True, sil, ch, db_term)

    # Helper function to determine optimal number of clusters and algorithm
    def get_optimal_clusters(data, min_k=2, max_k=10):
        # Use provided parameters or defaults
        max_k = max_clusters_override if max_clusters_override is not None else min(max_k, max_clusters)
        min_k = max(min_k, min_clusters)
        
        if data.shape[0] < max_k * 3:  # Ensure enough samples for clustering
            max_k = max(min_k, data.shape[0] // 3)
        
        # If forcing specific number of clusters, return that
        if force_clusters is not None:
            return force_clusters
        
        # Try different cluster counts and models
        best_k = min_k
        best_algo = 'kmeans'
        best_tuple = (False, float('-inf'), float('-inf'), float('-inf'))
        
        # Cache for metrics to avoid recomputation
        metric_cache = {}
        
        for k in range(min_k, max_k + 1):
            try:
                # KMeans metrics
                cache_key_km = (data.shape, 'kmeans', k)
                if cache_key_km in metric_cache:
                    km_tuple = metric_cache[cache_key_km]
                else:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km_labels = kmeans.fit_predict(data)
                    km_tuple = _metrics_for_labels(data, km_labels)
                    metric_cache[cache_key_km] = km_tuple
                
                # GMM metrics
                cache_key_gmm = (data.shape, 'gmm', k)
                if cache_key_gmm in metric_cache:
                    gmm_tuple = metric_cache[cache_key_gmm]
                else:
                    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
                    gmm_labels = gmm.fit_predict(data)
                    gmm_tuple = _metrics_for_labels(data, gmm_labels)
                    metric_cache[cache_key_gmm] = gmm_tuple

                # Choose best between KMeans and GMM using lexicographic comparison
                # Prioritize valid silhouette, then silhouette value, then CH, then -DB
                if km_tuple > best_tuple:
                    best_tuple = km_tuple
                    best_k = k
                    best_algo = 'kmeans'
                if gmm_tuple > best_tuple:
                    best_tuple = gmm_tuple
                    best_k = k
                    best_algo = 'gmm'
            except Exception as e:
                print(f"Failed to cluster with k={k}: {e}")
                continue
        
        print(f"Optimal clusters: {best_k} using {best_algo} (sil={best_tuple[1]:.3f}, CH={best_tuple[2]:.2f}, -DB={best_tuple[3]:.3f})")
        return best_k, best_algo
    
    # If activations is a single array (not a dictionary)
    if not isinstance(activations, dict):
        scaler = StandardScaler()
        normalized = scaler.fit_transform(activations)
        if normalized.shape[1] > pca_components:
            pca = PCA(n_components=pca_components, random_state=42)
            reduced = pca.fit_transform(normalized)
        else:
            reduced = normalized

        # Find optimal number of clusters and model
        n_clusters, algo = get_optimal_clusters(reduced, max_clusters)
        
        # Perform clustering with optimal number using best algorithm
        if algo == 'gmm':
            model = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
            clusters = model.fit_predict(reduced)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = model.fit_predict(reduced)
        print(f"Clustering complete with {n_clusters} clusters using {algo}")
        # Return clusters and pseudo-centers for compatibility
        centers = None
        if hasattr(model, 'cluster_centers_'):
            centers = model.cluster_centers_
        elif hasattr(model, 'means_'):
            centers = model.means_
        return clusters, centers
    
    # If activations is a dictionary
    clustering_results = {}
    for layer, act_data in activations.items():
        # Cluster all layers
        scaler = StandardScaler()
        normalized = scaler.fit_transform(act_data)
        if normalized.shape[1] > pca_components:
            pca = PCA(n_components=pca_components, random_state=42)
            reduced = pca.fit_transform(normalized)
        else:
            reduced = normalized

        # Find optimal number of clusters for this layer
        print(f"\nFinding optimal clusters for layer: {layer}")
        n_clusters, algo = get_optimal_clusters(reduced, max_clusters)
        
        # Perform clustering with optimal number using best algorithm
        if algo == 'gmm':
            model = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
            clustering_results[layer] = model.fit_predict(reduced)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clustering_results[layer] = model.fit_predict(reduced)
        print(f"{layer}: clustering complete with {n_clusters} clusters using {algo}")
        
    return clustering_results

def create_latent_cluster_dataset(clustering_results, labels):
    """
    Create a one-hot encoded latent cluster dataset.
    Works with variable cluster counts per layer.
    """
    df_list = []
    
    # Ensure all cluster assignments have the same length
    min_length = min(len(clusters) for clusters in clustering_results.values())
    
    # Truncate labels to match the minimum length
    if len(labels) > min_length:
        labels = labels[:min_length]
    
    # Process each layer's clusters
    for layer, clusters in clustering_results.items():
        # Truncate clusters to match the minimum length
        if len(clusters) > min_length:
            clusters = clusters[:min_length]
            
        # Create one-hot encoding for this layer's clusters
        cluster_series = pd.Series(clusters, name=layer)
        one_hot = pd.get_dummies(cluster_series, prefix=f"{layer}_cluster")
        
        # Add the one-hot encoded columns to the list
        df_list.append(one_hot)
    
    # Combine all one-hot encoded columns
    latent_df = pd.concat(df_list, axis=1)
    latent_df = latent_df.fillna(0).astype(int)
    
    # Add labels if provided
    if labels is not None:
        latent_df["label"] = labels
    
    # Print debug information
    print("\nDEBUG: Latent cluster dataset info:")
    print(f"Number of samples: {len(latent_df)}")
    print(f"Number of features: {len(latent_df.columns)}")
    print("\nFeature columns:")
    for col in latent_df.columns:
        print(f"- {col}")
    
    return latent_df


def compute_statistical_transition_graph(latent_df):
    """
    Build a directed graph with edge weights as empirical transition probabilities
    between layer-wise clusters P(G_{L+1,j} | G_{L,i}).
    """
    df = latent_df.copy()
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    # Identify layers and their columns
    layer_to_cols = {}
    for col in df.columns:
        if '_cluster_' in col:
            layer = col.split('_cluster_')[0]
            layer_to_cols.setdefault(layer, []).append(col)
    ordered_layers = sorted(layer_to_cols.keys())
    G = nx.DiGraph()
    # Ensure all node names exist
    for cols in layer_to_cols.values():
        for c in cols:
            G.add_node(c)
    # For each adjacent layer pair compute transitions
    for i in range(len(ordered_layers) - 1):
        L = ordered_layers[i]
        L1 = ordered_layers[i + 1]
        cols_L = layer_to_cols[L]
        cols_L1 = layer_to_cols[L1]
        X = df[cols_L].values.astype(int)
        Y = df[cols_L1].values.astype(int)
        # One-hot sanity: clamp to {0,1}
        X = (X > 0).astype(int)
        Y = (Y > 0).astype(int)
        # For each cluster i in L, compute P(Y_j | X_i)
        for idx_i, col_i in enumerate(cols_L):
            xi = X[:, idx_i]
            denom = xi.sum()
            if denom == 0:
                continue
            for idx_j, col_j in enumerate(cols_L1):
                yj = Y[:, idx_j]
                num = np.logical_and(xi == 1, yj == 1).sum()
                p = num / float(denom)
                if p > 0:
                    G.add_edge(col_i, col_j, weight=p)
    return G


def visualize_causal_graph(causal_graph, title="Causal Graph"):
    """
    Create a visualization of the causal graph with enhanced readability and information.
    
    Args:
        causal_graph: The graph to visualize (networkx DiGraph or causallearn graph)
        title: Title for the visualization
    """
    if causal_graph is None:
        print("No causal graph to visualize.")
        return
        
    G = nx.DiGraph()
    
    # Handle both causallearn graph and networkx graph
    if hasattr(causal_graph, 'get_graph_edges'):
        edges = causal_graph.get_graph_edges()
    else:
        edges = causal_graph.edges()
    
    # Create a mapping of layer names to their cluster numbers
    layer_clusters = {}
    for edge in edges:
        try:
            if hasattr(edge, 'node1') and hasattr(edge, 'node2'):
                source = str(edge.node1)
                target = str(edge.node2)
            else:
                source, target = edge
                
            # Extract layer name and cluster number from node names
            source_parts = source.split('_cluster_')
            target_parts = target.split('_cluster_')
            
            if len(source_parts) == 2 and len(target_parts) == 2:
                source_layer = source_parts[0]
                target_layer = target_parts[0]
                
                if source_layer not in layer_clusters:
                    layer_clusters[source_layer] = set()
                if target_layer not in layer_clusters:
                    layer_clusters[target_layer] = set()
                    
                source_cluster = source_parts[1]
                target_cluster = target_parts[1]
                
                layer_clusters[source_layer].add(source_cluster)
                layer_clusters[target_layer].add(target_cluster)
                
                G.add_edge(source, target)
        except Exception as e:
            print(f"DEBUG: Cannot process edge: {edge}, Error: {e}")
            continue
    
    if not layer_clusters:
        print("No valid layer clusters found in the graph.")
        # Return early as we can't visualize an empty graph
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "No valid causal relationships could be determined.\nCheck console for details.", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        save_current_figure(title)
        return
    
    # Create a hierarchical layout
    layer_order = sorted(layer_clusters.keys())
    pos = {}
    
    # Position nodes in a more structured layout
    for i, layer in enumerate(layer_order):
        y_pos = 1 - (i / max(1, len(layer_order) - 1))  # Normalize y position between 0 and 1
        clusters = sorted(list(layer_clusters[layer]), 
                        key=lambda x: int(x) if x.isdigit() else 0)  # Sort clusters numerically
        
        # Distribute clusters horizontally
        for j, cluster in enumerate(clusters):
            node_name = f"{layer}_cluster_{cluster}"
            x_pos = (j + 0.5) / max(1, len(clusters))  # Space clusters evenly
            pos[node_name] = (x_pos, y_pos)
    
    # Draw the graph with improved visual elements
    plt.figure(figsize=(14, 10))
    plt.suptitle(f"{title}\nRelationships Between Neural Network Layers", fontsize=16, y=0.98)
    
    # Draw edges with varying thickness based on importance
    # If we have edge weights, use them to determine thickness
    edge_weights = {}
    for u, v, data in G.edges(data=True):
        if 'weight' in data:
            # Normalize weight to be between 0.5 and 4
            weight = abs(data['weight'])
            edge_weights[(u, v)] = 0.5 + 3.5 * min(weight, 1.0)
        else:
            edge_weights[(u, v)] = 1.0  # Default thickness
    
    # Sort edges by weight for better visualization (heavier edges on bottom)
    sorted_edges = sorted(G.edges(), key=lambda e: edge_weights.get((e[0], e[1]), 1.0))
    
    # Draw edges with curved lines for better visibility
    for source, target in sorted_edges:
        # Calculate edge thickness
        width = edge_weights.get((source, target), 1.0)
        # Use curved edges with varying colors
        nx.draw_networkx_edges(
            G, pos, edgelist=[(source, target)],
            width=width, 
            edge_color='gray',
            alpha=0.7,
            connectionstyle='arc3,rad=0.1',  # Slightly curved
            arrows=True, arrowsize=15
        )
    
    # Draw nodes with different colors for each layer
    layer_colors = plt.cm.tab10(np.linspace(0, 1, len(layer_order)))
    for i, layer in enumerate(layer_order):
        layer_nodes = [node for node in G.nodes() if node.startswith(layer)]
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=layer_nodes, 
            node_color=[layer_colors[i]] * len(layer_nodes),
            node_size=1200,
            alpha=0.8
        )
    
    # Draw labels with improved formatting
    labels = {}
    for node in G.nodes():
        layer, cluster = node.split('_cluster_')
        # Count the number of edges for this node
        in_edges = len(list(G.in_edges(node)))
        out_edges = len(list(G.out_edges(node)))
        
        # Create a more descriptive label
        labels[node] = f"{layer}\nCluster {cluster}\n({in_edges} in, {out_edges} out)"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
    
    # Add a legend for layers
    legend_elements = []
    for i, layer in enumerate(layer_order):
        legend_elements.append(plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor=layer_colors[i],
            markersize=10, label=f"{layer}"
        ))
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add explanatory notes
    plt.figtext(0.5, 0.02, 
               "This graph shows how different clusters in neural network layers influence each other.\n" +
               "Nodes represent clusters of similar activations, edges represent causal relationships.\n" +
               "Thicker edges indicate stronger relationships. Numbers in parentheses show incoming and outgoing connections.",
               ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for the footer text
    save_current_figure(title)

def architecture_mask(all_cols):
    """
    Create a mask to respect neural network layer architecture.
    Only allows edges from layer i to layer i+1.
    """
    # layer order by name
    layers = {}
    for c in all_cols:
        L = c.split('_cluster_')[0]
        layers.setdefault(L, []).append(c)
    ordered = sorted(layers.keys())
    allowed = set()
    for i in range(len(ordered)-1):
        for u in layers[ordered[i]]:
            for v in layers[ordered[i+1]]:
                allowed.add((u, v))
    return allowed

def causal_discovery_all_layers(latent_df, method='pc'):
    """
    Perform causal discovery on the latent cluster dataset.
    
    Args:
        latent_df: DataFrame with one-hot encoded cluster columns
        method: Method to use for causal discovery (default: 'pc')
        
    Returns:
        causal_graph: The discovered causal graph
    """
    print("Performing causal discovery on all layers...")
    
    # Use the provided DataFrame directly
    df = latent_df.copy()
    
    # Remove label column if present
    all_columns = df.columns.tolist()
    if "label" in df.columns:
        df = df.drop("label", axis=1)
    
    # Check for problematic columns (constant/near-constant values or high correlations)
    print(f"DataFrame shape before filtering: {df.shape}")
    problem_columns = []
    
    # Check for columns with very low variance (compute once)
    print("DEBUG: Checking for low variance columns...")
    column_vars = df.var()
    for col in df.columns:
        if column_vars[col] < 0.005:  # Less strict variance threshold
            print(f"DEBUG: Column {col} has very low variance: {column_vars[col]}")
            problem_columns.append(col)
    
    # Compute correlation matrix only once for debugging and filtering
    print("DEBUG: Computing correlation matrix...")
    corr_matrix = df.corr()
    
    # Check for highly correlated columns that might cause singularity
    print("DEBUG: Checking for highly correlated columns...")
    high_corr_pairs = []
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i+1:]:
            corr_val = abs(corr_matrix.loc[col1, col2])
            if corr_val > 0.95:  # Less strict correlation threshold
                high_corr_pairs.append((col1, col2, corr_val))
                print(f"DEBUG: High correlation between {col1} and {col2}: {corr_val}")
                # Only mark one of the columns as problematic
                if col2 not in problem_columns and col1 not in problem_columns:
                    problem_columns.append(col2)
    
    print(f"DEBUG: Found {len(high_corr_pairs)} highly correlated column pairs")
    
    # Remove problematic columns
    print(f"Found {len(problem_columns)} problematic columns that could cause singularity issues")
    for col in problem_columns:
        print(f"Removing problematic column: {col}")
    
    # Keep only non-problematic columns
    filtered_df = df.drop(columns=problem_columns)
    
    if filtered_df.shape[1] < 2:
        print("Not enough valid columns left for causal discovery")
        return create_hierarchical_causal_graph(all_columns)
    
    print(f"DataFrame shape after filtering: {filtered_df.shape}")
    
    # Create a mapping from original column names to X1, X2, etc.
    remaining_columns = filtered_df.columns.tolist()
    column_mapping = {f"X{i+1}": col for i, col in enumerate(remaining_columns)}
    reverse_mapping = {col: f"X{i+1}" for i, col in enumerate(remaining_columns)}
    
    print(f"Column mapping for causal discovery: {column_mapping}")
    
    # Convert to numpy array with discrete values (0/1) for CI tests on binary features
    data = filtered_df.values.astype(int)
    
    try:
        if method == 'pc':
            # Use PC algorithm with optimized parameter selection
            print("Using PC algorithm for causal discovery...")
            
            # Try different alpha values systematically
            alpha_values = [0.01, 0.05, 0.1, 0.2, 0.3]
            best_cg = None
            best_edges = []
            best_alpha = 0.05
            
            for alpha in alpha_values:
                try:
                    # Prefer discrete CI test (G^2 / Chi-square) for one-hot binary columns
                    indep_test = None
                    if CAUSAL_LEARN_AVAILABLE:
                        try:
                            indep_test = gsq
                        except Exception:
                            indep_test = None
                    cg = pc(data, alpha=alpha, indep_test=indep_test, verbose=False)
                    edges = cg.G.get_graph_edges()
                    print(f"DEBUG: PC algorithm with alpha={alpha} found {len(edges)} edges")
                    
                    # Prefer results with moderate number of edges (not too sparse, not too dense)
                    if len(edges) > 0:
                        if best_cg is None or (len(edges) >= 2 and len(edges) <= data.shape[1] * 2):
                            best_cg = cg
                            best_edges = edges
                            best_alpha = alpha
                            
                except Exception as e:
                    print(f"PC algorithm failed with alpha={alpha}: {e}")
                    continue
            
            # If no good results, try with stable parameters
            if not best_edges:
                print("Trying PC algorithm with stable parameters...")
                try:
                    cg = pc(data, alpha=0.1, indep_test=indep_test, stable=True, uc_rule=1, uc_priority=1, verbose=False)
                    edges = cg.G.get_graph_edges()
                    print(f"DEBUG: PC algorithm with stable parameters found {len(edges)} edges")
                    if len(edges) > 0:
                        best_cg = cg
                        best_edges = edges
                        best_alpha = 0.1
                except Exception as e:
                    print(f"PC algorithm with stable parameters failed: {e}")
            
            # Use the best result found
            if best_cg is not None:
                cg = best_cg
                edges = best_edges
                print(f"Using best PC result with alpha={best_alpha} and {len(edges)} edges")
            else:
                print("All PC algorithm attempts failed")
                edges = []
                
            # Rename nodes back to original names
            causal_graph = nx.DiGraph()
            for edge in edges:
                # Map back from pc node ids (integers) to column names
                try:
                    if hasattr(edge, 'node1') and hasattr(edge, 'node2'):
                        # causallearn edge
                        src_idx = int(str(edge.node1).replace('X', '')) - 1 if str(edge.node1).startswith('X') else int(str(edge.node1))
                        tgt_idx = int(str(edge.node2).replace('X', '')) - 1 if str(edge.node2).startswith('X') else int(str(edge.node2))
                        if 0 <= src_idx < len(remaining_columns) and 0 <= tgt_idx < len(remaining_columns):
                            source = remaining_columns[src_idx]
                            target = remaining_columns[tgt_idx]
                            causal_graph.add_edge(source, target)
                    else:
                        # networkx-like tuple
                        source = column_mapping.get(str(edge[0]), str(edge[0]))
                        target = column_mapping.get(str(edge[1]), str(edge[1]))
                        causal_graph.add_edge(source, target)
                except Exception as e:
                    print(f"DEBUG: Failed mapping edge {edge}: {e}")
            
            # If still no meaningful edges, fall back to hierarchical structure
            if not causal_graph.edges():
                print("PC algorithm couldn't find meaningful edges, using hierarchical structure")
                return create_hierarchical_causal_graph(all_columns)
            
            # Optional: Apply architecture mask to respect NN layer direction
            try:
                allowed = architecture_mask(remaining_columns)
                masked_graph = nx.DiGraph()
                for edge in causal_graph.edges():
                    if edge in allowed:
                        masked_graph.add_edge(edge[0], edge[1])
                
                if masked_graph.edges():
                    print(f"Architecture mask applied: {len(masked_graph.edges())} edges kept from {len(causal_graph.edges())} total")
                    return masked_graph
                else:
                    print("Architecture mask too restrictive, returning full graph")
            except Exception as e:
                print(f"Architecture mask failed: {e}, returning full graph")
                
            return causal_graph
            
        else:
            # If method not recognized, use hierarchical structure
            print(f"Causal discovery method '{method}' not recognized, using hierarchical structure")
            return create_hierarchical_causal_graph(all_columns)
            
    except Exception as e:
        print(f"Error during causal discovery: {e}")
        import traceback
        traceback.print_exc()
        
        # Create a fallback graph with hierarchical structure
        print("Creating hierarchical structure as fallback")
        return create_hierarchical_causal_graph(all_columns)
    
def create_hierarchical_causal_graph(all_columns):
    """
    Create a hierarchical causal graph based on layer order.
    Used as a fallback when statistical causal discovery fails.
    
    Args:
        all_columns: List of column names with layer and cluster information
        
    Returns:
        hierarchical_graph: A networkx DiGraph with hierarchical structure
    """
    print("Creating hierarchical causal graph as fallback...")
    hierarchical_graph = nx.DiGraph()
    
    # Group columns by layer
    layer_clusters = {}
    for col in all_columns:
        parts = col.split('_cluster_')
        if len(parts) == 2:
            layer = parts[0]
            cluster = parts[1]
            if layer not in layer_clusters:
                layer_clusters[layer] = set()
            layer_clusters[layer].add(cluster)
    
    # Sort layers by name to establish a hierarchy
    layer_order = sorted(layer_clusters.keys())
    print(f"Layer order for hierarchical graph: {layer_order}")
    
    # Connect clusters between adjacent layers with sparse forward-only edges
    for i in range(len(layer_order) - 1):
        current_layer = layer_order[i]
        next_layer = layer_order[i + 1]
        
        current_indices = sorted(list(layer_clusters[current_layer]), key=lambda x: int(x) if str(x).isdigit() else 0)
        next_indices = sorted(list(layer_clusters[next_layer]), key=lambda x: int(x) if str(x).isdigit() else 0)
        
        for idx, current_cluster in enumerate(current_indices):
            # Connect to same-index and nearest neighbors in next layer when available
            preferred = []
            if idx < len(next_indices):
                preferred.append(next_indices[idx])
            if idx - 1 >= 0:
                preferred.append(next_indices[max(0, idx - 1)])
            if idx + 1 < len(next_indices):
                preferred.append(next_indices[min(len(next_indices) - 1, idx + 1)])
            
            for next_cluster in set(preferred):
                source = f"{current_layer}_cluster_{current_cluster}"
                target = f"{next_layer}_cluster_{next_cluster}"
                hierarchical_graph.add_edge(source, target)
    
    print(f"Created hierarchical graph with {len(hierarchical_graph.edges())} edges")
    return hierarchical_graph

def visualize_clusters(activations, clustering_results, layer_name, apply_pca=True,
                       pca_components=50, tsne_perplexity=30, tsne_n_iter=1000):
    """
    Visualize clusters in the latent space using t-SNE.
    
    Args:
        activations: Dictionary of activations from model layers
        clustering_results: Dictionary of clustering results
        layer_name: Name of the layer to visualize
        apply_pca: Whether to apply PCA for dimensionality reduction
        pca_components: Number of PCA components to use
        tsne_perplexity: Perplexity parameter for t-SNE
        tsne_n_iter: Number of iterations for t-SNE
    """
    data = activations[layer_name]
    
    # Apply dimensionality reduction if needed
    if apply_pca and data.shape[1] > pca_components:
        pca = PCA(n_components=pca_components, random_state=42)
        data = pca.fit_transform(data)
    
    # Use t-SNE for visualization
    # Bound perplexity to valid range: (5 .. n_samples/3]
    max_perp = max(5, data.shape[0] // 3)
    perplexity = max(5, min(tsne_perplexity, max_perp))
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=tsne_n_iter, random_state=42, init='pca', learning_rate='auto')
    emb_2d = tsne.fit_transform(data)
    
    # Get cluster assignments and count
    cluster_assign = clustering_results[layer_name]
    n_clusters = len(np.unique(cluster_assign))
    
    # Create figure with better size and resolution
    plt.figure(figsize=(10, 8))
    
    # Set different colormaps based on cluster count
    if n_clusters <= 10:
        cmap = plt.cm.get_cmap('tab10', n_clusters)
    elif n_clusters <= 20:
        cmap = plt.cm.get_cmap('tab20', n_clusters)
    else:
        cmap = plt.cm.get_cmap('viridis', n_clusters)
    
    # Plot each cluster with a different color
    for i in range(n_clusters):
        mask = cluster_assign == i
        plt.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=[cmap(i)], label=f'Cluster {i}',
            s=50, alpha=0.7, edgecolors='w', linewidths=0.5
        )
    
    # Add labels and legend
    plt.title(f"{layer_name} Clusters (t-SNE)\n{n_clusters} clusters found", fontsize=14)
    
    # Add legend only if not too many clusters
    if n_clusters <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Add a colorbar instead for many clusters
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_clusters-1))
        sm.set_array([])
        plt.colorbar(sm, label="Cluster")
    
    # Add descriptive text
    plt.figtext(0.5, -0.05, 
               f"This visualization shows how the {n_clusters} clusters are arranged in the latent space.\n" +
               "Points that are close together have similar activation patterns.",
               ha='center', fontsize=10)
    
    plt.tight_layout()
    save_current_figure(f"Clusters_{layer_name}")
    
    # Optional: Add distribution analysis of cluster sizes
    cluster_sizes = np.bincount(cluster_assign)
    plt.figure(figsize=(10, 4))
    plt.bar(range(n_clusters), cluster_sizes, color=cmap(range(n_clusters)))
    plt.title(f"Cluster Size Distribution - {layer_name}", fontsize=14)
    plt.xlabel("Cluster")
    plt.ylabel("Number of samples")
    plt.xticks(range(n_clusters))
    plt.figtext(0.5, -0.05, 
               "This shows the number of samples in each cluster.\n" +
               "Uneven distributions may indicate some features are more common than others.",
               ha='center', fontsize=10)
    plt.tight_layout()
    save_current_figure(f"Cluster_Sizes_{layer_name}")


# ---------------------------
# New Functions for SHAP and Feature Analysis

def visualize_learned_features(activations, layer_name, n_components=3):
    """
    Visualize the learned features of a layer using PCA and ICA.
    Handles both convolutional and dense layer activations with flexible reshaping.
    
    Args:
        activations: Dictionary of activations from model layers
        layer_name: Name of the layer to visualize
        n_components: Number of components to extract and display
    """
    act_data = activations[layer_name]
    
    # Determine if this is likely a convolutional layer (heuristic)
    is_conv = False
    if layer_name.startswith("conv") or layer_name.startswith("pool"):
        is_conv = True
    
    # Alternative detection: Check if the activation dimensions correspond to typical conv patterns
    if not is_conv:
        # Check if the activations can be reshaped into a square-like shape
        # Most CNN activations will have dimensions that can form a grid
        try:
            n_samples = act_data.shape[0]
            features_per_sample = act_data.shape[1]
            
            # Common patterns for CNN layers (NxCxHxW flattened to NxD)
            # Test if the features can be arranged in a square
            feature_root = int(np.sqrt(features_per_sample))
            if feature_root**2 == features_per_sample:
                is_conv = True
                print(f"Detected convolutional pattern in {layer_name} based on dimensions")
        except:
            pass
    
    print(f"Processing {layer_name}: {'convolutional' if is_conv else 'dense'} layer")
    
    if is_conv:
        # For convolutional layers, try multiple reshaping approaches
        try:
            # Try some common reshaping patterns for CNNs
            sample_count = act_data.shape[0]
            total_features = act_data.shape[1]
            
            # First attempt: Try to arrange as square feature maps
            feature_dim = int(np.sqrt(total_features))
            if feature_dim * feature_dim == total_features:
                # Reshape to (samples, height, width)
                reshaped = act_data.reshape(sample_count, feature_dim, feature_dim)
                
                # Plot average activation map
                plt.figure(figsize=(8, 6))
                plt.title(f"Average Feature Map for {layer_name}")
                plt.imshow(np.mean(reshaped, axis=0), cmap="viridis")
                plt.colorbar(label="Activation")
                plt.figtext(0.5, -0.05, 
                           f"This shows the average activation pattern across all {sample_count} samples.\n" +
                           "Bright areas indicate features that are consistently activated.",
                           ha='center', fontsize=10)
                save_current_figure(f"Average_Activation_{layer_name}")
                
                # Plot PCA components as images
                pca = PCA(n_components=n_components, random_state=42)
                pca_result = pca.fit_transform(act_data)
                plt.figure(figsize=(12, 4))
                plt.suptitle(f"Principal Components for {layer_name}", fontsize=14)
                
                for i in range(n_components):
                    plt.subplot(1, n_components, i+1)
                    component_img = pca.components_[i].reshape(feature_dim, feature_dim)
                    plt.imshow(component_img, cmap="viridis")
                    plt.title(f"PC {i+1} ({pca.explained_variance_ratio_[i]:.2%})")
                
                plt.tight_layout()
                plt.figtext(0.5, -0.05, 
                           "These components represent the main patterns the layer has learned.\n" +
                           "Each image shows a different aspect of the layer's specialization.",
                           ha='center', fontsize=10)
                save_current_figure(f"PCA Components for {layer_name}")
                return
                
            # Second attempt: Try to find other common CNN dimensions
            # Common patterns: height × width (square), or N × channels × height × width
            possible_shapes = []
            for h in range(2, 300):  # Up to 300px height
                if total_features % h == 0:
                    w = total_features // h
                    possible_shapes.append((h, w))
            
            # Prefer shapes that are closer to square (common in CNNs)
            possible_shapes.sort(key=lambda x: abs(x[0] - x[1]))
            
            if possible_shapes:
                h, w = possible_shapes[0]
                print(f"Reshaping to {h}×{w} (closest to square)")
                reshaped = act_data.reshape(sample_count, h, w)
                
                # Plot average activation map
                plt.figure(figsize=(8, 6))
                plt.title(f"Average Feature Map for {layer_name} ({h}×{w})")
                plt.imshow(np.mean(reshaped, axis=0), cmap="viridis")
                plt.colorbar(label="Activation")
                plt.figtext(0.5, -0.05, 
                           f"This shows the average activation pattern across all {sample_count} samples.\n" +
                           "Bright areas indicate features that are consistently activated.",
                           ha='center', fontsize=10)
                save_current_figure(f"Average_Activation_{layer_name}")
                
                return
                
            # If we get here, we couldn't find a suitable reshaping
            raise ValueError("Could not find suitable dimensions for reshaping")
                
        except Exception as e:
            print(f"Error in visualizing learned features for {layer_name}: {e}")
            # Create a fallback visualization showing the raw activations
            plt.figure(figsize=(10, 6))
            plt.plot(act_data[:100, :50].T)  # Plot first 50 features for first 100 samples
            plt.title(f"Raw Activations for {layer_name}")
            plt.xlabel("Feature Index")
            plt.ylabel("Activation Value")
            plt.figtext(0.5, -0.05, 
                       "Showing raw activation values due to reshaping error.\n" +
                       "This provides an alternative view of the layer's learned features.",
                       ha='center', fontsize=10)
            save_current_figure(f"Raw Activations for {layer_name}")
    else:
        # For non-convolutional layers, show the original line plots
        try:
            pca = PCA(n_components=n_components, random_state=42)
            pca_components = pca.fit_transform(act_data)
            plt.figure(figsize=(10, 4))
            plt.suptitle(f"Principal Components for {layer_name}\nMost significant learned patterns", fontsize=14, y=1.05)
            for i in range(n_components):
                plt.plot(pca_components[:, i], label=f'PC {i+1} ({pca.explained_variance_ratio_[i]:.2%} variance)')
            plt.title("Activation patterns across samples")
            plt.legend()
            plt.figtext(0.5, -0.05, 
                       "These components show the most important patterns in the layer's activations.\n" +
                       "Each line represents a different learned pattern across all samples.",
                       ha='center', fontsize=10)
            save_current_figure(f"PCA Components for {layer_name}")
            
            ica = FastICA(n_components=n_components, random_state=42)
            ica_components = ica.fit_transform(act_data)
            plt.figure(figsize=(10, 4))
            plt.suptitle(f"Independent Components for {layer_name}\nUnique activation patterns", fontsize=14, y=1.05)
            for i in range(n_components):
                plt.plot(ica_components[:, i], label=f'IC {i+1}')
            plt.title("Independent activation patterns across samples")
            plt.legend()
            plt.figtext(0.5, -0.05, 
                       "These components show independent patterns in the layer's activations.\n" +
                       "Each line represents a unique feature the layer has learned.",
                       ha='center', fontsize=10)
            save_current_figure(f"ICA Components for {layer_name}")
            
        except Exception as e:
            print(f"Error in visualizing learned features for {layer_name}: {e}")
            # Create a simple fallback visualization
            plt.figure(figsize=(10, 6))
            # Sample a subset if there are too many features
            if act_data.shape[1] > 50:
                indices = np.random.choice(act_data.shape[1], 50, replace=False)
                plt.plot(act_data[:100, indices].T)
            else:
                plt.plot(act_data[:100].T)
            plt.title(f"Raw Activations for {layer_name}")
            plt.xlabel("Sample Index (first 100 samples)")
            plt.ylabel("Activation Value")
            plt.figtext(0.5, -0.05, 
                       "Showing raw activation values due to error in PCA/ICA decomposition.\n" +
                       "Each line represents a different feature in the layer.",
                       ha='center', fontsize=10)
            save_current_figure(f"Raw Activations for {layer_name}")

def explain_clusters_with_shap(model, dataloader, clustering_results, activations, device, layer_name, is_image=True, background_size=20):
    """
    Explain clusters identified in the latent space using SHAP values.
    
    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        clustering_results: Dictionary of clustering results
        activations: Dictionary of activations
        device: Device to run model on
        layer_name: Layer name to explain
        is_image: Whether the data is image data
        background_size: Number of samples to use as background for SHAP
    
    Returns:
        Dictionary of cluster explanations
    """
    try:
        # Store original model state
        original_model_training = model.training
        model.eval()
        
        # Get cluster assignments for the layer
        if layer_name not in clustering_results:
            print(f"Layer '{layer_name}' not found in clustering results")
            return {}
        
        cluster_ids = clustering_results[layer_name]
        num_clusters = np.max(cluster_ids) + 1
        
        # Collect all inputs and their cluster assignments
        all_inputs = []
        all_cluster_ids = []
        
        for i, batch in enumerate(dataloader):
            if len(batch) == 3:  # Metric learning triplet
                inputs, _, _ = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            else:
                inputs, _ = batch[0].to(device), batch[1]
                
            # Get model activations for this batch
            with torch.no_grad():
                _ = model(inputs)
                
            # Get the activations for this layer and batch
            batch_activations = model.activations[layer_name].cpu().numpy()
            
            # For each sample in the batch, determine its cluster
            for j in range(batch_activations.shape[0]):
                # Store the input and its cluster
                all_inputs.append(inputs[j].cpu().numpy())
                
                # Use actual cluster assignment from clustering results
                global_idx_start = i * inputs.shape[0]
                if global_idx_start + j < len(cluster_ids):
                    all_cluster_ids.append(cluster_ids[global_idx_start + j])
                else:
                    # If we've gone beyond the clustering assignments, use a random assignment
                    all_cluster_ids.append(random.randint(0, num_clusters-1))
                    
            if len(all_inputs) >= 100:  # Limit the number of samples to process for efficiency
                break
        
        # Convert to numpy arrays
        if is_image:
            all_inputs = np.array(all_inputs)
        else:
            all_inputs = np.array(all_inputs)
        all_cluster_ids = np.array(all_cluster_ids)
        
        # Create a background dataset for SHAP
        background_indices = np.random.choice(len(all_inputs), min(background_size, len(all_inputs)), replace=False)
        background = torch.tensor(all_inputs[background_indices], device=device)
        
        # Create an explainer
        if is_image:
            # For CNNs, try using GradientExplainer first as it's more likely to work
            try:
                explainer = shap.GradientExplainer(model, background)
            except Exception as e:
                print(f"GradientExplainer failed for layer {layer_name}, falling back to DeepExplainer: {e}")
                try:
                    explainer = shap.DeepExplainer(model, background)
                except Exception as e:
                    print(f"Both SHAP explainers failed for layer {layer_name}: {e}")
                    if original_model_training:
                        model.train()
                    return {}
            
            # Get SHAP values for a subset
            try:
                test_indices = np.random.choice(len(all_inputs), min(20, len(all_inputs)), replace=False)
                test_data = torch.tensor(all_inputs[test_indices], device=device)
                shap_values = explainer.shap_values(test_data)
            except Exception as e:
                print(f"Failed to get SHAP values for layer {layer_name}: {e}")
                if original_model_training:
                    model.train()
                return {}
            
            # Group SHAP values by cluster
            cluster_explanations = {}
            for cluster_id in range(num_clusters):
                # Get indices of samples belonging to this cluster
                cluster_indices = np.where(all_cluster_ids[test_indices] == cluster_id)[0]
                
                if len(cluster_indices) == 0:
                    continue
                    
                # Get average SHAP values for this cluster
                if isinstance(shap_values, list):
                    # For multi-class models
                    cluster_shap = [shap_values[i][cluster_indices].mean(axis=0) for i in range(len(shap_values))]
                else:
                    # For single-output models
                    cluster_shap = shap_values[cluster_indices].mean(axis=0)
                    
                # Store the explanation
                cluster_explanations[cluster_id] = {
                    'shap_values': cluster_shap,
                    'num_samples': len(cluster_indices)
                }
        else:
            # For tabular data, use KernelExplainer which is more reliable
            try:
                def model_predict(x):
                    with torch.no_grad():
                        inputs = torch.tensor(x, dtype=torch.float32).to(device)
                        return model(inputs).cpu().numpy()
                
                # Use a subset of data as background
                explainer = shap.KernelExplainer(model_predict, all_inputs[background_indices])
                
                # Get SHAP values for a subset
                test_indices = np.random.choice(len(all_inputs), min(20, len(all_inputs)), replace=False)
                test_data = all_inputs[test_indices]
                
                shap_values = explainer.shap_values(test_data)
            except Exception as e:
                print(f"SHAP KernelExplainer failed for layer {layer_name}: {e}")
                if original_model_training:
                    model.train()
                return {}
            
            # Group SHAP values by cluster
            cluster_explanations = {}
            for cluster_id in range(num_clusters):
                # Get indices of samples belonging to this cluster
                cluster_indices = np.where(all_cluster_ids[test_indices] == cluster_id)[0]
                
                if len(cluster_indices) == 0:
                    continue
                    
                # Get average SHAP values for this cluster
                if isinstance(shap_values, list):
                    # For multi-class models
                    cluster_shap = [shap_val[cluster_indices].mean(axis=0) for shap_val in shap_values]
                else:
                    # For single-output models
                    cluster_shap = shap_values[cluster_indices].mean(axis=0)
                    
                # Store the explanation
                cluster_explanations[cluster_id] = {
                    'shap_values': cluster_shap,
                    'num_samples': len(cluster_indices)
                }
        
        # Restore original model state
        if original_model_training:
            model.train()
        
        return cluster_explanations
    except Exception as e:
        print(f"SHAP cluster explanation failed for layer {layer_name}: {e}")
        # Restore original model state
        if original_model_training:
            model.train()
        return {}

def visualize_cluster_explanations(cluster_explanations, is_image=True, feature_names=None, max_clusters=5):
    """
    Visualize SHAP explanations for clusters
    
    Args:
        cluster_explanations: Dictionary of cluster explanations
        is_image: Whether the data is image data
        feature_names: Names of features (for tabular data)
        max_clusters: Maximum number of clusters to visualize
    """
    num_clusters = len(cluster_explanations)
    clusters_to_vis = min(num_clusters, max_clusters)
    
    if is_image:
        plt.figure(figsize=(15, 4 * clusters_to_vis))
        
        for i, (cluster_id, explanation) in enumerate(list(cluster_explanations.items())[:clusters_to_vis]):
            plt.subplot(clusters_to_vis, 1, i + 1)
            
            # For images, visualize the average SHAP values
            shap_values = explanation['shap_values']
            
            if isinstance(shap_values, list):
                # For multi-class models, visualize the first class
                shap_image = shap_values[0]
            else:
                shap_image = shap_values
                
            # If shap_image is 4D (batch, height, width, channels), take the first image
            if shap_image.ndim == 4:
                shap_image = shap_image[0]
                
            # For RGB images, sum across channels
            if shap_image.ndim == 3 and shap_image.shape[2] == 3:
                shap_image = np.sum(shap_image, axis=2)
                
            plt.imshow(shap_image, cmap='bwr')
            plt.colorbar()
            plt.title(f"Cluster {cluster_id} ({explanation['num_samples']} samples)")
            plt.axis('off')
            
        plt.tight_layout()
        save_current_figure("Cluster SHAP Explanations (Images)")
    else:
        # For tabular data, create a bar chart of feature importance
        for cluster_id, explanation in list(cluster_explanations.items())[:clusters_to_vis]:
            plt.figure(figsize=(10, 6))
            
            shap_values = explanation['shap_values']
            
            if isinstance(shap_values, list):
                # For multi-class models, visualize the first class
                shap_values = shap_values[0]
                
            # Sort by absolute value
            indices = np.argsort(np.abs(shap_values))[::-1]
            
            # Get feature names if available
            if feature_names is not None:
                sorted_features = [feature_names[i] for i in indices]
            else:
                sorted_features = [f"Feature {i}" for i in indices]
                
            # Plot top features
            num_features = min(10, len(sorted_features))
            plt.barh(range(num_features), shap_values[indices[:num_features]])
            plt.yticks(range(num_features), sorted_features[:num_features])
            plt.title(f"Cluster {cluster_id} Feature Importance ({explanation['num_samples']} samples)")
            plt.xlabel("SHAP Value (Impact on Model Output)")
            plt.tight_layout()
            save_current_figure(f"Cluster_{cluster_id}_SHAP_Explanation")


def create_integrated_gcfn(model, activation_dict, cluster_data_dict, causal_graph, title="Integrated GCFN"):
    """
    Create an integrated GCFN visualization that shows clusters and their causal relationships.
    
    Args:
        model: The neural network model
        activation_dict: Dictionary of activations for each layer
        cluster_data_dict: Dictionary of cluster data for each layer
        causal_graph: The causal graph showing relationships between clusters
        title: Title for the visualization
    """
    print("Generating integrated GCFN visualization...")
    
    if causal_graph is None or not causal_graph.edges():
        print("No causal graph available for integrated visualization")
        return
    
    # Create a figure for the integrated visualization
    plt.figure(figsize=(20, 16))
    plt.suptitle(title, fontsize=20, y=0.98)
    
    # Get all layer names from the causal graph
    all_layers = set()
    for node in causal_graph.nodes():
        if '_cluster_' in node:
            layer_name = node.split('_cluster_')[0]
            all_layers.add(layer_name)
    
    layer_order = sorted(all_layers)
    print(f"Layers for integrated visualization: {layer_order}")
    
    # Create a directory for the integrated visualization
    integrated_dir = os.path.join('figures', 'integrated_gcfn')
    os.makedirs(integrated_dir, exist_ok=True)
    
    # Generate individual visualizations for each cluster
    cluster_images = {}
    for layer_name in layer_order:
        if layer_name not in activation_dict or layer_name not in cluster_data_dict:
            continue
            
        activations = activation_dict[layer_name]
        
        # Get cluster assignments
        cluster_data = cluster_data_dict[layer_name]
        num_clusters = cluster_data.shape[1]
        
        # Recreate cluster assignments from cluster data
        # This is a simplification - ideally we would store cluster assignments when creating cluster_data
        cluster_assignments = np.argmax(cluster_data, axis=1)
        
        # Generate a t-SNE visualization for this layer
        fig = plt.figure(figsize=(10, 8))
        
        # Apply PCA for dimensionality reduction
        pca_components = min(50, activations.shape[1])
        pca = PCA(n_components=pca_components)
        activations_pca = pca.fit_transform(activations)
        
        # Apply t-SNE
        perplexity = min(30, activations.shape[0] // 5)  # Adjust perplexity based on sample size
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
        activations_tsne = tsne.fit_transform(activations_pca)
        
        # Plot clusters
        plt.scatter(
            activations_tsne[:, 0], activations_tsne[:, 1],
            c=cluster_assignments, cmap='viridis', 
            alpha=0.7, s=30
        )
        
        plt.title(f"Layer: {layer_name} (t-SNE Visualization)")
        plt.colorbar(label='Cluster')
        plt.tight_layout()
        
        # Save the figure for this layer
        layer_filename = os.path.join(integrated_dir, f"{layer_name}_tsne.png")
        plt.savefig(layer_filename, dpi=100)
        plt.close(fig)
        
        # Generate individual visualizations for each cluster
        for cluster_idx in range(num_clusters):
            # Find samples belonging to this cluster
            cluster_samples = activations[cluster_assignments == cluster_idx]
            
            if len(cluster_samples) < 5:  # Skip if too few samples
                continue
                
            # Create a visualization of feature distributions for this cluster
            fig = plt.figure(figsize=(8, 6))
            
            # Generate a compact feature visualization
            if cluster_samples.shape[1] > 20:
                # Too many features, show a PCA reduction
                pca = PCA(n_components=min(5, cluster_samples.shape[1]))
                reduced_features = pca.fit_transform(cluster_samples)
                
                # Plot distributions of principal components
                for i in range(reduced_features.shape[1]):
                    plt.hist(reduced_features[:, i], alpha=0.5, bins=20, 
                             label=f"PC{i+1} (var: {pca.explained_variance_ratio_[i]:.2f})")
            else:
                # Directly visualize feature distributions
                for i in range(min(5, cluster_samples.shape[1])):
                    plt.hist(cluster_samples[:, i], alpha=0.5, bins=20, 
                             label=f"Feature {i+1}")
            
            plt.title(f"{layer_name} - Cluster {cluster_idx} Features")
            plt.legend()
            plt.tight_layout()
            
            # Save the figure for this cluster
            cluster_filename = os.path.join(integrated_dir, f"{layer_name}_cluster_{cluster_idx}.png")
            plt.savefig(cluster_filename, dpi=100)
            plt.close(fig)
            
            # Store the filename for this cluster
            node_name = f"{layer_name}_cluster_{cluster_idx}"
            cluster_images[node_name] = cluster_filename
    
    # Create a directed graph visualization
    G = causal_graph.copy()
    
    # Create a position layout for the graph
    pos = {}
    for i, layer in enumerate(layer_order):
        y_pos = 1 - (i / max(1, len(layer_order) - 1))  # Normalize y position between 0 and 1
        
        # Find all nodes for this layer
        layer_nodes = [node for node in G.nodes() if node.startswith(f"{layer}_cluster_")]
        
        # Calculate how many clusters we have for this layer
        num_clusters = len(layer_nodes)
        
        for j, node in enumerate(sorted(layer_nodes)):
            x_pos = (j + 0.5) / max(1, num_clusters)  # Space clusters evenly
            pos[node] = (x_pos, y_pos)
    
    # Create a directed graph visualization
    fig = plt.figure(figsize=(18, 14))
    plt.suptitle("Causal Graph with Integrated Features", fontsize=16)
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, 
        edge_color='gray',
        alpha=0.6,
        arrows=True, 
        arrowsize=15,
        connectionstyle='arc3,rad=0.1'
    )
    
    # Draw nodes with different colors for each layer
    layer_colors = plt.cm.tab10(np.linspace(0, 1, len(layer_order)))
    for i, layer in enumerate(layer_order):
        layer_nodes = [node for node in G.nodes() if node.startswith(layer)]
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=layer_nodes, 
            node_color=[layer_colors[i]] * len(layer_nodes),
            node_size=1500,
            alpha=0.8
        )
    
    # Add labels with node info
    labels = {}
    for node in G.nodes():
        layer, cluster = node.split('_cluster_')
        # Count the number of edges for this node
        in_edges = len(list(G.in_edges(node)))
        out_edges = len(list(G.out_edges(node)))
        
        # Create a more descriptive label
        labels[node] = f"{layer}\nCluster {cluster}\n({in_edges} in, {out_edges} out)"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
    
    # Add a legend
    legend_elements = []
    for i, layer in enumerate(layer_order):
        legend_elements.append(plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor=layer_colors[i],
            markersize=10, label=f"{layer}"
        ))
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.axis('off')
    
    # Save the graph visualization
    graph_filename = os.path.join(integrated_dir, "causal_graph.png")
    plt.savefig(graph_filename, dpi=100)
    plt.close(fig)
    
    # Generate an HTML visualization that combines all images
    html_filename = os.path.join(integrated_dir, "integrated_gcfn.html")
    
    with open(html_filename, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ text-align: center; color: #333; margin-bottom: 30px; }}
        h2 {{ color: #444; margin-top: 30px; }}
        .causal-graph {{ text-align: center; margin: 30px 0; }}
        .causal-graph img {{ max-width: 100%; border: 1px solid #ddd; }}
        .layer-section {{ margin: 40px 0; }}
        .clusters {{ display: flex; flex-wrap: wrap; justify-content: center; }}
        .cluster {{ margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: white; }}
        .cluster h3 {{ margin-top: 0; color: #555; }}
        .cluster img {{ width: 100%; max-width: 400px; }}
        .tsne-plot {{ text-align: center; margin: 20px 0; }}
        .tsne-plot img {{ max-width: 500px; border: 1px solid #ddd; }}
        .footer {{ text-align: center; margin-top: 50px; color: #777; font-size: 0.8em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="causal-graph">
            <h2>Causal Graph Overview</h2>
            <p>This graph shows causal relationships between clusters across different layers.</p>
            <img src="{os.path.basename(graph_filename)}" alt="Causal Graph">
        </div>
""")
        
        # Add sections for each layer
        for layer in layer_order:
            # Check if we have a t-SNE visualization for this layer
            layer_tsne = os.path.join(integrated_dir, f"{layer}_tsne.png")
            if os.path.exists(layer_tsne):
                f.write(f"""
        <div class="layer-section">
            <h2>Layer: {layer}</h2>
            
            <div class="tsne-plot">
                <h3>t-SNE Visualization</h3>
                <img src="{os.path.basename(layer_tsne)}" alt="{layer} t-SNE">
            </div>
            
            <h3>Individual Clusters</h3>
            <div class="clusters">
""")
                
                # Add all clusters for this layer
                layer_nodes = [node for node in G.nodes() if node.startswith(f"{layer}_cluster_")]
                for node in sorted(layer_nodes):
                    if node in cluster_images and os.path.exists(cluster_images[node]):
                        f.write(f"""
                <div class="cluster">
                    <h3>{node}</h3>
                    <img src="{os.path.basename(cluster_images[node])}" alt="{node}">
                    <p>Connections: {len(list(G.in_edges(node)))} in, {len(list(G.out_edges(node)))} out</p>
""")
                        
                        # Add incoming connections
                        if G.in_edges(node):
                            f.write("<p>Influenced by:</p><ul>")
                            for source, _ in G.in_edges(node):
                                f.write(f"<li>{source}</li>")
                            f.write("</ul>")
                        
                        # Add outgoing connections
                        if G.out_edges(node):
                            f.write("<p>Influences:</p><ul>")
                            for _, target in G.out_edges(node):
                                f.write(f"<li>{target}</li>")
                            f.write("</ul>")
                        
                        f.write("</div>")
                
                f.write("""
            </div>
        </div>
""")
        
        # Close the HTML file
        f.write("""        
        <div class="footer">
            <p>Generated by Causal Clusters for Deep Learning Models</p>
        </div>
    </div>
</body>
</html>
""")
    
    print(f"Integrated GCFN visualization saved to {html_filename}")
    
    # Also add a link to this visualization in the main HTML report
    return html_filename

# ---------------------------
# Main Execution

def get_transform(dataset_name):
    """
    Get appropriate transformations based on dataset type.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'lung_cancer', 'insurance', etc.)
        
    Returns:
        transform_func: Dataset-specific transform function
    """
    if dataset_name == 'lung_cancer':
        # For image datasets
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        # For tabular datasets, return a simple normalization function
        def tabular_transform(x):
            # Convert to tensor and normalize if needed
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=torch.float32)
            return x
        
        return tabular_transform




def main_flexible_clustering():
    """Flexible Clustering Demo - Lung Cancer Dataset"""
    print("Flexible Clustering Demo - Lung Cancer Dataset")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Load dataset
    print("Loading lung cancer dataset...")
    try:
        batch_size = 16  # Smaller batch size for faster processing
        dataloader, dataset = get_image_dataloader(
            image_dir='IQ-OTHNCCD',
            batch_size=batch_size,
            shuffle=True,
            metric_learning=True
        )
        print(f"Dataset loaded successfully: {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Step 2: Load or create model
    model_path = "lung_cancer_model.pth"
    num_classes = 3  # lung cancer classes
    model = SimpleCNN(num_classes=num_classes).to(device)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    # Step 3: Extract activations
    from time import time
    start_time = time()
    print("Extracting activations from model layers...")
    try:
        activations, labels = extract_activations(model, dataloader, device, metric_learning=True)
        print(f"Extracted activations for {len(activations)} layers in {time() - start_time:.2f} seconds")
        
        # Print information about each layer's activations
        for layer_name, layer_acts in activations.items():
            print(f"Layer {layer_name}: shape {layer_acts.shape}, mean {layer_acts.mean():.4f}, std {layer_acts.std():.4f}")
    except Exception as e:
        print(f"Error extracting activations: {e}")
        return
    
    # Step 4: Flexible clustering
    print("\nPerforming flexible clustering with optimal cluster counts...")
    try:
        # Maximum number of clusters to consider per layer
        max_clusters = 8  
        
        # Perform clustering with optimized number of clusters per layer
        start_time = time()
        clustering_results = perform_clustering_optimized(
            activations, 
            max_clusters=max_clusters,
            pca_components=min(50, min(layer.shape[1] for layer in activations.values()) - 1)
        )
        print(f"Clustering completed in {time() - start_time:.2f} seconds")
        
        # Check the optimal cluster count for each layer
        for layer_name, clusters in clustering_results.items():
            n_clusters = len(np.unique(clusters))
            print(f"Layer {layer_name}: {n_clusters} optimal clusters")
    except Exception as e:
        print(f"Error in clustering: {e}")
        return
    
    # Step 5: Create latent cluster dataset
    print("\nCreating latent cluster dataset...")
    try:
        cluster_df = create_latent_cluster_dataset(clustering_results, labels)
        print("Latent cluster dataset created successfully")
        print(f"Dataset shape: {cluster_df.shape}")
        print("First few rows:")
        print(cluster_df.head())
    except Exception as e:
        print(f"Error creating latent cluster dataset: {e}")
        return
    
    # Step 6: Visualize clusters and features
    print("\nGenerating visualizations...")
    try:
        # Visualize clusters
        for layer_name in activations.keys():
            print(f"Visualizing clusters for layer {layer_name}...")
            perplexity = min(30, activations[layer_name].shape[0] // 5)
            perplexity = max(5, perplexity)  # Ensure perplexity is at least 5
            
            visualize_clusters(
                activations, 
                clustering_results, 
                layer_name,
                apply_pca=True, 
                pca_components=min(50, activations[layer_name].shape[1] - 1),
                tsne_perplexity=perplexity
            )
        
        # Visualize learned features
        for layer_name in activations.keys():
            print(f"Visualizing learned features for layer {layer_name}...")
            visualize_learned_features(activations, layer_name, n_components=3)
    except Exception as e:
        print(f"Error in visualization: {e}")
    
    # Step 7: Causal discovery
    print("\nPerforming causal discovery...")
    try:
        causal_graphs = causal_discovery_all_layers(
            cluster_df, 
            layer_names=list(activations.keys())
        )
        print("Causal discovery completed")
    except Exception as e:
        print(f"Error in causal discovery: {e}")
    
    # Step 8: Generate HTML report
    print("\nGenerating HTML report...")
    write_html_page("flexible_clusters_report.html")
    
    print("\nAnalysis complete! Check the flexible_clusters_report.html for results.")


def main_test_clustering():
    """Test clustering functionality with optimal cluster detection"""
    # Load image dataset and model
    image_dir = "Data/Image/IQ-OTHNCCD"
    try:
        dataloader, dataset = get_image_dataloader(
            image_dir=image_dir,
            batch_size=16,
            shuffle=True,
            metric_learning=True
        )
        print(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Load or create model
    model_path = "lung_cancer_model.pth"
    num_classes = len(dataset.class_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=num_classes).to(device)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("Model file not found")
        return
    
    # Extract activations
    try:
        activations, labels = extract_activations(model, dataloader, device, metric_learning=True)
        print(f"Extracted activations for {len(activations)} layers")
    except Exception as e:
        print(f"Error extracting activations: {e}")
        return
    
    # For each layer, find optimal clusters and visualize
    clustering_results = {}
    for layer_name, layer_activations in activations.items():
        print(f"\nAnalyzing layer: {layer_name}")
        print(f"Layer shape: {layer_activations.shape}")
        
        # Use the existing clustering function with optimal cluster detection
        clustering_results = perform_clustering_optimized(
            {layer_name: layer_activations}, 
            max_clusters=8,
            pca_components=min(50, layer_activations.shape[1] - 1)
        )
        cluster_labels = clustering_results[layer_name]
        
        # Visualize clusters using existing function
        visualize_clusters(
            {layer_name: layer_activations}, 
            clustering_results, 
            layer_name,
            apply_pca=True,
            pca_components=min(50, layer_activations.shape[1] - 1)
        )
    
    print("\nClustering analysis complete!")


def main_integrated_gcfn():
    """Generate Integrated GCFN Visualization"""
    parser = argparse.ArgumentParser(description="Generate Integrated GCFN Visualization")
    parser.add_argument('--dataset', type=str, default='lung_cancer', 
                      choices=['lung_cancer', 'insurance', 'aging_clock'],
                      help='Dataset to use (lung_cancer, insurance, or aging_clock)')
    parser.add_argument('--model_path', type=str, default='lung_cancer_model.pth',
                      help='Path to pre-trained model')
    parser.add_argument('--max_clusters', type=int, default=8, 
                      help='Maximum number of clusters per layer')
    parser.add_argument('--output_html', type=str, default='integrated_gcfn.html',
                      help='Output HTML file name')
    parser.add_argument('--batch_size', type=int, default=16, 
                      help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                     help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found. Please train a model first.")
        return
    
    # Setup based on dataset type
    is_image_data = args.dataset == 'lung_cancer'
    
    # Load dataset and model
    print(f"Loading {args.dataset} dataset...")
    if is_image_data:
        # Load image dataset
        dataloader, _ = get_image_dataloader(
            image_dir='IQ-OTHNCCD',
            batch_size=args.batch_size,
            shuffle=True,
            metric_learning=True
        )
        # Create model
        model = SimpleCNN(num_classes=3).to(device)
    else:
        # For tabular datasets (simplified - in practice you'd need to load the real data)
        print("Error: This script currently only supports the lung_cancer dataset.")
        print("Please run main.py with --integrated flag for other datasets.")
        return
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Extract activations
    print("Extracting activations from model layers...")
    activations, labels = extract_activations(model, dataloader, device, metric_learning=is_image_data)
    
    # Perform clustering
    print("Performing clustering with optimal cluster counts...")
    max_pca_components = min(50, min(layer.shape[1] for layer in activations.values()) - 1)
    max_pca_components = max(2, max_pca_components)  # Ensure at least 2 components
    
    clustering_results = perform_clustering_optimized(
        activations, 
        max_clusters=args.max_clusters, 
        pca_components=max_pca_components
    )
    
    # Generate integrated GCFN visualization
    print("Generating integrated GCFN visualization...")
    
    # Create cluster data dict (one-hot per layer)
    cluster_data_dict = {}
    for layer, clusters in clustering_results.items():
        cluster_series = pd.Series(clusters, name=f"{layer}_cluster")
        one_hot = pd.get_dummies(cluster_series, prefix=f"{layer}_cluster")
        cluster_data_dict[layer] = one_hot.values
    
    # Create causal graph
    latent_df = create_latent_cluster_dataset(clustering_results, labels)
    causal_graph = causal_discovery_all_layers(latent_df)
    
    create_integrated_gcfn(
        model,
        activations,
        cluster_data_dict,
        causal_graph,
        title=args.output_html
    )
    
    print(f"Integrated GCFN visualization created successfully! View it at: {args.output_html}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Causal Clusters for Deep Learning Models')
    parser.add_argument('--dataset', type=str, default='lung_cancer', help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, default=None, help='Pre-trained model path')
    parser.add_argument('--flexible', action='store_true', help='Use flexible clustering (determine optimal clusters)')
    parser.add_argument('--integrated', action='store_true', help='Generate integrated GCFN visualization')
    parser.add_argument('--image_dir', type=str, default='IQ-OTHNCCD', help='Path to image dataset directory')
    parser.add_argument('--fallback', action='store_true', help='Automatically fallback to insurance dataset if image dataset is not found')
    args = parser.parse_args()
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Seed for reproducibility
    seed_everything(42)
    
    # Get dataset based on dataset type
    if args.dataset == 'lung_cancer':
        # For image dataset
        print(f"Loading {args.dataset} dataset (image data) from {args.image_dir}...")
        try:
            # Explicitly set metric_learning to False for consistency
            train_loader, train_dataset = get_image_dataloader(args.image_dir, 
                                                            batch_size=args.batch_size, 
                                                            metric_learning=False)
            test_loader, test_dataset = get_image_dataloader(args.image_dir, 
                                                           batch_size=args.batch_size, 
                                                           metric_learning=False)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            
            if args.fallback:
                print("\nFalling back to 'insurance' dataset since lung cancer dataset was not found...")
                args.dataset = 'insurance'
                print(f"Loading {args.dataset} dataset (tabular data)...")
                train_loader, train_dataset = get_tabular_dataloader('insurance', args.batch_size)
                test_loader, test_dataset = get_tabular_dataloader('insurance', args.batch_size)
            else:
                print("\nTo use the lung cancer dataset, please:")
                print("1. Download a dataset like 'IQ-OTHNCCD' or another lung cancer image dataset")
                print("2. Organize it with subdirectories for each class (benign, malignant, normal)")
                print("3. Run the script again with: --image_dir /path/to/your/dataset")
                print("\nAlternatively, you can use the built-in insurance dataset with:")
                print("python CC.py --dataset insurance --flexible --integrated")
                print("\nOr add --fallback to automatically switch to the insurance dataset when images aren't found:")
                print("python CC.py --dataset lung_cancer --flexible --integrated --fallback")
                sys.exit(1)
    else:
        # For tabular datasets
        print(f"Loading {args.dataset} dataset (tabular data)...")
        train_loader, train_dataset = get_tabular_dataloader(args.dataset, args.batch_size)
        test_loader, test_dataset = get_tabular_dataloader(args.dataset, args.batch_size)
    
    # Get model
    if args.dataset == 'lung_cancer':
        # For lung cancer image dataset
        if args.model is None:
            # Train model
            model = SimpleCNN(num_classes=3).to('cuda' if torch.cuda.is_available() else 'cpu')
            train_network(model, train_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                         args.epochs, args.lr, use_metric_learning=False)
        else:
            # Try to load pre-trained model
            try:
                model = SimpleCNN(num_classes=3).to('cuda' if torch.cuda.is_available() else 'cpu')
                model.load_state_dict(torch.load(args.model, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
                model.eval()
                print(f"Successfully loaded pre-trained model: {args.model}")
            except Exception as e:
                print(f"Error loading pre-trained model: {e}")
                print("Architecture mismatch detected. This could be due to changes in the model architecture.")
                print("Options:")
                print("1. Train a new model")
                print("2. Use a compatible model")
                
                print("\nTraining a new model...")
                model = SimpleCNN(num_classes=3).to('cuda' if torch.cuda.is_available() else 'cpu')
                train_network(model, train_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                             args.epochs, args.lr, use_metric_learning=False)
                
                # Save the new model
                new_model_path = "lung_cancer_model_new.pth"
                torch.save(model.state_dict(), new_model_path)
                print(f"New model saved to {new_model_path}")
    else:
        # For tabular datasets
        # Get input dimensions from first batch
        for batch in train_loader:
            input_dim = batch[0].shape[1]
            break
            
        if args.model is None:
            # Train model
            model = SimpleMLP(input_dim=input_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
            train_network(model, train_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                         args.epochs, args.lr)
        else:
            # Load pre-trained model
            model = SimpleMLP(input_dim=input_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(args.model, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
            model.eval()
    
    # Get activations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Extracting activations using device: {device}")
    activation_dict, labels = extract_activations(model, test_loader, device, 
                                         metric_learning=False)  # Set to False for consistency
    
    # Perform flexible or fixed clustering
    
    if args.flexible:
        print("Using flexible clustering approach (determining optimal number of clusters)...")
        cluster_data_dict = {}
        for layer_name, activations in activation_dict.items():
            print(f"Processing layer: {layer_name}")
            # Find optimal number of clusters with caching
            clustering_result = perform_clustering_optimized(activations, max_clusters=10, pca_components=50)
            
            if clustering_result is not None:
                cluster_assignments, silhouette_score, num_clusters = clustering_result
                print(f"Layer {layer_name}: Using {num_clusters} clusters (silhouette score: {silhouette_score:.4f})")
                
                # Visualize clusters
                visualize_clusters(activations, cluster_assignments, layer_name)
                
                # Create dataset for each layer
                cluster_data = create_latent_cluster_dataset(activations, cluster_assignments, num_clusters)
                cluster_data_dict[layer_name] = cluster_data
            else:
                print(f"Clustering failed for layer {layer_name}")
        
        # Perform causal discovery on all layers
        causal_graph = causal_discovery_all_layers(cluster_data_dict)
        
        # Visualize causal graph
        visualize_causal_graph(causal_graph, "Causal Graph (Flexible Clustering)")
        
        # Visualize learned features
        for layer_name, activations in activation_dict.items():
            if layer_name in cluster_data_dict:
                visualize_learned_features(activations, layer_name, cluster_data_dict[layer_name].shape[1])
    else:
        # Using fixed 3 clusters approach
        print("Using fixed clustering approach (3 clusters per layer)...")
        # Original clustering approach, using fixed 3 clusters
        num_clusters = 3
        all_clusters = {}
        all_cluster_data = {}
        
        for layer_name, activations in activation_dict.items():
            print(f"Processing layer: {layer_name}")
            
            # Perform clustering with forced cluster count
            clustering_result = perform_clustering_optimized(activations, 
                                                            max_clusters=num_clusters, 
                                                            pca_components=50,
                                                            force_clusters=num_clusters)
            
            if clustering_result is not None:
                cluster_assignments, silhouette_score, _ = clustering_result
                print(f"Layer {layer_name}: Fixed {num_clusters} clusters (silhouette score: {silhouette_score:.4f})")
                
                # Save cluster assignments
                all_clusters[layer_name] = cluster_assignments
                
                # Visualize clusters
                visualize_clusters(activations, cluster_assignments, layer_name)
                
                # Create cluster dataset
                cluster_data = create_latent_cluster_dataset(activations, cluster_assignments, num_clusters)
                all_cluster_data[layer_name] = cluster_data
        
        # Combine cluster data across layers
        combined_data = pd.DataFrame()
        layer_names = []
        
        for layer_name, cluster_data in all_cluster_data.items():
            layer_names.append(layer_name)
            # Create column names with layer and cluster information
            col_names = [f"{layer_name}_cluster_{i}" for i in range(cluster_data.shape[1])]
            df = pd.DataFrame(cluster_data, columns=col_names)
            
            # Combine with existing data
            if combined_data.empty:
                combined_data = df
            else:
                combined_data = pd.concat([combined_data, df], axis=1)
        
        # Perform causal discovery on combined dataset
        causal_graph = causal_discovery_all_layers(all_cluster_data)
        
        # Visualize causal graph
        visualize_causal_graph(causal_graph, "Causal Graph (Fixed 3 Clusters)")
        
        # Visualize learned features
        for layer_name, activations in activation_dict.items():
            if layer_name in all_cluster_data:
                visualize_learned_features(activations, layer_name, num_clusters)
    
    # Generate integrated GCFN visualization if requested
    if args.integrated:
        if args.flexible:
            create_integrated_gcfn(model, activation_dict, cluster_data_dict, causal_graph, "Integrated GCFN (Flexible Clustering)")
        else:
            create_integrated_gcfn(model, activation_dict, all_cluster_data, causal_graph, "Integrated GCFN (Fixed Clustering)")
    
    # Generate HTML report
    write_html_page()

















