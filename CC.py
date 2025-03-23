# Standard Imports
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import re

# Additional imports for training, clustering, visualization, and causal discovery
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score

# For causal discovery (if installed)
try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.GraphUtils import GraphUtils
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    print("Warning: causallearn is not installed. Causal discovery functions will be limited.")

import networkx as nx

# For SHAP explanations (make sure to install shap via pip)
import shap

import sys


def write_html_page(output_html="gcfn.html"):
    html_content = """<html>
<head><meta charset="utf-8"><title>GCFN Results</title></head>
<body>
<h1>Global Causal Feature Network (GCFN) Outputs</h1>
"""

    for fig_name, fig_title in figure_files:
        html_content += f'<h2>{fig_title}</h2>\n'
        html_content += f'<img src="{fig_name}" style="max-width:600px;"><br/><br/>\n'
        
    html_content += "</body></html>"

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"All figures and references saved to '{output_html}'.")


figure_count = 1
figure_files = []

def save_current_figure(title="Figure"):
    """Save the current matplotlib figure to a PNG and remember its file name."""
    global figure_count
    fig_name = f"figure_{figure_count}.png"
    plt.title(title)
    plt.savefig(fig_name, dpi=100, bbox_inches='tight')
    plt.close()  # close the figure to free memory
    figure_files.append((fig_name, title))
    figure_count += 1




# ---------------------------
# Dataset Definitions

class LungCancerDataset(Dataset):
    def __init__(self, image_dir, transform=None, metric_learning=False):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.metric_learning = metric_learning

        # Map folder names to labels (expects folders: benign, malignant, normal)
        self.class_map = {
            folder_name: idx
            for idx, folder_name in enumerate(os.listdir(image_dir))
            if os.path.isdir(os.path.join(image_dir, folder_name))
        }

        self.image_paths = []
        self.labels = []
        self.class_to_images = {lbl: [] for lbl in self.class_map.values()}

        for class_name, label in self.class_map.items():
            class_dir = os.path.join(image_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, file_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    self.class_to_images[label].append(img_path)

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
        flattened_size = 128 * 28 * 28  # for input size 224x224

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
            if metric_learning:
                anchor, _, _ = batch
                anchor = anchor.to(device)
                _ = model(anchor)
                label_list.extend([0] * anchor.size(0))
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                _ = model(inputs)
                label_list.extend(labels.cpu().numpy())

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

    # When called from generate_gcfn_report, perform_clustering functions, we only want activations
    if 'generate_gcfn_report' in sys._getframe(1).f_code.co_name or 'perform_clustering' in sys._getframe(1).f_code.co_name:
        return layer_outputs
    
    return layer_outputs, np.array(label_list)


def perform_clustering_optimized(activations, max_clusters=10, pca_components=50):
    """
    Perform clustering on activations with automatic selection of optimal number of clusters.
    
    Args:
        activations: Dictionary of activations or a single activation array
        max_clusters: Maximum number of clusters to consider
        pca_components: Number of PCA components to use for dimensionality reduction
        
    Returns:
        If activations is a dictionary: Dictionary of cluster labels for each layer
        If activations is an array: Tuple of (labels, centroids)
    """
    # If activations is a single array (not a dictionary)
    if not isinstance(activations, dict):
        scaler = StandardScaler()
        normalized = scaler.fit_transform(activations)
        if normalized.shape[1] > pca_components:
            pca = PCA(n_components=pca_components, random_state=42)
            reduced = pca.fit_transform(normalized)
        else:
            reduced = normalized

        best_n = 2
        best_score = -1
        best_kmeans = None
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(reduced)
            score = silhouette_score(reduced, clusters)
            if score > best_score:
                best_score = score
                best_n = k
                best_kmeans = kmeans
        print(f"Best number of clusters: {best_n}, silhouette score: {best_score:.4f}")
        # Return both labels and centroids
        return best_kmeans.labels_, best_kmeans.cluster_centers_
        
    # If activations is a dictionary
    clustering_results = {}
    for layer, act_data in activations.items():
        scaler = StandardScaler()
        normalized = scaler.fit_transform(act_data)
        if normalized.shape[1] > pca_components:
            pca = PCA(n_components=pca_components, random_state=42)
            reduced = pca.fit_transform(normalized)
        else:
            reduced = normalized

        best_n = 2
        best_score = -1
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(reduced)
            score = silhouette_score(reduced, clusters)
            if score > best_score:
                best_score = score
                best_n = k
        print(f"{layer}: best_n={best_n}, silhouette={best_score:.4f}")
        final_kmeans = KMeans(n_clusters=best_n, random_state=42)
        clustering_results[layer] = final_kmeans.fit_predict(reduced)
    return clustering_results

def create_latent_cluster_dataset(clustering_results, labels):
    """
    Create a one-hot encoded latent cluster dataset.
    Each layer's one-hot encoding now drops the first column to avoid collinearity.
    """
    df_list = []
    
    # Ensure all cluster assignments have the same length
    min_length = min(len(clusters) for clusters in clustering_results.values())
    
    # Truncate labels to match the minimum length
    if len(labels) > min_length:
        labels = labels[:min_length]
    
    for layer, clusters in clustering_results.items():
        # Truncate clusters to match the minimum length
        if len(clusters) > min_length:
            clusters = clusters[:min_length]
            
        cluster_series = pd.Series(clusters, name=layer)
        one_hot = pd.get_dummies(cluster_series, prefix=f"{layer}_cluster", drop_first=True)
        df_list.append(one_hot)
    
    latent_df = pd.concat(df_list, axis=1)
    latent_df["label"] = labels
    
    return latent_df

def get_nodes_from_edge(edge):
    if isinstance(edge, str):
        edge_clean = edge.replace('-->', '->').replace('---', '->')
        parts = edge_clean.split('->')
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        else:
            raise ValueError("Edge string not in expected format: " + edge)
    elif hasattr(edge, 'node1') and hasattr(edge, 'node2'):
        print("DEBUG: Using node1 and node2 attributes:", edge.node1, edge.node2)
        source = edge.node1.name if hasattr(edge.node1, 'name') else str(edge.node1)
        target = edge.node2.name if hasattr(edge.node2, 'name') else str(edge.node2)
        return source, target
    else:
        raise ValueError("Edge object is not recognized: " + str(edge))

def visualize_causal_graph(causal_graph, title="Causal Graph"):
    if causal_graph is None:
        print("No causal graph to visualize.")
        return
    G = nx.DiGraph()
    edges = causal_graph.get_graph_edges()
    for edge in edges:
        try:
            source, target = get_nodes_from_edge(edge)
        except Exception as e:
            print("DEBUG: Cannot extract nodes from edge:", edge, "Error:", e)
            continue
        G.add_edge(str(source), str(target))
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_size=500, node_color="lightblue", arrows=True)
    plt.title(title)
    # Save using the given title instead of an undefined variable
    save_current_figure(title)

def causal_discovery_all_layers(cluster_df, layer_names):
    causal_graphs = {}
    try:
        data_np = cluster_df.drop(columns=["label"]).to_numpy()
        if CAUSAL_LEARN_AVAILABLE:
            pc_result = pc(data_np)
            causal_graphs["all_layers"] = pc_result.G
            visualize_causal_graph(pc_result.G, title="All Layers Causal Graph")
        else:
            print("Warning: causallearn is not installed. Causal discovery functions will be limited.")
            # Create a simple correlation-based graph as fallback
            X = data_np
            corr_matrix = np.corrcoef(X.T)
            G = nx.DiGraph()
            
            # Add nodes
            columns = [col for col in cluster_df.columns if col != "label"]
            for col in columns:
                G.add_node(col)
            
            # Helper function to extract layer number
            def get_layer_depth(col_name):
                # For column names like "conv1_cluster_1", extract the number after the layer name
                parts = col_name.split('_')
                if len(parts) >= 1:
                    layer_name = parts[0]  # e.g., "conv1"
                    # Extract the numeric part (if any)
                    numeric_part = ''.join(c for c in layer_name if c.isdigit())
                    if numeric_part:
                        return int(numeric_part)
                # Default to 0 if we can't extract a number
                return 0
            
            # Add edges based on correlation threshold
            for i, col1 in enumerate(columns):
                for j, col2 in enumerate(columns):
                    if i != j and abs(corr_matrix[i, j]) > 0.3:  # Correlation threshold
                        # Direction based on column names (layer depth)
                        layer1 = get_layer_depth(col1)
                        layer2 = get_layer_depth(col2)
                        if layer1 < layer2:
                            G.add_edge(col1, col2, weight=corr_matrix[i, j])
            
            causal_graphs["all_layers"] = G
            # Visualize the correlation-based graph
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", 
                   arrows=True, arrowsize=15, node_shape='o', font_size=8)
            plt.title("Correlation-based Graph (Fallback)")
            save_current_figure("Correlation-based Graph (Fallback)")
    except Exception as e:
        print(f"Error in causal discovery: {e}")
    return causal_graphs

def visualize_clusters(activations, clustering_results, layer_name, apply_pca=True,
                       pca_components=50, tsne_perplexity=30, tsne_n_iter=1000):
    data = activations[layer_name]
    if apply_pca and data.shape[1] > pca_components:
        pca = PCA(n_components=pca_components, random_state=42)
        data = pca.fit_transform(data)
    # Use max_iter instead of n_iter to avoid the FutureWarning
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, max_iter=tsne_n_iter, random_state=42)
    emb_2d = tsne.fit_transform(data)
    cluster_assign = clustering_results[layer_name]
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_assign, cmap="tab10", s=5)
    plt.title(f"{layer_name} Clusters (t-SNE)")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    save_current_figure(f"{layer_name} Clusters (t-SNE)")


# ---------------------------
# New Functions for SHAP and Feature Analysis

def explain_images_with_shap(model, dataloader, device, background_size=50):
    """
    Use SHAP to explain where the model is looking on two images.
    If the dataset returns a triplet (as in metric learning), only the first element (anchor) is used.
    """
    try:
        # Store original model state
        original_model_training = model.training
        model.eval()  # Make sure the model is in eval mode
        
        batch = next(iter(dataloader))
        # Use only anchor images if dataset returns a triplet
        if isinstance(batch, (tuple, list)) and len(batch) >= 3:
            images = batch[0]
        else:
            images, _ = batch

        print("DEBUG: Raw images shape:", images.shape)  # Should be (N, 3, 224, 224)
        images = images.to(device)
        background = images[:background_size]
        test_images = images[:2]
        print("DEBUG: Test images shape (before transpose):", test_images.shape)
        
        # For CNNs, try using GradientExplainer first as it's more likely to work
        # with complex CNN architectures with pooling layers
        try:
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(test_images)
        except Exception as e:
            print(f"GradientExplainer failed, falling back to DeepExplainer: {e}")
            # If GradientExplainer fails, try DeepExplainer
            try:
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(test_images)
            except Exception as e:
                print(f"Both SHAP explainers failed: {e}")
                # If both explainers fail, create a simple visualization placeholder
                plt.figure(figsize=(12, 6))
                for i in range(min(2, test_images.shape[0])):
                    plt.subplot(1, 2, i+1)
                    # Convert tensor to numpy
                    img = test_images[i].cpu().detach().numpy().transpose(1, 2, 0)
                    # Normalize for display
                    img = (img - img.min()) / (img.max() - img.min())
                    plt.imshow(img)
                    plt.title(f"Image {i+1} (SHAP analysis failed)")
                    plt.axis('off')
                plt.tight_layout()
                save_current_figure("SHAP Analysis Failed - Original Images")
                # Restore original model state
                if original_model_training:
                    model.train()
                return
        
        # Print type and shape info for shap_values
        if isinstance(shap_values, list):
            for i, sv in enumerate(shap_values):
                print(f"DEBUG: shap_values[{i}] shape before transpose: {sv.shape}")
        else:
            print("DEBUG: shap_values shape before transpose:", shap_values.shape)
        
        # Convert test_images from (N, 3, 224, 224) to (N, 224, 224, 3)
        test_images_np = test_images.cpu().numpy()
        test_images_np = np.transpose(test_images_np, (0, 2, 3, 1))
        print("DEBUG: Test images shape after transpose:", test_images_np.shape)
        
        # Fix shap values dimensions
        if isinstance(shap_values, list):
            new_shap_values = []
            for sv in shap_values:
                # Transpose each element from (N, 3, 224, 224) to (N, 224, 224, 3)
                if sv.ndim == 4:
                    sv_new = np.transpose(sv, (0, 2, 3, 1))
                    new_shap_values.append(sv_new)
                else:
                    new_shap_values.append(sv)
            shap_values = new_shap_values
        else:
            # If it's a single array, transpose from (N, 3, 224, 224) to (N, 224, 224, 3)
            if shap_values.ndim == 4:
                shap_values = np.transpose(shap_values, (0, 2, 3, 1))
        
        # Optionally, print value range for a sanity check
        if isinstance(shap_values, list) and len(shap_values) > 0:
            print("DEBUG: shap_values[0] min/max:", shap_values[0].min(), shap_values[0].max())
        elif not isinstance(shap_values, list):
            print("DEBUG: shap_values min/max:", shap_values.min(), shap_values.max())
        
        # Use SHAP's visualization function
        plt.figure(figsize=(12, 6))
        try:
            shap.image_plot(shap_values, test_images_np)
            save_current_figure("SHAP Explanation")
        except Exception as e:
            print(f"Error in SHAP visualization: {e}")
            # Fallback visualization if shap.image_plot fails
            plt.figure(figsize=(12, 6))
            if isinstance(shap_values, list):
                sv_display = np.abs(shap_values[0]).sum(axis=3)  # Sum across color channels
                for i in range(min(2, sv_display.shape[0])):
                    plt.subplot(1, 2, i+1)
                    plt.imshow(sv_display[i], cmap='hot')
                    plt.title(f"SHAP values for image {i+1}")
                    plt.axis('off')
            else:
                sv_display = np.abs(shap_values).sum(axis=3)  # Sum across color channels
                for i in range(min(2, sv_display.shape[0])):
                    plt.subplot(1, 2, i+1)
                    plt.imshow(sv_display[i], cmap='hot')
                    plt.title(f"SHAP values for image {i+1}")
                    plt.axis('off')
            plt.tight_layout()
            save_current_figure("SHAP Explanation (Fallback)")
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, f"SHAP explanation unavailable\n{str(e)}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        save_current_figure("SHAP Explanation Failed")
    
    # Restore original model state
    if original_model_training:
        model.train()

def visualize_learned_features(activations, layer_name, n_components=5):
    """
    Analyze the learned features from a given layer using PCA and ICA.
    Plots the first few principal components and independent components.
    """
    act_data = activations[layer_name]
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca_components = pca.fit_transform(act_data)
    # Plot PCA components
    plt.figure(figsize=(10, 4))
    for i in range(n_components):
        plt.plot(pca_components[:, i], label=f'PC {i+1}')
    plt.title(f"PCA Components for {layer_name}")
    plt.legend()
    save_current_figure(f"PCA Components for {layer_name}")
    
    # Apply ICA
    ica = FastICA(n_components=n_components, random_state=42)
    ica_components = ica.fit_transform(act_data)
    # Plot ICA components
    plt.figure(figsize=(10, 4))
    for i in range(n_components):
        plt.plot(ica_components[:, i], label=f'IC {i+1}')
    plt.title(f"ICA Components for {layer_name}")
    plt.legend()
    save_current_figure(f"ICA Components for {layer_name}")

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
        
        cluster_assignments = clustering_results[layer_name]
        num_clusters = np.max(cluster_assignments) + 1
        
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
                # Find the closest cluster centroid
                sample_act = batch_activations[j].flatten()
                
                # Store the input and its cluster
                all_inputs.append(inputs[j].cpu().numpy())
                
                # For simplicity, we'll assign a random cluster (this should be replaced with actual assignment)
                # In practice, you'd use the clustering model to predict the cluster
                if i*inputs.shape[0] + j < len(cluster_assignments):
                    all_cluster_ids.append(cluster_assignments[i*inputs.shape[0] + j])
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

def generate_gcfn_report(model, dataloader, clustering_results, activations, device, is_image=None):
    """
    Generate a complete GCFN report with all visualizations
    
    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        clustering_results: Dictionary of clustering results
        activations: Dictionary of activations
        device: Device to run model on
        is_image: Whether the data is image data. If None, will try to determine from the model
    """
    # Determine if we're dealing with image data if not specified
    if is_image is None:
        is_image = isinstance(model, SimpleCNN)
    
    # 1. First visualize the clustering of latent space for each layer
    for layer_name in activations.keys():
        visualize_clusters(activations, clustering_results, layer_name,
                        apply_pca=True, pca_components=50,
                        tsne_perplexity=30, tsne_n_iter=1000)
    
    # 2. Visualize the learned features for each layer
    for layer_name in activations.keys():
        visualize_learned_features(activations, layer_name, n_components=5)
    
    # 3. Generate SHAP explanations for clusters in each layer
    try:
        for layer_name in activations.keys():
            print(f"Generating explanations for clusters in layer {layer_name}...")
            try:
                cluster_explanations = explain_clusters_with_shap(
                    model, dataloader, clustering_results, activations, 
                    device, layer_name, is_image=is_image)
                
                if cluster_explanations:
                    # Get feature names for tabular data if available
                    feature_names = None
                    if not is_image and hasattr(dataloader.dataset, 'feature_cols'):
                        feature_names = dataloader.dataset.feature_cols
                    
                    visualize_cluster_explanations(
                        cluster_explanations, 
                        is_image=is_image,
                        feature_names=feature_names
                    )
            except Exception as e:
                print(f"Error generating SHAP explanations for layer {layer_name}: {e}")
                # Generate a fallback visualization for this layer
                if is_image:
                    plt.figure(figsize=(8, 6))
                    plt.text(0.5, 0.5, f"SHAP explanation unavailable for layer {layer_name}\nError: {str(e)[:100]}...", 
                            horizontalalignment='center', verticalalignment='center', fontsize=12)
                    plt.axis('off')
                    save_current_figure(f"SHAP explanation unavailable - {layer_name}")
    except Exception as e:
        print(f"Error in SHAP explanation step: {e}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "SHAP explanations unavailable\nSee console for details", 
                horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        save_current_figure("SHAP explanations unavailable")
    
    # 4. Generate causal graph visualization
    try:
        causal_graphs = causal_discovery_all_layers(
            create_latent_cluster_dataset(clustering_results, np.zeros(len(next(iter(clustering_results.values()))))), 
            layer_names=list(activations.keys()))
    except Exception as e:
        print(f"Error in causal discovery step: {e}")
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "Causal discovery unavailable\nSee console for details", 
                horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        save_current_figure("Causal discovery unavailable")
    
    # 5. Generate the HTML report
    write_html_page("gcfn_complete_report.html")
    print("GCFN report generation complete!")

# ---------------------------
# Main Execution

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For testing/example - Uncomment the dataset you want to test
    
    # 1. Image Dataset Example with SimpleCNN
    train_loader, test_loader = get_image_dataloader('IQ-OTHNCCD', batch_size=16)
    dataloader = test_loader  # For visualization and explanation
    
    model = SimpleCNN().to(device)
    
    # Check if a trained model exists, if not, train one
    model_path = 'lung_cancer_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        train_network(model, train_loader, test_loader, epochs=5, lr=0.001, device=device)
        torch.save(model.state_dict(), model_path)
        print(f"Trained model saved to {model_path}")
    
    # Extract activations, perform clustering, and generate report
    acts = extract_activations(model, dataloader, device)
    
    # Perform clustering on each layer
    clustering_res = {}
    for layer_name in acts.keys():
        print(f"Clustering layer: {layer_name}")
        labels, centroids = perform_clustering_optimized(acts[layer_name], n_clusters=10, 
                                                      apply_pca=True, pca_components=50)
        clustering_res[layer_name] = labels
    
    # Generate the complete GCFN report with all visualizations
    generate_gcfn_report(model, dataloader, clustering_res, acts, device)
    
    # Or generate individual visualizations:
    # Visualize clusters for a chosen layer (e.g., 'conv1')
    visualize_clusters(acts, clustering_res, layer_name='conv1',
                    apply_pca=True, pca_components=50,
                    tsne_perplexity=30, tsne_n_iter=1000)
    
    # Analyze learned features for a chosen layer (e.g., 'fc1') using PCA and ICA
    visualize_learned_features(acts, layer_name='fc1', n_components=5)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up your image dataset (expects folders like 'Bengin cases', 'Normal cases', 'Malignant cases')
image_dir = "Data/Image/IQ-OTHNCCD"
dataloader, dataset = get_image_dataloader(
    image_dir=image_dir,
    batch_size=16,
    shuffle=True,
    metric_learning=True  # For triplet training; set False for standard classification
)
num_classes = len(dataset.class_map)
print("Class Map:", dataset.class_map)

model_path = "lung_cancer_model.pth"
model = SimpleCNN(num_classes=num_classes).to(device)

if os.path.exists(model_path):
    print(f"Model file '{model_path}' found. Loading model...")
    # Optionally, you can set weights_only=True to limit unpickling risks.
    model.load_state_dict(torch.load(model_path, map_location=device))  # Option: add weights_only=True if using newer torch versions.
    model.eval()
    print("Model loaded successfully.")
else:
    print(f"No existing model found. Training a new model...")
    train_network(model, dataloader, device, num_epochs=5, lr=1e-3, use_metric_learning=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

# Extract activations from the trained (or loaded) model
acts, labels = extract_activations(model, dataloader, device, metric_learning=True)

# Perform clustering with PCA preprocessing
clustering_res = perform_clustering_optimized(acts, max_clusters=10, pca_components=50)
cluster_df = create_latent_cluster_dataset(clustering_res, labels)
print("Cluster assignments (first 5 samples):")
print(cluster_df.head())

# Run causal discovery on the clusters using the keys from 'acts'
causal_graphs = causal_discovery_all_layers(cluster_df, layer_names=list(acts.keys()))

# Visualize clusters for each hooked layer using keys from 'acts'
for layer in acts.keys():
    visualize_clusters(acts, clustering_res, layer,
                       apply_pca=True, pca_components=50,
                       tsne_perplexity=30, tsne_n_iter=1000)

# Use SHAP to explain two images from the dataset
explain_images_with_shap(model, dataloader, device, background_size=50)

# Analyze learned features for a chosen layer (e.g., 'fc1') using PCA and ICA
visualize_learned_features(acts, layer_name='fc1', n_components=5)

# Generate GCFN report
generate_gcfn_report(model, dataloader, clustering_res, acts, device)

write_html_page("gcfn.html")

















