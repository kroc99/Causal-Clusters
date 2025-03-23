import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from CC import (
    LungCancerDataset, InsuranceDataset, AgingClockDataset, 
    SimpleCNN, SimpleMLP, 
    train_network, extract_activations, perform_clustering_optimized, 
    generate_gcfn_report, get_image_dataloader, get_tabular_dataloader,
    create_latent_cluster_dataset, causal_discovery_all_layers,
    visualize_causal_graphs, visualize_clusters, visualize_learned_features,
    write_html_page
)

def main():
    parser = argparse.ArgumentParser(description="Causal Clusters: DNN Explainability through Causal Graphs")
    parser.add_argument('--dataset', type=str, default='lung_cancer', 
                        choices=['lung_cancer', 'insurance', 'aging_clock'],
                        help='Dataset to use (lung_cancer, insurance, or aging_clock)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_clusters', type=int, default=10, help='Number of clusters per layer')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Step 1: Load dataset and create model
    print(f"Loading {args.dataset} dataset...")
    is_image_data = args.dataset == 'lung_cancer'
    is_classification = True  # Default for lung_cancer and insurance
    
    if args.dataset == 'lung_cancer':
        # Image dataset
        train_loader, test_loader = get_image_dataloader('IQ-OTHNCCD', args.batch_size)
        # Use a smaller subset for explanation to improve performance 
        # and avoid SHAP errors with large image datasets
        small_batch_size = min(8, args.batch_size)
        small_test_loader, _ = get_image_dataloader('IQ-OTHNCCD', small_batch_size, shuffle=True)
        dataloader = small_test_loader  # Use smaller test set for explanation
        
        # Create model - SimpleCNN with 3 classes for lung cancer
        model = SimpleCNN(num_classes=3).to(device)
        input_shape = (3, 224, 224)  # RGB images
        
    elif args.dataset == 'insurance':
        # Tabular dataset
        train_loader, test_loader = get_tabular_dataloader('insurance', args.batch_size)
        dataloader = test_loader
        
        # Get feature dimensions from dataset
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[1]
        
        # Create model
        model = SimpleMLP(input_size=input_size, hidden_sizes=[64, 32], output_size=1, 
                          classification=is_classification).to(device)
        input_shape = (input_size,)
        
    elif args.dataset == 'aging_clock':
        # Tabular dataset - aging clock
        train_loader, test_loader = get_tabular_dataloader('aging_clock', args.batch_size)
        dataloader = test_loader
        
        # Get feature dimensions from dataset
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[1]
        
        # Set classification mode based on dataset format
        try:
            dataset = AgingClockDataset()
            is_classification = 'AgeRange' in dataset.data.columns
        except:
            is_classification = False
        
        # Create model
        output_size = 5 if is_classification else 1  # 5 age categories or regression
        model = SimpleMLP(input_size=input_size, hidden_sizes=[64, 32], output_size=output_size, 
                          classification=is_classification).to(device)
        input_shape = (input_size,)
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Step 2: Train or load model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pre-trained model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"Training model on {args.dataset} for {args.epochs} epochs...")
        train_network(model, train_loader, test_loader, epochs=args.epochs, 
                      lr=args.lr, device=device)
        
        if args.save_model:
            os.makedirs('models', exist_ok=True)
            model_filename = f"{args.dataset}_model.pth"
            model_path = os.path.join('models', model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    # Step 3: Extract activations
    print("Extracting activations from model layers...")
    activations = extract_activations(model, dataloader, device)
    
    # Step 4: Perform clustering on activations
    print("Performing clustering on layer activations...")
    clustering_results = {}
    
    for layer_name in activations.keys():
        print(f"  Clustering layer: {layer_name}")
        labels, centroids = perform_clustering_optimized(
            activations[layer_name], 
            max_clusters=args.num_clusters,
            pca_components=min(50, activations[layer_name].shape[1] - 1)
        )
        clustering_results[layer_name] = labels
    
    # Step 5: Generate GCFN Report with all visualizations
    print("Generating full GCFN report with all visualizations...")
    generate_gcfn_report(
        model, 
        dataloader, 
        clustering_results, 
        activations, 
        device,
        is_image=is_image_data
    )
    
    print("Analysis complete! Check the gcfn_complete_report.html for results.")

if __name__ == "__main__":
    main() 