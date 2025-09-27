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
    get_image_dataloader, get_tabular_dataloader,
    create_latent_cluster_dataset, causal_discovery_all_layers,
    visualize_causal_graph, visualize_clusters, visualize_learned_features,
    write_html_page, create_integrated_gcfn,
    main_flexible_clustering, main_test_clustering, main_integrated_gcfn
)

def main():
    parser = argparse.ArgumentParser(description="Causal Clusters: DNN Explainability through Causal Graphs")
    parser.add_argument('--dataset', type=str, default='lung_cancer', 
                        choices=['lung_cancer', 'insurance', 'aging_clock'],
                        help='Dataset to use (lung_cancer, insurance, or aging_clock)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_clusters', type=int, default=8, help='Maximum number of clusters per layer')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--integrated', action='store_true', help='Generate the integrated GCFN visualization')
    parser.add_argument('--flexible', action='store_true', help='Run flexible clustering demo')
    parser.add_argument('--test', action='store_true', help='Run test clustering with optimal cluster detection')
    parser.add_argument('--integrated-only', action='store_true', help='Run only integrated GCFN generation')
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.flexible:
        print("Running flexible clustering demo...")
        main_flexible_clustering()
        return
    
    if args.test:
        print("Running test clustering with optimal cluster detection...")
        main_test_clustering()
        return
    
    if args.integrated_only:
        print("Running integrated GCFN generation only...")
        main_integrated_gcfn()
        return
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Step 1: Load dataset and create model
    print(f"Loading {args.dataset} dataset...")
    is_image_data = args.dataset == 'lung_cancer'
    is_classification = True  # Default for lung_cancer and insurance
    
    if args.dataset == 'lung_cancer':
        # Image dataset
        train_loader, train_dataset = get_image_dataloader('IQ-OTHNCCD', args.batch_size)
        # Use a smaller subset for explanation to improve performance 
        # and avoid SHAP errors with large image datasets
        small_batch_size = min(8, args.batch_size)
        dataloader, _ = get_image_dataloader('IQ-OTHNCCD', small_batch_size, shuffle=True)
        
        # Create model - SimpleCNN with 3 classes for lung cancer
        model = SimpleCNN(num_classes=3).to(device)
        input_shape = (3, 224, 224)  # RGB images
        
    elif args.dataset == 'insurance':
        # Tabular dataset
        train_loader, train_dataset = get_tabular_dataloader('insurance', args.batch_size)
        dataloader, _ = get_tabular_dataloader('insurance', args.batch_size)
        
        # Get feature dimensions from dataset
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[1]
        
        # Create model
        model = SimpleMLP(input_dim=input_size, hidden_dim=64, output_dim=1 if not is_classification else 3).to(device)
        input_shape = (input_size,)
        
    elif args.dataset == 'aging_clock':
        # Tabular dataset - aging clock
        train_loader, train_dataset = get_tabular_dataloader('aging_clock', args.batch_size)
        dataloader, _ = get_tabular_dataloader('aging_clock', args.batch_size)
        
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
        model = SimpleMLP(input_dim=input_size, hidden_dim=64, output_dim=output_size).to(device)
        input_shape = (input_size,)
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Step 2: Train or load model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pre-trained model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"Training model on {args.dataset} for {args.epochs} epochs...")
        train_network(model, train_loader, device, num_epochs=args.epochs, lr=args.lr, use_metric_learning=is_image_data)
        
        if args.save_model:
            os.makedirs('models', exist_ok=True)
            model_filename = f"{args.dataset}_model.pth"
            model_path = os.path.join('models', model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    # Step 3: Extract activations
    print("Extracting activations from model layers...")
    if is_image_data:
        activations, labels = extract_activations(model, dataloader, device, metric_learning=True)
    else:
        activations, labels = extract_activations(model, dataloader, device, metric_learning=False)
    
    # Step 4: Perform clustering on activations with optimal cluster count
    print("Performing clustering on layer activations with optimal cluster count...")
    try:
        # Calculate maximum possible PCA components based on activation dimensions
        max_pca_components = min(50, min(layer.shape[1] for layer in activations.values()) - 1)
        max_pca_components = max(2, max_pca_components)  # Ensure at least 2 components
        
        # Perform clustering with dynamic optimal cluster counts
        clustering_results = perform_clustering_optimized(
            activations, 
            max_clusters=args.max_clusters, 
            pca_components=max_pca_components
        )
        
        # Create latent cluster dataset
        cluster_df = create_latent_cluster_dataset(clustering_results, labels)
        print("Cluster assignments (head):")
        print(cluster_df.head())
        
        # Step 5: Generate GCFN Report with all visualizations
        print("Generating full GCFN report with all visualizations...")
        
        # Visualize clusters for each layer
        for layer_name in activations.keys():
            try:
                # Adjust perplexity based on sample size to avoid warnings
                perplexity = min(30, activations[layer_name].shape[0] // 5)
                perplexity = max(5, perplexity)  # Ensure perplexity is at least 5
                
                # Adjust PCA components based on data dimensionality
                pca_components = min(50, activations[layer_name].shape[1] - 1)
                pca_components = max(2, pca_components)  # Ensure at least 2 components
                
                visualize_clusters(
                    activations, 
                    clustering_results, 
                    layer_name,
                    apply_pca=True, 
                    pca_components=pca_components,
                    tsne_perplexity=perplexity
                )
            except Exception as e:
                print(f"Error visualizing clusters for layer {layer_name}: {e}")
        
        # Visualize learned features for each layer
        for layer_name in activations.keys():
            try:
                visualize_learned_features(activations, layer_name, n_components=3)
            except Exception as e:
                print(f"Error visualizing learned features for layer {layer_name}: {e}")
        
        # Run causal discovery on the clusters
        try:
            causal_graphs = causal_discovery_all_layers(
                cluster_df, 
                layer_names=list(activations.keys())
            )
        except Exception as e:
            print(f"Error in causal discovery: {e}")
        
        # Generate HTML report
        write_html_page("gcfn_report.html")
        
        # Generate integrated GCFN if requested
        if args.integrated:
            print("\nGenerating integrated GCFN visualization...")
            create_integrated_gcfn(
                model, 
                dataloader, 
                clustering_results, 
                activations, 
                device, 
                list(activations.keys()),
                is_image=is_image_data,
                output_html="integrated_gcfn.html"
            )
        
        print("Analysis complete! Check the gcfn_report.html for results.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 