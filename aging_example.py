import torch
import os
import matplotlib.pyplot as plt

from CC import (
    AgingClockDataset, SimpleMLP, 
    train_network, extract_activations, perform_clustering_optimized, 
    generate_gcfn_report, get_tabular_dataloader,
    visualize_clusters, visualize_learned_features,
    write_html_page
)

def main():
    print("Causal Clusters Example - Aging Clock Dataset")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Load aging clock dataset
    print("Loading aging clock dataset...")
    train_loader, test_loader = get_tabular_dataloader('aging_clock', batch_size=32)
    dataloader = test_loader  # For explanation to avoid data leakage
    
    # Get dataset properties to determine model configuration
    dataset = AgingClockDataset()
    feature_names = dataset.feature_columns
    is_classification = 'AgeRange' in dataset.data.columns
    print(f"Dataset loaded: {len(dataset)} samples, {len(feature_names)} features")
    print(f"Task type: {'Classification' if is_classification else 'Regression'}")
    
    # Step 2: Create and train model
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[1]
    output_size = 5 if is_classification else 1  # 5 age categories or regression
    
    model = SimpleMLP(
        input_size=input_size, 
        hidden_sizes=[64, 32], 
        output_size=output_size, 
        classification=is_classification
    ).to(device)
    
    # Check if a trained model exists
    model_path = 'aging_clock_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("Training aging clock model...")
        train_network(model, train_loader, test_loader, epochs=20, lr=0.001, device=device)
        torch.save(model.state_dict(), model_path)
        print(f"Trained model saved to {model_path}")
    
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
            max_clusters=10,
            pca_components=min(50, activations[layer_name].shape[1] - 1)
        )
        clustering_results[layer_name] = labels
    
    # Step 5: Generate complete GCFN report
    print("Generating full GCFN report with all visualizations...")
    generate_gcfn_report(
        model, 
        dataloader, 
        clustering_results, 
        activations, 
        device,
        is_image=False
    )
    
    print("Analysis complete! Check the gcfn_complete_report.html for results.")

if __name__ == "__main__":
    main() 