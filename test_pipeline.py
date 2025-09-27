#!/usr/bin/env python3
"""
Test script to verify the fixed pipeline works end-to-end
"""

import torch
import os
import sys

# Add current directory to path
sys.path.append('.')

from CC import (
    get_image_dataloader, SimpleCNN, extract_activations,
    perform_clustering_optimized, create_latent_cluster_dataset,
    causal_discovery_all_layers, visualize_causal_graph,
    compute_statistical_transition_graph,
    write_html_page
)

def test_pipeline():
    """Test the basic pipeline with a small dataset"""
    print("Testing Causal Clusters Pipeline...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model exists
    model_path = "lung_cancer_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        print("Please train a model first using: python main.py --dataset lung_cancer --epochs 5")
        return False
    
    try:
        # Step 1: Load dataset
        print("Loading dataset...")
        dataloader, dataset = get_image_dataloader(
            image_dir='IQ-OTHNCCD',
            batch_size=4,  # Very small batch for testing
            shuffle=True,
            metric_learning=True
        )
        print(f"Dataset loaded: {len(dataset)} samples")
        
        # Step 2: Load model
        print("Loading model...")
        model = SimpleCNN(num_classes=3).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully")
        
        # Step 3: Extract activations
        print("Extracting activations...")
        activations, labels = extract_activations(model, dataloader, device, metric_learning=True)
        print(f"Extracted activations for {len(activations)} layers")
        for layer_name, layer_acts in activations.items():
            print(f"  {layer_name}: shape {layer_acts.shape}")
        
        # Step 4: Perform clustering
        print("Performing clustering...")
        clustering_results = perform_clustering_optimized(
            activations, 
            max_clusters=4,  # Small number for testing
            pca_components=min(10, min(layer.shape[1] for layer in activations.values()) - 1)
        )
        print(f"Clustering completed for {len(clustering_results)} layers")
        
        # Step 5: Create latent dataset
        print("Creating latent cluster dataset...")
        latent_df = create_latent_cluster_dataset(clustering_results, labels)
        print(f"Latent dataset shape: {latent_df.shape}")
        print("Columns:", list(latent_df.columns))
        
        # Step 6: Statistical transitions and causal discovery
        print("Computing statistical transition graph...")
        stat_graph = compute_statistical_transition_graph(latent_df)
        visualize_causal_graph(stat_graph, "Statistical Transition Graph (Pipeline Test)")
        print("Performing causal discovery...")
        causal_graph = causal_discovery_all_layers(latent_df)
        if causal_graph is not None:
            print(f"Causal graph has {len(causal_graph.nodes())} nodes and {len(causal_graph.edges())} edges")
        else:
            print("Causal graph is None (this might be expected for small datasets)")
        
        # Step 7: Generate report
        print("Generating HTML report...")
        write_html_page("test_pipeline_report.html")
        
        print("\n✅ Pipeline test completed successfully!")
        print("Check 'test_pipeline_report.html' for results.")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
