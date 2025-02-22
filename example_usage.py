import pandas as pd
import matplotlib.pyplot as plt
from vector_field_processor import VectorFieldProcessor

def plot_results(processor):
    """Plot the vector field, attractors, and basins."""
    plt.figure(figsize=(12, 4))
    
    # Plot vector field
    plt.subplot(131)
    plt.quiver(
        range(processor.grid_size),
        range(processor.grid_size),
        processor.smoothed_field['mu_dx'],
        processor.smoothed_field['mu_dy']
    )
    plt.title('Smoothed Vector Field')
    
    # Plot attractors and basins
    plt.subplot(132)
    plt.imshow(processor.basins, cmap='viridis')
    plt.scatter(
        processor.attractors[:, 1],
        processor.attractors[:, 0],
        c='red',
        s=100,
        label='Attractors'
    )
    plt.colorbar(label='Basin Label')
    plt.title('Basins and Attractors')
    
    # Plot some streamlines
    plt.subplot(133)
    for trajectory in processor.streamlines[:100]:  # Plot first 100 for clarity
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 1], trajectory[:, 0], 'b-', alpha=0.1)
    plt.scatter(
        processor.attractors[:, 1],
        processor.attractors[:, 0],
        c='red',
        s=100,
        label='Attractors'
    )
    plt.title('Streamlines')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load your trajectory data
    df = pd.read_csv("./local_data/trace_w_attractors-final-we-rev3.csv")
    
    # Initialize processor
    processor = VectorFieldProcessor(grid_size=100, smoothing_radius=1)
    
    # Run the complete pipeline
    processor.create_vector_field(df)
    processor.apply_bayesian_smoothing()
    processor.identify_attractors(step_size=0.1, eps=0.5, min_samples=5)
    processor.clean_basins_gpu()
    
    # Get features for predictive modeling
    features = processor.get_features()
    print("\nExtracted Features:")
    print(f"Number of attractors: {features['n_attractors']}")
    print(f"Basin sizes: {features['basin_sizes']}")
    print(f"Mean vector magnitude: {features['vector_field_stats']['mean_magnitude']:.3f}")
    print(f"Mean vector variance: {features['vector_field_stats']['variance']:.3f}")
    
    # Visualize results
    plot_results(processor)

if __name__ == "__main__":
    main()
