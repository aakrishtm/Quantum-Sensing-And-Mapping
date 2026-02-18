import matplotlib.pyplot as plt


def visualize_prediction(density_grid, ground_truth, prediction_prob, prediction_binary,
                         save_path=None, show=True, verbose=True):
    """Visualize input, ground truth, and prediction"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Input density grid
    im0 = axes[0, 0].imshow(density_grid, cmap='inferno', origin='upper')
    axes[0, 0].set_title('Input Density Grid')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], label='Density')

    # Ground truth
    im1 = axes[0, 1].imshow(ground_truth, cmap='binary', origin='upper', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth Tunnel Mask')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])

    # Prediction probability
    im2 = axes[1, 0].imshow(prediction_prob, cmap='hot', origin='upper', vmin=0, vmax=1)
    axes[1, 0].set_title('Prediction Probability')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], label='Probability')

    # Prediction binary
    im3 = axes[1, 1].imshow(prediction_binary, cmap='binary', origin='upper', vmin=0, vmax=1)
    axes[1, 1].set_title('Prediction Binary (threshold=0.5)')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
