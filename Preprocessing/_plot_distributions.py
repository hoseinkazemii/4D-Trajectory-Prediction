import matplotlib.pyplot as plt


# Visualize the original and scaled data
def _plot_distributions(original, scaled):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Histogram for original data
    axes[0].hist(original, bins=30, alpha=0.7, color='blue', label='Original')
    axes[0].set_title('Histogram of Original Data')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    
    # Histogram for scaled data
    axes[1].hist(scaled, bins=30, alpha=0.7, color='green', label='Scaled')
    axes[1].set_title('Histogram of Scaled Data')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    # Box plot for original data
    axes[2].boxplot(original, vert=False, patch_artist=True, boxprops=dict(facecolor='blue', color='blue'))
    axes[2].set_title('Box Plot of Original Data')
    axes[2].set_xlabel('Value')
    
    # Box plot for scaled data
    axes[3].boxplot(scaled, vert=False, patch_artist=True, boxprops=dict(facecolor='green', color='green'))
    axes[3].set_title('Box Plot of Scaled Data')
    axes[3].set_xlabel('Value')
    
    plt.tight_layout()
    plt.show()