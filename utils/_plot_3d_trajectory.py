import matplotlib.pyplot as plt

def _plot_3d_trajectory(y_test, y_pred, **params):
    verbose = params.get("verbose")
    sample_index = params.get("sample_index")
    coordinate = params.get("coordinate")
    if verbose:
        print("plotting the results...")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # Adjust labels and title based on the coordinate
    if coordinate == 'X':
        ax.set_ylabel('X')
    elif coordinate == 'Y':
        ax.set_ylabel('Y')
    elif coordinate == 'Z':
        ax.set_ylabel('Z')

    ax.plot(y_test[sample_index], label=f'True {coordinate} Trajectory')
    ax.plot(y_pred[sample_index], label=f'Predicted {coordinate} Trajectory')
    
    ax.set_xlabel('Time Step')
    ax.set_title(f'True vs Predicted {coordinate} Trajectory')
    ax.legend()
    plt.savefig(f"test_{coordinate}.png", dpi=300)
    plt.show()