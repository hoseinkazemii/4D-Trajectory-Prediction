import matplotlib.pyplot as plt


# Plot the results
def _plot_3d_trajectory(y_test, y_pred, **params):
    verbose = params.get("verbose")
    sample_index = params.get("sample_index")
    if verbose:
        print("plotting the results...")


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(y_test[sample_index, :, 0], y_test[sample_index, :, 1], y_test[sample_index, :, 2], label='True Trajectory')
    ax.plot(y_pred[sample_index, :, 0], y_pred[sample_index, :, 1], y_pred[sample_index, :, 2], label='Predicted Trajectory')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('True vs Predicted 3D Trajectory')
    ax.legend()
    plt.show()