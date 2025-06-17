import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns

def scenario_distance_matrix(arrays_list, scenario_to_trial_indices):
    scenario_aggregates = []
    for i in range(len(scenario_to_trial_indices)):
        trial_indices = scenario_to_trial_indices[i]
        all_xyz = np.concatenate([arrays_list[j]['XYZ'] for j in trial_indices], axis=0)
        scenario_aggregates.append(all_xyz)

    num_scenarios = len(scenario_aggregates)
    distance_matrix = np.zeros((num_scenarios, num_scenarios))

    for i in range(num_scenarios):
        for j in range(i, num_scenarios):
            dist_x = wasserstein_distance(scenario_aggregates[i][:, 0], scenario_aggregates[j][:, 0])
            dist_y = wasserstein_distance(scenario_aggregates[i][:, 1], scenario_aggregates[j][:, 1])
            dist_z = wasserstein_distance(scenario_aggregates[i][:, 2], scenario_aggregates[j][:, 2])
            avg_dist = np.mean([dist_x, dist_y, dist_z])

            distance_matrix[i, j] = avg_dist
            distance_matrix[j, i] = avg_dist

    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=[f"Scen {i}" for i in range(num_scenarios)],
                yticklabels=[f"Scen {i}" for i in range(num_scenarios)])
    plt.title("Scenario Distance Matrix (Wasserstein)")
    plt.tight_layout()
    plt.savefig("scenario_distance_matrix.png", dpi=300)
    plt.show()
    
    np.save("scenario_distance_matrix.npy", distance_matrix)

    return distance_matrix
