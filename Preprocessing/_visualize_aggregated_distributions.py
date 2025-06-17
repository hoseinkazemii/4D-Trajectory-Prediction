import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.cm import get_cmap

def _visualize_aggregated_distributions(arrays_list, trials_per_scenario=29, coord_key="XYZ", method="tsne", max_points_per_scenario=1000):
    print(f"Running {method.upper()} to reduce trajectory dimensionality and plot grouped scenario distributions...")

    total_trials = len(arrays_list)
    num_scenarios = total_trials // trials_per_scenario
    print(f"Detected {num_scenarios} scenarios (from {total_trials} trials, {trials_per_scenario} per scenario).")

    scenario_trajectories = []

    for i in range(num_scenarios):
        start = i * trials_per_scenario
        end = (i + 1) * trials_per_scenario
        scenario_trials = arrays_list[start:end]

        all_points = np.concatenate([trial[coord_key] for trial in scenario_trials], axis=0)

        if len(all_points) > max_points_per_scenario:
            idx = np.random.choice(len(all_points), max_points_per_scenario, replace=False)
            all_points = all_points[idx]

        scenario_trajectories.append(all_points)

    all_data = np.concatenate(scenario_trajectories, axis=0)

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca')
    else:
        raise ValueError(f"Unsupported method: {method}")

    reduced_all = reducer.fit_transform(all_data)

    start = 0
    reduced_by_scenario = []
    for points in scenario_trajectories:
        n = len(points)
        reduced_by_scenario.append(reduced_all[start:start + n])
        start += n

    cmap = get_cmap("tab10")
    plt.figure(figsize=(10, 8))
    for i, reduced_points in enumerate(reduced_by_scenario):
        color = cmap(i % 10)
        plt.scatter(reduced_points[:, 0], reduced_points[:, 1], label=f"Scenario {i}", s=10, alpha=0.6, color=color)

    plt.title(f"{method.upper()} of Scenario Trajectories")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{method.lower()}_scenario_distributions.png", dpi=300)
    plt.show()
