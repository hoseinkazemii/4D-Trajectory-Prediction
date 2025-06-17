import numpy as np

def _get_similar_scenarios(distance_matrix, target_index, top_k=3):
    distances = distance_matrix[target_index]
    sorted_indices = np.argsort(distances)
    similar_indices = [i for i in sorted_indices if i != target_index][:top_k]
    return similar_indices
