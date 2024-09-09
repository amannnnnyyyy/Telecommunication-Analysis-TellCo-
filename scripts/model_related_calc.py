import numpy as np

def compute_distance(df, centroid,features):
    return np.sqrt(((df[features] - centroid) ** 2).sum(axis=1))