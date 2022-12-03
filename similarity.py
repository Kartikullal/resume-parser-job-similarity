import numpy as np

#This is a function to calculate cosine similarity between 2 vectors
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))