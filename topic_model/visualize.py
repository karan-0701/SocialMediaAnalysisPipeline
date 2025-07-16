import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

def plot_dendrogram(topic_embeddings):
    linked = linkage(topic_embeddings, method='ward', metric='euclidean')

    plt.figure(figsize=(10, 5))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title("Dendrogram of Topic Embeddings")
    plt.xlabel("Topic Embeddings")
    plt.ylabel("Distance")
    plt.show()
