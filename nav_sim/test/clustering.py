import os
import random
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from multiprocessing import Pool
import math
import IPython as ipy

from sklearn.cluster import DBSCAN

def cluster(X, visualize):
    db = DBSCAN(eps=0.5, min_samples=100).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    cluster_centers = np.zeros((n_clusters_, 3))

    # print("Estimated number of clusters: %d" % n_clusters_)
    # print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        if(len(xy) > 0):
            cluster_centers[k, :] = np.mean(xy, axis=0)
            
        if visualize:
            # plt.plot(
            #     xy[:, 0],
            #     xy[:, 1],
            #     "o",
            #     markerfacecolor=tuple(col),
            #     markeredgecolor="k",
            #     markersize=14,
            # )
            ax.scatter(
                xy[:, 0], xy[:, 1], xy[:, 2],
                marker='o',
                color=col
            )

            xy = X[class_member_mask & ~core_samples_mask]
            ax.scatter(
                xy[:, 0], xy[:, 1], xy[:, 2],
                marker='^',
                color=col
            )
            # plt.show()
    if visualize:
        ax.set_aspect('equal')
        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.show()
    return cluster_centers