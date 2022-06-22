# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class ConstrainedDBSCAN:

    def __init__(self, eps=0.6, min_samples=4):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, must_link=None, cannot_link=None):

        neighbors_model = NearestNeighbors(radius=self.eps, metric='precomputed', n_jobs=-1)
        neighbors_model.fit(X)
        neighborhoods = neighbors_model.radius_neighbors(X, return_distance=False)

        labels = np.full(X.shape[0], -1, dtype=np.intp)

        cluster_id = 0
        for point in tqdm(range(X.shape[0])):
            if labels[point] == -1:
                if self.expand_cluster(labels, cluster_id, point, neighborhoods,
                                       must_link, cannot_link):
                    cluster_id += 1
        return labels

    def expand_cluster(self, labels, cluster_id, point, neighborhoods,
                       must_link, cannot_link):

        seeds = []
        neighb = neighborhoods[point]

        if len(neighb) < self.min_samples:
            return False

        if must_link[point]:
            for p in must_link[point]:
                if labels[p] == -1:
                    labels[p] = cluster_id
                    seeds.append(p)
        else:
            labels[point] = cluster_id
            seeds.append(point)

        while seeds:
            seed = seeds[0]
            if must_link[seed]:
                for p in must_link[seed]:
                    if labels[p] == -1:
                        labels[p] = cluster_id
                        seeds.append(p)
            neighb = neighborhoods[seed]
            if len(neighb) >= self.min_samples:
                for p in neighb:
                    if labels[p] == -1 and self._check_cannot_link_constraints(cannot_link, p, labels, cluster_id):
                        labels[p] = cluster_id
                        seeds.append(p)
            del seeds[0]
        return True

    def _check_cannot_link_constraints(self, cannot_link, point, labels, cluster_id):
        indices = np.where(labels == cluster_id)[0]
        cl = cannot_link[point]
        for i in indices:
            if i in cl:
                return False
        return True
