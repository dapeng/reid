import torch
import numpy as np


class ActiveLearning:
    def __init__(self, num_instance, budget):
        self.num_instance = num_instance
        self.budget = budget * self.num_instance
        self.ml_graph = dict()
        self.cl_graph = dict()
        for i in range(self.num_instance):
            self.ml_graph[i] = set()
            self.cl_graph[i] = set()

    @staticmethod
    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    def add_ml(self, i, j):
        self.add_both(self.ml_graph, i, j)
        visited = [False] * self.num_instance

        if self.ml_graph[i]:
            component = []
            self.dfs(i, self.ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        self.ml_graph[x1].add(x2)
                        for x3 in self.cl_graph[x2]:
                            self.add_both(self.cl_graph, x1, x3)

    def add_cl(self, i, j):
        self.add_both(self.cl_graph, i, j)
        for x in self.ml_graph[i]:
            self.add_both(self.cl_graph, x, j)
        for y in self.ml_graph[j]:
            self.add_both(self.cl_graph, i, y)
        for x in self.ml_graph[i]:
            for y in self.ml_graph[j]:
                self.add_both(self.cl_graph, x, y)

    def dfs(self, i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                self.dfs(j, graph, visited, component)
        component.append(i)

    def get_mask(self, mask_type=0):
        if mask_type == 0:
            mask = torch.zeros(self.num_instance, self.num_instance, dtype=torch.int)
        else:
            mask = np.zeros((self.num_instance, self.num_instance), dtype=np.intp)
        for i in range(self.num_instance):
            mask[i, list(self.ml_graph[i])] = 1
            mask[i, list(self.cl_graph[i])] = -1
        return mask

    def dus(self, pseudo_labels, gt_labels, dist_mat):
        """
        Dual Uncertainty Selection
        """
        if self.budget == 0:
            print('==> Budget has run out')
            return

        cluster_ids = set(pseudo_labels)
        if -1 in cluster_ids:
            cluster_ids.remove(-1)
        for id in cluster_ids:
            sp = []
            intra_idx = np.where(pseudo_labels == id)[0]
            inter_idx = np.where(pseudo_labels != id)[0]

            # uncertain postive pairs
            intra_dist_mat = dist_mat[np.ix_(intra_idx, intra_idx)]
            pos = np.unravel_index(intra_dist_mat.argmax(), intra_dist_mat.shape)
            sp.append((intra_idx[pos[0]], intra_idx[pos[1]]))

            # uncertain negative pairs
            inter_dist_mat = dist_mat[np.ix_(intra_idx, inter_idx)]
            neg = np.unravel_index(inter_dist_mat.argmin(), inter_dist_mat.shape)
            sp.append((intra_idx[neg[0]], inter_idx[neg[1]]))

            # TODO: check budget
            for (u, v) in sp:
                if u == v or u in self.ml_graph[v] or u in self.cl_graph[v]:
                    continue
                if gt_labels[u] == gt_labels[v]:
                    self.add_ml(u, v)
                else:
                    self.add_cl(u, v)
                self.budget -= 1

