import torch
from torch import nn
from torch.nn import functional as F
try:
    import faiss
    use_faiss = True
except:
    print("couldn't load faiss")
    use_faiss = False
import numpy as np


class KeyValueStore:
    def __init__(self, keys, values, use_faiss=use_faiss):
        if use_faiss:
            self.search = self._search_faiss
            self.values_index = faiss.IndexFlatL2(values)
        else:
            self.search = self._search_simple
            self.values = values
        self.keys = keys

    def _search_faiss(self, value, k):
        _, indices = self.values_index.search(value, k)
        return list(map(self.keys.__getitem__, indices))

    def _search_simple(self, value, k):
        scores = []
        value = np.array(value).reshape(1, -1)
        # assumes normalized
        for x in self.values:
            x = np.array(x).reshape(-1, 1)
            scores.append((value @ x)[0][0])
        return list(map(self.keys.__getitem__, np.argsort(scores)[-k:]))
