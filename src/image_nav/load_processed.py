import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import msgpack_numpy
import copy
import os
import quaternion
from habitat_sim import ShortestPath
import pickle


class Loader:
    def __init__(self, args):
        self.args = args
        self.datasets = {}

    def build_dataset(self):
        print("Loading {} data...".format(self.args.processed_data_file))
        split_data = pickle.load(
            open(self.args.image_nav_dir + self.args.processed_data_file, "rb")
        )
        print("Creating mp3d dataset...")
        self.datasets["test"] = Mp3dDataset(split_data)


class Mp3dDataset(Dataset):
    def __init__(self, data_list):
        self.max_edges = 60
        self.feat_size = 512
        self.edge_feat_size = 16
        self.data_list = data_list

    def __getitem__(self, index):
        instance = self.data_list[index]
        return (
            instance.traj,
            instance.scan,
            instance.floor,
            instance.start,
            instance.end,
            instance.adj_matrix,
            instance.edge_attr,
            instance.length_shortest,
        )

    def __len__(self):
        return len(self.data_list)
