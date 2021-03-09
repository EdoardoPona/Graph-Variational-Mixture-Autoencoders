import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def load_regions(WORKING_PATH, YEAR, one_hot=False):
    region_data = np.load(WORKING_PATH+str(YEAR)+'-'+str(YEAR+1)+'_region.npy', allow_pickle=True)
    region_data = ["unknown" if region is np.nan else region for region in region_data]
    region_data = LabelEncoder().fit_transform(region_data)
    if one_hot:
        return OneHotEncoder(sparse=False).fit_transform(np.array(region_data).reshape((-1, 1)))
    return region_data


def load_graph(WORKING_PATH, YEAR): 
    loader = np.load(WORKING_PATH+"weighted_procurement_"+str(YEAR)+"-"+str(YEAR+1)+".npz")
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

