import torch
from torchtext.data import to_map_style_dataset #Convert iterable-style dataset to map-style dataset
from torchdata.datapipes.iter import IterableWrapper
from torchtext.datasets import WikiText2, WikiText103
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def get_data_iterator(ds_name, ds_type, data_dir):
	"""ds_name: WikiText2, WikiText103, 
	    data_dir: path to the data directory,
	    ds_type: train, valid, test"""
	
	if ds_name == 'WikiText2':
		data_iter = WikiText2(root=data_dir, split=(ds_type))
	elif ds_name == "WikiText103":
		data_iter =  WikiText103(root=data_dir, split=(ds_type))
	else:
		raise ValueError("Choose dataset from: WikiText2, WikiText103")
	data_iter = to_map_style_dataset(data_iter)
	return data_iter

ds_name = 'WikiText2'
ds_type = 'train'
data_dir = 'data/'
data_iter = get_data_iterator(ds_name, ds_type, data_dir)
print(list(data_iter)[3])