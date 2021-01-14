import numpy as np
from scipy.stats import pearsonr
from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import logging
import sys
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
import torch
torch.manual_seed(int(sys.argv[1]))

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

R = 6371

def evaluate(c1, c2, scale_km=True):
        d = np.radians(c2-c1)
        a = np.sin(d[:,0]/2) * np.sin(d[:,0]/2) + np.cos(np.radians(c1[:,0])) * np.cos(np.radians(c2[:,0])) * np.sin(d[:,1]/2) * np.sin(d[:,1]/2)
        d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        if scale_km:
                return R * d
        else:
                return d

def median_dist(a, b):
        global scl
        a_tr = scl.inverse_transform(a)
        b_tr = scl.inverse_transform(b)
        d = evaluate(a_tr, b_tr)
        return np.median(d)

class GlobalScaler():
	def __init__(self):
		self.means = None
		self.stddev = None

	def fit_transform(self, data):
		self.means = np.mean(data, axis=0)
		centereddata = data - self.means
		self.stddev = np.std(centereddata)
		return centereddata / self.stddev

	def transform(self, data):
		return (data - self.means) / self.stddev

	def inverse_transform(self, data):
		return (data * self.stddev) + self.means

def load_data(path, size=-1):
	data=[]
	for line in open(path):
		x, y, text = line.strip().split('\t')
		x, y = float(x),float(y)
		data.append((text,(x,y)))
		if len(data) == size:
			break
	return data

# Preparing train data

train_data = load_data('bcms/train.txt')
dev_data = load_data('bcms/dev.txt')
scl=GlobalScaler()
train_y=scl.fit_transform([e[1] for e in train_data])
dev_y=scl.transform([e[1] for e in dev_data])

train_df = pd.DataFrame(zip([e[0] for e in train_data],train_y))
train_df.columns = ["text", "labels"]
print(train_df)

dev_df = pd.DataFrame(zip([e[0] for e in dev_data],dev_y))
dev_df.columns = ["text", "labels"]

# Setting optional model configuration
model_args = {
    "regression": True,
    "num_train_epochs": 3,
    "overwrite_output_dir": True,
    "best_model_dir": "bcms_csebert_output_"+sys.argv[1]+"/best_model",
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 0,
    "evaluate_during_training_verbose": True,
    #"early_stopping_metric": "median_dist",
    #"early_stopping_metric_minimize": True,
    "output_dir": 'bcms_output_'+sys.argv[1],
    "do_lower_case": True,
    "save_steps": 0,
    "train_batch_size": 64,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    #"random_init": False
}

# Create a ClassificationModel
model = MultiLabelClassificationModel(
    "bert",
    "EMBEDDIA/crosloengual-bert",
    num_labels=2,
    loss_fct="MSELoss",
    args=model_args,
)

# Train the model
model.train_model(train_df, eval_df = dev_df, median_dist = median_dist)
