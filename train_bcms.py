import numpy as np
from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import logging
from numpy.linalg import norm

R = 6371

def evaluate(c1, c2, scale_km=True):
        d = np.radians(c2-c1)
        a = np.sin(d[:,0]/2) * np.sin(d[:,0]/2) + np.cos(np.radians(c1[:,0])) * np.cos(np.radians(c2[:,0])) * np.sin(d[:,1]/2) * np.sin(d[:,1]/2)
        d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        if scale_km:
                return R * d
        else:
                return d

def mean_dist(a, b):
        global scl
        a_tr = scl.inverse_transform(a)
        b_tr = scl.inverse_transform(b)
        d = evaluate(a_tr, b_tr)
        return np.mean(d)

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
pickle.dump(scl,open('bcms.scaler','wb'))
dev_y=scl.transform([e[1] for e in dev_data])

train_df = pd.DataFrame(zip([e[0] for e in train_data],train_y))
train_df.columns = ["text", "labels"]

dev_df = pd.DataFrame(zip([e[0] for e in dev_data],dev_y))
dev_df.columns = ["text", "labels"]

model_args = {
    "regression": True,
    "num_train_epochs": 10,
    "overwrite_output_dir": True,
    "best_model_dir": "bcms_output/best_model",
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 0,
    "evaluate_during_training_verbose": True,
    "early_stopping_metric": "median_dist",
    "early_stopping_metric_minimize": True,
    "output_dir": 'bcms_output',
    "do_lower_case": True,
    "save_steps": 0,
    "learning_rate": 5e-4,
    "train_batch_size": 128,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
}

model = MultiLabelClassificationModel(
    "electra",
    "CLASSLA/bcms-bertic",
    num_labels=2,
    loss_fct="MAELoss",
    args=model_args,
)

model.train_model(train_df, eval_df = dev_df, median_dist = median_dist)
