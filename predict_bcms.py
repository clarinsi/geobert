from simpletransformers.classification import MultiLabelClassificationModel
import numpy as np
import pickle
import reverse_geocode

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
		return (np.asarray(data) * self.stddev) + self.means

scl=pickle.load(open('bcms.scaler','rb'))

# Setting optional model configuration
model_args = {
    "regression": True,
    "do_lower_case": True,
    "eval_batch_size": 64,
}

# Create a ClassificationModel
model = MultiLabelClassificationModel(
    "electra",
    "CLASSLA/bcms-bertic-geo",
    num_labels=2,
    loss_fct="MAELoss",
    args=model_args,
)


text = ['Kaj si rekel', 'Ne mogu to da uradim', 'Sjutra idemo na more', 'Skuvaj kahvu, bona!']
pred = model.predict(text)
pred_inv = scl.inverse_transform(pred)[0]
pred_rev = reverse_geocode.search(pred_inv)
for t, c, r in zip(text, pred_inv, pred_rev):
	print(t,c,r)
