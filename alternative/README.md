# Alternative scripts for geoBERT finetuning

These scripts provide an alternative interface to the `predict_bcms.py` script. They have been used for the constrained submissions at VarDial 2021.

Each run is defined by an entry in a JSON file. For example, the following code defines the experiment `regr_ch_constr30k_3`:
```
{
 "regr_ch_constr30k_3": {
    "_train_data": "ch/train.txt",
    "_dev_data": "ch/dev.txt",
    "_ynorm": "jointscale",
    "_model_type": "bert",
    "_model_name": "ch-constr-uncased-30k/best_model",
    "_lossfn": "MAELoss",
    "_save_predictions": true,
    "num_train_epochs": 60,
    "train_batch_size": 32,
    "max_seq_length": 128,
    "seed": 3
  }
}
```

An experiment is launched by specifying the JSON file as well as an optional regular expression that filters the experiments in the JSON file. For example, `python3 regression.py constr_regr.json "regr_ch"` runs all experiments from the `constr_regr.json` file whose keys contain the `regr_ch` substring.
