# Scripts for pre-training BERT models on the VarDial training data

First, create text-only files:
```cut -f 3 ch/train.txt > ch/train_textonly.txt
cut -f 3 ch/dev.txt > ch/dev_textonly.txt
```

Then pretrain a BERT model with a 30k vocabulary:
```python3 pretrain_constr.py ch 30000```
