# geoBERT

A tool for predicting geolocations of (Twitter, Jodel) messages. Winner of VarDial2020 and VarDial2021 shared tasks on geolocation prediction on social media.

We share code for training new models, as well as the winning models from VarDial2021.

## Application

To run models on examples, use code similar to the `predict_bcms.py` tailored to the BCMS subtask, with the following expected output:

```
Kaj si rekel [45.82933197 15.84926331] {'country_code': 'HR', 'city': 'Bestovje', 'country': 'Croatia'}
Ne mogu to da uradim [43.8143314  20.46512093] {'country_code': 'RS', 'city': 'Čačak', 'country': 'Serbia'}
Sjutra idemo na more [42.41667169 19.15543229] {'country_code': 'ME', 'city': 'Mojanovići', 'country': 'Montenegro'}
Skuvaj kahvu, bona! [43.66061352 19.42953163] {'country_code': 'BA', 'city': 'Rudo', 'country': 'Bosnia and Herzegovina'}
```

The winning system for the area of Bosnia, Croatia, Montenegro and Serbia can be found on HuggingFace under [bcms-bertic-geo](https://huggingface.co/classla/bcms-bertic-geo), for the area of Germany and Austria under [bert-base-german-dbmdz-uncased-geo](https://huggingface.co/classla/bert-base-german-dbmdz-uncased-geo), and for Switzerland under [swissbert-geo](https://huggingface.co/classla/swissbert-geo).

While the models are the state-of-the-art in geolocation prediction, their predictions are as good as the data they are trained on, so use them with reasonable caution!

## Training

To train a new model, use code similar to the `train_bcms.py` script. Training and dev data is supposed to be in a tab-separated (lat, lon, text) triple format, each message in one line.

The `alternative` folder provides an alternative implementation of the same script, with largely the same functionality.

The `pretrain_constrained` folder provides code to pre-train a BERT model from scratch. We used this code as a basis for our constrained submissions, but for real-world applications outside the limitations of a shared task you may rather want to use the unconstrained models referenced above.

## Citing

Please cite the following paper if you use the models or code:

```
@inproceedings{scherrer-ljubesic-2021-social,
    title = "Social Media Variety Geolocation with geo{BERT}",
    author = "Scherrer, Yves  and
      Ljube{\v{s}}i{\'c}, Nikola",
    booktitle = "Proceedings of the Eighth Workshop on NLP for Similar Languages, Varieties and Dialects",
    month = apr,
    year = "2021",
    address = "Kiyv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.vardial-1.16",
    pages = "135--140",
}
```
