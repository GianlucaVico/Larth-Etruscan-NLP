# Translation

This folder contains the code and notebooks for the translation part.

## Code

* `base_models.py`: base class for the translation models (except for the NNs)
* `random_models.py`: class for the random translation model
* `dictionary_model.py`: class for the dictionary model
* `ngram_models.py`: classes for the n-gram models
* `translation_utils.py`: methods to plot the results and compute the metrics
* `Larth/bigbird_attention.py`: BigBird attention mechanism (from Ithaca)
* `Larth/bigbird.py`: classes for the encoders and decoder (partially from Ithaca)
* `Larth/common_layers.py`: modules for NNs (from Ithaca)
* `Larth/data_utils.py`: methods to load the tokenizers and the dataset
* `Larth/decode.py`: beam search decoding
* `Larth/inference.py`: inference module for Larth
* `Larth/larth.py`: Larth translation model
* `Larth/run_train.py`: run Larth training from the terminal
* `Larth/train_utils.py`: functions used for training
* `Larth/train.py`: main train functions




## Notebooks

Changes in the may have broken some parts of the notebooks.

* `random_models.ipynb`: notebook to run the random model experiments and testing
* `dictionary_model.ipynb`: notebook to run the dictionary model experiments and testing
* `ibm_models.ipynb`: notebook to run the IBM models experiments
* `ngram_models.ipynb`: notebook to run the N-gram and Naive Bayes model experiments
* `Larth/Inference.ipynb`: notebook to run Larth for inference (not updated)
* `Larth/prepare_nn_data.ipynb`: notebook used to debug and prepare the dataloader part

## Others

* `etruscan_train.yml`: example of the train configuration file
* `model.yml`: example of the model configuration file
