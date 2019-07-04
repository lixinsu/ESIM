The code is modified from [repo](https://github.com/coetaur0/ESIM).

Major improvement:
- support fix embedding or finetune embedding
- fine tune part of embeddings in hypothese
- Antomatically inference some parameter from the checkpoint, deprecated the test config.



### Preprocess the data
Before the downloaded corpus and embeddings can be used in the ESIM model, they need to be preprocessed. This can be done with
the *preprocess_data.py* script in the *scripts/* folder of this repository. 

The script's usage is the following:
```
preprocess_data.py [-h] [--config CONFIG]
```
where `config` is the path to a configuration file defining the parameters to be used for preprocessing. A default configuration
file can be found in the *config/* folder of this repository.

### Train the model
The *train_model.py* script can be used to train the ESIM model on some training data and validate it on some validation data.

The script's usage is:
```
train_model.py [-h] [--config CONFIG] [--checkpoint CHECKPOINT]
```
where `config` is a configuration file (a default one is located in the *config/* folder), and `checkpoint` is an optional
checkpoint from which training can be resumed. Checkpoints are created by the script after each training epoch, with the name
*esim_\*.pth.tar*, where '\*' indicates the epoch's number.

### Test the model
The *test_model.py* script can be used to test the model on some test data.

Its usage is:
```
test_model.py [-h]
```
