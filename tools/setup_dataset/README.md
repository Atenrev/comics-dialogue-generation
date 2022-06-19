# Dataset setup 

If you want to build the dataset by yourself, use the scripts in this folder. These scripts are slightly modified from the originals so as not to use their original tokenizer. You can find the original scripts in the [COMICS repository](https://github.com/miyyer/comics).

## Dependencies
* python 2.7, 
* lasagne 
* theano,
* h5py
* cv2
* glob2

## Setup

```sh
# Run the setup script that will download all the files required for the dataset.
setup.sh

# Run the script that will create the folds for the dataset.
python text_cloze_minibatching.py
```