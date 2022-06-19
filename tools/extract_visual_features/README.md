# Visual features extraction

Just as [VL-T5 authors](https://github.com/j-min/VL-T5) do, we use [Hao Tan's Detectron2 implementation of 'Bottom-up feature extractor'](https://github.com/airsplay/py-bottom-up-attention).

We use the feature extractor which outputs 36 boxes per image and store them in hdf5 format.


## How to extract

### 1. Install Detectron2

Please follow [the original installation guide](https://github.com/airsplay/py-bottom-up-attention#installation).

*We strongly recommend that you create a new Conde environment for this task.*


### 2. Download the panel images

If you have not done so, please download the extracted panel images from [here](https://obj.umiacs.umd.edu/comics/index.html).


### 3. Manually extract & convert features

Just run the following command:

```sh
python comics_proposal.py 
    --batch_size=<batch size> 
    --dataset_path=<path to the dataset (defaults to 'datasets/COMICS/data')> 
    --out_dir=<path to the output directory (defaults to 'datasets/COMICS/frcnn_features')>
```
