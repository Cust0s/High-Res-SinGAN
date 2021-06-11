# High-Res-SinGAN

High-Res-SinGAN is an amplified version of [SinGAN](https://github.com/tamarott/SinGAN.git)
capable of producing images up to 500 x 500 pixels.

This repository is part of the *Neural Networks for Image Processing* course at DGIST.
For this course, several presentations were held. The presentation files can be found
under [/supplementary_files/](supplementary_files/). The two-page paper concluding this
project can be found [here]().

### SinGAN

SinGAN produces random sample images from a single natural image using a generative adverserial network.


### High Resultion Image Generation



## Installation

- Python 3.6

```
python -m pip install -r requirements.txt
```

### Random Sample Generation

## Standard Resolution

To train a model with the default parameters of SinGAN run:
```
python main_train.py --input_name cityscape.jpg
```
The training time should be about 40 minutes on an NVIDIA GeForce RTX 2070 SUPER.

## High Resolution

High resolution image generation can be achieved by modifying the structure of SinGAN. For this, we have 
introduced additional parameters that can be adjusted. To train a high resolution model run:
```
python
```


