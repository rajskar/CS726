# CS726
FgSegNet_MyVersion

# Project Statement/ Goal
In the paper titled “Foreground segmentation using convolutional neural networks for multiscale feature encoding”, authors tried to solve the problem background subtraction for various scenarios and environmental conditions. 

They have modelled a deep learning based model using CNNs and used multiscale features extracted from the image to classify the pixels in image into foreground and background.  As quoted by the authors “Our models are learning foreground objects from isolated frames, i.e. the time sequence is not considered during training. As a future work, we plan to redesign our network to learn from temporal data as well, and extend our FPM module.”

To elaborate, they are using the multi-scale features of a single image to train the model to classify the foreground and background pixels, which may not be an entirely correct idea considering the theme of background subtraction. The background or foreground in a image depends on the past frames. 

In this project, we want to try and make a deep learning model which takes the features of the past images also into consideration before classifying a pixel into foreground or background. 

Generally, all classical background subtraction models should work same for similar environmental conditions. So as a preliminary study, we used the pre-trained models of the authors for inference in an “OFFICE Scenario”. The office scenario was kept almost close to the OFFICE dataset of CDNet Dataset used for training. Still, we observed that the performance was poor. This inspired us to model a network which can also take the past-information into consideration to improve the inference performance on the different scenarios, other than the dataset used for training.

At the end of this project, we expect to have a more generic model for background subtraction

Original FgSegnet models were tested on the surveillance videos. But it was observed that this model does not mask off the static objects into background in a moderately different scenario indicating fitted more towards the moving objects in the dataset used for training. So when shown a traffic signal post, the model trained on the highway dataset could not mask it off as a background object inspite of being a static object in the image. Also when shown a single static image of text, the static text does not merge into the background. This is not a behavior to be exhibited by a Background Subtractor Algorithm. It is happening because the FgSegNet does not consider the temporal information in the videos.

Goal is to achieve the realistic behavior of Background Subtractor by considering the temporal information of the videos.

We plan to modify the network to take care of this problem in the presented model.

# Related Literature
Long Ang Lim, Hacer Yalim Keles, "Foreground segmentation using convolutional neural networks for multiscale feature encoding" at Pattern Recognition Letters, vol. 112, Pg. 256 - 262, 2018,issn: 0167-8655

# Approaches tried
We modelled and trained two different deep learning architectures
1. Using 3D convolutional neural networks
2. Using CNN-LSTM combination

# Experiments
## Code Describe
### Language and Environment
### Lines of Code
### Starting Code
### URL where my code starts

## Experimental Platform description
### Hardware Setup
### Duration and other details

## Experimental Results along with Commentary

## Effort
### Fraction of time in different parts of project
### Challenging part
### Fraction of work done by different team members

# Methodology
## Data Preparation

Modified existing FgSegNet with 3D convolutional Network. This has improved the performance 
