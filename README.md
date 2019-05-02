# CS726
FgSegNet_MyVersion

# Project Statement/ Goal
Original FgSegnet models were tested on the surveillance videos. But it was observed that this model does not mask off the static objects into background in a moderately different scenario indicating fitted more towards the moving objects in the dataset used for training. So when shown a traffic signal post, the model trained on the highway dataset could not mask it off as a background object inspite of being a static object in the image. Also when shown a single static image of text, the static text does not merge into the background. This is not a behavior to be exhibited by a Background Subtractor Algorithm. It is happening because the FgSegNet does not consider the temporal information in the videos.

Goal is to achieve the realistic behavior of Background Subtractor by considering the temporal information of the videos.

We plan to modify the network to take care of this problem in the presented model.

# Related Literature

# Approaches tried

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
