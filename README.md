# CS726
FgSegNet_MyVersion

# Project Statement
Original FgSegnet models were tested on the surveillance videos. But it was observed that this model does not mask off the static objects into background in a moderately different scenario indicating fitted more towards the moving objects in the dataset used for training. So when shown a traffic signal post, the model trained on the highway dataset could not mask it off as a background object inspite of being a static object in the image. Also when shown a single static image of text, the static text does not merge into the background.

We plan to modify the network to take care of this problem in the presented model.

# Methodology
## Data Preparation
Modified existing FgSegNet with 3D convolutional Network. This has improved the performance 
