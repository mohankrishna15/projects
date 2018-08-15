# Project: Semantic Segmentation 

Please see my final capstone project report [here](https://github.com/mohankrishna15/projects/blob/master/capstone/Project%20Report.pdf)

Please see my final capstone project proposal [here](https://github.com/mohankrishna15/projects/blob/master/capstone/proposal.pdf)


### Software requirements:

1. [python 3.x](https://www.python.org/)
2. [numpy](http://www.numpy.org/)
3. [pillow](https://pillow.readthedocs.io/en/latest/)
4. [tqdm](https://pypi.python.org/pypi/tqdm)
5. [tensorflow](https://www.tensorflow.org/)
6. [matplotlib](http://matplotlib.org/)

### Data

Dataset used for training and validation of the project is augmented PASCAL VOC dataset, available at
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

After downloading the dataset put the VOC_aug folder in the downloaded folder in augmented_dataset folder
 
so that the preprocessed data is present in aug_preprocessed_data in corresponding train_data and val_data folders.

### Run

Now train.py can be run, which takes the preprocessed input and trains the model, and saves the checkpoints and summary in the aug_supervisor_training folder. 
The number of epochs in train.py are configured to 50, which can be updated as needed. Images of the predictions during training are saved in images folder 





