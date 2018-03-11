Software requirements:
1. python 3.x
2. numpy
3. pillow
4. tqdm
5. tensorflow
6. matplotlib

Dataset used for training and validation of the project is augmented PASCAL VOC dataset, available at
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

After downloading the dataset put the VOC_aug folder in the downloaded folder in augmented_dataset folder
 
so that the preprocessed data is present in aug_preprocessed_data in corresponding train_data and val_data folders.

Now train.py can be run, which takes the preprocessed input and trains the model, and saves the checkpoints and summary in the aug_supervisor_training folder. 
The number of epochs in train.py are configured to 50, which can be updated as needed. Images of the predictions during training are saved in images folder 





