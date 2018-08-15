# Project: Portrait Segmentation 

It is basically the task segmenting foreground from the background, in this case the foreground is the images of a person/face.



### Data

But CELEBA dataset does not have the dataset used to train on the task of segmentation. But it has data to identify faces and 
different attributes of faces. So for our purposes we assume the bounding boxes of the faces as the foreground and rest of the image
as background.
 
Images are downloaded and unpacked into folder data.

list_bbox_celeba.txt - file having bounding box infomration
evalpartition.txt  - file having information on partitioning the dataset

are also downloaded

### Run

Preprocess.ipynb is run initially to preprocess the data and divide them into different batches for training, validation and testing purposes

train.ipynb has the code used to train the dataset 

evaluation.ipynb has the code used to evalute the trained model

Evaluation metric used in Mean IOU (Intersection over union)- This is an ideal metric for segmentation tasks, as it basically measures the amount 
of overlap between the predicted and ideal output.




