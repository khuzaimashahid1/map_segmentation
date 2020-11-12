# Indoor Map Segmentation

Multi class Segmentation of multiple types of indoor maps such as floor plans, grid maps etc. to segment walls, lobbies, and rooms. This project contains notebooks and scripts to train segmentation models for multiple datsets using tensorflow and Keras

# Data Exploration Notebook:

At first, let us go through the data exploration notebook in the notebooks folder. It was designed to read the old data (the data that was provided earlier) and its masks. Furthermore, it creates multiple 2-layer UNET segmentation models which are as follows:
1.	Complete Model that utilizes all masks
2.	Room masks only Model
3.	Lobby masks only Model
4.	Wall masks only Model
The multiple models were trained for 10 epochs and their outputs are visible in the notebook. It can be concluded easily that furniture masks are very few in number therefore we can skip them in final model. Furthermore, the model is still under trained, therefore more deep model (more number of layers) is needed and training time also needs to be adjusted. All the models and their respective model weights are saved in the notebooks folder in case cross referencing is needed.

# New Data Exploration Notebook:
Now, let us go through the new data exploration notebook in the notebooks folder. It was designed to read the new data (detailed flat maps) and its masks. Furthermore, it creates a 2-layer UNET segmentation model which was  trained for 10 epochs and its output is visible in the notebook. It can be concluded that the model is still under trained, therefore more deep model (more number of layers) is needed and training time also needs to be adjusted. All the models and their respective model weights are saved in the notebooks folder in case cross referencing is needed.   

# Loss Functions Test Case Notebooks:
In this stage, multiple loss functions were tested for both data and new data. These loss functions and their respective notebooks are:
1.	Jaccard Loss Models Notebook (JaccardLossModels)
2.	Binary Cross Entropy Loss Models Notebook (BinaryLossModels)
3.	Mean Squared Error Loss Models Notebook (MSELossModels)
The same 2 layer unet model was trained for 10 epochs , with 50 steps per epoch, with above mentioned loss functions. It can be clearly seen from the results in the notebooks that Binary Cross entropy gives the best estimation for the model to improve as it has the highest validation IOU or accuracy. MSE loss has lower loss values but the prediction it generates has very low confidence and is therefore blurred.  All the models and their respective model weights are saved in the notebooks folder in case cross referencing is needed.

# Segmentation Training Python File:
In this python script, multiple functions were written to test multiple configurations of UNET architecture by varying different hyper parameters of the model which include:
1.	Number of Layers
2.	Loss Function
3.	Number of Epochs
User can enter any combination of above hyper parameter in the main function and the mode will train and also plot its respective accuracy and losses plot. 
The configurations that were already tested are:
1.	Layers 1 to 5 models for both old data and new data with binary cross entropy loss trained for 20 epochs. The results of this configuration and weight files are saved in Test Cases folder in the Segmentation_Training Directory.
2.	Layers 3,4,5 models for both old data and new data with Mean Squared Error loss trained for 50 epochs. The results of this configuration and weight files are saved in Test Cases folder in the Segmentation_Training Directory.
It is quite discernible from the results of above configurations as saved in their respective folders in test cases directory that Binary Cross entropy gives best model accuracy and IOU. Furthermore, the layers sweet point is at 3,4 and 5 layer network.
For convenience, All 3,4,5 layers models haven been per-trained for both old data and new data.

# Classification Training Python File:
In this python script, multiple well known classification models were re trained (last 20 layers only) using transfer learning to classify a map as old data or new data (Binary Classification Problem). The tested models include:
1.	EfficientNet B0
2.	MobileNet
3.	MobileNetV2
4.	NASNetMobile
5.	VGG16
6.	VGG19
7.	ResNet50
8.	InceptionV3
9.	Xception
10.	InceptionResNetV2
11.	DenseNet121
12.	DenseNet169
13.	DenseNet201
14.	NASNetLarge
All these trained models and their weights along with their accuracy and loss plots are saved in the classification training folder.
It can be clearly seen from the loss and accuracy plots of all models listed above that only MobileNet model has generalized well to our data. Therefore, only it will be used in final pipeline.
# Test Models Python Script:
This script will be used to test the trained models. User can set number of layers for UNET test model in the main function. There are 4 types of testing that can be performed which are as follows:
1.	Testing the pipeline which includes classification first and then testing segmentation based on the respective UNET model i.e. old data or new data model.
2.	Testing only the classification model.
3.	Testing only segmentation model of data.
4.	Testing segmentation model of old data.
The testing is performed on all images in test folder. For classification, a csv file is generated with predicted results. For segmentation, the resulted masks are saved in separate folders along with overlays.