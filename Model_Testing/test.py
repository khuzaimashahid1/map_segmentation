import sys
import os
from PIL import Image
from keras_unet.models import custom_unet
from tensorflow.keras.losses import MeanSquaredError
from keras_unet.metrics import iou
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

def mask_to_rgba(mask, color="red"):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    
    Args:
        mask (numpy.ndarray): [description]
        color (str, optional): Check `MASK_COLORS` for available colors. Defaults to "red".
    
    Returns:
        numpy.ndarray: [description]
    """    
    
    assert(mask.ndim==3 or mask.ndim==2)

    h = mask.shape[0]
    w = mask.shape[1]
    zeros = np.zeros((h, w))
    ones = mask.reshape(h, w)
    if color == "red":
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == "green":
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == "blue":
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == "yellow":
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == "magenta":
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == "cyan":
        return np.stack((zeros, ones, ones, ones), axis=-1)
    elif color == "white":
        return np.stack((ones, ones, ones, ones), axis=-1)


def zero_pad_mask(mask, desired_size):
    """[summary]
    
    Args:
        mask (numpy.ndarray): [description]
        desired_size ([type]): [description]
    
    Returns:
        numpy.ndarray: [description]
    """
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask



def saveResultsNewData(imgs,masks):
    """[summary]
    This is a function that will save all results of new maps (detailed) into sepereate folders along with showing overlay of all masks
    """
    # print("New Data")
    # print("Images",imgs.max(),imgs.min())
    # print("Masks",masks.max(),masks.min())
    result_dir=os.path.join(os.getcwd(),'result_new_data')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    else:
        raise Exception("Sorry, Result Directory already exists from previous result. Kindly rename or delete it")
    
    if imgs.max()!=255:
        imgs=imgs*255
    if masks.max()!=255:
        masks=masks*255
    bathroom_masks=masks[:,:,:,0]
    furniture_masks=masks[:,:,:,1]
    hall_masks=masks[:,:,:,2]
    kitchen_masks=masks[:,:,:,3]
    room_masks=masks[:,:,:,4]
    terrace_masks=masks[:,:,:,5]
    window_masks=masks[:,:,:,6]
    
    for idx in range(imgs.shape[0]):
        file_dir=os.path.join(result_dir,str(idx+1))
        os.mkdir(file_dir)
        # Save original Image
        img = Image.fromarray(imgs[idx].astype('uint8'),"RGB")
        img.save(os.path.join(file_dir,'original.png'))
        # Save bathroom mask
        img = Image.fromarray((mask_to_rgba(zero_pad_mask(bathroom_masks[idx], desired_size=masks.shape[1]),color='white')).astype('uint8'),"RGBA")
        img.save(os.path.join(file_dir,'bathroom.png'))
        # Save furniture mask
        img = Image.fromarray((mask_to_rgba(zero_pad_mask(furniture_masks[idx], desired_size=masks.shape[1]),color='white')).astype('uint8'),"RGBA")
        img.save(os.path.join(file_dir,'furniture.png'))
        # Save hall mask
        img = Image.fromarray((mask_to_rgba(zero_pad_mask(hall_masks[idx], desired_size=masks.shape[1]),color='white')).astype('uint8'),"RGBA")
        img.save(os.path.join(file_dir,'hall.png'))
        # Save kitchen mask
        img = Image.fromarray((mask_to_rgba(zero_pad_mask(kitchen_masks[idx], desired_size=masks.shape[1]),color='white')).astype('uint8'),"RGBA")
        img.save(os.path.join(file_dir,'kitchen.png'))
        # Save room mask
        img = Image.fromarray((mask_to_rgba(zero_pad_mask(room_masks[idx], desired_size=masks.shape[1]),color='white')).astype('uint8'),"RGBA")
        img.save(os.path.join(file_dir,'room.png'))
        # Save terrace mask
        img = Image.fromarray((mask_to_rgba(zero_pad_mask(terrace_masks[idx], desired_size=masks.shape[1]),color='white')).astype('uint8'),"RGBA")
        img.save(os.path.join(file_dir,'terrace.png'))
        # Save window mask
        img = Image.fromarray((mask_to_rgba(zero_pad_mask(window_masks[idx], desired_size=masks.shape[1]),color='white')).astype('uint8'),"RGBA")
        img.save(os.path.join(file_dir,'window.png'))
        
        # Save overlapping masks
        fig=plt.figure(figsize=(2, 2))
        plt.imshow((imgs[idx]/255),cmap='gray')
        plt.imshow(mask_to_rgba(zero_pad_mask((bathroom_masks[idx]/255), desired_size=imgs.shape[1]),color='red'),cmap='gray',alpha=0.3)
        plt.imshow(mask_to_rgba(zero_pad_mask((hall_masks[idx]/255), desired_size=x.shape[1]),color='blue'),cmap='gray',alpha=0.5)
        plt.imshow(mask_to_rgba(zero_pad_mask((kitchen_masks[idx]/255), desired_size=x.shape[1]),color='yellow'),cmap='gray',alpha=0.5)
        plt.imshow(mask_to_rgba(zero_pad_mask((room_masks[idx]/255), desired_size=x.shape[1]),color='magenta'),cmap='gray',alpha=0.2)
        plt.imshow(mask_to_rgba(zero_pad_mask((terrace_masks[idx]/255), desired_size=x.shape[1]),color='cyan'),cmap='gray',alpha=0.5)
        plt.imshow(mask_to_rgba(zero_pad_mask((window_masks[idx]/255), desired_size=x.shape[1]),color='white'),cmap='gray',alpha=1)
        plt.imshow(mask_to_rgba(zero_pad_mask((furniture_masks[idx]/255), desired_size=x.shape[1]),color='green'),cmap='gray',alpha=0.2)
        plt.axis('off')
        plt.savefig(os.path.join(file_dir,'overlay.png'),dpi=384,bbox_inches="tight",pad_inches=0)
        plt.close(fig)
        


def saveResultsData(imgs,masks):
    """[summary]
    This is a function that will save all results of old maps into sepereate folders along with showing overlay of all masks
    """
    # print("Data")
    # print("Images",imgs.max(),imgs.min())
    # print("Masks",masks.max(),masks.min())
    result_dir=os.path.join(os.getcwd(),'result_data')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    else:
        raise Exception("Sorry, Result Directory already exists from previous result. Kindly rename or delete it")
    if imgs.max()!=255:
        imgs=imgs*255
    if masks.max()!=255:
        masks=masks*255
    room_masks=masks[:,:,:,0]
    lobby_masks=masks[:,:,:,1]
    wall_masks=masks[:,:,:,2]
    
    
    for idx in range(imgs.shape[0]):
        file_dir=os.path.join(result_dir,str(idx+1))
        os.mkdir(file_dir)
        # Save original Image
        img = Image.fromarray(imgs[idx].astype('uint8'),"RGB")
        img.save(os.path.join(file_dir,'original.png'))
        # Save Room mask
        img = Image.fromarray((mask_to_rgba(zero_pad_mask(room_masks[idx], desired_size=masks.shape[1]),color='white')).astype('uint8'),"RGBA")
        img.save(os.path.join(file_dir,'room.png'))
        # Save Lobby mask
        img = Image.fromarray((mask_to_rgba(zero_pad_mask(lobby_masks[idx], desired_size=masks.shape[1]),color='white')).astype('uint8'),"RGBA")
        img.save(os.path.join(file_dir,'lobby.png'))
        # Save wall mask
        img = Image.fromarray((mask_to_rgba(zero_pad_mask(wall_masks[idx], desired_size=masks.shape[1]),color='white')).astype('uint8'),"RGBA")
        img.save(os.path.join(file_dir,'wall.png'))
        
        # Save overlapping masks
        fig=plt.figure(figsize=(2, 2))
        imgs[idx]=imgs[idx]/255
        room_masks[idx]=room_masks[idx]/255
        lobby_masks[idx]=lobby_masks[idx]/255
        wall_masks[idx]=wall_masks[idx]/255
        plt.imshow((imgs[idx]),cmap='gray')
        plt.imshow(mask_to_rgba(zero_pad_mask((room_masks[idx]), desired_size=imgs.shape[1]),color='red'),cmap='gray',alpha=0.3)
        plt.imshow(mask_to_rgba(zero_pad_mask((lobby_masks[idx]), desired_size=x.shape[1]),color='blue'),cmap='gray',alpha=0.5)
        plt.imshow(mask_to_rgba(zero_pad_mask((wall_masks[idx]), desired_size=x.shape[1]),color='green'),cmap='gray',alpha=0.5)
        plt.axis('off')
        plt.savefig(os.path.join(file_dir,'overlay.png'),dpi=384,bbox_inches="tight",pad_inches=0)
        plt.close(fig)
        


def getTestData(datadir=os.path.join(os.getcwd(),'test')):
    """[summary]
    This is a function that get all images and masks stored in Test directory, converts it into numpy arrays
    """
    
    # Getting all images in the test directory and converting it into numpy arrays
    imgs_list = []
    filenames=[]
    onlyfiles = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
    for file in onlyfiles:
        currentfile=os.path.join(datadir, file)
        filenames.append(file)
        image=np.array(Image.open(currentfile).convert('RGB').resize((384,384)))
        imgs_list.append(image)
    x = np.asarray(imgs_list, dtype=np.float32)
    # print("Original")
    # print(x.max(),x.min())
    return x,filenames



def createModel(num_classes_input,num_layers_input,input_shape=(384,384,3),lossFunction='binary_crossentropy',metrics_input=[iou]):
    """[summary]
    This function creates and returns a model instance with given parametres.
    Args:
        num_classes_input (Number): [Number of classes for prediction. The Output of Model will have dimensions (?,?,num_classes_input)]
        num_layers_list (Number): [Number of layers for Deep UNET stacked together]
        input_shape (?,?,?): [Dimension of input]
        model_file_name (String): [Filename for model to be saved with].
        lossFunction     (Function,optional): [Loss function to be used for training] Defaults to Mean Squared Error
        metrics_input     (list,optional): [Metrics to be used for evauation]. Defaults are [iou]
    """
    
    # Creare Model
    model = custom_unet(
        input_shape,
        filters=32,
        use_batch_norm=True,
        dropout=0.3,
        dropout_change_per_layer=0.0,
        num_classes=num_classes_input,
        num_layers=num_layers_input
    )
    
    # Compile Model
    model.compile(
    optimizer=Adam(), 
    loss=lossFunction,
    metrics=metrics_input
    )
    return model


def loadModelsAndWeights(classification_model_name='MobileNet',num_layers_input=4):
    """[summary]
    This is a function that will load all models according to given arguments and run the test pipeline
    """

    # Load Segmentation Models
    segmentation_model_data=createModel(3,num_layers_input)
    segmentation_model_data.load_weights(str(num_layers_input)+"_layer_data_model.h5")
    segmentation_model_new_data=createModel(7,num_layers_input)
    segmentation_model_new_data.load_weights(str(num_layers_input)+"_layer_new_data_model.h5")
    
    # Load Classification Model
    json_file = open(str(classification_model_name)+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classification_model = model_from_json(loaded_model_json)
    classification_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    classification_model.load_weights(str(classification_model_name)+".h5")
    
    return classification_model,segmentation_model_data,segmentation_model_new_data
    
    
    
    
def testPipeline(x,filenames,classification_model,segmentation_model_data,segmentation_model_new_data):
    
    org_x=np.copy(x)
    # print("original",org_x.max(),org_x.min())
    # First Predict Class of images as Old data (0) or new Data (1)
    Y_pred=testClassification(x,filenames,classification_model) 
    # print("original",org_x.max(),org_x.min())
    # Normalize data for faster processing
    if org_x.max()==255:
        org_x=org_x/255
    # print(" normalized original",org_x.max(),org_x.min())
    # Seperate old data and new data images to further pass through respective unet models
    data_list=[]
    new_data_list=[]
    for i in range(len(Y_pred)):
        if Y_pred[i]==0:
            new_data_list.append(org_x[i,:,:,:])
        elif Y_pred[i]==1:
            data_list.append(org_x[i,:,:,:])
    data=np.asarray(data_list, dtype=np.float32)
    new_data=np.asarray(new_data_list, dtype=np.float32)
    # print("Pipeline")
    # print("Data : ",data.max(),data.min())
    # print("New Data : ",new_data.max(),new_data.min())
    
    # Test old data and new data images through their resective unet models and Save results in seperate folders   
    testDataSegmentationModel(data,segmentation_model_data)
    testNewDataSegmentationModel(new_data,segmentation_model_new_data)
    
def testClassification(x,filenames,classification_model):
    classification_model._make_predict_function()
    x_preprocessed=preprocess_input(x)
    Y_pred =classification_model.predict(x_preprocessed)
    Y_pred=np.where(Y_pred>0.5,1,0)
    csv=open("classification_results.csv","w")
    csv.write(" Filename , Predicted Class ID, Predicted Class Name"+"\n")
    for idx in range(len(filenames)):
        if Y_pred[idx]==0:
            className="New Data"
        elif Y_pred[idx]==1:
            className="Old Data"
        csv.write(str(filenames[idx])+','+str(Y_pred[idx])+','+className+"\n")
    csv.close()
    return Y_pred 

def testDataSegmentationModel(x,segmentation_model_data):

    # Predict on testing data
    y_pred_data=segmentation_model_data.predict(x,batch_size=8)
    # Save Results in result folder
    saveResultsData(x,y_pred_data)

def testNewDataSegmentationModel(x,segmentation_model_new_data):
    # Predict on testing data
    y_pred_new_data=segmentation_model_new_data.predict(x,batch_size=8)
    # Save Results in result folder
    saveResultsNewData(x,y_pred_new_data)



# Define Main Function
if __name__ == "__main__":
    
    '''
    If You wish to test entire pipeline. Uncomment Below Code and comment all else
    '''
    # Get Test Data
    x,filenames=getTestData()
    # Load All Models and Weights
    classification_model,segmentation_model_data,segmentation_model_new_data=loadModelsAndWeights(classification_model_name='MobileNet',num_layers_input=5)
    # Test Pipeline
    testPipeline(x,filenames,classification_model,segmentation_model_data,segmentation_model_new_data)
    
    '''
    If You wish to test only classification Model. Uncomment Below Code and comment all else
    '''
    # # Get Test Data
    # x,filenames=getTestData()
    # # Load All Models and Weights
    # classification_model,segmentation_model_data,segmentation_model_new_data=loadModelsAndWeights(classification_model_name='MobileNet',num_layers_input=4)
    # # Test Classification Model
    # testClassification(x,filenames,classification_model)

    '''
    If You wish to test only Segmentation Model of old data. Uncomment Below Code and comment all else
    '''
    # # Get Test Data
    # x,filenames=getTestData()
    # # Load All Models and Weights
    # classification_model,segmentation_model_data,segmentation_model_new_data=loadModelsAndWeights(classification_model_name='MobileNet',num_layers_input=4)
    # # Test Old data segmentation Model
    # testDataSegmentationModel(x,segmentation_model_data)

    '''
    If You wish to test only Segmentation Model of new data. Uncomment Below Code  and comment all else
    '''
    # # Get Test Data
    # x,filenames=getTestData()
    # # Load All Models and Weights
    # classification_model,segmentation_model_data,segmentation_model_new_data=loadModelsAndWeights(classification_model_name='MobileNet',num_layers_input=5)
    # # Test New Data Segmentation Mdoel
    # testNewDataSegmentationModel(x,segmentation_model_new_data)