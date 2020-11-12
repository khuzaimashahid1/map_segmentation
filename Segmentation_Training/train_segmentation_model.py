import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import argparse
import datetime
from PIL import Image
from keras_unet.utils import plot_imgs
from sklearn.model_selection import train_test_split
from keras_unet.utils import get_augmented
from keras_unet.models import custom_unet
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras.optimizers import Adam
from keras_unet.losses import jaccard_distance
from keras_unet.metrics import iou,iou_thresholded,jaccard_coef,dice_coef
from keras_unet.utils import plot_segm_history
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError



def getData(datadir=os.path.join(os.path.dirname(os.getcwd()),'Data','updated_masks')):
    """[summary]
    This is a function that get all images and masks stored in Data (Old Data) directory, converts it into numpy arrays
    and normalizes the data to have values between 0 and 1 for faster model convergence.
    """

    # Getting All folders in Directory of Data
    output = [dI for dI in os.listdir(datadir) if os.path.isdir(os.path.join(datadir,dI))]
    # Transforming all images and their respective masks into numpy arrays
    imgs_list = []
    masks_list = []
    room_masks=0
    lobby_masks=0
    furniture_masks=0
    wall_masks=0
    for folder in output:
        currentfolder=os.path.join(datadir,folder)
        onlyfiles = [f for f in os.listdir(currentfolder) if os.path.isfile(os.path.join(currentfolder, f))]
        flagroom=False
        flaglobby=False
        flagwall=False
        flagfurniture=False
        for file in onlyfiles:
            if 'real' in file:
                currentfile=os.path.join(currentfolder, file)
                image=np.array(Image.open(currentfile).convert('RGB').resize((384,384)))
                imgs_list.append(image)
            if 'room' in file:
                currentfile=os.path.join(currentfolder, file)
                room=np.array(Image.open(currentfile).resize((384,384)))
                room = room[:, :, 3]
                flagroom=True
            if 'lobby' in file:
                currentfile=os.path.join(currentfolder, file)
                lobby=np.array(Image.open(currentfile).resize((384,384)))
                lobby = lobby[:, :, 3]
                flaglobby=True
            if 'wall' in file:
                currentfile=os.path.join(currentfolder, file)
                wall=np.array(Image.open(currentfile).resize((384,384)))
                wall = wall[:, :, 3]
                flagwall=True
            
        mask=np.zeros((384,384,3))
        mask[:,:,0]=image[:,:,0]
        if flagroom:
            mask[:,:,0]=room
            room_masks+=1
        else:
            mask[:,:,0]=0
        if flaglobby:
            mask[:,:,1]=lobby
            lobby_masks+=1
        else:
            mask[:,:,1]=0
        if flagwall:
            mask[:,:,2]=wall
            wall_masks+=1
        else:
            mask[:,:,2]=0
        masks_list.append(mask)
    imgs_np = np.asarray(imgs_list)
    masks_np = np.asarray(masks_list)
    x = np.asarray(imgs_np, dtype=np.float32)/255
    y = np.asarray(masks_np, dtype=np.float32)/255
    return x,y

def getNewData(datadir=os.path.join(os.path.dirname(os.getcwd()),'Data_new','mask')):
    """[summary]
    This is a function that get all images and masks stored in Data_New (Detailed Flap maps) directory, converts it into numpy arrays
    and normalizes the data to have values between 0 and 1 for faster model convergence.
    """
    # Getting All folders in Directory of Data
    output = [dI for dI in os.listdir(datadir) if os.path.isdir(os.path.join(datadir,dI))]
    # Transforming all images and their respective masks into numpy arrays
    imgs_list = []
    masks_list = []
    bathroom_masks=0
    furniture_masks=0
    hall_masks=0
    kitchen_masks=0
    room_masks=0
    terrace_masks=0
    window_masks=0

    for folder in output:
        currentfolder=os.path.join(datadir,folder)
        onlyfiles = [f for f in os.listdir(currentfolder) if os.path.isfile(os.path.join(currentfolder, f))]
    #     print(folder,len(onlyfiles))
        flagbathroom=False
        flagfurniture=False
        flaghall=False
        flagkitchen=False
        flagroom=False
        flagterrace=False
        flagwindow=False
        for file in onlyfiles:
            if 'Bathroom' in file:
                currentfile=os.path.join(currentfolder, file)
                bathroom=np.array(Image.open(currentfile).resize((384,384)))
                bathroom = bathroom[:, :, 3]
                flagbathroom=True
            elif 'Furniture' in file:
                currentfile=os.path.join(currentfolder, file)
                furniture=np.array(Image.open(currentfile).resize((384,384)))
                furniture = furniture[:, :, 3]
                flagfurniture=True
            elif 'Hall' in file:
                currentfile=os.path.join(currentfolder, file)
                hall=np.array(Image.open(currentfile).resize((384,384)))
                hall = hall[:, :, 3]
                flaghall=True
            elif 'Kitchen' in file:
                currentfile=os.path.join(currentfolder, file)
                kitchen=np.array(Image.open(currentfile).resize((384,384)))
                kitchen = kitchen[:, :, 3]
                flagkitchen=True
            elif 'Room' in file:
                currentfile=os.path.join(currentfolder, file)
                room=np.array(Image.open(currentfile).resize((384,384)))
                room = room[:, :, 3]
                flagroom=True
            elif 'Terrace' in file:
                currentfile=os.path.join(currentfolder, file)
                terrace=np.array(Image.open(currentfile).resize((384,384)))
                terrace = terrace[:, :, 3]
                flagterrace=True
            elif 'Window' in file:
                currentfile=os.path.join(currentfolder, file)
                window=np.array(Image.open(currentfile).resize((384,384)))
                window = window[:, :, 3]
                flagwindow=True
            elif 'Other Area' in file:
                continue
            else:
                currentfile=os.path.join(currentfolder, file)
                image=np.array(Image.open(currentfile).convert('RGB').resize((384,384)))
    #             print(currentfile)
                imgs_list.append(image)
        mask=np.zeros((384,384,7))
        #print(mask.shape)
        if flagbathroom:
            mask[:,:,0]=bathroom
            bathroom_masks+=1
        else:
            mask[:,:,0]=0
        if flagfurniture:
            mask[:,:,1]=furniture
            furniture_masks+=1
        else:
            mask[:,:,1]=0
        if flaghall:
            mask[:,:,2]=hall
            hall_masks+=1
        else:
            mask[:,:,2]=0
        if flagkitchen:
            mask[:,:,3]=kitchen
            kitchen_masks+=1
        else:
            mask[:,:,3]=0
        if flagroom:
            mask[:,:,4]=room
            room_masks+=1
        else:
            mask[:,:,4]=0
        if flagterrace:
            mask[:,:,5]=terrace
            terrace_masks+=1
        else:
            mask[:,:,5]=0
        if flagwindow:
            mask[:,:,6]=window
            window_masks+=1
        else:
            mask[:,:,6]=0
        masks_list.append(mask)
    imgs_np = np.asarray(imgs_list)
    masks_np = np.asarray(masks_list)
    x = np.asarray(imgs_np, dtype=np.float32)/255
    y = np.asarray(masks_np, dtype=np.float32)/255
    return x,y
    # print("Bathoom Masks :",bathroom_masks)
    # print("Furniture Masks :",furniture_masks)
    # print("Hall Masks :",hall_masks)
    # print("kithcen Masks :",kitchen_masks)
    # print("Room Masks :",room_masks)
    # print("Terrace Masks :",terrace_masks)
    # print("Window Masks :",window_masks)


def train(x_train, x_val, y_train, y_val,num_classes_input,num_layers_input,batch_size_input,num_epochs_input,lossFunction,metrics_input):
    """[summary]
    This is a function that creates and trains a model with given configurations and data.
    Args:
        x_train, x_val, y_train, y_val (Numpy Array): [Data for training]
        num_classes_input (Number): [Number of classes for prediction. The Output of Model will have dimensions (?,?,num_classes_input)]
        num_layers_input (Number): [Number of layers for Deep UNET Stacked Together]
        num_epochs_input (Number): [Number of epochs to be trained]. 
        batch_size_input (Number): [Batch Size for training]. 
    """

    # Prepare Training generator for Augmenting the data
    train_gen=getAugmentedData(x_train,y_train,batch_size_input)
    # Model Definition
    input_shape = x_train[0].shape
    timestamp= datetime.datetime.now()
    if num_classes_input==7:
        # model_filename = 'segm_model_new_data?time='+str(timestamp)+'_layers='+str(num_layers_input)+'_batch_size='+str(batch_size_input)+'_epochs='+str(num_epochs_input)+'.h5'
        model_filename = str(num_layers_input)+'_layer_new_data_model.h5'
    elif num_classes_input==3:
        # model_filename = 'segm_model_data?time='+str(timestamp)+'_layers='+str(num_layers_input)+'_batch_size='+str(batch_size_input)+'_epochs='+str(num_epochs_input)+'.h5'
        model_filename = str(num_layers_input)+'_layer_data_model.h5'
    model=createModel(num_classes_input,num_layers_input,input_shape,lossFunction,metrics_input)
    try:
        model.load_weights(model_filename)
    except:
        pass
    # Define Callbacks
    callbacks = [
        ModelCheckpoint(model_filename, verbose=1, monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir='./logs')
        ]
    # Model Training
    history = model.fit(
    train_gen,
    validation_data=(x_val, y_val),
    steps_per_epoch=100,
    epochs=num_epochs_input,
    callbacks=callbacks
    )
    
    return history.history


def trainConfiguration(num_classes_input,num_layers_list=[1,2,3,4,5],num_epochs_input=20,batch_size_input=8,lossFunction=MeanSquaredError(),metrics_input=[iou]):
    """[summary]
    This is a function that tests different models with different configuration as specified by its arguments. It outputs multiple models
    and their metrics vs losses plots seperately and together.
    Args:
        num_classes_input (Number): [Number of classes for prediction. The Output of Model will have dimensions (?,?,num_classes_input). It also signifies which data to read]. 
        num_layers_list (list, optional): [A list of Number of layers to test for Deep UNET Stacked Together]. Defaults to [1,2,3,4,5]
        num_epochs_input (Number, optional): [Number of epochs to be trained]. Defaults to 20.
        batch_size_input (Number, optional): [Batch Size for training]. Defaults to 8.
        lossFunction     (Function, optional): [Loss function to be used for training]. Defaults to Mean Squared Error
        metrics_input     (list, optional): [Metrics to be used for training]. Defaults to iou (Intersection Over Union)
    """
    # Read Data based on num_classes_input argument
    if num_classes_input==7:
        x,y=getNewData()
        dataType="New Data"
    elif num_classes_input==3:
        x,y=getData()
        dataType="Data"
    else:
        raise Exception("Sorry, Given num_classes_input value does not match any data and model")
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=0)
    histories=[]
    # Train Model with given configurations and plot histories of different models 
    for i in range(len(num_layers_list)):
        history=train(x_train, x_val, y_train, y_val,num_classes_input,num_layers_list[i],batch_size_input,num_epochs_input,lossFunction,metrics_input)
        histories.append(history)
        plot_segm_history(history,str(num_layers_list[i])+" Layer Model for "+dataType,metrics=["iou", "val_iou"], losses=["loss", "val_loss"])

    # Create 2 figures. One for Metrics and Loss over Epochs. One for Metrics and Loss over # of layers
    fig,a=  plt.subplots(2,2)
    fig1,a1=plt.subplots(2)
    a[0][0].set_title('IOU Over Epochs')
    a[0][1].set_title('Validation IOU Over Epochs')
    a[1][0].set_title('Loss Over Epochs')
    a[1][1].set_title('Validation Loss Over Epochs')
    a1[0].set_title('Metrics over Layers')
    a1[1].set_title('Loss over Layers')
    for ax in a1.flat:
        ax.set(xlabel='# of Layers')
    for ax in a.flat:
        ax.set(xlabel='# of epochs')
    maxIOU=[]
    maxValIOU=[]
    minLoss=[]
    minValLoss=[]
    # Create Metrics and Losses over Epochs Plot for N layer Model
    for idx,history in enumerate(histories):
        maxIOU.append(max(history['iou']))
        maxValIOU.append(max(history['val_iou']))
        minLoss.append(min(history['loss']))
        minValLoss.append(min(history['val_loss']))
        a[0][0].plot(history['iou'],label=str(num_layers_list[idx])+" Layer Model")
        a[0][1].plot(history['val_iou'],label=str(num_layers_list[idx])+" Layer Model")
        a[1][0].plot(history['loss'],label=str(num_layers_list[idx])+" Layer Model")
        a[1][1].plot(history['val_loss'],label=str(num_layers_list[idx])+" Layer Model")
    a[0][0].legend()
    a[0][1].legend()
    a[1][0].legend()
    a[1][1].legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.suptitle('Metrics and Losses over Epochs for '+dataType)
    fig.savefig('Metrics and Losses over Epochs for '+dataType)

    # Create Metrics and Losses over Number of Layers
    a1[0].plot(maxIOU,label="IOU")
    a1[0].plot(maxValIOU,label="val_IOU")
    a1[1].plot(minLoss,label="Loss")
    a1[1].plot(minValLoss,label="val_Loss")
    a1[0].legend()
    a1[1].legend()
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.85)
    fig1.suptitle('Metrics And Losses over Layers for '+dataType)
    fig1.savefig('Metrics And Losses over Layers for '+dataType)


def plot_segm_history(history,fname, metrics=["iou", "val_iou"], losses=["loss", "val_loss"]):
    """[summary]
    This function plots history of a model i.e. its metrics and losses. Output is a picture of plots. 
    Args:
        history (Dictionary): [Dictionary Containing metrics and losses over epochs]
        metrics (list, optional): [Names of metrics to plot]. Defaults to ["iou", "val_iou"].
        losses (list, optional): [Names of losses to plot]. Defaults to ["loss", "val_loss"].
    """
    
    # summarize history for metrics
    fig,(ax1,ax2)=  plt.subplots(2)
    ax1.set_title('Metrics Over Epochs')
    for metric in metrics:
        ax1.plot(history[metric], label=metric)
    ax1.legend()
    # summarize history for loss
    ax2.set_title('Losses Over Epochs')
    for loss in losses:
        ax2.plot(history[loss], label=loss)
    ax2.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.suptitle(fname)
    fig.savefig(fname)

def createModel(num_classes_input,num_layers_input,input_shape,lossFunction,metrics_input):
    """[summary]
    This function creates and returns a model instance with given parametres.
    Args:
        num_classes_input (Number): [Number of classes for prediction. The Output of Model will have dimensions (?,?,num_classes_input)]
        num_layers_list (Number): [Number of layers for Deep UNET stacked together]
        input_shape (?,?,?): [Dimension of input]
        lossFunction     (Function): [Loss function to be used for training]
        metrics_input     (list): [Metrics to be used for training].
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

def getAugmentedData(x_train, y_train,batch_size_input):
    """[summary]
    This is a function that creates a training generator which will give augmented data in batches of given configuration.
    Args:
        x_train, y_train(Numpy Arrays): [Data to be augmented]
        batch_size_input (Number): [Batch Size for training]. 
    """

    # Prepare Training generator for Augmenting the data
    train_gen = get_augmented(
        x_train, y_train, batch_size=batch_size_input,
        data_gen_args = dict(
            rotation_range=5.,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=40,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='constant'
        ))
    train_gen = (pair for pair in train_gen)
    return train_gen

# Define Main Function
if __name__ == "__main__":
    trainConfiguration(num_classes_input=3,num_layers_list=[3,4,5],num_epochs_input=20,batch_size_input=8,lossFunction='binary_crossentropy',metrics_input=[iou,'accuracy'])
    trainConfiguration(num_classes_input=7,num_layers_list=[3,4,5],num_epochs_input=20,batch_size_input=8,lossFunction='binary_crossentropy',metrics_input=[iou,'accuracy'])
    

