import os
import cv2
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset




# Function to Import the data
def import_train_folder_dataset_2(root_path, resized_side=(224,224)):
    class_names = os.listdir(root_path)
    
    img_data_files=[]
    label_data_files=[]
    for pos, img_class in enumerate(class_names):
        for img in os.listdir(os.path.join(root_path,img_class)):
            path_= os.path.join(root_path,img_class,img)
            
            img = cv2.imread(path_)
            resized = cv2.resize(img, resized_side)
            #norm_image = resized/255
            #norm_image = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # Normalize img
            #reorder_img = np.moveaxis(resized, -1, 0)# Reorder Axis to be Chanel * H * W
            img_data_files.append(resized)
            label_data_files.append(pos)
    return( (np.array(img_data_files),np.array(label_data_files)) )


# Create the dataset object
class Data_2(Dataset):
    def __init__(self, X_train, y_train, transform):
        # convert to tensor
        self.X = X_train
        # Convert to tensor
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        
        self.transform = transform
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        item = self.X[index]
        item = self.transform(item)
        return item, self.y[index]

    

def plot_loss_accuracy(train_loss, val_loss, train_accuracy, val_accuracy):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,6))

    fig.suptitle('Horizontally stacked subplots')
    ax1.plot(train_loss, label="Train_loss")
    ax1.plot(val_loss, label="Validation_loss")
    ax1.title.set_text("Loss")
    ax1.legend(loc="best")

    ax2.plot(train_accuracy, label="Train_Accuracy")
    ax2.plot(val_accuracy, label="Validation_Accuracy")
    ax2.title.set_text("Accuracy")
    ax2.legend(loc="best")

    plt.show()
    
    
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots(figsize=(15,6))
    # make a plot
    ax.plot(train_loss, color="red", marker="o", label="Train_loss")
    ax.plot(val_loss, color="orange", marker="o", label="Validation_loss")
    # set x-axis label
    ax.set_xlabel("Epoch", fontsize = 14)
    # set y-axis label
    ax.set_ylabel("Loss Function",
                  color="red",
                  fontsize=14)


    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(train_accuracy,color="blue",marker="^", label="Train_Accuracy")
    ax2.plot(val_accuracy,color="green",marker="^", label="Validation_Accuracy")

    ax2.set_ylabel("Accuracy",color="blue",fontsize=14)

    ax.legend(loc="center left")
    ax2.legend(loc="center right")
    plt.show()



def accuracy_given_set(modelpy, valid_loader, device, name="Validation"):
    # Evaluate the Validation Set
    modelpy.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in valid_loader:
            outputs = modelpy(X.to(device))
            predictions = torch.argmax(outputs, dim=1)

            total += y.size(0)
            correct +=(predictions == y.to(device)).sum().item()
        
    print(f'Accuracy of the network on the {total} {name} instances: {100 * correct / total}%')

# Import the test dataset, with the names
def import_test_folder_dataset_2(root_path, resized_side=(224,224)):
    #class_names = os.listdir(root_path)
    
    img_data_files=[]
    img_data_names=os.listdir(root_path)
    for img in img_data_names:
        path_= os.path.join(root_path,img)
            
        img = cv2.imread(path_)
        resized = cv2.resize(img, resized_side)
        img_data_files.append(resized)
        
    return( (np.array(img_data_files),np.array(img_data_names)) )

# Create the dataset object
class Data_test_2(Dataset):
    def __init__(self, X_train, y_train, transform):
        # convert to tensor
        self.X = X_train
        # Convert to tensor
        self.y = y_train
        
        self.transform = transform
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        item = self.X[index]
        item = self.transform(item)
        return item, self.y[index]
    

# make predictions
def make_predictions(modelpy, dataset, device):
    modelpy.eval()
    predictions_list = []
    name_list = []
    with torch.no_grad():
        for x,y in dataset:
            x_tensor = x#torch.tensor(x, dtype=torch.float)# convert the array to tensor
            x_tensor = x_tensor[None, :] # Add 1 dimension as batch 1
            outputs = modelpy(x_tensor.to(device))
            #print(outputs)
            predictions = torch.argmax(outputs, dim=1)
            #print(predictions)
            predictions_list.append(predictions.item())
            name_list.append(y)
    return(predictions_list, name_list)

# Save Predictions
def save_predictions_as_csv(names, predictions, name="placeholder.csv"):
    df = pd.DataFrame(list(zip(names, predictions)))#, columns =['Name', 'class']
    #save
    df.to_csv(name, sep=';', header=False,index=False )

    
# Save model Checkpoint    
def save_model(epochs, time, model, optimizer, criterion, path):
    """
    Function to save the trained model to disk.
    """
    # Remove the last model checkpoint if present.
    torch.save({
                'epoch': epochs+1,
                'time': time,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path)
    
def save_metrics(train_loss, val_loss, train_accuracy, val_accuracy, path):
    # Method to save the results as a csv. Method by Alejandro C Parra Garcia
    dict = {'train_loss': train_loss, 'val_loss': val_loss, 'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy}  
    df = pd.DataFrame(dict)  
    df.to_csv(path)