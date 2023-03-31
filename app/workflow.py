import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split
from model.alexnet import AlexNet
from Dataloader.custom_dataset import MPIIGaze
from Dataloader.dataloader import data_loader
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def test_train_split(dataset, test_size = 0.3):
    train_size = 1 - test_size
    return random_split(dataset, lengths=[train_size,test_size])

def create_dataloader(dataset, batch_size = 4, shuffle = True):
    return data_loader(dataset, batch_size, shuffle)

def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


def train_model(model, train_data_loader, val_data_loader):
    """ method to train the model

    Parameters
    ----------
    model : AlexNet model
        the model to be trained
    train_data_loader : DataLoader
        the training data loader
    val_data_loader : DataLoader
        the validation data loader

    Returns
    -------
    LOSS: dictionary, checkpoint_epoch: int 
        the LOSS dictionary contains the training information and chekpoint epoch contains the latest epoch 
    """
    start_time = time.time()
    epoch = 10

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists("/kaggle/input/models/latest_alexnet_model_checkpoint.pt"):
        checkpoint = torch.load(
            "/kaggle/input/models/latest_alexnet_model_checkpoint.pt"
            , map_location = device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        check_point_epoch = checkpoint['epoch']+1
        LOSS = checkpoint['loss']
        theta_training_loss = LOSS['theta_training_loss']
        theta_training_mae = LOSS['theta_training_mae']
        theta_validation_loss = LOSS['theta_validation_loss']
        theta_validation_mae = LOSS['theta_validation_mae']

        phi_training_loss = LOSS['phi_training_loss']
        phi_training_mae = LOSS['phi_training_mae']
        phi_validation_loss = LOSS['phi_validation_loss']
        phi_validation_mae = LOSS['phi_validation_mae']
        
        theta_min_loss = min(theta_validation_loss)
        phi_min_loss = min(phi_validation_loss)
    else:
        theta_training_loss = []
        theta_training_mae = []
        theta_validation_loss = []
        theta_validation_mae = []

        phi_training_loss = []
        phi_training_mae = []
        phi_validation_loss = []
        phi_validation_mae = []
        
        theta_min_loss = 100000
        phi_min_loss = 100000
        
        check_point_epoch = 1

    model.train()
    for i in range(check_point_epoch,epoch+check_point_epoch):
        theta_train_loss = []
        theta_train_mae = []
        theta_val_loss =[]
        theta_val_mae = []

        phi_train_loss = []
        phi_train_mae = []
        phi_val_loss =[]
        phi_val_mae = []

        data_loop = tqdm(train_data_loader, leave=True)
        for x, y_theta, y_phi  in data_loop:
            data_loop.set_description(f"Epoch {i}")
            optimizer.zero_grad()
            theta_pred, phi_pred = model(x)
            theta_loss = loss_fn(theta_pred, y_theta)
            theta_loss.backward(retain_graph=True)
            phi_loss = loss_fn(phi_pred, y_phi)
            phi_loss.backward()
            optimizer.step()

            theta_train_loss.append(theta_loss.item())
            theta_train_mae.append(mean_absolute_error(y_theta, theta_pred).item())

            phi_train_loss.append(phi_loss.item())
            phi_train_mae.append(mean_absolute_error(y_phi, phi_pred).item())

            data_loop.set_postfix(
                    train_loss = (sum(theta_train_loss) / len(theta_train_loss), sum(phi_train_loss) / len(phi_train_loss)),
                    train_MAE = (sum(theta_train_mae) / len(theta_train_mae), sum(phi_train_mae) / len(phi_train_mae))
                    )
            
        theta_training_loss.append(sum(theta_train_loss) / len(theta_train_loss))
        theta_training_mae.append(sum(theta_train_mae)/len(theta_train_mae))

        phi_training_loss.append(sum(phi_train_loss) / len(phi_train_loss))
        phi_training_mae.append(sum(phi_train_mae)/len(phi_train_mae))

        data_loop = tqdm(val_data_loader, leave = True)
        with torch.no_grad():
            for x, y_theta, y_phi in data_loop:
                theta_pred, phi_pred = model(x)
                
                theta_loss = loss_fn(theta_pred, y_theta)
                theta_val_loss.append(theta_loss.item())
                theta_val_mae.append(mean_absolute_error(y_theta, theta_pred).item())

                phi_loss = loss_fn(phi_pred, y_phi)
                phi_val_loss.append(phi_loss.item())
                phi_val_mae.append(mean_absolute_error(y_phi, phi_pred).item())

                data_loop.set_postfix(
                    train_loss = (sum(theta_train_loss) / len(theta_train_loss), sum(phi_train_loss) / len(phi_train_loss)),
                    train_MAE = (sum(theta_train_mae) / len(theta_train_mae), sum(phi_train_mae) / len(phi_train_mae)),
                    val_loss = (sum(theta_val_loss) / len(theta_val_loss), sum(phi_val_loss) / len(phi_val_loss)),
                    val_MAE = (sum(theta_val_mae) / len(theta_val_mae), sum(phi_val_mae) / len(phi_val_mae)),
                    )
                
            theta_validation_loss.append(sum(theta_val_loss) / len(theta_val_loss))
            theta_validation_mae.append(sum(theta_val_mae)/len(theta_val_mae))

            phi_validation_loss.append(sum(phi_val_loss) / len(phi_val_loss))
            phi_validation_mae.append(sum(phi_val_mae)/len(phi_val_mae))
        LOSS = {
            'theta_training_loss': theta_training_loss,
            'theta_training_mae': theta_training_mae,
            'theta_validation_loss': theta_validation_loss,
            'theta_validation_mae': theta_validation_mae,

            'phi_training_loss': phi_training_loss,
            'phi_training_mae': phi_training_mae,
            'phi_validation_loss': phi_validation_loss,
            'phi_validation_mae': phi_validation_mae
        }
        torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
                }, "/kaggle/working/latest_alexnet_model_checkpoint.pt")
        if theta_min_loss > theta_validation_loss[-1] and phi_min_loss > phi_validation_loss[-1]:
            theta_min_loss = theta_validation_loss[-1]
            phi_min_loss = phi_validation_loss[-1]
            torch.save(model, "/kaggle/working/best_alexnet_model.pt")
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
                }, "/kaggle/working/best_alexnet_model_checkpoint.pt")

    end_time = time.time()
    print((end_time-start_time)//60, 'min',(end_time-start_time)-((end_time-start_time)//60)*60, 'sec')
    return LOSS, checkpoint_epoch


if __name__ =='__main__':
    # Initial Assignement
    torch.manual_seed(100)
    dataset = MPIIGaze(mpii_dir=".../Dataset/MPIIGaze")
    model = AlexNet()

    # Test train split
    train_data, val_data = test_train_split(dataset)

    # Creating dataloader
    train_data_loader= create_dataloader(train_data)
    val_data_loader = create_dataloader(val_data)

    # Training and performance visualization
    metrics, checkpoint_epoch = train_model(model,train_data_loader, val_data_loader)


