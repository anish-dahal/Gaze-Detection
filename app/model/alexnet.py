import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0)
        self.relu1 = nn.ReLU()
        self.response_Norm1 = nn.LocalResponseNorm(size = 5,k = 2)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size = 5,  stride = 1, padding = 'same')
        self.relu2 = nn.ReLU()
        self.response_Norm2 = nn.LocalResponseNorm(size = 5,k = 2)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.conv3 =  nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 'same')
        self.relu3 = nn.ReLU()

        self.conv4 =  nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 'same')
        self.relu4 = nn.ReLU()

        self.conv5 =  nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 'same')
        self.relu5 = nn.ReLU()
        self.response_Norm3 = nn.LocalResponseNorm(size = 5,k = 2)
        self.max_pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2)

        self.theta_dropout1 = nn.Dropout(0.5)
        self.theta_fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.theta_relu1 = nn.ReLU()

        self.phi_dropout1 = nn.Dropout(0.5)
        self.phi_fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.phi_relu1 = nn.ReLU()

        self.theta_dropout2 = nn.Dropout(0.5)
        self.theta_fc2 = nn.Linear(4096, 4096)
        self.theta_relu2 = nn.ReLU()

        self.phi_dropout2 = nn.Dropout(0.5)
        self.phi_fc2 = nn.Linear(4096, 4096)
        self.phi_relu2 = nn.ReLU()

        self.theta_fc3 = nn.Linear(4096,1)
        self.phi_fc3 = nn.Linear(4096,1)
    
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.response_Norm1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.response_Norm2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.response_Norm3(x)
        x = self.max_pool3(x)

        x = torch.flatten(x, start_dim=1)

        # Theta Side Head
        theta_x = self.theta_dropout1(x)
        theta_x = self.theta_fc1(theta_x)
        theta_x = self.theta_relu1(theta_x)

        theta_x = self.theta_dropout2(theta_x)
        theta_x = self.theta_fc2(theta_x)
        theta_x = self.theta_relu2(theta_x)
        
        theta_x = self.theta_fc3(theta_x)

        # Phi Side Head
        phi_x = self.phi_dropout1(x)
        phi_x = self.phi_fc1(phi_x)
        phi_x = self.phi_relu1(phi_x)

        phi_x = self.phi_dropout2(phi_x)
        phi_x = self.phi_fc2(phi_x)
        phi_x = self.phi_relu2(phi_x)
        
        phi_x = self.phi_fc3(phi_x)
        return theta_x, phi_x
