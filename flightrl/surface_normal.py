import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import numpy as np
import timeit    
import cv2

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.convDelYDelZ = nn.Conv2d(1, 1, 3)
        self.convDelXDelZ = nn.Conv2d(1, 1, 3)


        #self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        nb_channels = 1
        h, w = 129, 129
        x = torch.randn(1, nb_channels, h, w)

        delzdelxweights = torch.tensor([[0., 0., 0.],
                                        [-1., 0., 1.],
                                        [0., 0., 0.]])
        delzdelxweights = delzdelxweights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
        delzdelx = F.conv2d(x, delzdelxweights)

        delzdelyweights = torch.tensor([[0., -1., 0.],
                                        [0., 0., 0.],
                                        [0., 1., 0.]])
        delzdelyweights = delzdelyweights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)

        delzdely = F.conv2d(x, delzdelyweights)

        #delxdelz = -self.convDelXDelZ(x)
        #delydelz = -self.convDelYDelZ(x)
        return x


net = Net()
params = list(net.parameters())
print(len(params))
print(params[0].size())
print(net)

def main():
    # b = torch.tensor([ 5.0,  5.0, 5.0, 5.0])
    # a = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
    #             [ 0.1815, -1.0111,  0.9805, -1.5923],
    #             [ 0.1062,  1.4581,  0.7759, -1.2344],
    #             [-0.1830, -0.0313,  1.1908, -1.4757]])
    with open("depth_images.json",'r') as file_:
        data_dict = json.load(file_)
    time_idx = 6; idx = 0 
    img = np.array(data_dict[str(time_idx)][idx], dtype=float)
    normal_img = np.zeros((192,192,1),dtype=float)
    start = timeit.default_timer()
    for i in range(img.shape[0]-1):
        for j in range(img.shape[0]-1):
            if i ==0 or j==0:
                normal_img[i,j,0] = 0
                #normal_img[i,j,1] = 0
                #normal_img[i,j,2] = 0
            else:
                dzdx = img[i+1,j] - img[i-1,j]
                dzdy = img[i,j+1] - img[i,j-1]
                normal = np.zeros((3,1),dtype=float)
                normal[0] = -dzdx
                normal[1] = -dzdy
                normal[2] = 1.0000

                norm = np.linalg.norm(normal)
                normal = normal/norm
                #normal = normal + 1.0
                #if(normal[2] > 0):
                #    print("no")
                normal_img[i,j,0] = int((normal[2]+1.0)*127.5)
                #normal_img[i,j,1] = int((normal[1]+1.0)*127.5)*0
                #normal_img[i,j,2] = int((normal[2]+1.0)*127.5)*0 
         
    cv2.imwrite("surface_img_"+str(time_idx)+"__"+str(idx)+".jpg", normal_img)           
    #grayImage = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("gray_surface_img_"+str(time_idx)+"__"+str(idx)+".jpg", grayImage)
if __name__ == "__main__":
    pass
    main()