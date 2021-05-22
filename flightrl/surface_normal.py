from os import stat
from os.path import supports_unicode_filenames
from numpy.lib.npyio import mafromtxt
import torch
from torch.functional import norm
#from torch._C import double
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
        nb_channels = x.shape[0]
        h, w = x.shape[-2:]
        #x = torch.randn(1, nb_channels, h, w)

        delzdelxkernel = torch.tensor([[0.00000, 0.00000, 0.00000],
                                        [-1.00000, 0.00000, 1.00000],
                                        [0.00000, 0.00000, 0.00000]])
        delzdelxkernel = delzdelxkernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
        delzdelx = F.conv2d(x, delzdelxkernel)

        delzdelykernel = torch.tensor([[0.00000, -1.00000, 0.00000],
                                        [0.00000, 0.00000, 0.00000],
                                        [0.0000, 1.00000, 0.00000]])
        delzdelykernel = delzdelykernel.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)

        delzdely = F.conv2d(x, delzdelykernel)

        delzdelz = torch.ones(delzdely.shape)

        print("cuda", delzdelx.is_cuda)
        mag = torch.sqrt(torch.square(delzdelx) + torch.square(delzdely) + torch.square(delzdelz))
        surface_norm = torch.stack((-delzdelx,-delzdely, delzdelz),2)
        print(mag)
        print("1. ", torch.div(-delzdelx, mag))
        print("2. ", torch.div(-delzdely,mag))
        print("3. ", torch.div(delzdelz,mag))

        print(surface_norm.shape, mag.shape)
        
        surface_norm = torch.div(surface_norm, mag)
        print('out',surface_norm)
        surface_norm_viz = torch.mul(torch.add(surface_norm, 1.00000),127 )
        return surface_norm_viz



#params = list(net.parameters())
# print(len(params))
# print(params[0].size())
# print(net)

def main():
    # b = torch.tensor([ 5.0,  5.0, 5.0, 5.0])
# a = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
#             [ 0.1815, -1.0111,  0.9805, -1.5923],
#             [ 0.1062,  1.4581,  0.7759, -1.2344],
#             [-0.1830, -0.0313,  1.1908, -1.4757]])
    net = Net()
    with open("depth_images.json",'r') as file_:
        data_dict = json.load(file_)
    time_idx = 15; idx = 1 

    
    img = np.array(data_dict[str(time_idx)][idx], dtype=np.float32)
    start = timeit.default_timer()
    normal_img = np.zeros((129,129,3),dtype=np.float32)
    for i in range(img.shape[0]-1):
        for j in range(img.shape[0]-1):
            if i ==0 or j==0:
                normal_img[i,j,0] = 0
                normal_img[i,j,1] = 0
                normal_img[i,j,2] = 0
            else:
                dzdx = img[i+1,j] - img[i-1,j]
                dzdy = img[i,j+1] - img[i,j-1]
                normal = np.zeros((3,1),dtype=np.float32)
                normal[0] = -dzdx
                normal[1] = -dzdy
                normal[2] = 1.0000

                norm = np.linalg.norm(normal)
                normal = normal/norm
                #normal = normal + 1.0
                #if(normal[2] > 0):
                #    print("no")
                normal_img[i,j,0] = int((normal[0]+1.00000)*127.5) #normal[0]#
                normal_img[i,j,1] = int((normal[1]+1.00000)*127.5) #normal[1]#
                normal_img[i,j,2] = int((normal[2]+1.00000)*127.5) #normal[2]#

    end = timeit.default_timer()     
    print("1",normal_img[1:-1,1:-1,0])
    print('2',normal_img[1:-1,1:-1,1])
    print('3',normal_img[1:-1,1:-1,2])
    print("time", end-start)
    #cv2.imwrite("surface_img_"+str(time_idx)+"__"+str(idx)+".jpg", normal_img)   
    
    
    
    
    
    
    
    
    
    
    
    start = timeit.default_timer()
    img = torch.from_numpy(img)
    img = img[None,None,:,:]
    print('image',img.shape)

    normal_output = 255*net.forward(img).numpy()[0,0,:,:,:]
    
    normal_img = np.zeros((126,126,1),dtype=float)
    normal_img[:,:,0] = (normal_output[0,:,:] +1.00)*127.5  
    #normal_img[:,:,1] = normal_output[1,:,:]
    #normal_img[:,:,2] = normal_output[2,:,:]
    print('normal',normal_output.shape)
    end = timeit.default_timer()
    print("torch time", end-start)
    cv2.imwrite("torch_surface_img_"+str(time_idx)+"__"+str(idx)+".jpg", normal_img)    
    #print(normal_img.channels())
    #normal_img = normal_img
    #

    
        
    #grayImage = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("gray_surface_img_"+str(time_idx)+"__"+str(idx)+".jpg", grayImage)
if __name__ == "__main__":
    pass
    main()