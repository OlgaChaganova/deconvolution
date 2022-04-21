import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import image_gradients


###### ADDED #######
class GradLoss(torch.nn.Module): 
    def __init__(self):
        super(GradLoss, self).__init__()
        self.norm = lambda x: torch.linalg.norm(x, ord=1, dim=[2, 3])

    def forward(self, x_gt, x_curr):
        '''
        Parameters:
        - x_gt: ground truth image (BS x CH x H x W)
        - x_curr: current approximation of the x_gt (BS x CH x H x W)
        '''
        assert x_gt.shape == x_curr.shape, 'x_gt and x_curr must to be of the same dimensions'
        
        dx_gt_y, dx_gt_x = image_gradients(x_gt) # torch.metrics
        dx_curr_y, dx_curr_x = image_gradients(x_curr)
        
        loss = self.norm(dx_gt_y - dx_curr_y) + self.norm(dx_gt_x - dx_curr_x)
        
        return loss.mean()
#####################


class LogLoss(torch.nn.Module):
    def __init__(self):
        super(LogLoss, self).__init__()

    def forward(self, x, y):
        error = torch.pow(x - y, 2)
        sampleSize = error.size()
        error = error.view(sampleSize[0], -1)
        # print(error.size())
        mean_error = error.mean(1)
        # print(mean_error.size())
        logloss = torch.log(mean_error)
        return logloss.mean()


###### ADDED #######  
class DeconvLoss(torch.nn.Module):
    def __init__(self, tau=1):
        super(DeconvLoss, self).__init__()
        self.tau = tau # importance weight for the L_grad term
        self.L_mse = LogLoss()
        self.L_grad = GradLoss()

    def forward(self, x_gt, output):
        '''
        Parameters:
        - x_gt: ground truth image (BS x CH x H x W)
        - output: output of the model (list with <num_steps> elements, where each element has the shape of (BS x CH x H x W))
        '''
        S = len(output) # number of the model's steps
        loss = 0
        for i in range(1, S+1):
            loss += i*(self.L_mse(x_gt, output[i-1]) + self.tau * self.L_grad(x_gt, output[i-1]))

        return loss / S    
#####################    
        
    
class ImgDiffComputer(torch.nn.Module):
    def __init__(self):
        super(ImgDiffComputer, self).__init__()

        k = np.zeros([1, 1, 2, 1], dtype=np.float32)
        k[0, 0, 0, 0] = 1
        k[0, 0, 1, 0] = -1

        self.k = torch.nn.Parameter(
            torch.from_numpy(k), requires_grad=False)

    def forward(self, x):

        xsize = x.size()

        xr = x.view(-1, 1, xsize[2], xsize[3])

        diff1 = F.conv2d(xr, self.k, padding=0)
        diff2 = F.conv2d(xr, self.k.transpose(3, 2), padding=0)

        size1 = diff1.size()
        size2 = diff2.size()

        return diff1.view(-1, 3, size1[2],
                          size1[3]), diff2.view(-1, 3, size2[2], size2[3])
