from __future__ import print_function, absolute_import
import torch
import os
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data.make_kernel import kernel_sim_spline
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from pathlib import Path
from operator import itemgetter
from skimage.color import rgb2gray
import pyDOE as doe # LHS
from scipy.stats.distributions import uniform
from tqdm import tqdm
from skimage import util, filters
from scipy import signal

from sweet.sweet import Sweet
from sweet.util.fft import shift, fft_conv


fixed_params = {
    'pupil_diam' : 2.5,
    'A' : 0
}


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gkern(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D


class ToTensor(object):
    def __call__(self, sample):
        y, k, kt, x_gt = sample['y'], sample['k'], sample['kt'], sample['x_gt']
        img_ch_num = len(y.shape)
        if img_ch_num == 2:
            # x0 = y
            # x_gt = x_gt
            y = y
        elif img_ch_num == 3:
            # x0 = y
            # x0 = x0.transpose(2, 0, 1)
            # x_gt = x_gt.transpose((2, 0, 1))
            y = y.transpose((2, 0, 1))

        return torch.from_numpy(y).float(), \
            torch.from_numpy(k.reshape(1, k.shape[0], k.shape[1])).float(), \
            torch.from_numpy(kt.reshape(1, k.shape[0], k.shape[1])).float()


class BlurryImageDataset(Dataset):
    """Blur image dataset -- implemented in the original paper"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_name_list = [
            name for name in os.listdir(self.root_dir)
            if os.path.isfile(os.path.join(self.root_dir, name)) and name.endswith('.mat')
        ]
        # print(self.file_name_list)
        self.file_name_list.sort()
        self.TensorConverter = ToTensor()

    def __len__(self):
        return len([name for name in os.listdir(self.root_dir) \
                    if os.path.isfile(os.path.join(self.root_dir, name)) and name.endswith('.mat') ])

    def __getitem__(self, idx):
        """get .mat file"""
        mat_name = self.file_name_list[idx]
        sample = sio.loadmat(os.path.join(self.root_dir, mat_name))
        if self.transform:
            sample = self.transform(sample)

        return self.TensorConverter(sample), mat_name

    
    
class BlurryImageDatasetOnTheFlySweet(Dataset):
    '''
    Implemented to use an eye's PSF from Sweet.
    PSF is not cropped.
    Convolution is perfomed with fft_conv.
    '''
    def __init__(self,
                 file_name_list,
                 is_rgb=True,
                 k_size=256,
                 patch_size=256,
                 max_num_images=None):
        assert is_rgb == False, 'RGB images currently not supported'
        
        self.file_name_list = sorted(file_name_list)
        self.is_rgb = is_rgb
        self.k_size = k_size
        self.patch_size = 256
        self.params = []
        self.num_params = 0

        if max_num_images is not None and max_num_images < len(
                self.file_name_list):
            self.file_name_list = self.file_name_list[:max_num_images]
        
    @staticmethod
    def get_psf(S, A, C, pupil_diam):
        sweet = Sweet()
        sweet.set_eye_prescription(
            S = S,  
            A = A,
            C = C,  
        )
        sweet.set_experiment_params(
            pupil_diam = pupil_diam,  
            view_dist = 100.0,      
            canvas_size_h = 10.0,  
        )
        psf = sweet._psf()
        return psf
    
    
    def params_init(self, params):
        self.params = params
        self.num_params = len(params)
      
        
    def params_random_init(self, num_params): #LHS      
        params = doe.lhs(4, samples=num_params) # order: S, A, C, pupil_diam
        loc =   [-8, fixed_params['A'], -3, fixed_params['pupil_diam']]
        scale = [4,  fixed_params['A'],  3, fixed_params['pupil_diam']]
        for j in range(4):
            if j == 1:
                params[:, j] = fixed_params['A']
            elif j == 3:
                params[:, j] = fixed_params['pupil_diam']
            else:
                params[:, j] = uniform(loc=loc[j], scale=scale[j]).ppf(params[:, j])
        params = params.tolist()
        for i in tqdm(range(num_params)):
            psf = BlurryImageDatasetOnTheFlySweet.get_psf(*params[i])
            psf = psf / psf.sum()        ###### ADD NORMALIZING
            params[i].append(psf)   
        self.params = params
        self.num_params = num_params

        
    def __len__(self):
        return len(self.file_name_list)

    
    def __getitem__(self, idx):
        i = np.random.choice(np.arange(self.num_params))
        k = self.params[i][-1]
        img_name = self.file_name_list[idx]
        sample = plt.imread(img_name)

        if sample.shape[0] < self.patch_size or sample.shape[
                1] < self.patch_size:
            return self.__getitem__((idx - 1) % (self.__len__()))
        patches = image.extract_patches_2d(sample,
                                           [self.patch_size, self.patch_size],
                                           max_patches=1)
        sample = patches[0, ...]
        sample = sample.astype(np.float32) / 255.0
        if not self.is_rgb:
            sample = rgb2gray(sample)
        y = fft_conv(sample, k)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        x_gt = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        k = torch.tensor(k, dtype=torch.float32).unsqueeze(0)
        kt = torch.flip(k, [1, 2])
        return y, x_gt, k, kt     
    
    

# class BlurryImageDatasetOnTheFlySweet(Dataset):
#     '''
#     Uses an eye's PSF from Sweet.
#     PSF is not cropped.
#     Convolution is perfomed with fft_conv.
#     '''
#     def __init__(self,
#                  file_name_list,
#                  is_rgb=True,
#                  k_size=256,
#                  patch_size=256,
#                  max_num_images=None):
#         assert is_rgb == False, 'RGB images currently not suooorted'
        
#         self.file_name_list = sorted(file_name_list)
#         self.is_rgb = is_rgb
#         self.k_size = k_size
#         self.patch_size = 256
#         self.params = []
#         self.num_params = 0

#         if max_num_images is not None and max_num_images < len(
#                 self.file_name_list):
#             self.file_name_list = self.file_name_list[:max_num_images]
        
        
#     def get_psf(self, S, A, C, pupil_diam):
#         sweet = Sweet()
#         sweet.set_eye_prescription(
#             S = S,  
#             A = A,
#             C = C,  
#         )
#         sweet.set_experiment_params(
#             pupil_diam = pupil_diam,  
#             view_dist = 100.0,      
#             canvas_size_h = 10.0,  
#         )
#         psf = sweet._psf()
#         return psf
    
    
#     def params_init(self, params):
#         self.params = params
#         self.num_params = len(params)
      
        
#     def params_random_init(self, num_params): #LHS
#         params = doe.lhs(4, samples=num_params) # order: S, A, C, diam
#         loc = [-3, 0, -3, 2]
#         scale = [2, 180, 2, 2]
#         for j in range(4):
#             params[:, j] = uniform(loc=loc[j], scale=scale[j]).ppf(params[:, j])
#         params = params.tolist()
#         for i in tqdm(range(num_params)):
#             psf = self.get_psf(*params[i])
#             psf = psf / psf.sum()        ###### ADD NORMALIZING
#             params[i].append(psf)   
#         self.params = params
#         self.num_params = num_params

        
#     def __len__(self):
#         return len(self.file_name_list)

    
#     def __getitem__(self, idx):
#         i = np.random.choice(np.arange(self.num_params))
#         k = self.params[i][-1]
#         img_name = self.file_name_list[idx]
#         sample = plt.imread(img_name) # uint8

#         if sample.shape[0] < self.patch_size or sample.shape[
#                 1] < self.patch_size:
#             return self.__getitem__((idx - 1) % (self.__len__()))
#         patches = image.extract_patches_2d(sample,
#                                            [self.patch_size, self.patch_size],
#                                            max_patches=1)
#         sample = patches[0, ...]
#         sample = sample.astype(np.float32) / 255.0
#         if not self.is_rgb:
#             sample = rgb2gray(sample)
#         y = fft_conv(sample, k)
#         y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
#         x_gt = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
#         k = torch.tensor(k, dtype=torch.float32).unsqueeze(0)
#         kt = torch.flip(k, [1, 2])
#         return y, x_gt, k, kt      


    
    
# class BlurryImageDatasetOnTheFly(Dataset):
#     '''
#     Might be used a random PSF from kernel or an eye's PSF form Sweet.
#     In case of Sweet PSF is cropped to the size of (k_size, k_size).
#     Convolution is perfomed with F.conv2d.
#     '''
#     def __init__(self,
#                  file_name_list,
#                  is_rgb=True,
#                  k_size=41,
#                  k_from_sweet=False,
#                  sp_size=[11, 16, 21, 26, 31],
#                  num_spl_ctrl=[3, 4, 5, 6],
#                  patch_size=256,
#                  max_num_images=None,
#                  sigma=None,
#                  use_gauss_blur=False):
        
#         self.file_name_list = sorted(file_name_list)
#         self.is_rgb = is_rgb
#         self.k_size = k_size
#         self.k_from_sweet = k_from_sweet
#         self.sp_size = sp_size
#         self.num_spl_ctrl = num_spl_ctrl
#         self.patch_size = 256
#         self.params = []
#         self.num_params = 0
#         self.use_gauss_blur=use_gauss_blur
#         self.sigma = sigma

# #         self.file_name_list = [name for name in os.listdir(self.root_dir)
# #                                if os.path.isfile(os.path.join(self.root_dir, name))]
# #         self.file_name_list.sort()

#         if max_num_images is not None and max_num_images < len(
#                 self.file_name_list):
#             self.file_name_list = self.file_name_list[:max_num_images]
        
        
#     def get_psf(self, S, A, C, pupil_diam):
#         sweet = Sweet()
#         sweet.set_eye_prescription(
#             S = S,  # Spherical refraction error (in diopters)
#             A = A,  # Axis of the cylindrical refraction error (in degrees)
#             C = C,  # Cylindrical refraction error (in diopters)
#         )
#         sweet.set_experiment_params(
#             pupil_diam = pupil_diam,  # Pupil diameter in the experiment (in mm)
#             view_dist = 100.0,  # Viewing distance, how far is the participant from the monitor (in cm)
#             canvas_size_h = 10.0,  # Canvas size,how big is image (with padding) on monitor (in cm)
#         )
#         psf = sweet._psf()
#         return psf
    
    
#     def params_init(self, params):
#         self.params = params
#         self.num_params = len(params)
      
        
#     def params_random_init(self, num_params): #LHS; 0.7 sec
#         params = doe.lhs(4, samples=num_params) # order: S, A, C, diam
#         loc = [-3, 0, -3, 2]
#         scale = [2, 180, 2, 2]
        
#         for j in range(4):
#             params[:, j] = uniform(loc=loc[j], scale=scale[j]).ppf(params[:, j])
#         params = params.tolist()
        
#         for i in tqdm(range(num_params)):
#             psf = self.get_psf(*params[i])
#             params[i].append(psf)   
#         self.params = params
#         self.num_params = num_params

        
#     def __len__(self):
#         return len(self.file_name_list)

    
#     def __getitem__(self, idx):
#         sp_size = int(np.random.choice(self.sp_size))
#         num_spl_ctrl = int(np.random.choice(self.num_spl_ctrl))
# #         print(f'sp_size: {sp_size}, num_spl_ctrl:{num_spl_ctrl}')
#         if not self.k_from_sweet and not self.use_gauss_blur:
#             k = kernel_sim_spline(sp_size, self.k_size, num_spl_ctrl, 1)
#             if self.sigma is not None:
#                 k = filters.gaussian(k, sigma=self.sigma)
#         elif self.k_from_sweet:
#             i = np.random.choice(np.arange(self.num_params))
#             k = self.params[i][-1]
#             if self.k_size < k.shape[0]:
#                 # shift
#                 k = shift(k)
#                 #crop
#                 crop_width = (k.shape[0] - self.k_size)//2
#                 k = util.crop(k, crop_width=crop_width)[:self.k_size, :self.k_size]
#             k = k / k.sum()
#         elif self.use_gauss_blur:
#             k = gkern(self.k_size, self.sigma)
#             k = k / k.sum()
                
#         k = np.reshape(k, [1, 1, self.k_size, self.k_size])
#         k = k.astype(np.float32)
            
#         img_name = self.file_name_list[idx]
# #         sample = plt.imread(os.path.join(self.root_dir, img_name))
#         sample = plt.imread(img_name)

#         if sample.shape[0] < self.patch_size or sample.shape[
#                 1] < self.patch_size:
#             return self.__getitem__((idx - 1) % (self.__len__()))
#         patches = image.extract_patches_2d(sample,
#                                            [self.patch_size, self.patch_size],
#                                            max_patches=1)
# #         print((sample != patches).sum())
#         sample = patches[0, ...]
#         sample = sample.astype(np.float32) / 255.0
#         if not self.is_rgb:
#             sample = rgb2gray(sample)
#             sample = np.expand_dims(sample, 2)
            
#         sample = np.expand_dims(np.transpose(sample, [2, 0, 1]), 1)
#         sample = torch.from_numpy(sample.astype(np.float32))  # n x c x w x h
#         hks = (self.k_size) // 2

#         with torch.no_grad():
#             k = torch.from_numpy(k)
# #             sample = torch.nn.functional.pad(sample, (hks, hks, hks, hks),
# #                                         mode='replicate')
#             y = torch.nn.functional.conv2d(sample, k) # convolution
#             nl = np.random.uniform(0.003, 0.015)
#             y = y + nl * torch.randn_like(y)  # adding noise
#             if not self.k_from_sweet and not self.use_gauss_blur:
#                 y = torch.clamp(y * 255.0, 0, 255)
#                 y = y.type(torch.ByteTensor)
#                 y = y.float() / 255.0
# #                 y = torch.nn.functional.pad(y, (hks, hks, hks, hks),
# #                                             mode='replicate')
#             else:
# #                 print(y.shape)  # [1, 1, 216, 216]
#                 y = y / y.max()
#             y = y.squeeze(1)
#             x_gt = sample.squeeze(1)[:, hks:(-hks), hks:(-hks)]
#             k = k.squeeze(0)
#             kt = torch.flip(k, [1, 2])

#         return y, x_gt, k, kt     
    
    
def get_data(data_dir: str,
             is_silent: bool = False) -> (list, list, list):
    """
    :param data_dir: path to data
    :param is_silent: silent mode or not
    :return: lists with paths to train, valid and test files
    """
    train_dir = Path(os.path.join(data_dir, 'train'))
    test_dir = Path(os.path.join(data_dir, 'test'))

    train_files = list(train_dir.rglob('*.jpg'))
    test_files = list(test_dir.rglob('*.jpg'))
        
    ids = np.random.permutation(len(test_files))
    N = len(test_files) // 2

    valid_files = list(itemgetter(*ids[:N])(test_files))
    test_files = list(itemgetter(*ids[N:])(test_files))
    
    if not is_silent:
        print('Files are loaded.')
        print('Train size: ', len(train_files), '\t', 'Valid size: ', len(valid_files), '\t', 'Test size: ', len(test_files))
    
    return train_files, valid_files, test_files 



def get_dataloaders(train_files: list,
                    valid_files: list,
                    dataset: torch.utils.data.Dataset,
                    num_params: int,
                    params_filename: str) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """
    Function to get dataloaders (works only with Sweet dataset).
    
    :param: train_files: list with paths to train data
    :param: valid_files: list with paths to valid data
    :param: dataset: object of Dataset class
    :param: num_params: number of eye parameters generated on each epoch
    :param: params_filename: name of file to save params generated during the training
    :return: train dataloder, valid dataloader
    """
    train_dataset = dataset(file_name_list=train_files,
                            is_rgb=False,
                            k_size=256)
    train_dataset.params_random_init(num_params)

    valid_dataset = dataset(file_name_list=valid_files,
                            is_rgb=False,
                            k_size=256)
    valid_dataset.params_init(train_dataset.params)  

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, drop_last=False)
    
    save_params(params_filename, train_dataset)
    
    return train_loader, valid_loader


def save_params(filename, dataset):
    with open(filename, 'a') as f:
        for param in dataset.params:
            for p in param:
                f.write(str(p))
                f.write('\n')
            f.write('\n')