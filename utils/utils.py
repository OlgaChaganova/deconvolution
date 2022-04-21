import torch
import matplotlib.pyplot as plt

from sweet.util.fft import shift


MODEL_DIR = '/home/jovyan/learn-optimizer-rgdn/trained models/'
SHIFT=True


def viz_test(model: torch.nn.Module, 
             data_loader: torch.utils.data.DataLoader, 
             use_cuda: bool,
             is_rgb: bool,
             figsize: tuple=(25, 25)):
    """
    Vizualization on test dataset.
    
    :param model: pytorch model of neural network
    :param dataloader: dataloader for the test dataset
    :param use_cuda: use cuda or not
    :param is_rgb: if images have 3 channels or not (gray images)
    :param figsize: size of the  plt.figure
    """
    
    model.eval()
    
    y, x_gt, k, kt = next(iter(data_loader))
    bs = len(y)
    
    if use_cuda:
        y = y.cuda()
        k = k.cuda()
        kt = kt.cuda()
        model = model.cuda()
    else:
        model = model.cpu()
    
    res = model(y, k, kt)
    
    fig, axs = plt.subplots(nrows=bs, ncols=4, figsize=figsize)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)
    
    if is_rgb:
        for j in range(len(y)):
            x_hat = res[-1][j].detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            for i in range(4):
                axs[j][i].axis('off')
            
            if SHIFT:
                kj = shift(k[j].squeeze(0).cpu().numpy())
            else:
                kj = k[j].squeeze(0).cpu().numpy()

            axs[j][0].imshow(kj, cmap='gray')
            axs[j][0].set_title("PSF")

            axs[j][1].imshow(x_gt[j].permute(1, 2, 0).cpu().numpy())
            axs[j][1].set_title("Ground truth")

            axs[j][2].imshow(x_hat)
            axs[j][2].set_title("Recovered")

            axs[j][3].imshow(y[j].permute(1, 2, 0).cpu().numpy())
            axs[j][3].set_title("Blurred image")
    else:
        for j in range(len(y)):
            x_hat = res[-1][j].detach().cpu().squeeze(0).numpy()
            for i in range(4):
                axs[j][i].axis('off')
            
            if SHIFT:
                kj = shift(k[j].squeeze(0).cpu().numpy())
            else:
                kj = k[j].squeeze(0).cpu().numpy()
                
            axs[j][0].imshow(kj, cmap='gray')
            axs[j][0].set_title("PSF")

            axs[j][1].imshow(x_gt[j][0, :, :].cpu().numpy(), cmap='gray')
            axs[j][1].set_title("Ground truth")

            axs[j][2].imshow(x_hat, cmap='gray')
            axs[j][2].set_title("Recovered")

            axs[j][3].imshow(y[j][0, :, :].cpu().numpy(), cmap='gray')
            axs[j][3].set_title("Blurred image")
    plt.show() 
        

def save_trained_model(model: torch.nn.Module, 
                       opt: torch.optim.Optimizer,
                       epoch: int, 
                       model_name: str,
                       start_idx: int):
    """
    :param model: pytorch model to save
    :param opt: optimizer
    :param epoch: number of the epoch on which the model is saved
    :param: model_name: name of the model
    :param start_idx: number of epochs before (0 if model is trained from scratch)
    """
    
    try:
        torch.save({
            'state_dict': model.module.state_dict(),
            'optimizer_state_dict': opt.state_dict()
        }, MODEL_DIR + str(model_name) + "-" + str(start_idx + epoch))
        print('Model is DataParallel object.')
        
    except AttributeError:
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict()
        }, MODEL_DIR + str(model_name) + "-" + str(start_idx + epoch))
        
    name = str(model_name) + "-" + str(start_idx + epoch)
    print('\nModel\'s weights were saved. Name: ', name, '\n')

    
def load_trained_model(model: torch.nn.Module, 
                       opt: torch.optim.Optimizer,
                       name: str) -> (torch.nn.Module, torch.optim):  
    """
    :param model: pytorch model to save
    :param opt: optimizer
    :param: model_name: name of the file where the model is stored
    :return: model with pretrained weights, optimizer
    """
    
    path = MODEL_DIR + name
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer