import torch
import numpy as np


def fft(inp, shape=None):
    if shape is None:
        shape = inp.shape
    spectrum = torch.fft.fft2(inp)
    return spectrum


def ifft(inp, shape=None):
    if shape is None:
        shape = inp.shape

    res = torch.fft.ifft2(inp)
    return res


def shift(inp, dim=None, inverse=False):
    """Shift image(s) to center
    Args:
        inp(np.ndarray or tf): input images
        axes(tuple): axes to shift across
            beware, if inp is array of images, default shift change the images order
        inverse(bool)
    """
    if inverse:
        return torch.fft.ifftshift(inp, dim=dim)
    else:
        return torch.fft.fftshift(inp, dim=dim)
    
    
def ishift(inp, dim=None):
    return shift(inp, dim=dim, inverse=True)


def fft_conv(inp1, inp2, use_numpy=True, scale_output=True):
    """
    inputs are shaped: [N, N] of [B, N, N]
    """
    if use_numpy:
        if inp1.shape != inp2.shape:
            kernel = np.fft.fftshift(inp2)
            sz = (inp1.shape[0] - kernel.shape[0], inp1.shape[1] - kernel.shape[1])  # total amount of padding
            kernel = np.pad(
                kernel,
                (((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)),
                'constant'
            )
            inp2 = np.fft.ifftshift(kernel)
        
        inp1_c = torch.from_numpy(inp1.copy()).type(torch.complex64)
        inp2_c = torch.from_numpy(inp2.copy()).type(torch.complex64)
        
    else:
        if inp1.shape != inp2.shape:
            kernel = shift(inp2)
            sz = (inp1.shape[-2] - kernel.shape[-2], inp1.shape[-1] - kernel.shape[-1])  # total amount of padding
            print(sz)
            kernel = torch.nn.functional.pad(
                kernel,
                ((sz[0]+1)//2, sz[0]//2, (sz[1]+1)//2, sz[1]//2),
                'constant',
            )
            inp2 = ishift(kernel)
            print(inp2.shape)
    
        inp1_c = inp1.type(torch.complex64)
        inp2_c = inp2.type(torch.complex64)
    
    batch_size = inp1_c.shape[0] if len(inp1_c.shape) == 4 else 1

    f1 = fft(inp1_c)
    f2 = fft(inp2_c, shape=inp1.shape)
    convolved = ifft(f1*f2, shape=inp1.shape)
    res = torch.abs(convolved)
    
    if use_numpy:
        if scale_output:
            return (res * np.sum(inp1) / torch.sum(res)).numpy()
        else:
            return res.numpy()
    else:
        if scale_output:
            return (res * torch.sum(inp1) / (torch.sum(res) * batch_size))
        else:
            return res


def scale(f):
    return torch.log(1 + torch.abs(f))



def retinal(image, psf):
    return fft_conv(image, psf)