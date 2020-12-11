import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import colors

import logging
import warnings

from .vis import create_color_map
from .config import config
# from tensorshow import config_dict

logger = logging.getLogger('TensorShow')
logger.setLevel(logging.INFO)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# def display(vis, backend='plt', mode='image'):
#     if backend=='matplotlib':
#         try:
#             from matplotlib import pyplot as plt 
#         except ImportError:
#             raise Exception("The backend is set to \"plt\" but Matplotlib was not installed. Try to install using \'pip install matplotlib\'.")
#         display_plt(vis, mode)
#     else:
#         raise NotImplementedError("Currently only support plt as display backend.")

def display_plt(vis, mode, cmap=None, **kwargs):
    if mode == 'image':
        plt.imshow(vis)
    elif mode == 'grayscale':
        plt.imshow(vis, cmap='gray')
    elif mode == 'categorical_mask':
        plt.imshow(vis, cmap=cmap)
    if not config.get('inline'):
        plt.show()


def show(x, 
         mode='auto',
         auto_permute=True,
         display=True,
         **kwargs):
    try:
        import torch 
    except ImportError:
        raise Exception("Pytorch installation not found.")
    
    if isinstance(x, (torch.Tensor)):
        x = x.detach().clone().cpu().numpy()
    elif isinstance(x, (np.ndarray)):
        x = x.copy()
    else:
        raise NotImplementedError("Does not support input type \"{}\"".format(type(x)))
    
    shape = x.shape
    ndim = len(shape)
    if ndim > 3:
        raise TypeError("tensor with more than 3 dimensions are currently not supported.")
    
    if auto_permute:
        if (ndim == 3) and (shape[0] in [1, 2, 3]): # For C, H, W kind of array.
            logger.debug('Detected input shape {} is in CHW format, TensorShow will automatically convert it to HWC format'.format(shape))
            x = np.transpose(x, (1,2,0))

    if ndim == 2:
        x = np.expand_dims(x, axis=-1)

    shape = x.shape

    if mode == 'auto':
        if shape[-1] == 3:
            
            return vis_image(x, display=display, **kwargs)
        if shape[-1] == 2:
            return vis_flow(x, display=display, **kwargs)
        if shape[-1] == 1:
            if (x.min() >= 0) and (x.max() <= 1):
                return vis_grayscale(x, display=display, **kwargs)
            if isinteger(np.unique(x)).all(): # If values are all integer
                return vis_categorical_mask(x, display=display, **kwargs)
        
    elif mode.lower()=='image':
        return vis_image(x, display=display, **kwargs)
    elif mode.lower()=='grayscale':
        return vis_grayscale(x, display=display, **kwargs)
    elif mode.lower()=='categorical_mask':
        raise vis_categorical_mask(x, display=display, **kwargs)
    elif mode.lower()=='flow':
        raise vis_flow(x, display=display, **kwargs)
    else:
        raise ValueError("mode {} is not supported.".format(mode))
        
        
def isinteger(x):
    """
    Function to check if np array has integer. Work regardless of type.
    e.g. isinteger(np.array([0., 1.5. 1.])) >>> array([ True, False,  True], dtype=bool)
    
    """
    return np.equal(np.mod(x, 1), 0)
        

def within_0_1(x):
    return (x.min() >= 0) and (x.max() <= 1)


def within_0_255(x):
    return (x.min() >= 0) and (x.max() <= 255)

        
def auto_unnormalize_image(x):
    all_int = isinteger(np.unique(x)).all()
    range_0_1 = within_0_1(x)
    range_0_255 = within_0_255(x)
    has_negative = (x.min() < 0)
    
    if has_negative:
        logger.debug('Detects input has negative values, auto rescaling input to 0-1.')
        return rescale_0_1(x)
    if range_0_255 and all_int and (not range_0_1): # if image is all integer and between 0 - 255. Normalize it to 0-1.
        logger.debug('Detects input are all integers within range 0-255. Divided all values by 255.')
        return x / 255.
    if range_0_1:
        logger.debug('Inputs already within 0-1, no unnormalization is performed.')
        return x
    logger.debug('Auto rescaling input to 0-1.')
    return rescale_0_1(x)


def rescale_0_1(x):
    """
    Rescaling tensor to 0-1 using min-max normalization
    """
    return (x - x.min()) / (x.max() - x.min())

        
def unnormalize(x, mean=None, std=None):
    """
    General Channel-wise mean std unnormalization. Expect input to be (H, W, C)
    """
    assert (len(x.shape) == 3), "Unnormalization only support (H, W, C) format input, got {}".format(x.shape)
    C = x.shape[-1]
    assert (len(mean) == C) and (len(std) == C), "Number of mean and std values must equals to number of channels."
    
    for i in range(C):
        x[:,:,i] = x[:,:,i] * mean[i] - std[i]
        
    return x
        
        
def vis_image(x, 
              unnormalize='auto',
              display=True,
              **kwargs):
    """
    : x: ndarray (H, W, 3).
    : unnormalize: 'auto', 'imagenet', 'imagenet_scaled'
    : mean: image mean for unnormalization.
    : std: image std for unnormalization
    : display: whether to display image using matplotlib
    """
    shape = x.shape
    ndim = len(shape)
    assert ndim == 3, "vis_image only support 3D array in (H, W, C) format."
    
    user_mean = config.get('image_mean')
    user_std = config.get('image_std')
    
    if (user_mean is not None) or (user_std is not None):
        if user_mean == None:
            user_mean = [0] * x.shape[-1] # Initialize mean to 0 if not specified.
        if user_std == None:
            user_std = [1.] * x.shape[-1] # Initialize std to 1 if not specified.
        x = unnormalize(x, user_mean, user_std)
        
    elif unnormalize=='auto':
        x = auto_unnormalize_image(x)
        
    elif unnormalize=='imagenet':
        if (x.max() > 2.66) or (x.min() < -2.17): 
            # A quick validation to check if the image was normalized to 0-1 
            # before substracting imagenet mean and std
            x = x / 255.
        x = unnormalize(x, IMAGENET_MEAN, IMAGENET_STD)
        
    else:
        raise NotImplementedError("Unsupported unnormalization profile \"{}\"".format(unnormalize))
    
    assert x is not None
    
    if config.get('color_mode') == 'bgr':
        x = x[:,:,::-1]
    
    if display:
        display_plt(x, mode='image', **kwargs)

    
def vis_flow(x):
    pass


def vis_grayscale(x, display=True, **kwargs):
    assert (len(x.shape) == 3) and (x.shape[-1] == 1)
    x = np.squeeze(x, -1)
    
    # rescale to [0-1]
    if not within_0_1(x):
        warnings.warn('Original input range is not 0-1 when using grayscale mode. Auto-rescaling it to 0-1 by default.')
        x = rescale_0_1(x)
    
    if display:
        display_plt(x, mode='grayscale', **kwargs)
    


def vis_categorical_mask(x, display=True, max_N=256):
    assert (len(x.shape) == 3) and (x.shape[-1] == 1)
    assert isinteger(np.unique(x)).all(), "Input has to contain only integers in categorical mask mode."
    
    N = int(x.max()) + 1
    
    if x.max() > max_N:
        warnings.warn('The maximum value in input is {} which is greater than the default max_N ({}), TensorShow will automatically adjust max_N to {}.'.format(x.max(), max_N, x.max()))
        max_N = x.max() + 1
        
    color_list = create_color_map(N=max_N, normalized=True)
    color_list = np.concatenate([color_list, np.ones((max_N, 1)).astype(np.float32)], axis=1)
    
    if x.min() < 0:
        warnings.warn('Input has negative value when trying to visualize as categorical mask, which will all be converted to -1 and displayed in white.')
        x[x<0] = -1 # Map all negative value to -1
        color_list = np.concatenate([np.ones((1,4)).astype(np.float32), color_list], axis=0) # appending an extra value.
        N = N + 1
    
    cmap = colors.ListedColormap(color_list, N=N)
    if display:
        display_plt(x, mode='categorical_mask', cmap=cmap)