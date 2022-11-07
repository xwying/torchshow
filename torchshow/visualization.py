import numpy as np
import matplotlib
from matplotlib import colors, animation, pyplot as plt, rcParams
# from PIL import Image
from .utils import isinteger, within_0_1, within_0_255, isnotebook
from .config import config
from .flow import flow_to_color
import warnings
import logging
from datetime import datetime
import os
import copy
from numbers import Number


logger = logging.getLogger('TorchShow')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

AXES_TITLE_FONTDICT = {'fontsize': rcParams['axes.titlesize'],
                       'fontweight': rcParams['axes.titleweight'],
                       'color': rcParams['axes.titlecolor'],
                       'verticalalignment': 'bottom',
                       'horizontalalignment': 'center'}


def create_color_map(N=256, normalized=False):
    
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def set_window_title(fig, title):
    """
    Set the title of the figure window (effective when using a interactive backend.)
    """
    # fig.canvas.set_window_title(title)
    if matplotlib.__version__ < '3.4':
        fig.canvas.set_window_title(title)
    else:
        fig.canvas.manager.set_window_title(title)

def imshow(ax, vis, alpha=None, extent=None, show_rich_info=True):
    max_rows, max_cols = vis['raw'].shape[:2]
    
    def format_coord(x, y):
        """
        We display x-y coordinate as integer.
        """
        col = int(x + 0.5)
        row = int(y + 0.5)
        if not (0<=col<max_cols and 0<=row<max_rows):
            return ""
        return 'Mode='+vis['mode'] + ', Shape='+vis['shape'] + ', X=%1d, Y=%1d' % (col, row)
    
    def prepare_data(data):
        try:
            data[0]
        except (TypeError, IndexError):
            data = [data]
        return data
    
    def get_cursor_data(event):
        col = int(event.xdata + 0.5)
        row = int(event.ydata + 0.5)
        if not (0<=col<max_cols and 0<=row<max_rows):
            return ""
        raw_data = prepare_data(vis['raw'][row, col])
        disp_data = prepare_data(vis['disp'][row, col])
        
        return raw_data, disp_data
    
    def format_cursor_data(data):
        try:
            raw, disp = data
        except:
            return ""
        raw_data_str = ', '.join('{:0.3g}'.format(item) for item in raw
                                 if isinstance(item, Number))
        disp_data_str = ', '.join('{:0.3g}'.format(item) for item in disp
                                 if isinstance(item, Number))
        return "Raw=[{}], Disp=[{}]".format(raw_data_str, disp_data_str)
        # return "testest [" + str(data) + "]"
    
    artist = ax.imshow(vis['disp'], alpha=alpha, extent=extent, **vis['plot_cfg'])
    
    if show_rich_info:
        artist.get_cursor_data = get_cursor_data
        artist.format_cursor_data = format_cursor_data
        ax.format_coord = format_coord


def display_plt(vis_list, **kwargs):
    nrows = len(vis_list)
    ncols = max([len(l) for l in vis_list])
    show_axis = kwargs.get('show_axis', False)
    tight_layout = kwargs.get('tight_layout', True)
    suptitle = kwargs.get('suptitle', None)
    axes_title = kwargs.get('axes_title', None)
    # show_title = kwargs.get('show_title', False)
    # title_pattern = kwargs.get('title_pattern', "{img_id}")
    figsize = kwargs.get('figsize', None)
    dpi = kwargs.get('dpi', None)
    title_namespace = {}
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    show_rich_info = config.get('show_rich_info')
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    set_window_title(fig, 'TorchShow')
    if suptitle:
        fig.suptitle(suptitle)
    
    for i, plots_per_row in enumerate(vis_list):
        for j, vis in enumerate(plots_per_row):
            # axes[i, j].imshow(vis, **plot_cfg)
            imshow(axes[i,j], vis, show_rich_info=show_rich_info)
            title_namespace["img_id"] = i*ncols+j
            title_namespace["img_id_from_1"] = title_namespace["img_id"] + 1
            title_namespace["row"] = i
            title_namespace["column"] = j
            if axes_title is not None:
                axes[i, j].set_title(axes_title.format(**title_namespace))
           
    if not show_axis:
        for ax in axes.ravel():
            ax.axis('off')
            
    if tight_layout:
        fig.tight_layout()
    
    if kwargs.get('save', False):
        file_path = kwargs.get('file_path', None)
        if file_path is None:
            os.makedirs('_torchshow', exist_ok=True)
            cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            file_path = '_torchshow/'+cur_time+'.png'
        dirname = os.path.dirname(file_path)
        if dirname != '':
            os.makedirs(dirname, exist_ok=True)
        fig.savefig(file_path, bbox_inches = 'tight', pad_inches=0)
        plt.close(fig)
    else: # If the image is saved by ts.save() it will not call plt.show()
        if not isnotebook():
            plt.show()
    

def overlay_plt(vis_list, alpha, save_as, extent, **kwargs):
    show_axis = kwargs.get('show_axis', False)
    tight_layout = kwargs.get('tight_layout', True)
    suptitle = kwargs.get('suptitle', None)
    axes_title = kwargs.get('axes_title', None)
    
    figsize = kwargs.get('figsize', None)
    dpi = kwargs.get('dpi', None)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot()
    set_window_title(fig, 'TorchShow')
    if suptitle:
        fig.suptitle(suptitle)
    
    assert len(vis_list) == len(alpha)
    
    for vis, a in zip(vis_list, alpha):
        imshow(ax, vis, alpha=a, extent=extent, show_rich_info=False)
        
    if not show_axis:
        ax.axis('off')
            
    if tight_layout:
        fig.tight_layout()
        
    if save_as != None:
        assert isinstance(save_as, str)
        dirname = os.path.dirname(save_as)
        if dirname!='':
            os.makedirs(dirname, exist_ok=True)
        fig.savefig(save_as, bbox_inches = 'tight', pad_inches=0)
        plt.close(fig)
    else:
        if not isnotebook():
            plt.show()
            
        
        

def animate_plt(video_list, **kwargs):
    """
    : video_list: [t, row, col, image]
    """
    nrows = len(video_list[0])
    ncols = max([len(l) for l in video_list[0]])
    
    show_axis = kwargs.get('show_axis', False)
    tight_layout = kwargs.get('tight_layout', True)
    suptitle = kwargs.get('suptitle', None)
    # show_title = kwargs.get('show_title', False)
    # title_pattern = kwargs.get('title_pattern', "{img_id}")
    # title_namespace = {}
    figsize = kwargs.get('figsize', None)
    dpi = kwargs.get('dpi', None)
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    set_window_title(fig, 'TorchShow')
    if suptitle:
        fig.suptitle(suptitle)

    plots = []
    
    # Initialization
    for i, plots_per_row in enumerate(video_list[0]):
        for j, vis in enumerate(plots_per_row):
            if vis is not None:
                plot = axes[i, j].imshow(vis['disp'], **vis['plot_cfg'])
                plots.append(plot)
            else:
                plots.append(None)
    
    if not show_axis:
        for ax in axes.ravel():
            ax.axis('off')
            
    if tight_layout:
        fig.tight_layout()            
    
    def run(frames_at_t):
        for i, plots_per_row in enumerate(frames_at_t):
            for j, vis in enumerate(plots_per_row):
                # axes[i, j].imshow(vis, **plot_cfg)
                if vis is not None:
                    # axes[i,j].figure.canvas.draw()
                    plots[i*ncols+j].set_data(vis['disp'])
        fig.canvas.draw()
        return plots
           
    # if tight_layout:
    #     fig.tight_layout()
    
    ani = animation.FuncAnimation(fig, run, video_list, blit=True, interval=5, repeat=True)
    
    if not config.get('inline'):
        plt.show()
    return ani


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

        
def unnormalize_with_mean_and_std(x, mean, std):
    """
    General Channel-wise mean std unnormalization. Expect input to be (H, W, C)
    """
    x = x.copy()
    assert (len(x.shape) == 3), "Unnormalization only support (H, W, C) format input, got {}".format(x.shape)
    C = x.shape[-1]
    assert (len(mean) == C) and (len(std) == C), "Number of mean and std values must equals to number of channels."
    
    for i in range(C):
        x[:,:,i] = x[:,:,i] * std[i] + mean[i]
        
    return x
        
        
def vis_image(x, unnormalize='auto', **kwargs):
    """
    : x: ndarray (H, W, 3).
    : unnormalize: 'auto', 'imagenet', 'imagenet_scaled'
    : mean: image mean for unnormalization.
    : std: image std for unnormalization
    : display: whether to display image using matplotlib
    """
    vis = dict()
    vis['raw'] = copy.deepcopy(x)
    vis['shape'] = str(x.shape)
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
        x = unnormalize_with_mean_and_std(x, user_mean, user_std)
        
    elif unnormalize=='auto':
        x = auto_unnormalize_image(x)
        
    elif unnormalize=='imagenet':
        if (x.max() > 2.66) or (x.min() < -2.17): 
            # A quick validation to check if the image was normalized to 0-1 
            # before substracting imagenet mean and std
            x = x / 255.
        x = unnormalize_with_mean_and_std(x, IMAGENET_MEAN, IMAGENET_STD)
        
    else:
        raise NotImplementedError("Unsupported unnormalization profile \"{}\"".format(unnormalize))
    
    assert x is not None
    
    if config.get('color_mode') == 'bgr':
        x = x[:,:,::-1]
        vis['mode'] = 'Image(BGR)'
    else:
        vis['mode'] = 'Image(RGB)'
    
    plot_cfg = dict()
    
    vis['disp'] = x
    vis['plot_cfg'] = plot_cfg
    return vis

    
def vis_flow(x, **kwargs):
    vis = dict()
    vis['raw'] = copy.deepcopy(x)
    vis['shape'] = str(x.shape)
    x = flow_to_color(x)
    plot_cfg = dict()
    vis['disp'] = x
    vis['plot_cfg'] = plot_cfg
    vis['mode'] = 'Flow'
    return vis


def vis_grayscale(x, **kwargs):
    vis = dict()
    assert (len(x.shape) == 3) and (x.shape[-1] == 1)
    x = np.squeeze(x, -1)
    vis['raw'] = copy.deepcopy(x)
    vis['shape'] = str(x.shape)
    # rescale to [0-1]
    if not within_0_1(x):
        warnings.warn('Original input range is not 0-1 when using grayscale mode. Auto-rescaling it to 0-1 by default.')
        x = rescale_0_1(x)
    vis['disp'] = x
    cmap = kwargs.get("cmap", "gray")
    plot_cfg = dict(cmap=cmap)
    
    if isinteger(np.unique(x)).all():
        plot_cfg['interpolation'] = 'nearest'
        vis['mode'] = 'Binary'
    else:
        vis['mode'] = 'Gray'
    vis['plot_cfg'] = plot_cfg
    
    return vis


def vis_categorical_mask(x, max_N=256, **kwargs):
    assert (len(x.shape) == 3) and (x.shape[-1] == 1)
    assert isinteger(np.unique(x)).all(), "Input has to contain only integers in categorical mask mode."
    vis = dict()
    
    x = np.squeeze(x, -1)
    vis['raw'] = copy.deepcopy(x)
    vis['shape'] = str(x.shape)
    N = int(x.max()) + 1
    
    if x.max() > max_N:
        warnings.warn('The maximum value in input is {} which is greater than the default max_N ({}). Automatically adjust max_N to {}.'.format(x.max(), max_N, x.max()))
        max_N = x.max() + 1
        
    color_list = create_color_map(N=max_N, normalized=True)
    color_list = np.concatenate([color_list, np.ones((max_N, 1)).astype(np.float32)], axis=1)
    
    if x.min() < 0:
        warnings.warn('Input has negative value when trying to visualize as categorical mask, which will all be converted to -1 and displayed in white.')
        x[x<0] = -1 # Map all negative value to -1
        color_list = np.concatenate([np.ones((1,4)).astype(np.float32), color_list], axis=0) # appending an extra value.
        N = N + 1
    
    cmap = colors.ListedColormap(color_list, N=N)
    
    x = cmap(x.astype(np.int), alpha=None, bytes=True)[:,:,:3]
    # print(x.shape)
    plot_cfg = dict( interpolation="nearest")
    
    vis['disp'] = x
    vis['plot_cfg'] = plot_cfg
    vis['mode'] = 'Categorical'
    return vis


if __name__ == "__main__":
    print(create_color_map())