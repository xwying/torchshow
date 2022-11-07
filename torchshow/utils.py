import numpy as np
import os
import sys
import importlib
import warnings
import numbers

_EXIF_ORIENT = 274


def isnumber(x):
    return isinstance(x, numbers.Number)


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


def calculate_grid_layout(N, img_H, img_W, nrow=None, ncol=None):
    """
    Function to calculate grid_layout
    """
    if (nrow != None and ncol == None):
        ncol = int(np.ceil(N / nrow))
    elif (nrow == None and ncol != None):
        nrow = int(np.ceil(N / ncol))
    else:
        N_sqrt = np.sqrt(N)
        if img_H >= img_W:
            nrow = int(np.floor(N_sqrt))
            ncol = int(np.ceil(N/nrow))
        else:
            ncol = int(np.floor(N_sqrt))
            nrow = int(np.ceil(N/ncol))
    return nrow, ncol
    
    
def tensor_to_array(x):
    # Recursively perform tensor to array conversion
    # ====== PyTorch Tensor =======
    try:
        torch = import_if_not_yet("torch")
    except ImportError:
        pass
    else:
        if isinstance(x, torch.Tensor):
            return x.detach().clone().cpu().numpy()
    
    # ====== PIL Image =======
    try: 
        Image = import_if_not_yet(module_name="Image", package_name="PIL")
    except ImportError:
        pass
    else:
        if isinstance(x, Image.Image):
            return np.asarray(x)
    
    # ====== Numpy Array =======
    if isinstance(x, np.ndarray):
        return x.copy()
    
    # ====== Filename =======
    elif isinstance(x, str): # Handling filename
        if os.path.exists(x):
            if x.split('.')[-1].lower() == 'flo': # Handling optical flow files.
                return read_flow(x)
            else:
                image = read_image_PIL(x)
                return np.asarray(image)
        else:
            raise FileNotFoundError(f"{x} is not a file.")
    # ====== Recursively processing list of inputs
    elif isinstance(x, list):
        return [tensor_to_array(e) for e in x]
    else:
        raise TypeError('Found unsupported type ', type(x))


def isnotebook():
    # Check if running in ipython/jupyter notebook environment
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
def import_if_not_yet(module_name, package_name=""):
    module_key = f"{package_name}.{module_name}".lstrip('.')
    if module_key not in sys.modules:
        return importlib.import_module(module_name, package_name)
    else:
        return sys.modules[module_key]

def read_image_PIL(filename):
    try: 
        Image = import_if_not_yet(module_name="Image", package_name="PIL")
    except ImportError:
        raise ImportError("TorchShow opens image files using PIL which is not yet installed.")
    else:
        image = Image.open(filename)
        mode = image.mode
        if mode == 'RGBA': # Convert RGBA to RGB since torchshow will handle 4 channel images differently.
            mode = 'RGB'
            
        """
        The following code is for correctly applying the exif orientation. 
        This code is modified based on https://github.com/facebookresearch/detectron2/blob/2b98c273b240b54d2d0ee6853dc331c4f2ca87b9/detectron2/data/detection_utils.py#L119
        """
        
        if not hasattr(image, "getexif"):
            return image

        try:
            exif = image.getexif()
        except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
            exif = None

        if exif is None:
            return image

        orientation = exif.get(_EXIF_ORIENT)
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)

        if method is not None:
            warnings.warn(f"TorchShow has detected orientation information in the EXIF of file {filename} and the corresponding transformed has been applied. Please be aware of this issue if you want to load this file with PIL in your code.")
            image = image.transpose(method)
            
        return image.convert(mode)


def read_flow(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            print('Reading %d x %d flo file' % (w, h))
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h, w, 2))
        return data2D


if __name__ == '__main__':
    for N in range(1,100):
        row, col = calculate_grid_layout(N, 10, 10)
        print(N, row, col, row*col)