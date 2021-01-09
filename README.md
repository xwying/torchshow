# TensorShow

`Note: The package is still under development and may have many bugs.`

Tensorshow visualizes your data in one line of code. It is developed to helped debugging Computer Vision project.

Tensorshow automatically infers the type of a tensor such as RGB images, grayscale images, binary masks, categorical masks (automatically apply color palette), etc. and perform necessary unnormalization if needed. 


# Installation

```bash
git clone https://github.com/xwying/tensorshow.git
cd tensorshow
pip install .
```

TODO: support installation via pip.

# Usage 

## Basic Usage
```python
import tensorshow as ts
ts.show(tensor) # Visualize a tensor in one-line
```

## Examples
Please check out the tutorial for more examples.

## Use in notebook
By default the ts.show() will call `plt.imshow()` followed by `plt.show()` to display the result. When using notebook environment with `%inline` display enabled. Running the following code will tell tensorshow to not run `plt.show()`.

```python
import tensorshow as ts
ts.use_inline(True)
```
## Visualizing BGR image
By default tensorflow visualize image tensor in the RGB mode, you can switch the setting to BGR in case you are using opencv to load the image.
```python
ts.set_color_mode('bgr)
```
## 

## Unnormalization
The image tensor may have been preprocessed with a normalization function. If not specified, tensorshow will automatically rescale it to 0-1. 


To change the preset to imagenet normalization. Use the following code.
```python
ts.show(tensor, unnormalize='imagenet')
```

To use a customize mean and std value, use the following command. 
```python
ts.set_image_mean([0., 0., 0.])
ts.set_image_std([1., 1., 1.])
```
Note that once this is set, tensorshow will use this value for the following visualization. This is useful because usually only a single normalization preset will be used for the entire dataset.