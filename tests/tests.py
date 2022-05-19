import torchshow as ts
import torch 
import logging

ts.use_inline(False)

print("1.1 Single 3-channel image between 0-1")
rgb_img = torch.rand((3, 100, 100))
print(rgb_img.min(), rgb_img.max())
ts.show(rgb_img)

print("1.2 Single 3-channel image between 0-255")
rgb_img_2 = rgb_img * 100
print(rgb_img_2.min(), rgb_img_2.max())
ts.show(rgb_img_2)

print("1.3 Single 3-channel image with value larger than 255")
rgb_img_3 = rgb_img * 300
print(rgb_img_3.min(), rgb_img_3.max())
ts.show(rgb_img_3)

print("1.4 Single 3-channel image with value smaller than 0")
rgb_img_4 = rgb_img - 0.5
print(rgb_img_4.min(), rgb_img_4.max())
ts.show(rgb_img_4)

print("1.5 Single 3-channel image with value smaller than 0 and larger than 255")
rgb_img_5 = rgb_img * 500 - 100
print(rgb_img_5.min(), rgb_img_5.max())
ts.show(rgb_img_5)

print("2.1 Single 1-Channel image between 0-1")
gray_img = torch.rand((1, 100, 100))
print(gray_img.min(), gray_img.max())
ts.show(gray_img)

print("2.2 Single 1-Channel image between 0-255")
gray_img_2 = gray_img * 100
print(gray_img_2.min(), gray_img_2.max())
ts.show(gray_img_2)

print("2.3 Single 1-Channel image with value smaller than 0 and larger than 255")
gray_img_3 = gray_img *  500 - 100
print(gray_img_3.min(), gray_img_3.max())
ts.show(gray_img_3)

print("2.4 Single 1-Channel image with binary value")
gray_img_4 = torch.randint(0, 2, (1, 100, 100))
print(gray_img_4.unique())
ts.show(gray_img_4)