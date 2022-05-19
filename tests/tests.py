import sys
import torchshow as ts
import torch 
import numpy as np
import logging

ts.use_inline(False)

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
            

def test(section):
    rgb_img = torch.rand((3, 100, 100))
    if section <= 1:
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

    gray_img = torch.rand((1, 100, 100))
    if section <=2:
        print("2.1 Single 1-Channel image between 0-1")
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

        print("2.5 Single 1-Channel image with integer value >= 0")
        gray_img_5 = torch.randint(0, 100, (1, 100, 100))
        print(gray_img_5.unique())
        ts.show(gray_img_5)

        print("2.6 Single 1-Channel image with both positive and negative integer value")
        gray_img_6 = torch.randint(-50, 100, (1, 100, 100))
        print(gray_img_6.unique())
        ts.show(gray_img_6)

    flow = read_flow("./example_flow/example1.flo")
    if section <= 3:
        print("3.1 Single 2-Channel normal optical flow")
        print(flow.shape, flow.min(), flow.max())
        ts.show(flow)

        print("3.2 Single 2-Channel with random values")
        flow_2 = torch.rand((2, 100, 100)) * 200 - 100
        print(flow_2.min(), flow_2.max())
        ts.show(flow_2)

    img_n = torch.rand(16, 100, 100)
    if section <=4:
        print("4.1 Single n-channel tensor (n>3")
        print(img_n.min(), img_n.max())
        ts.show(img_n)
        ts.show(img_n, ncols=3)
        ts.show(img_n, nrows=5)

        print("4.2 4D tensors")
        batch = torch.rand(8, 3, 100, 100) * 500 - 250
        print(batch.min(), batch.max())
        ts.show(batch)
        ts.show(batch, ncols=3)
        ts.show(batch, nrows=4)

    if section <=5:
        print("5.1 Custom Layout")
        grid = [[rgb_img, gray_img, flow]]
        ts.show(grid)
        grid2 = [[rgb_img, gray_img], 
                 [flow]]
        ts.show(grid2)

    if section <=6:
        print("6.1 Video Clip")
        video = torch.rand(16, 3, 100, 100)
        print(video.min(), video.max())
        ts.show_video(video)
        video2 = torch.rand(8, 1, 100, 100)
        video3 = torch.rand(13, 2, 100, 100)
        ts.show_video([[video, video2], 
                       [video3]])
        video4 = torch.rand(13,5, 100, 100)
        print("6.2 ts.show_video with image") 
        # This test produces unwanted results. Ignore it at this moment unless requested.
        ts.show_video(rgb_img)
        ts.show_video(flow)
        ts.show_video([[video, video2], 
                       [rgb_img, flow]])
        print("6.3 Video Clip Edge Cases")
        try:
            ts.show_video(video4)
        except Exception as e: # This should raise an error
            print(e)
        try:
            ts.show_video([[video, video2], 
                           [video3, video4]])
        except Exception as e: # This should raise an error
            print(e)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        section = sys.argv[1]
    else:
        section = 0
    test(int(section))