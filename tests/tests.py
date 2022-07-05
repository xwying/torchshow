import sys
import torchshow as ts
import torch 
import torchvision
import numpy as np
import logging
from PIL import Image

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
        ts.save(rgb_img)

        print("1.2 Single 3-channel image between 0-255")
        rgb_img_2 = rgb_img * 100
        print(rgb_img_2.min(), rgb_img_2.max())
        ts.show(rgb_img_2)
        ts.save(rgb_img_2)

        print("1.3 Single 3-channel image with value larger than 255")
        rgb_img_3 = rgb_img * 300
        print(rgb_img_3.min(), rgb_img_3.max())
        ts.show(rgb_img_3)
        ts.save(rgb_img_3)

        print("1.4 Single 3-channel image with value smaller than 0")
        rgb_img_4 = rgb_img - 0.5
        print(rgb_img_4.min(), rgb_img_4.max())
        ts.show(rgb_img_4)
        ts.save(rgb_img_4)

        print("1.5 Single 3-channel image with value smaller than 0 and larger than 255")
        rgb_img_5 = rgb_img * 500 - 100
        print(rgb_img_5.min(), rgb_img_5.max())
        ts.show(rgb_img_5)
        ts.save(rgb_img_5)

    gray_img = torch.rand((1, 100, 100))
    category_mask = np.array(Image.open('test_data/example_category_mask.png'))
    if section <=2:
        print("2.1 Single 1-Channel image between 0-1")
        print(gray_img.min(), gray_img.max())
        ts.show(gray_img)
        ts.save(gray_img)

        print("2.2 Single 1-Channel image between 0-255")
        gray_img_2 = gray_img * 100
        print(gray_img_2.min(), gray_img_2.max())
        ts.show(gray_img_2)
        ts.save(gray_img_2)

        print("2.3 Single 1-Channel image with value smaller than 0 and larger than 255")
        gray_img_3 = gray_img *  500 - 100
        print(gray_img_3.min(), gray_img_3.max())
        ts.show(gray_img_3)
        ts.save(gray_img_3)

        print("2.4 Single 1-Channel image with binary value")
        gray_img_4 = torch.eye(100).unsqueeze(0)
        gray_img_4[:, 20:40, 20:40] = 1
        gray_img_4[:, 60:80, 60:80] = 1 
        print(gray_img_4.unique())
        ts.show(gray_img_4)
        ts.save(gray_img_4)

        print("2.5 Single 1-Channel image with integer value >= 0")
        gray_img_5 = torch.randint(0, 100, (1, 100, 100))
        print(gray_img_5.unique())
        ts.show(gray_img_5)
        ts.save(gray_img_5)

        print("2.6 Single 1-Channel image with both positive and negative integer value")
        gray_img_6 = torch.randint(-50, 100, (1, 100, 100))
        print(gray_img_6.unique())
        ts.show(gray_img_6)
        ts.save(gray_img_6)
        
        print("2.7 Single 1-Channel normal categorical mask")
        print(np.unique(category_mask))
        ts.show(category_mask)
        ts.save(category_mask)

    flow = read_flow("./test_data/example_flow.flo")
    if section <= 3:
        print("3.1 Single 2-Channel normal optical flow")
        print(flow.shape, flow.min(), flow.max())
        ts.show(flow)
        ts.save(flow)

        print("3.2 Single 2-Channel with random values")
        flow_2 = torch.rand((2, 100, 100)) * 200 - 100
        print(flow_2.min(), flow_2.max())
        ts.show(flow_2)
        ts.save(flow_2)

    img_n = torch.rand(16, 100, 100)
    if section <=4:
        print("4.1 Single n-channel tensor (n>3")
        print(img_n.min(), img_n.max())
        ts.show(img_n)
        ts.save(img_n)
        ts.show(img_n, ncols=3)
        ts.save(img_n, ncols=3)
        ts.show(img_n, nrows=5)
        ts.save(img_n, nrows=5)

        print("4.2 4D tensors")
        batch = torch.rand(8, 3, 100, 100) * 500 - 250
        print(batch.min(), batch.max())
        ts.show(batch)
        ts.save(batch)
        ts.show(batch, ncols=3)
        ts.save(batch, ncols=3)
        ts.show(batch, nrows=4)
        ts.save(batch, nrows=4)
        ts.show(batch, axes_title="Image ID: {img_id_from_1}")
        ts.show(batch, nrows=4, axes_title="{img_id}-{img_id_from_1}-{row}-{column}")
        ts.show(batch, nrows=4, axes_title="{img_id}-{img_id_from_1}-{row}-{column}", suptitle="Figure 1")

    if section <=5:
        print("5.1 Custom Layout")
        grid = [[rgb_img, gray_img, flow]]
        ts.show(grid)
        ts.save(grid)
        grid2 = [[rgb_img, gray_img], 
                 [flow]]
        ts.show(grid2)
        ts.save(grid2)

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
                       [rgb_img, flow]], suptitle="Video Example")
        vis_list = ts.show_video([[video, video2], 
                       [rgb_img, flow]], display=False)
        print(len(vis_list), len(vis_list[0]), len(vis_list[0][0]))
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

    if section <=7:
        print("7 Return vis_list if display=False")
        vis_list = ts.show(img_n, display=False)
        print(len(vis_list), len(vis_list[0]))
        vis_list = ts.show(img_n, display=False, nrows=3)
        print(len(vis_list), len(vis_list[0]))
 
    if section <=8:
        print("8 Change Unnormalization Presets")
        rgb_img_numpy = rgb_img.permute(1,2,0).numpy()
        def test_normalize(MEAN, STD):
            transform = torchvision.transforms.Normalize(MEAN, STD)
            rgb_img_0 = transform(rgb_img)
            ts.set_image_mean(MEAN)
            ts.set_image_std(STD)
            rgb_img_1 = ts.show(rgb_img_0, display=False)[0][0]['disp']
            assert np.allclose(rgb_img_1, rgb_img_numpy, atol=1e-7)  # will be False if atol=1e-8

        test_normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        test_normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        test_normalize([0.4851231531212364564, 0.4561231523135436, 0.406123412312452343], [0.2293453455673435, 0.22434531445342, 0.22534534472423])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        section = sys.argv[1]
    else:
        section = 0
    test(int(section))