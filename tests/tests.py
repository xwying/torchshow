import torchshow as ts
import torch 

ts.use_inline(False)

test_img = torch.rand((3, 100, 100)) * 50
# ts.show(test_img)

test_grayscale = torch.rand((1, 100, 100))
# ts.show(test_grayscale)

test_category_mask = torch.randint(-1, 25, (1, 10, 10))
# ts.show(test_category_mask)

test_flow = torch.rand((2, 100, 100))
# ts.headless(True)
ts.save(test_flow)
ts.save(test_flow, './test_fig/vis/test.png')

# feature_maps = torch.rand((64, 32, 32))

# ts.show([[test_img, test_grayscale], [test_category_mask]])
# ts.show(feature_maps)

# ts.show(feature_maps, nrows=4)

# ts.show(feature_maps, title_pattern="channel {img_id_from_1}")
# ts.show(test_grayscale)
# ts.show(test_category_mask)

# video1 = torch.rand((8, 3, 100, 100))
# video2 = torch.rand((15, 3, 100, 100))

# ts.show_video(video2)
# ts.show(video1)
# ts.show_video([[video1, video2], [video1, video2]])