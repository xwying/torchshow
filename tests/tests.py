import tensorshow as ts
import torch 

ts.use_inline(False)

# test_img = torch.rand((3, 100, 100)) * 50
# ts.show(test_img)

# test_grayscale = torch.rand((1, 100, 100))
# ts.show(test_grayscale)

test_category_mask = torch.randint(-1, 25, (1, 10, 10))
ts.show(test_category_mask)