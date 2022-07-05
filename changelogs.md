# Changelogs

## Next Version
- Fix bugs when unnormalize with customize mean and std

## [2022-06-30] v0.4.2
- You can specify the `figsize`, `dpi`, and `subtitle` parameter when calling ts.show(). 
- Add some missing APIs to `ts.save()`.
- Revisit the option to add axes titles.
- Add tight_layout option to `ts.show_video` (enabled by default).
- Fix some bugs.
- Create API Reference Page.

## [2022-05-21] v0.4.1
- Now you can get richer information from a pixel (e.g. raw pixel value) by hovering the mouse over the pixels.
- Fix the unexpected colors around the contour while visualizing categorical masks.

## [2022-05-19] v0.4.0
- TorchShow will now automatically check if running in an ipython environment (e.g. jupyter notebook). Remove `ts.inline()` since it is no longer needed.
- Fix a bug where binary mask will be inferred as categorical mask.
- Optimize the logic to handle a few corner cases.


## [2021-08-23] v0.3.2 
- Adding `ts.save(tensor)` API for saving figs instead of showing them. This is more convenient compared to the headless mode. - Remove surrounding white spaces of the saved figures. 
- ts.headless() has been removed. Use ts.save() instead.

## [2021-06-14] v0.3.1 
- Fixes some bugs. 
- Now support headless mode useful for running on server without display. After setting `ts.headless(True)`, calling `ts.show(tensor)` will save the figure under `./_torchshow/`.

## [2021-04-25] v0.3.0 
- Adding optical flow support.