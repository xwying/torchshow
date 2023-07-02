# Changelogs

## [2023-07-02] v0.5.1
- Fix `np.int` depreciation issues ([#13](https://github.com/xwying/torchshow/pull/13)).
- Allow specifying `nrows` and `ncols` when visualizing a list of tensors ([#17](https://github.com/xwying/torchshow/pull/17)).
- Fix unexpected white spaces when saving figures ([#19](https://github.com/xwying/torchshow/pull/19)).

## [2022-11-07] v0.5.0
- Support specifying the color map for grayscale image. 
- Support PIL Image.
- Support filenames. 
- Addinng `ts.overlay()` API which can be used to blend multiple plots.
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