# Changelogs

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