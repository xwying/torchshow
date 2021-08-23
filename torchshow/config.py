class Config():
    def __init__(self):
        self.config_dict = {'backend': 'matplotlib', # May support more backend in the future
                            'inline': False, # Set it to true if using jupyter notebook
                            'image_mean': None,
                            'image_std': None,
                            'color_mode': 'rgb',
                            } 

    def set(self, key, value):
        if key in self.config_dict.keys():
            self.config_dict[key] = value
        
    def get(self, key):
        return self.config_dict.get(key, None)

config = Config()


def use_inline(value: bool):
    if value:
        config.set('inline', True)
    else:
        config.set('inline', False)
        
def set_image_mean(mean: list):
    assert len(mean)==3
    config.set('image_mean', mean)
    
def set_image_std(mean: list):
    assert len(mean)==3
    config.set('image_mean', mean)
    
def set_color_mode(value='rgb'):
    assert value in ['rgb', 'bgr']
    config.set('color_mode', value)