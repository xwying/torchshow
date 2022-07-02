class Config():
    def __init__(self):
        self.config_dict = {'backend': 'matplotlib', # May support more backend in the future
                            'image_mean': None,
                            'image_std': None,
                            'color_mode': 'rgb',
                            'show_rich_info': True,
                            } 

    def set(self, key, value):
        if key in self.config_dict.keys():
            self.config_dict[key] = value
        
    def get(self, key):
        return self.config_dict.get(key, None)

config = Config()
        
def set_image_mean(mean: list):
    assert len(mean)==3
    config.set('image_mean', mean)
    
def set_image_std(std: list):
    assert len(std)==3
    config.set('image_std', std)
    
def set_color_mode(value='rgb'):
    assert value in ['rgb', 'bgr']
    config.set('color_mode', value)
    
def show_rich_info(flag: bool):
    if flag:
        config.set('show_rich_info', True)
    else:
        config.set('show_rich_info', False)