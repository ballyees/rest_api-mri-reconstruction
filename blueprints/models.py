from ..helper import load_model
class Models:
    def __init__(self):
        self.model_16_3 = load_model('../models/checkpoints_rein-unet-ssim_loss-test-3x-1_1')
        self.model_16_5 = load_model('../models/checkpoints_rein-unet-ssim_loss-test-5x-1_1')
        self.model_16_8 = load_model('../models/checkpoints_rein-unet-ssim_loss-test-8x-1_1')
        self.model_64_3 = load_model('../models/checkpoints_rein-unet-ssim_loss-test-3x-64-1_1')
        self.model_64_5 = load_model('../models/checkpoints_rein-unet-ssim_loss-test-5x-64-1_1')
        self.model_64_8 = load_model('../models/checkpoints_rein-unet-ssim_loss-test-8x-64-1_1')
        print('call init')
        
models = Models()