import helper
class Models:
    def __init__(self):
        self.model_16_3 = helper.load_keras_model('./models/checkpoints_rein-unet-ssim_loss-test-3x-1_1')
        # self.model_16_5 = helper.load_keras_model('./models/checkpoints_rein-unet-ssim_loss-test-5x-1_1')
        # self.model_16_8 = helper.load_keras_model('./models/checkpoints_rein-unet-ssim_loss-test-8x-1_1')
        # self.model_64_3 = helper.load_keras_model('./models/checkpoints_rein-unet-ssim_loss-test-3x-64-1_1')
        # self.model_64_5 = helper.load_keras_model('./models/checkpoints_rein-unet-ssim_loss-test-5x-64-1_1')
        # self.model_64_8 = helper.load_keras_model('./models/checkpoints_rein-unet-ssim_loss-test-8x-64-1_1')
        print('call init')
        
models = Models()