import abc
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import tensorflow as tf
try:
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass
class Loss(metaclass=abc.ABCMeta):
    def __init__(self, name='metric', short_name='m'):
        self.name = name
        self.short_name = short_name
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fn') and 
                callable(subclass.fn) or 
                NotImplemented)

    @abc.abstractmethod
    def fn(self, y_true, y_pred):
        """function for calculate error"""
        raise NotImplementedError

    def get_name(self):
        return self.name
    
    def get_shot_name(self):
        return self.short_name
    
    def __call__(self, y_true, y_pred):
        return self.fn(y_true, y_pred)
    
    def __str__(self):
        return f'{self.name} ({self.short_name})'
    
    def __repr__(self):
        return f'{self.name} ({self.short_name})'
    
class MILoss(Loss):
    def __init__(self):
        super().__init__(name='mutual_information_loss', short_name='mi_loss')
        
    def norm_to_discrete(self, data):
        return tf.cast(data * 255, tf.uint8)
    
    def fn(self, y_true, y_pred):
        yt = tf.reshape(y_true, -1)
        yp = tf.reshape(y_pred, -1)
        yt = self.norm_to_discrete(yt)
        yp = self.norm_to_discrete(yp)
        mi = normalized_mutual_info_score(yt, yp)
        return tf.abs(tf.math.log(mi))
    
class SSIMLoss(Loss):
    def __init__(self, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, use_log=True):
        super().__init__(name='structural_similarity_loss', short_name='ssim_loss')
        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2
        self.use_log = use_log
        
    def fn(self, y_true, y_pred):
        ssim = tf.image.ssim(tf.cast(y_true, y_pred.dtype), y_pred, self.max_val, filter_size=self.filter_size, filter_sigma=self.filter_sigma, k1=self.k1, k2=self.k2)
        ssim = tf.reduce_mean(ssim)
        if self.use_log is True:
            return tf.abs(tf.math.log(ssim))
        else:
            return 1 - ssim
        
class PSNRLoss(Loss):
    def __init__(self, max_val=1.0, val=1):
        super().__init__(name='peak_signal_to_noise_ratio_loss', short_name='psnr_loss')
        self.max_val = max_val
        self.val = val
    def fn(self, y_true, y_pred):
        psnr = tf.reduce_mean(tf.image.psnr(y_true, y_pred, self.max_val))
        if self.val is None:
            return psnr * -1
        else:
            return self.val / psnr
        
class Corr2DLoss(Loss):
    def __init__(self, use_abs=False, use_log=True):
        super().__init__(name='correlation_coefficient_2d_loss', short_name='corr2d_loss')
        self.use_abs = use_abs
        self.use_log = use_log
        
    def fn(self, y_true, y_pred):
        dims = len(y_true.shape)
        yt_mean = tf.reduce_mean(tf.cast(y_true, y_pred.dtype), axis=tf.range(1, dims))
        yh_mean = tf.reduce_mean(y_pred, axis=tf.range(1, dims))
        z_yt = y_true - yt_mean
        z_yh = y_pred - yh_mean
        sum_z1 = tf.square(z_yt)
        sum_z2 = tf.square(z_yh)
        corr = tf.reduce_sum(z_yt * z_yh, axis=tf.range(1, dims)) / tf.sqrt(tf.reduce_sum(sum_z1, axis=tf.range(1, dims)) * tf.reduce_sum(sum_z2, axis=tf.range(1, dims)))
        if self.use_abs is True:
            corr = tf.reduce_mean(tf.abs(corr))
        else:
            corr = tf.reduce_mean(corr)
        corr = tf.abs(corr)
        if self.use_log is True:
            return tf.abs(tf.math.log(corr))
        else:
            return 1 - corr
        
class MPLLoss(Loss):
    def __init__(self, use_percent=False):
        super().__init__(name='magnitude_power_loss', short_name='mpl_loss')
        self.use_percent = use_percent
        
    def images_to_kspace(self, images):
        img = np.fft.ifftshift(images)
        img = np.fft.fftshift(np.fft.fftn(img, axes=(-2, -1), norm='ortho'))
        return img

    def fn(self, y_true, y_pred):
        feq_true = self.images_to_kspace(y_true)
        feq_pred = self.images_to_kspace(y_pred)
        power_true = tf.reduce_sum(tf.square(tf.abs(feq_true)))
        power_pred = tf.reduce_sum(tf.square(tf.abs(feq_pred)))
        power_loss = power_pred / power_true
        if self.use_percent is True:
            return tf.abs((1 - power_loss) * 100)
        else:
            return tf.abs(1 - power_loss)

class NRMSELoss(Loss):
    def __init__(self):
        super().__init__(name='normalize_root_mean_squared_error_loss', short_name='nrmse_loss')
        
    def fn(self, y_true, y_pred):
        y_mean = tf.reduce_mean(y_true)
        mse = tf.reduce_mean(tf.square(y_true-y_pred))
        return tf.cast(tf.math.sqrt(mse), y_mean.dtype) / y_mean
    
class RMSELoss(Loss):
    def __init__(self):
        super().__init__(name='root_mean_squared_error_loss', short_name='rmse_loss')
        
    def fn(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true-y_pred))
        return tf.math.sqrt(mse)

class MSELoss(Loss):
    def __init__(self):
        super().__init__(name='mean_squared_error_loss', short_name='mse_loss')
        
    def fn(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true-y_pred))
class NMSELoss(Loss):
    def __init__(self):
        super().__init__(name='normalize_mean_squared_error_loss', short_name='nmse_loss')
        
    def fn(self, y_true, y_pred):
        return tf.norm(y_true-y_pred) / tf.norm(tf.cast(y_true, y_pred.dtype))
    

class MAPELoss(Loss):
    def __init__(self):
        super().__init__(name='mean_absolute_percentage_error_loss', short_name='mape_loss')
        
    def fn(self, y_true, y_pred):
        error = tf.abs((y_true - y_pred) / tf.cast(y_true, y_pred.dtype))
        error = tf.reduce_mean(error)
        return error