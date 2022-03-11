import abc
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import tensorflow as tf
try:
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass
class Metric(metaclass=abc.ABCMeta):
    
    def __init__(self, name='metric', short_name='m', mode=1):
        self.name = name
        self.short_name = short_name
        self.mode = mode
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fn') and 
                callable(subclass.fn) or 
                NotImplemented)

    @abc.abstractmethod
    def fn(self, y_true, y_pred):
        """function for calculate metric"""
        raise NotImplementedError

    def get_name(self):
        return self.name
    
    def get_shot_name(self):
        return self.short_name   
     
    def get_mode(self):
        '''get mode lower is better or higher is better; 0 is lower, 1 is higher'''
        return self.mode
    
    def __call__(self, y_true, y_pred):
        return self.fn(y_true, y_pred)
    
    def __str__(self):
        return f'{self.name} ({self.short_name})'
    
    def __repr__(self):
        return f'{self.name} ({self.short_name})'
    
    
class MI(Metric):
    def __init__(self):
        super().__init__(name='mutual_information', short_name='mi', mode=1)
        
    def norm_to_discrete(self, data):
        return np.uint8(data * 255)
    
    def fn(self, y_true, y_pred):
        yt = np.array(y_true).reshape(-1)
        yp = np.array(y_pred).reshape(-1)
        yt = self.norm_to_discrete(yt)
        yp = self.norm_to_discrete(yp)
        mi = normalized_mutual_info_score(yt, yp)
        return tf.constant(mi)
    
class SSIM(Metric):
    def __init__(self, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, mode=1):
        super().__init__(name='structural_similarity', short_name='ssim')
        self.max_val = max_val
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.k1 = k1
        self.k2 = k2
        
    def fn(self, y_true, y_pred):
        ssim = tf.image.ssim(tf.cast(y_true, y_pred.dtype), y_pred, self.max_val, filter_size=self.filter_size, filter_sigma=self.filter_sigma, k1=self.k1, k2=self.k2)
        ssim = tf.reduce_mean(ssim)
        return ssim

class PSNR(Metric):
    def __init__(self, max_val=1.0, is_loss=False, val=1):
        super().__init__(name='peak_signal_to_noise_ratio', short_name='psnr', mode=1)
        self.max_val = max_val
        self.is_loss = is_loss
        self.val = val
    def fn(self, y_true, y_pred):
        psnr = tf.reduce_mean(tf.image.psnr(y_true, y_pred, self.max_val))
        return psnr
        
class Corr2D(Metric):
    def __init__(self, reduce_abs=False, use_abs=True, mode=1):
        super().__init__(name='correlation_coefficient_2d', short_name='corr2d', mode=mode)
        self.use_abs = use_abs
        self.reduce_abs = reduce_abs
        
    def fn(self, y_true, y_pred):
        dims = len(y_true.shape)
        yt_mean = tf.reduce_mean(tf.cast(y_true, y_pred.dtype), axis=tf.range(1, dims))
        yh_mean = tf.reduce_mean(y_pred, axis=tf.range(1, dims))
        z_yt = y_true - yt_mean
        z_yh = y_pred - yh_mean
        sum_z1 = tf.square(z_yt)
        sum_z2 = tf.square(z_yh)
        corr = tf.reduce_sum(z_yt * z_yh, axis=tf.range(1, dims)) / tf.sqrt(tf.reduce_sum(sum_z1, axis=tf.range(1, dims)) * tf.reduce_sum(sum_z2, axis=tf.range(1, dims)))
        if self.reduce_abs is True:
            corr = tf.reduce_mean(tf.abs(corr))
        else:
            corr = tf.reduce_mean(corr)
        if self.use_abs is True:
            corr = tf.abs(corr)
        return corr
        
class MPL(Metric):
    def __init__(self, use_percent=False):
        super().__init__(name='magnitude_power_loss', short_name='mpl', mode=1)
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
            return (1 - power_loss) * 100
        else:
            return (1 - power_loss)

class NRMSE(Metric):
    def __init__(self):
        super().__init__(name='normalize_root_mean_squared_error', short_name='nrmse', mode=0)
        
    def fn(self, y_true, y_pred):
        y_mean = tf.reduce_mean(y_true)
        mse = tf.reduce_mean(tf.square(y_true-y_pred))
        return tf.cast(tf.math.sqrt(mse), y_mean.dtype) / y_mean
    
class RMSE(Metric):
    def __init__(self):
        super().__init__(name='root_mean_squared_error', short_name='rmse', mode=0)
        
    def fn(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true-y_pred))
        return tf.math.sqrt(mse)

class MSE(Metric):
    def __init__(self):
        super().__init__(name='mean_squared_error', short_name='mse', mode=0)
        
    def fn(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true-y_pred))
class NMSE(Metric):
    def __init__(self):
        super().__init__(name='normalize_mean_squared_error', short_name='nmse', mode=0)
        
    def fn(self, y_true, y_pred):
        return tf.norm(y_true-y_pred) / tf.norm(tf.cast(y_true, y_pred.dtype))

class MAPE(Metric):
    def __init__(self):
        super().__init__(name='mean_absolute_percentage_error', short_name='mape', mode=0)
        
    def fn(self, y_true, y_pred):
        error = tf.abs((y_true - y_pred) / tf.cast(y_true, y_pred.dtype))
        error = tf.reduce_mean(error)
        return error