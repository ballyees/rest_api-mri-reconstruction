import cv2
import tensorflow as tf
try:
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass
import tensorflow_addons as tfa
from helper import *
import numpy as np
from unet import *
import matplotlib.pyplot as plt
from metrics import *

class ReinUNet:
    def __init__(self, input_shape=(640, 320, 1), num_filters=16, n=2, out_activation='sigmoid', drop_prob=0.0, lr=0.001, sync_period=6, slow_step_size=0.5):
        assert len(input_shape) == 3
        out_activation = out_activation.lower()
        assert out_activation in ['sigmoid', 'tanh']
        self.models = []
        self.optimizers = []
        # self.losses = NormalizeRMSE()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.n = n
        use_relu = out_activation == 'sigmoid'
        self.use_relu = use_relu
        self.out_activation = out_activation
        for i in range(n):
            self.models.append(build_unet(input_shape, outdim=1, num_filters=num_filters, out_activation=out_activation, use_relu=use_relu, drop_prob=drop_prob))
            radam = tfa.optimizers.RectifiedAdam(learning_rate=lr)
            ranger = tfa.optimizers.Lookahead(radam, sync_period=sync_period, slow_step_size=slow_step_size)
            self.optimizers.append(ranger)
            
    def set_loss(self, loss):
        self.losses = loss
        
    def set_optimizers(self, optimizers):
        self.optimizers = optimizers
        
    def predict(self, X, batch_size=None, out_process=False):
        """model prediction process from MR image data 

        Args:
            X (np.ndarray): source
            batch_size (None, int): if have mini batch in prediction process is mean yield data from source. Defaults to None.
            out_process (bool): if it true mean return n-dimensions of image on end channel. Defaults to False.
        """        
        testing_size = X.shape[0]
        if batch_size is None:
            batch_size = testing_size
        for bi in range(0, testing_size, batch_size):
            if out_process is True:
                outputs_dims = []
            for i in range(self.n):
                if i == 0:
                    out = self.models[i](X[bi:bi+batch_size])
                else:
                    out = self.models[i](out)
                if out_process is True: outputs_dims.append(out)
            if out_process is True:
                out = np.concatenate(outputs_dims, -1)
            if batch_size is None:
                return out
            else:
                yield out

    def get_models(self, model_idx=None):
        if model_idx is None:
            return self.models
        else:
            return self.models[model_idx]

    def input_image_preprocessing(self, image):
        # resize
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0])) # cv2 resize use (height, width)
        # apply smoothing filter
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # normalize
        if self.use_relu is True:  # sigmoid
            image = image / 255.
        else:  # tanh
            image = (image - 127.5) / 127.5
        if len(image.shape) == 2:  # gray scale image
            image = np.expand_dims(image, -1)
        return image

    def output_image_preprocessing(self, image):
        # resize
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0])) # cv2 resize use (height, width)
        # normalize
        if self.use_relu is True:  # sigmoid
            image = image / 255.
        else:  # tanh
            image = (image - 127.5) / 127.5
        if len(image.shape) == 2:  # gray scale image
            image = np.expand_dims(image, -1)
        return image

    def outputs_postprocessing(self, y):
        y = np.squeeze(y)
        if self.use_relu is True:  # sigmoid
            y = np.round(y * 255).astype(np.uint8)
        else:  # tanh
            y = np.round((y * 127.5) + 127.5).astype(np.uint8)
        return y

    def save_image(self, path, image, use_plt=True):
        if use_plt is True:
            plt.imsave(path, image, cmap='gray')
        else:
            cv2.imwrite(path, image)
            
    def concat_n_images(self, images, axis=1, norm=True):
        assert len(images) > 1
        if norm is True:
            for i in range(len(images)):
                images[i] = norm_minmax(images[i])
        return np.concatenate(images, axis=axis)
        
            
    def save_models(self, path='checkpoints_rein-unet'):
        for i, model in enumerate(self.models, 1):
            model.save(f'{path}_{i}')
    
    def load_models(self, path='checkpoints_rein-unet'):
        for i in range(self.n):
            self.models[i] = tf.keras.models.load_model(f'{path}_{i+1}')
    
    def fit_path_generators(self, src_path, dest_path, epochs=100, batch_size=16, shuffle=True):
        assert len(src_path) == len(dest_path)
        training_size = len(src_path)
        # training
        for ep in range(epochs):
            print(f'start epochs: {ep}')
            idx = np.arange(0, training_size)
            if shuffle is True:
                np.random.shuffle(idx)
            history_losses = []
            for bi in range(0, training_size, batch_size):
                losses = []
                size = idx[bi:bi+batch_size].shape[0]
                out = self.read_multiple_images(src_path[idx[bi:bi+batch_size]])
                y = self.read_multiple_images(dest_path[idx[bi:bi+batch_size]], False)
                
                for i in range(self.n):
                    with tf.GradientTape() as tape:
                        out = self.models[i](out, training=True)
                        loss = self.losses(y, out)
                    losses.append(loss.numpy() * size)
                    gradients = tape.gradient(loss, self.models[i].trainable_variables)
                    self.optimizers[i].apply_gradients(zip(gradients, self.models[i].trainable_variables))
                del out, y, gradients
                losses = [l/size for l in losses]
                history_losses.append(losses)
                print(f'loss on epochs: {ep}-{bi:04d}:{bi+size:04d}, {losses}')
        return history_losses
    
    def evaluate_path_generator(self, src_path, dest_path, batch_size=None, model_idx=0):
        """evaluate models from inputs X (source) and outputs y (destination) and return loss in this models

        Args:
            src_path (np.ndarray): path source images
            dest_path (np.ndarray): path destination images
            batch_size (None, int): [description]. Defaults to None.
        """        
        assert len(src_path) == len(dest_path)
        testing_size = len(src_path)
        if batch_size is None:
            batch_size = testing_size
        losses = []
        for bi in range(0, testing_size, batch_size):
            size = src_path[bi:bi+batch_size].shape[0]
            out = self.read_multiple_images(src_path[bi:bi+batch_size])
            y = self.read_multiple_images(dest_path[bi:bi+batch_size], False)
            for i in range(model_idx):
                out = self.models[i](out)
            loss = self.losses(y, out)
            losses.append(loss * size)
            print(f'loss: {bi:04d}:{bi+size:04d}, {loss}')
            del out, y
        return np.sum(losses) / testing_size
    
    def compare_predict_path_generators(self, src_path, dest_path, prefix_path='compare', all_process=False, norm=True):
        """model prediction process from MR image data 

        Args:
            src_path (np.ndarray): path source images
            dest_path (np.ndarray): path destination images
        """        
        assert len(src_path) == len(dest_path)
        testing_size = len(src_path)
        for bi in range(0, testing_size):
            X = self.read_multiple_images(src_path[bi:bi+1])
            y = self.read_multiple_images(dest_path[bi:bi+1], False)
            all_img = np.zeros((self.n, *self.input_shape))
            for i in range(self.n):
                if i == 0:
                    out = self.models[i](X)
                else:
                    out = self.models[i](out)
                all_img[i] = out
            X = np.squeeze(X)
            y = np.squeeze(y)
            if all_process is True:
                all_img = np.squeeze(all_img)
                img = self.concat_n_images([X, y, *all_img], norm=norm) # input(compressed), ground trust, all prediction images
            else:
                out = np.squeeze(out)
                img = self.concat_n_images([X, y, out], norm=norm) # input(compressed), ground trust, prediction
            self.save_image(f'{prefix_path}_{bi}.png', img)
            del X, y, out
            if bi % 100 == 0:
                tf.keras.backend.clear_session()
        
    def compare_loss_predict_path_generators(self, src_path, dest_path, show_loss=True):
        """model prediction process from MR image data 

        Args:
            src_path (np.ndarray): path source images
            dest_path (np.ndarray): path destination images
        """        
        assert len(src_path) == len(dest_path)
        testing_size = len(src_path)
        # metrics setup
        mi = MI()
        ssim = SSIM()
        nrmse = NormalizeRMSE()
        metrics = [nrmse, mi, ssim, psnr, loss_power_tf_img]
        metrics_name = [nrmse.name, mi.name, ssim.name, 'psnr', 'loss_power']
        metrics_mode = [0, 1, 1, 1, 1] # 0 is min, 1 is max
        sum_loss_of_metrics = np.zeros((len(metrics), self.n+1))
        wins = np.zeros((len(metrics), self.n), dtype=np.uint)
        for bi in range(0, testing_size):
            X = self.read_multiple_images(src_path[bi:bi+1])
            y = self.read_multiple_images(dest_path[bi:bi+1], False)
            all_img = np.zeros((self.n, *self.input_shape))
            for i in range(self.n):
                if i == 0:
                    out = self.models[i](X)
                else:
                    out = self.models[i](out)
                all_img[i] = out
            for i, metric in enumerate(metrics):
                loss_input = metric(y, X).numpy()
                loss_prediction = np.zeros(self.n)
                for j, img in enumerate(all_img):
                    loss_prediction[j] = metric(y, img).numpy()
                if metrics_mode[i] == 0:
                    wins[i] = wins[i] + (loss_input > loss_prediction)
                else:
                    wins[i] = wins[i] + (loss_input < loss_prediction)
                sum_loss_of_metrics[i] = sum_loss_of_metrics[i] + np.array([loss_input, *loss_prediction])
                if show_loss is True:
                    print(f'{metrics_name[i]}-{bi:04}:', loss_input, loss_prediction)
            del X, y, out, all_img
            if bi % 100 == 0:
                tf.keras.backend.clear_session()
        print(f'total compare size: {testing_size}')
        print('win inputs:', '(row)', ', '.join([n for n in metrics_name]), '|', '(col) pred_1, pred_2, ..., pred_n')
        print(wins)
        print('avg loss:', '(row)', ', '.join([n for n in metrics_name]), '|', '(col) input, pred_1, pred_2, ..., pred_n')
        print(sum_loss_of_metrics / testing_size)
    
    def evaluate_metrics_path_generators(self, src_path, dest_path, metrics=None, show_loss=True):
        """model prediction process from MR image data 

        Args:
            src_path (np.ndarray): path source images
            dest_path (np.ndarray): path destination images
        """        
        assert len(src_path) == len(dest_path)
        testing_size = len(src_path)
        # metrics setup
        if metrics is None:
            metrics = [MI(), SSIM(), PSNR(), Corr2D(), MPL(), NRMSE(), RMSE(), MSE(), NMSE(), MAPE()]
        metrics_name = [m.get_shot_name() for m in metrics]
        metrics_mode = [m.get_mode() for m in metrics] # 0 is min, 1 is max
        sum_loss_of_metrics = np.zeros((len(metrics), self.n+1))
        wins = np.zeros((len(metrics), self.n), dtype=np.uint)
        for bi in range(0, testing_size):
            X = self.read_multiple_images(src_path[bi:bi+1])
            y = self.read_multiple_images(dest_path[bi:bi+1], False)
            all_img = np.zeros((self.n, *self.input_shape))
            for i in range(self.n):
                if i == 0:
                    out = self.models[i](X)
                else:
                    out = self.models[i](out)
                all_img[i] = out
            for i, metric in enumerate(metrics):
                loss_input = metric(y, X).numpy()
                loss_prediction = np.zeros(self.n)
                for j, img in enumerate(all_img):
                    loss_prediction[j] = metric(y, img).numpy()
                if metrics_mode[i] == 0:
                    wins[i] = wins[i] + (loss_input > loss_prediction)
                else:
                    wins[i] = wins[i] + (loss_input < loss_prediction)
                sum_loss_of_metrics[i] = sum_loss_of_metrics[i] + np.array([loss_input, *loss_prediction])
                if show_loss is True:
                    print(f'{metrics_name[i]}-{bi:04}:', loss_input, loss_prediction)
            del X, y, out, all_img
            if bi % 100 == 0:
                tf.keras.backend.clear_session()
        print(f'total compare size: {testing_size}')
        print('win inputs:', '(row)', ', '.join([n for n in metrics_name]), '|', '(col) pred_1, pred_2, ..., pred_n')
        print(wins)
        print('win rate (%):')
        print((wins / testing_size) * 100)
        print('avg loss:', '(row)', ', '.join([n for n in metrics_name]), '|', '(col) input, pred_1, pred_2, ..., pred_n')
        print(sum_loss_of_metrics / testing_size)
        print('----------'*10)
    
    def read_multiple_images(self, path, is_inputs=True, imread_mode=cv2.IMREAD_GRAYSCALE):
        size = len(path)
        images = np.zeros((size, *self.input_shape))
        for i, p in enumerate(path):
            img = cv2.imread(p, imread_mode)
            if is_inputs is True:
                images[i] = self.input_image_preprocessing(img)
            else:
                images[i] = self.output_image_preprocessing(img)
        return images
            