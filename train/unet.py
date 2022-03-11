import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, LeakyReLU, Dropout, Layer
from tensorflow.keras.models import Model
try:
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass
from helper import *
import numpy as np

def conv_block(inputs, num_filters, use_relu=True, drop_prob=0.0):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    if use_relu:
        x = Activation("relu")(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(drop_prob)(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    if use_relu:
        x = Activation("relu")(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(drop_prob)(x)
    return x

def encoder_block(inputs, num_filters, use_relu, drop_prob):
    x = conv_block(inputs, num_filters, use_relu, drop_prob)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters, use_relu, drop_prob):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, use_relu, drop_prob)
    return x

def build_unet(inputs_shape, outdim=1, num_filters=64, out_activation='sigmoid', use_relu=True, drop_prob=0.0):
    inputs = Input(inputs_shape)

    s1, p1 = encoder_block(inputs, num_filters, use_relu, drop_prob)
    s2, p2 = encoder_block(p1, num_filters<<1, use_relu, drop_prob)
    s3, p3 = encoder_block(p2, num_filters<<2, use_relu, drop_prob)
    s4, p4 = encoder_block(p3, num_filters<<3, use_relu, drop_prob)

    b1 = conv_block(p4, num_filters<<4, use_relu, drop_prob)

    d1 = decoder_block(b1, s4, num_filters<<3, use_relu, drop_prob)
    d2 = decoder_block(d1, s3, num_filters<<2, use_relu, drop_prob)
    d3 = decoder_block(d2, s2, num_filters<<1, use_relu, drop_prob)
    d4 = decoder_block(d3, s1, num_filters, use_relu, drop_prob)

    outputs = Conv2D(outdim, 1, padding="same", activation=out_activation)(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

def build_vnet(inputs_shape, outdim=1, num_filters=64, out_activation='sigmoid', use_relu=True, drop_prob=0.0):
    inputs = Input(inputs_shape)

    s1, p1 = encoder_block(inputs, num_filters, use_relu, drop_prob)
    s2, p2 = encoder_block(p1, num_filters<<1, use_relu, drop_prob)
    s3, p3 = encoder_block(p2, num_filters<<2, use_relu, drop_prob)
    s4, p4 = encoder_block(p3, num_filters<<3, use_relu, drop_prob)
    
    b1, z_mean, z_log_var = conv_var_block(p4, num_filters<<4, use_relu, drop_prob)

    d1 = decoder_block(b1, s4, num_filters<<3, use_relu, drop_prob)
    d2 = decoder_block(d1, s3, num_filters<<2, use_relu, drop_prob)
    d3 = decoder_block(d2, s2, num_filters<<1, use_relu, drop_prob)
    d4 = decoder_block(d3, s1, num_filters, use_relu, drop_prob)

    outputs = Conv2D(outdim, 1, padding="same", activation=out_activation)(d4)

    model = Model(inputs, [outputs, z_mean, z_log_var], name="U-Net")
    return model

def build_unet_with_mask(inputs_shape, mask_shape, outdim=1, num_filters=64, out_activation='sigmoid', use_relu=True, drop_prob=0.0):
    inputs = Input(inputs_shape)
    mask = Input(mask_shape)

    s1, p1 = encoder_block(inputs, num_filters, use_relu, drop_prob)
    s2, p2 = encoder_block(p1, num_filters<<1, use_relu, drop_prob)
    s3, p3 = encoder_block(p2, num_filters<<2, use_relu, drop_prob)
    s4, p4 = encoder_block(p3, num_filters<<3, use_relu, drop_prob)

    b1 = conv_block(p4, num_filters<<4, use_relu, drop_prob)

    d1 = decoder_block(b1, s4, num_filters<<3, use_relu, drop_prob)
    d2 = decoder_block(d1, s3, num_filters<<2, use_relu, drop_prob)
    d3 = decoder_block(d2, s2, num_filters<<1, use_relu, drop_prob)
    d4 = decoder_block(d3, s1, num_filters, use_relu, drop_prob)

    outputs = Conv2D(outdim, 1, padding="same", activation=out_activation)(d4)

    model = Model(inputs=[inputs, mask], outputs=[outputs, mask], name="U-Net")
    return model

def build_unet_n_deep(inputs_shape, outdim=1, num_filters=64, out_activation='sigmoid', use_relu=True, drop_prob=0.0, n=2, keep_dims=True):
    assert n > 1 and type(n) == int
    assert type(keep_dims) == bool
    inputs = Input(inputs_shape)
    for i in range(n):
        if i != 0:
            s1, p1 = encoder_block(outputs, num_filters, use_relu, drop_prob)
        else:
            s1, p1 = encoder_block(inputs, num_filters, use_relu, drop_prob)
        s2, p2 = encoder_block(p1, num_filters<<1, use_relu, drop_prob)
        s3, p3 = encoder_block(p2, num_filters<<2, use_relu, drop_prob)
        s4, p4 = encoder_block(p3, num_filters<<3, use_relu, drop_prob)

        b1 = conv_block(p4, num_filters<<4, use_relu, drop_prob)

        d1 = decoder_block(b1, s4, num_filters<<3, use_relu, drop_prob)
        d2 = decoder_block(d1, s3, num_filters<<2, use_relu, drop_prob)
        d3 = decoder_block(d2, s2, num_filters<<1, use_relu, drop_prob)
        d4 = decoder_block(d3, s1, num_filters, use_relu, drop_prob)
        if keep_dims:
            if i+1 == n:
                outputs = Conv2D(outdim, 1, padding="same", activation=out_activation)(d4)
            else:
                outputs = Conv2D(inputs_shape[-1], 1, padding="same", activation=out_activation)(d4)
        else:
            outputs = Conv2D(outdim, 1, padding="same", activation=out_activation)(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

class UNetReconstruction:
    def __init__(
        self,
        inputs_shape,
        loss,
        eval_function,
        speed_up,
        lr=0.007,
        batch_size=5,
        coil=15,
        outdim1=15,
        outdim2=1,
        num_filters=16,
        out_activation='tanh',
        use_relu=False,
        drop_prob=0.05,
        n1=3,
        n2=3,
        keep_dims=True
    ):
        self.model_coil = build_unet_n_deep((*inputs_shape, coil*2), outdim=outdim1, num_filters=num_filters, out_activation=out_activation, use_relu=use_relu, drop_prob=drop_prob, n=n1, keep_dims=keep_dims)
        self.model_compress = build_unet_n_deep((*inputs_shape, 1), outdim=outdim2, num_filters=num_filters, out_activation=out_activation, use_relu=use_relu, drop_prob=drop_prob, n=n2, keep_dims=keep_dims)
        self.model_coil.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss)
        self.model_compress.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss)
        self.mask = create_mask_speed_up(inputs_shape, speed_up)
        self.inputs_shape = inputs_shape
        self.batch_size = batch_size
        self.coil = coil
        self.loss = loss
        self.eval_function = eval_function
        self.model_format_name = 'models/k{k_fold}_{speed}_{model_name}_{type_}.h5'
        self.speed_up = speed_up
        self.model_name_coil = 'coil_model'
        self.model_name_compress = 'compress_model'
    
    def load_model(self, fname_coil, fname_compress, custom_objects):
        self.model_coil = tf.keras.models.load_model(fname_coil, custom_objects=custom_objects)
        self.model_compress = tf.keras.models.load_model(fname_compress, custom_objects=custom_objects)
        return True
    
    def fit(self, file_names, k_fold, epochs=1, shuffle=True):
        training_size = len(file_names)
        idx = np.arange(0, training_size)
        if shuffle is True:
            np.random.shuffle(idx)
        for ep in range(1, epochs+1):
            loss = []
            for ifname in range(0, training_size, self.batch_size):
                kspace = read_kspace_multiple_file(file_names[idx[ifname:ifname+self.batch_size]], self.inputs_shape)
                error = self.train_on_batch(kspace)
                del kspace
                iend = ifname+self.batch_size
                iend = iend if iend < training_size else training_size
                print(f'({ifname+1:03d}-{iend:03d} of {training_size:03d}_{k_fold})\tcoil_loss: {error[0]:e}, compress_loss: {error[1]:e}')
                loss.append(error)
                if ifname % (self.batch_size*10) == 0:
                    tf.keras.backend.clear_session()
            tf.keras.backend.clear_session()
            loss = np.mean(loss, axis=0) # coil, compress
            file_names = [
                self.model_format_name.format(k_fold=k_fold, speed=self.speed_up, model_name=self.model_name_coil, type_='train'),
                self.model_format_name.format(k_fold=k_fold, speed=self.speed_up, model_name=self.model_name_compress, type_='train')
            ]
            _min, save_model = multi_model_save([self.model_coil, self.model_compress], file_names, _min, loss[1])
            # backup model
            file_names = [
                self.model_format_name.format(k_fold=k_fold, speed=self.speed_up, model_name=self.model_name_coil, type_='backup'),
                self.model_format_name.format(k_fold=k_fold, speed=self.speed_up, model_name=self.model_name_compress, type_='backup')
            ]
            multi_model_save([self.model_coil, self.model_compress], file_names, 1000, -1)
            print('---------'*10)
            print(f'(training) loss epochs {ep:03d}: {loss[0]:e} {loss[1]:e}, save model: {save_model}')
            print('---------'*10)
    
    def train_on_batch(self, kspace):
        error = training_process(self.model_coil, self.model_compress, kspace, self.mask)
        return error
    
    def predict(self):
        pass
    
    def evaluate(self, file_names, k_fold):
        testing_size = len(file_names)
        loss = []
        for ifname in range(0, testing_size, self.batch_size):
            kspace = read_kspace_multiple_file(file_names[ifname:ifname+self.batch_size], self.inputs_shape)
            pred = self.prediction_process(kspace * self.mask)
            images = kspace_to_images(kspace)
            images = norm_images_center_value(images)
            error = self.eval_function(images, pred)
            del kspace
            iend = ifname+self.batch_size
            iend = iend if iend < testing_size else testing_size
            print(f'({ifname+1:03d}-{iend:03d} of {testing_size:03d}_{k_fold})\tloss: {error:e}')
            loss.append(error)
        tf.keras.backend.clear_ssession()
        loss = np.mean(loss)
        return loss
    
    def __preprocessing_step1(self, kspace):
        # data preprocessing
        kspace = norm_kspace(kspace)
        # _, images, coil_images = coil_sensitivities(kspace)
        coil_images = kspace_to_images(kspace, rss=False)
        images = fastmri.rss(coil_images, dim=1)
        sensitivities_compress, images_compress, coil_images_compress = coil_sensitivities(kspace * self.mask) # under-sampling
        # transform images to range [-1, 1]
        images = norm_images_center_value(images)
        coil_images = norm_images_center_value(coil_images)
        sensitivities_compress = norm_images_center_value(sensitivities_compress)
        images_compress = norm_images_center_value(images_compress)
        coil_images_compress, norm_values = norm_images_center_value(coil_images_compress, keep_inverse=True)
        # concatenate coil and sensitivities for train
        X1 = tf.concat([coil_images_compress, sensitivities_compress], axis=1) # data shape (None, 15, n, m)
        X1 = np.moveaxis(X1.numpy(), 1, -1)
        y1 = np.moveaxis(coil_images.numpy(), 1, -1)
        return X1, y1, images, norm_values
    
    def __preprocessing_step2(self, kspace, coil_output, norm_values):
        coil_output = np.moveaxis(coil_output, -1, 1)
        coil_output = inverse_norm_images_center_value(coil_output, *norm_values)
        # transform predict muti-coil images to frequency domain
        coil_output = images_to_kspace(coil_output)
        # apply mask
        coil_output = (kspace * self.mask) + (coil_output * ~self.mask)
        # transform prediction images from ifft to fft
        coil_output = kspace_to_images(coil_output)
        coil_output = np.expand_dims(coil_output, -1)
        pass
    
    def training_process(self, model_coil, model_compress, kspace, mask):
        # data preprocessing
        kspace = norm_kspace(kspace)
        # _, images, coil_images = coil_sensitivities(kspace)
        coil_images = kspace_to_images(kspace, rss=False)
        images = fastmri.rss(coil_images, dim=1)
        sensitivities_compress, images_compress, coil_images_compress = coil_sensitivities(kspace * mask) # under-sampling
        # transform images to range [-1, 1]
        images = norm_images_center_value(images)
        coil_images = norm_images_center_value(coil_images)
        sensitivities_compress = norm_images_center_value(sensitivities_compress)
        images_compress = norm_images_center_value(images_compress)
        coil_images_compress, norm_values = norm_images_center_value(coil_images_compress, keep_inverse=True)
        # concatenate coil and sensitivities for train
        X = tf.concat([coil_images_compress, sensitivities_compress], axis=1) # data shape (None, 15, n, m)
        X = np.moveaxis(X.numpy(), 1, -1)
        y = np.moveaxis(coil_images.numpy(), 1, -1)
        # train model muti-coil
        h = model_coil.fit(X, y, epochs=1, verbose=0)
        model_coil_error = h.history['loss'][0]
        # prediction multi-coil images
        coil_output = model_coil.predict(X)
        coil_output = np.moveaxis(coil_output, -1, 1)
        coil_output = inverse_norm_images_center_value(coil_output, *norm_values)
        # transform predict muti-coil images to frequency domain
        coil_output = images_to_kspace(coil_output)
        # apply mask
        coil_output = (kspace * mask) + (coil_output * ~mask)
        # transform prediction images from ifft to fft
        coil_output = kspace_to_images(coil_output)
        coil_output = np.expand_dims(coil_output, -1)
        # train model reconstruction
        h = model_compress.fit(coil_output, images.numpy(), epochs=1, verbose=0)
        model_compress_error = h.history['loss'][0]
        return model_coil_error, model_compress_error

    def prediction_process(self, model_coil, model_compress, kspace, mask):
        # data preprocessing
        kspace = norm_kspace(kspace)
        sensitivities, images, coil_images = coil_sensitivities(kspace)
        sensitivities = norm_images_center_value(sensitivities)
        images = norm_images_center_value(images)
        coil_images, norm_values = norm_images_center_value(coil_images, keep_inverse=True)
        X = tf.concat([coil_images, sensitivities], axis=1) # data shape (None, 15, n, m)
        X = np.moveaxis(X.numpy(), 1, -1)
        # prediction multi-coil images
        coil_output = model_coil.predict(X)
        coil_output = np.moveaxis(coil_output, -1, 1)
        coil_output = inverse_norm_images_center_value(coil_output, *norm_values)
        # transform predict muti-coil images to frequency domain
        coil_output = images_to_kspace(coil_output)
        # apply mask
        coil_output = (kspace * mask) + (coil_output * ~mask)
        # transform prediction images from ifft to fft
        coil_output = kspace_to_images(coil_output).numpy()
        coil_output = np.expand_dims(coil_output, -1)
        # prediction model reconstruction
        pred = model_compress.predict(coil_output)
        pred = np.squeeze(pred)
        return pred
    