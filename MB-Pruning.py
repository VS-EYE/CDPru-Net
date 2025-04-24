import pandas as pd
import numpy as np
import time, pickle, glob, os
import random
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras import layers, models
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from keras.callbacks import *
from keras.models import Model
from tensorflow.keras.models import *
from keras.applications import *
from tensorflow.keras.layers import *
from sklearn import metrics
from sklearn.metrics import *
from keras_flops import get_flops

Width_Imgs, Heigth_Imgs = 224,224
Channal_Imags = 3
BATCH_SIZE = 64
shapeImage = (Width_Imgs, Heigth_Imgs, Channal_Imags)


TrainDS_Path = "/root/public/tf211/P5/DS/PP/Train"
ValidDS_Path = "/root/public/tf211/P5/DS/PP/Validate"
TestDS_Path  =  "/root/public/tf211/P5/DS/PP/Test"


trgen = ImageDataGenerator(rescale=1. / 255)


validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = trgen.flow_from_directory(
    TrainDS_Path,
    #classes=['Dementia' ,'Non Demented'],
    classes = ['benign','BA-impetigo','colon_aca','Colorectal cancer', 'COVID' ,'Esophagitis', 'glioma' , 'High_squamous', 'Low_squamous','Lung_Opacity','malignant','meningioma','Negative','pituitary','Pylorus','SCC','VI-chickenpox', 'Viral Pneumonia'],
    target_size=(Width_Imgs, Heigth_Imgs),
    batch_size=BATCH_SIZE,
    shuffle=True )

valid_gen = validation_datagen.flow_from_directory(
    ValidDS_Path,
    target_size=(Width_Imgs, Heigth_Imgs),
    batch_size=BATCH_SIZE,
      classes =['benign','BA-impetigo','colon_aca','Colorectal cancer', 'COVID' ,'Esophagitis', 'glioma' , 'High_squamous', 'Low_squamous','Lung_Opacity','malignant','meningioma','Negative','pituitary','Pylorus','SCC','VI-chickenpox', 'Viral Pneumonia'],
    shuffle=True)


test_gen = test_datagen.flow_from_directory(
    TestDS_Path,
       classes = ['benign','BA-impetigo','colon_aca','Colorectal cancer', 'COVID' ,'Esophagitis', 'glioma' , 'High_squamous', 'Low_squamous','Lung_Opacity','malignant','meningioma','Negative','pituitary','Pylorus','SCC','VI-chickenpox', 'Viral Pneumonia'],
    target_size=(Width_Imgs, Heigth_Imgs),
    
    batch_size=BATCH_SIZE,
    shuffle=False)



# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath= '/root/public/tf211/CU-MDP/P2/Pruning/ADD-DB.h5',
                                  save_best_only=True,
                                  monitor = "val_accuracy",
                                  verbose=1)
reducelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=3,
    verbose=0,
    mode="auto",
    min_delta=0.00001,
    cooldown=0,
    min_lr=0)


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


METRICS = [keras.metrics.CategoricalAccuracy(name='accuracy'),
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall'),
           keras.metrics.AUC(name='auc')]

def inception_module(x, filters):
    # 1x1 Convolution
    conv1x1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)

    # 3x3 Convolution
    conv3x3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)

    # 5x5 Convolution
    conv5x5 = Conv2D(filters, (5, 5), padding='same', activation='relu')(x)

    # Max Pooling
    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    max_pool_conv = Conv2D(filters, (1, 1), padding='same', activation='relu')(max_pool)

    # Concatenate the outputs
    inception_output = Concatenate()([conv1x1, conv3x3, conv5x5, max_pool_conv])

    return inception_output

def inception_moduleM(x, filters):
    # 1x1 Convolution
    conv1x1 = Conv2D(filters, (1, 1), padding='same')(x)
    conv1x1 = LeakyReLU(alpha=0.1)(conv1x1)

    # 3x3 Convolution
    conv3x3 = Conv2D(filters, (3, 3), padding='same')(x)
    conv3x3 = LeakyReLU(alpha=0.1)(conv3x3)

    # 5x5 Convolution
    conv5x5 = Conv2D(filters, (5, 5), padding='same')(x)
    conv5x5 = LeakyReLU(alpha=0.1)(conv5x5)

    # Max Pooling
    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    max_pool_conv = Conv2D(filters, (1, 1), padding='same')(max_pool)
    max_pool_conv = LeakyReLU(alpha=0.1)(max_pool_conv)

    # Concatenate the outputs
    inception_output = Concatenate()([conv1x1, conv3x3, conv5x5, max_pool_conv])

    return inception_output


def naive_inception_moduleM(input_tensor, num_filters):
    # 1x1 Convolution for dimension reduction
    conv1x1 = Conv2D(num_filters, (1, 1), padding='same', activation='gelu')(input_tensor)
    conv1x1 = LeakyReLU(alpha=0.2)(conv1x1)

    # 3x3 Convolution followed by 1x1 Convolution for dimension reduction
    conv3x3 = Conv2D(num_filters, (1, 1), padding='same', activation='gelu')(input_tensor)
    conv3x3 = LeakyReLU(alpha=0.2)(conv3x3)
    conv3x3 = Conv2D(num_filters, (3, 3), padding='same', activation='gelu')(conv3x3)
    conv3x3 = LeakyReLU(alpha=0.2)(conv3x3)

    # 5x5 Convolution followed by 1x1 Convolution for dimension reduction
    conv5x5 = Conv2D(num_filters, (1, 1), padding='same', activation='gelu')(input_tensor)
    conv5x5 = LeakyReLU(alpha=0.2)(conv5x5)
    conv5x5 = Conv2D(num_filters, (5, 5), padding='same', activation='gelu')(conv5x5)
    conv5x5 = LeakyReLU(alpha=0.2)(conv5x5)

    # Max pooling followed by 1x1 Convolution for dimension reduction
    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    max_pool_conv = Conv2D(num_filters, (1, 1), padding='same', activation='gelu')(max_pool)
    max_pool_conv = LeakyReLU(alpha=0.2)(max_pool_conv)
    
  

    # Concatenate the outputs along the channel axis
    inception_module = concatenate([conv1x1, conv3x3, conv5x5, max_pool_conv], axis=-1)

    return inception_module

def naive_inception_module(input_tensor, num_filters):
    # 1x1 Convolution for dimension reduction
    conv1x1 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)

    # 3x3 Convolution followed by 1x1 Convolution for dimension reduction
    conv3x3 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
    conv3x3 = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(conv3x3)

    # 5x5 Convolution followed by 1x1 Convolution for dimension reduction
    conv5x5 = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
    conv5x5 = Conv2D(num_filters, (5, 5), padding='same', activation='relu')(conv5x5)

    # Max pooling followed by 1x1 Convolution for dimension reduction
    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    max_pool_conv = Conv2D(num_filters, (1, 1), padding='same', activation='relu')(max_pool)
    
    # Concatenate the outputs along the channel axis
    inception_module = concatenate([conv1x1, conv3x3, conv5x5, max_pool_conv], axis=-1)

    return inception_module


def cbam_block(cbam_feature, ratio=8):

   
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
   
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
   
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
   
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
   
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
   
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
   
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature
   
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1
   
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
       
    return multiply([input_feature, cbam_feature])

def se_block(input_tensor, ratio=16):
    num_channels = input_tensor.shape[-1]
    squeeze = GlobalAveragePooling2D()(input_tensor)
    excitation = Dense(num_channels // ratio, activation='relu')(squeeze)
    excitation = Dense(num_channels, activation='sigmoid')(excitation)
    excitation = Reshape((1, 1, num_channels))(excitation)
    scaled_input = multiply([input_tensor, excitation])
    return scaled_input


def residual_block(input_tensor, filters, kernel_size=(3, 3), strides=(1, 1), activation='relu'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    if strides != (1, 1) or input_tensor.shape[-1] != filters:
        shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding='same')(input_tensor)
        shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
    else:
        x = Add()([x, input_tensor])
    x = Activation(activation)(x)
    return x

def dense_block(x, num_layers, growth_rate):

    for _ in range(num_layers):
        # Each layer is connected to all previous layers via concatenation
        y = Dense(growth_rate, activation='relu')(x)
        x = concatenate([x, y])
    
    return x

    

def conv_block(inputs, filters, kernel_size, strides=(1, 1)):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    #x = ReLU()(x)
    x = LeakyReLU()(x)
    return x

def depthwise_conv_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1), depth_multiplier=1):
    x = DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #x = LeakyReLU()(x)
    
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #x = LeakyReLU()(x)
    return x



class FourierTransformLayer(Layer):
    def __init__(self, **kwargs):
        super(FourierTransformLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Use TensorFlow's FFT implementation
        fft_result = tf.signal.fft(tf.cast(inputs, tf.complex64))
        return tf.math.real(fft_result)  # Return only the real part or modify as needed

    def compute_output_shape(self, input_shape):
        # The output shape is the same as input shape but might need adjustment based on your FFT implementation
        return input_shape

class STFTLayer(Layer):
    def __init__(self, frame_length=256, frame_step=128, **kwargs):
        super(STFTLayer, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step

    def call(self, inputs):
        # Compute the Short-Time Fourier Transform
        stft_result = tf.signal.stft(inputs, frame_length=self.frame_length, frame_step=self.frame_step)
        return tf.math.abs(stft_result)  # Return the magnitude of the STFT

    def compute_output_shape(self, input_shape):
        # The output shape will depend on the input shape and STFT parameters
        return (input_shape[0], (input_shape[1] - self.frame_length) // self.frame_step + 1, self.frame_length // 2 + 1)  # Adjust this as per your STFT implementation

    def get_config(self):
        config = super(STFTLayer, self).get_config()
        config.update({
            'frame_length': self.frame_length,
            'frame_step': self.frame_step,
        })
        return config


class ChirpletTransformLayer(Layer):
    def __init__(self, **kwargs):
        super(ChirpletTransformLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Placeholder for the Chirplet Transform
        # Implement your Chirplet Transform logic here
        chirplet_result = self.chirplet_transform(inputs)
        return chirplet_result

    def chirplet_transform(self, x):
        # Perform Chirplet Transform here
        # This is a placeholder. You need to implement the actual transform logic.
        # This function should return the transformed tensor.
        
        # Example: Just returning the input for demonstration
        # Replace this with actual Chirplet transform logic
        return x  # This should be replaced with the Chirplet result

    def compute_output_shape(self, input_shape):
        # The output shape should be the same as the input shape
        return input_shape


def MobileNetV1(input_shape=(224, 224, 3), n_classes=18):
    input = Input(shape=input_shape)


    ## Block - 1 InitialDownsamplingBlock
    x = conv_block(input, 32, (3, 3), strides=(2, 2))
    x = depthwise_conv_block(x, 64, depth_multiplier=1)
    x = depthwise_conv_block(x, 128, strides=(2, 2))
    x = depthwise_conv_block(x, 128, strides=(1, 1))


    ## Block - 2 MidDownsamplingBlock
    x = depthwise_conv_block(x, 256, strides=(2, 2))
    x = depthwise_conv_block(x, 256, strides=(1, 1))

    ## Block - 3 BottleneckBlock
    x = depthwise_conv_block(x, 512, strides=(2, 2))
    x = depthwise_conv_block(x, 512, strides=(1, 1))
    x1 = depthwise_conv_block(x, 512, strides=(1, 1))
    #x1 = dense_block(x, num_layers = 4, growth_rate = 8)
    
    x = depthwise_conv_block(x1, 512, strides=(1, 1))
    x2 = depthwise_conv_block(x, 512, strides=(1, 1))
    #x2 = dense_block(x, num_layers = 4, growth_rate = 8)
    #x = depthwise_conv_block(x, 512, strides=(1, 1))
    #x3 = inception_moduleM(x, filters=16)

    ## Block - 4 FinalDownsamplingBlock    
    x = depthwise_conv_block(x2, 1024, strides=(2, 2))
    #x = depthwise_conv_block(x, 1024, strides=(1, 1))
   
    #x5 = concatenate([x1, x2], axis=-1) 
    
    #input_ = tf.expand_dims(x5, axis = 1)
    #x5 = ConvLSTM2D(filters=64, kernel_size=(1,1),padding = "same")(input_)

    #x = Flatten()(x)


    ### ClassifierBlock
    #x = STFTLayer(frame_length=256, frame_step=64)(x)
    x = GlobalAveragePooling2D()(x)
    #x = FourierTransformLayer()(x) 
   
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=input, outputs=x)
 #   model.load_weights('/root/public/tf211/P5/model_weights.h5', by_name=True)
    return model

# Create the model
model = MobileNetV1()

optim_1 = Adam(learning_rate=0.001)
n_epochs = 20

# Compile the model
model.compile(optimizer=optim_1, loss='categorical_crossentropy', metrics=METRICS)

start = time.time()
#model = model.fit(train_gen , batch_size=BATCH_SIZE, epochs=n_epochs, validation_data=valid_gen , callbacks=[tl_checkpoint_1, reducelr], verbose=1)
Elapsed = time.time()-start
print (f'Training time: {hms_string(Elapsed)}')

#model= load_model("/root/public/tf211/CU-MDP/P2/Pruning/ADD-DB.h5")
#model = load_model('/root/public/tf211/CU-MDP/P2/Pruning/ADD-FT.h5', custom_objects={'FourierTransformLayer': FourierTransformLayer})
print (model.summary())


start = time.time()
prediction = np.argmax(model.predict(test_gen), axis=1)
print('Test Data accuracy: ',accuracy_score(test_gen.classes, prediction)*100)
Elapsed = time.time()-start
print (f'Training time: {hms_string(Elapsed)}')
print("F1-Score", f1_score(test_gen.classes, prediction, average='macro'))
print("Recall", recall_score(test_gen.classes, prediction, average='macro'))
print("Precision", precision_score(test_gen.classes, prediction, average='macro'))
report = classification_report(test_gen.classes, prediction, output_dict=True)

# Format the results to four decimal places
for label, metrics in report.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:  # Ignore summary lines
        for metric, value in metrics.items():
            report[label][metric] = f"{value:.4f}"

# Print the formatted report
print("Classification Report (with 4 decimal places):")
for label, metrics in report.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:  # Ignore summary lines
        print(f"{label}: Precision = {metrics['precision']}, Recall = {metrics['recall']}, F1-score = {metrics['f1-score']}")
print(confusion_matrix(test_gen.classes, prediction))
print("Cohen", cohen_kappa_score(test_gen.classes, prediction))
flops = get_flops(model, batch_size=64)
# Print FLOPS in billions (G)
#print(f"FLOPS: {flops / 10 ** 9:.03} G")
# Print FLOPS in millions (M)
print(f"FLOPS: {flops / 10 ** 6:.03} M")






