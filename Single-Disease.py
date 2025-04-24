import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import time, pickle, glob, os, shutil
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
from sklearn.metrics import cohen_kappa_score
#from vit_keras import vit
#from keras_flops import get_flops

Width_Imgs, Heigth_Imgs = 224, 224
Channal_Imags = 3
BATCH_SIZE = 64
shapeImage = (Width_Imgs, Heigth_Imgs, Channal_Imags)

TrainDS_Path =  "/root/public/tf211/CU/Train"
ValidDS_Path =  "/root/public/tf211/CU/Validate"
TestDS_Path =   "/root/public/tf211/CU/Test"

#trgen = ImageDataGenerator(rescale=1. / 255)


# For training data with augmentations
trgen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,  # Apply horizontal flip
    vertical_flip=True,    # Apply vertical flip
    rotation_range=40      # Apply random rotations between -40 and 40 degrees
)


validation_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = trgen.flow_from_directory(
    TrainDS_Path,
    #classes = ['benign','BA-impetigo','colon_aca','Colorectal cancer', 'COVID' ,'Esophagitis', 'glioma' , 'High_squamous', 'Low_squamous','Lung_Opacity','malignant','meningioma','Negative','pituitary','Pylorus','SCC','VI-chickenpox', 'Viral Pneumonia'],
    classes = ['benign','malignant','normal'],
    target_size=(Width_Imgs, Heigth_Imgs),
    batch_size=BATCH_SIZE,
    shuffle=True)

valid_gen = validation_datagen.flow_from_directory(
    ValidDS_Path,
    classes = ['benign','malignant','normal'],
    target_size=(Width_Imgs, Heigth_Imgs),
    batch_size=BATCH_SIZE,
    shuffle=True)

test_gen = test_datagen.flow_from_directory(
    TestDS_Path,
    classes = ['benign','malignant','normal'],
    target_size=(Width_Imgs, Heigth_Imgs),
    batch_size=BATCH_SIZE,
    shuffle=False)

# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath='/root/public/tf211/Models/M7.h5',
                                  save_best_only=True,
                                  monitor="val_accuracy",
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


def conv_block(inputs, filters, kernel_size, strides=(1, 1), name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', name=f"M1_{name}_conv")(inputs)
    x = BatchNormalization(name=f"M1_{name}_batchnorm")(x)
    x = LeakyReLU(name=f"M1_{name}_leakyrelu")(x)
    return x

def depthwise_conv_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1), depth_multiplier=1, name=None):
    x = DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=depth_multiplier, padding='same', name=f"M1_{name}_depthwiseconv")(inputs)
    x = BatchNormalization(name=f"M1_{name}_batchnorm")(x)
    x = Activation('relu', name=f"M1_{name}_relu")(x)
    
    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name=f"M1_{name}_conv1x1")(x)
    x = BatchNormalization(name=f"M1_{name}_batchnorm_conv1x1")(x)
    x = Activation('relu', name=f"M1_{name}_relu_conv1x1")(x)
    return x

def dense_block(x, num_layers, growth_rate, name=None):
    for i in range(num_layers):
        y = Dense(growth_rate, activation='relu', name=f"M1_{name}_dense_{i+1}")(x)
        x = concatenate([x, y], axis=-1, name=f"M1_{name}_concat_{i+1}")
    return x



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
    

def MobileNetV1(input_shape=(224, 224, 3), n_classes=3):
    input = Input(shape=input_shape, name="M1_input")

    ## Block - 1 InitialDownsamplingBlock
    x = conv_block(input, 32, (3, 3), strides=(2, 2), name="conv_block_1")
    x = depthwise_conv_block(x, 64, depth_multiplier=1, name="depthwise_conv_block_1")
    x = depthwise_conv_block(x, 128, strides=(2, 2), name="depthwise_conv_block_2")
    x = depthwise_conv_block(x, 128, strides=(1, 1), name="depthwise_conv_block_3")

    ## Block - 2 MidDownsamplingBlock
    x = depthwise_conv_block(x, 256, strides=(2, 2), name="depthwise_conv_block_4")
    x = depthwise_conv_block(x, 256, strides=(1, 1), name="depthwise_conv_block_5")

    ## Block - 3 BottleneckBlock
    x = depthwise_conv_block(x, 512, strides=(2, 2), name="depthwise_conv_block_6")
    x = depthwise_conv_block(x, 512, strides=(1, 1), name="depthwise_conv_block_7")
    
    # Dense Block with 4 layers and growth rate 8
    x1 = dense_block(x, num_layers=4, growth_rate=8, name="dense_block_1")
    #x1 = depthwise_conv_block(x, 512, strides=(1, 1), name="depthwise_conv_block_8")
    
    x = depthwise_conv_block(x1, 512, strides=(1, 1), name="depthwise_conv_block_9")
    
    # Dense Block with 4 layers and growth rate 8
    x2 = dense_block(x, num_layers=4, growth_rate=8, name="dense_block_2")
    #x2 = depthwise_conv_block(x, 1024, strides=(1, 1), name="depthwise_conv_block_10")

    ## Block - 4 FinalDownsamplingBlock    
    x = depthwise_conv_block(x2, 1024, strides=(2, 2), name="depthwise_conv_block_11")
    
    ### ConvLSTM Layer (for spatial-temporal feature extraction)
    x = Reshape((1, x.shape[1], x.shape[2], x.shape[3]))(x)  # Reshape to fit ConvLSTM2D input
    x = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=False, name="M1_ConvLSTM")(x)
    
    ### ClassifierBlock
    #x = GlobalAveragePooling2D(name="M1_global_avg_pooling")(x)
    x = Flatten(name="M1_flatten")(x)
    #x = ChirpletTransformLayer()(x)
    x = Dense(512, name="M1_dense_1")(x)
    x = BatchNormalization(name="M1_batch_norm")(x)
    x = Activation('relu', name="M1_activation")(x)
    x = Dropout(0.2, name="M1_dropout")(x)
    
    # Remove the original output layer
    # Add a new output layer with the correct number of classes
    x = Dense(n_classes, activation='softmax', name="M1_output")(x)
    
    model = Model(inputs=input, outputs=x, name="M1_MobileNetV1")
    
    # Load the weights, excluding the final classification layers
    model.load_weights('/root/public/tf211/ADD-DB.h5', by_name=True, skip_mismatch=True)
  
    return model
    
# Create the model
model = MobileNetV1()

optim_1 = Adam(learning_rate=0.001)
n_epochs = 30


# Compile the model
model.compile(optimizer=optim_1, loss='categorical_crossentropy', metrics=METRICS)

start = time.time()
model = model.fit(train_gen , batch_size=BATCH_SIZE, epochs=n_epochs, validation_data=valid_gen , callbacks=[tl_checkpoint_1, reducelr], verbose=1)
Elapsed = time.time()-start
print (f'Training time: {hms_string(Elapsed)}')

# ImageSize = Width_Imgs, Heigth_Imgs
# First we'll train the model without Fine-tuning
#Train_model = create_model(shapeImage, n_classes, optim_1)
#print(model.summary())

model = keras.models.load_model('/root/public/tf211/Models/M7.h5', custom_objects={'ChirpletTransformLayer': ChirpletTransformLayer})
# Get the weights
#model_weights = model.get_weights()

#print(model.summary())
start = time.time()
prediction = np.argmax(model.predict(test_gen), axis=1)
print('Test Data accuracy: ', accuracy_score(test_gen.classes, prediction) * 100)

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


Elapsed = time.time() - start
print(f'Training time: {hms_string(Elapsed)}')
