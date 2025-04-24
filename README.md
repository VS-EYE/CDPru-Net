# CDPru-Net
This repository contains an implementation of Novel ClinicaNet Weights for Accurate Multi-Disease to Single-Disease Diagnosis for image classification with a custom Chirplet Transform Layer. The model is used for classifying medical images into categories diverse medical disease. The model is trained using TensorFlow and Keras, with data augmentation techniques applied to the training set.

Requirements
Python 3.x
TensorFlow 2.x
Keras
Scikit-learn
Matplotlib
Numpy
OpenCV

Other dependencies as listed in the requirements.txt
Installation
You can set up the environment by running the following commands:
pip install -r requirements.txt

Alternatively, you can manually install the dependencies:
pip install tensorflow keras scikit-learn matplotlib numpy opencv-python
Dataset:
The dataset is expected to be organized in the following directory structure:
- Train/
- Validate/
- Test/

Model Architecture
Depthwise Convolution: Optimized convolution for efficiency.
Dense Block: A series of densely connected layers to improve learning capacity.
Chirplet Transform Layer: A custom layer that applies the Chirplet Transform to extract additional features from the data.
ConvLSTM Layer: For spatial-temporal feature extraction in case of dynamic image data.

Custom Layers:
Chirplet Transform Layer: This layer is designed for applying Chirplet Transform on the input tensor. The actual transform logic is currently a placeholder and needs to be implemented for specific applications.

Model Training
Data Augmentation
The model training includes data augmentation to improve generalization:
Horizontal flip
Vertical flip
Random rotations between 90 degrees

Callbacks
ModelCheckpoint: Saves the best model based on validation accuracy.
ReduceLROnPlateau: Reduces the learning rate when the validation loss plateaus.


start = time.time()
model = model.fit(
    train_gen, 
    batch_size=BATCH_SIZE, 
    epochs=n_epochs, 
    validation_data=valid_gen, 
    callbacks=[tl_checkpoint_1, reducelr], 
    verbose=1
)

Model Evaluation

After training, the model can be evaluated on the test data:

model = keras.models.load_model('/path/to/saved/model.h5')
prediction = np.argmax(model.predict(test_gen), axis=1)
accuracy = accuracy_score(test_gen.classes, prediction) * 100
print("Test Data accuracy: ", accuracy)


Results
After evaluating the model on the test set, you can print the detailed classification metrics:
print("Classification Report (with 4 decimal places):")


for label, metrics in report.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"{label}: Precision = {metrics['precision']}, Recall = {metrics['recall']}, F1-score = {metrics['f1-score']}")

Model File
The trained model is saved as CDPru-Net.h5. You can load it using the following code:
model = keras.models.load_model('/path/to/saved/model.h5', custom_objects={'ChirpletTransformLayer': ChirpletTransformLayer})
