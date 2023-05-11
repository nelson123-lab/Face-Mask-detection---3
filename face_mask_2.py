import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

# Load and preprocess the dataset
data = np.load('dataset.npy')
labels = np.load('labels.npy')

# Split the data into training and testing sets
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)

# Preprocess the images
trainX = tf.keras.applications.mobilenet_v2.preprocess_input(trainX)
testX = tf.keras.applications.mobilenet_v2.preprocess_input(testX)

# Create data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the MobileNetV2 base model
baseModel = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation='sigmoid')(headModel)

# Create the model by combining the base and head
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the base layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
opt = Adam(learning_rate=1e-4)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
history = model.fit(
    datagen.flow(trainX, trainY, batch_size=32),
    steps_per_epoch=len(trainX) // 32,
    validation_data=(testX, testY),
    validation_steps=len(testX) // 32,
    epochs=20
)

# Evaluate the model
preds = model.predict(testX)
preds = np.where(preds > 0.5, 1, 0)
print(classification_report(testY, preds))

# Save the model
model.save('face_mask_detection_model.h5')
