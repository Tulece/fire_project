# detection_incendie/detection/model/train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Définir les chemins de données
train_dir = 'detection_incendie/data/train'
test_dir = 'detection_incendie/data/test'

# Préparation des données avec Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Générateurs de données
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Construire le modèle (par exemple, en utilisant un modèle pré-entraîné)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False  # Geler les couches du modèle de base

# Ajouter des couches personnalisées
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10
)

# Sauvegarder le modèle
model.save('detection_incendie/detection/model/fire_detection_model.h5')
