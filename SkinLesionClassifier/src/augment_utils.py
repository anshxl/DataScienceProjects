# Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import h5py
from PIL import Image
import io
import pandas as pd

# Function to load and preprocess image (resizing and standardization)
def byte_string_to_resized_image_array(byte_string, target_size=(128, 128)):
    image_bytes = io.BytesIO(byte_string)
    image = Image.open(image_bytes)
    image = image.resize(target_size)
    image_array = np.array(image).astype(np.float32)  # Convert to float for TensorFlow
    return image_array

# Function to extract image data from HDF5 file and return it in array format
def load_image_from_hdf5(isic_id, hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        byte_string = f[isic_id][()]
        image_array = byte_string_to_resized_image_array(byte_string)
    return image_array

# Function to create data generators for benign and malignant cases
def data_generator(metadata, benign_ids, malignant_ids, hdf5_file, batch_size=32, target_size=(128, 128), malignant_augment_limit=4000):
    benign_count = 0
    malignant_count = 0
    
    while True:
        images = []
        labels = []
        
        for index, row in metadata.iterrows():
            isic_id = row['isic_id']
            target = row['target']  # Binary label
            
            # Load image based on isic_id
            image_array = load_image_from_hdf5(isic_id, hdf5_file)
            
            # Apply augmentation for malignant cases
            if isic_id in malignant_ids:
                if malignant_count < malignant_augment_limit:  # Augment only until the limit is reached
                    augmented_image = malignant_datagen.random_transform(image_array)  # Augmentation
                    images.append(augmented_image)
                    labels.append(1)  # Malignant label
                    malignant_count += 1  # Keep track of augmented malignant samples
                else:
                    # After reaching the limit, continue but don't augment malignant images
                    standardized_image = malignant_datagen.standardize(image_array)  # Just standardize after the limit
                    images.append(standardized_image)
                    labels.append(1)  # Still malignant label

            # Process benign cases without augmentation
            elif isic_id in benign_ids:
                benign_image = benign_datagen.standardize(image_array)  # Standardize without augmentation
                images.append(benign_image)
                labels.append(0)  # Benign label
                benign_count += 1
            
            # Yield the batch when enough samples are collected
            if len(images) == batch_size:
                print(f"Benign samples processed: {benign_count}, Malignant samples augmented: {malignant_count}")
                yield np.array(images), np.array(labels)
                images = []  # Reset for the next batch
                labels = []
        
        # Optional: print the final counts of benign and malignant samples processed
        print(f"Total Benign samples processed: {benign_count}, Malignant samples augmented: {malignant_count}")

