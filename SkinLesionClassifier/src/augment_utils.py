# Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import h5py
from PIL import Image
import io
import pandas as pd
from load_utils import load_image_from_hdf5, byte_string_to_resized_image_array

def data_generator(metadata, benign_ids, malignant_ids, hdf5_file, batch_size=32, 
                                          benign_datagen=None, malignant_datagen=None, 
                                          target_size=(128, 128), malignant_augmentations_per_sample=10, 
                                          num_epochs=10, steps_per_epoch=None):
    """
    A data generator that iterates through the dataset for each epoch, 
    generating multiple augmentations per malignant sample.
    """
    if steps_per_epoch is None:
        steps_per_epoch = len(metadata) // batch_size  # Calculate steps if not provided

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Initialize counters and batch lists
        benign_count = 0
        malignant_count = 0
        step = 0
        
        images, labels, metadata_features = [], [], []
        
        for index, row in metadata.iterrows():
            isic_id = row['isic_id']
            target = row['target']  # Binary label
            meta_features = row.drop(labels=['isic_id', 'target']).values.astype(np.float32)  # Metadata features
            
            # Load image based on isic_id
            image_array = load_image_from_hdf5(isic_id, hdf5_file, target_size=target_size)
            
            # Handle malignant samples with multiple augmentations
            if isic_id in malignant_ids:
                for _ in range(malignant_augmentations_per_sample):
                    augmented_image = malignant_datagen.random_transform(image_array)
                    images.append(augmented_image)
                    labels.append(1)  # Malignant label
                    metadata_features.append(meta_features)
                    malignant_count += 1

                    # Check if batch size is reached
                    if len(images) == batch_size:
                        yield [np.array(images), np.array(metadata_features)], np.array(labels)
                        images, labels, metadata_features = [], [], []  # Reset for next batch
                        step += 1
                        if step >= steps_per_epoch:
                            break
            
            # Handle benign samples without augmentation
            elif isic_id in benign_ids:
                benign_image = benign_datagen.standardize(image_array)
                images.append(benign_image)
                labels.append(0)  # Benign label
                metadata_features.append(meta_features)
                benign_count += 1

                # Check if batch size is reached
                if len(images) == batch_size:
                    yield [np.array(images), np.array(metadata_features)], np.array(labels)
                    images, labels, metadata_features = [], [], []  # Reset for next batch
                    step += 1
                    if step >= steps_per_epoch:
                        break
        
        # Optional: print batch summary
        print(f"Total benign samples processed: {benign_count}, Total malignant samples augmented: {malignant_count}")

def test_data_generator(metadata, hdf5_file, batch_size=32, datagen=None, target_size=(128, 128)):
    images = []
    metadata_features = []
    
    for index, row in metadata.iterrows():
        isic_id = row['isic_id']
        meta_features = row.drop(labels=['isic_id']).values.astype(np.float32)  # Metadata features only

        # Load and standardize the image
        image_array = load_image_from_hdf5(isic_id, hdf5_file)
        standardized_image = datagen.standardize(image_array)
        
        images.append(standardized_image)
        metadata_features.append(meta_features)

        # Yield each batch
        if len(images) == batch_size:
            yield [np.array(images), np.array(metadata_features)]
            images = []  # Reset for next batch
            metadata_features = []

    # Yield any remaining data in the final, incomplete batch
    if len(images) > 0:
        yield [np.array(images), np.array(metadata_features)]
