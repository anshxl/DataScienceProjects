import numpy as np
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
def load_image_from_hdf5(isic_id, hdf5_file, target_size=(128, 128)):
    with h5py.File(hdf5_file, 'r') as f:
        byte_string = f[isic_id][()]
        image_array = byte_string_to_resized_image_array(byte_string, target_size)
    return image_array