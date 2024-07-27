import io
from base64 import encodebytes

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons
from PIL import Image, ImageEnhance
from keras.models import load_model
from keras.preprocessing.image import image_utils
from tensorflow.keras.preprocessing import image as image_utils

from f1_score import F1Score
from preprocessing import preprocess_image


def preprocessing(img_path):
    image = Image.open(img_path)
    resized_image = image.resize((256, 256))
    normalized_image = np.asarray(resized_image) / 255.0
    enhancer = ImageEnhance.Contrast(resized_image)
    contrast_image = enhancer.enhance(2)
    image_array = np.asarray(resized_image)
    if len(image_array.shape) == 2:
        gray_image = image_array
    elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unexpected number of channels in input image")
    edges = cv2.Canny(gray_image, 100, 200)
    preprocess_image = cv2.merge([edges] * 3)
    _, segmented_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2RGB)
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    images = [
        image,
        Image.fromarray((normalized_image * 255).astype(np.uint8)),
        contrast_image,
        Image.fromarray(preprocess_image),
        Image.fromarray(segmented_image),
        Image.fromarray(image_array)
    ]
    return images


def get_response_image(pil_img):
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    return encoded_img


def predict(img_path, model_path):
    if model_path.find('efficientnet_b0_alz.h5') != -1:
        model = load_model(model_path, custom_objects={'F1Score': tensorflow_addons.metrics.F1Score})
        test_image1 = preprocess_image(img_path)
        test_image1.save(img_path)
    else:
        model = load_model(model_path, custom_objects={'F1Score': F1Score})
    test_image = image_utils.load_img(img_path, target_size=(224, 224))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image, batch_size=1)
    print(result)
    if result[0][1] == 0:
        return "Mild Demented"
    elif result[0][1] == 1:
        return "Moderate Demented"
    elif result[0][1] == 2:
        return "Non Demented"
    else:
        return "Very Mild Demented"


def plot_metrics(history):
    metrics = ['loss'] + [m for m in history.history.keys() if not m.startswith('val_')]
    fig, axes = plt.subplots(nrows=(len(metrics) + 1) // 2, ncols=2, figsize=(20, 5 * ((len(metrics) + 1) // 2)))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.plot(history.history[metric], label='Train')
        if f'val_{metric}' in history.history:
            ax.plot(history.history[f'val_{metric}'], label='Validation')
        ax.set_title(f'{metric.capitalize()} Over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    plt.show()
