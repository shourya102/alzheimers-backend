import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def resize_image(image, size=(256, 256)):
    return image.resize(size)


def normalize_image(image):
    return np.asarray(image) / 255.0


def adjust_contrast(image, factor=2):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def segment_image(image):
    image_array = np.asarray(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    _, segmented_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented_image = cv2.bitwise_not(segmented_image)
    return cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2RGB)


def ensure_3_channels(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    resized_image = resize_image(image)
    normalized_image = normalize_image(resized_image)
    normalized_image_pil = Image.fromarray((normalized_image * 255).astype(np.uint8))
    contrast_image = adjust_contrast(normalized_image_pil)
    segmented_image = segment_image(contrast_image)
    segmented_image_pil = Image.fromarray(segmented_image)
    final_image = ensure_3_channels(segmented_image_pil)
    return final_image


def preprocess_images(dir, output_dir):
    cls = os.listdir(dir)
    for cl in cls:
        images = os.listdir(os.path.join(dir, cl))
        for image in images:
            preprocessed = preprocess_image(os.path.join(dir, cl, image))
            preprocessed.save(os.path.join(output_dir, cl, image))


if __name__ == '__main__':
    preprocess_images('dataset/alzheimers-dataset/val', 'dataset/alzheimers-dataset/preprocessed_val')
