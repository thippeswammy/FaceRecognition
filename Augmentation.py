import os
import cv2
import shutil
import random
import numpy as np


def ChangesImage(img):
    output_img = [img]
    contrast_factor_dark = random.uniform(0.5, 0.9)
    enhancer = cv2.addWeighted(img, contrast_factor_dark, 0, 0, 0)
    output_img.append(enhancer)
    contrast_factor_bright = random.uniform(1.0, 1.6)
    enhancer = cv2.addWeighted(img, contrast_factor_bright, 0, 0, 0)
    output_img.append(enhancer)
    ram = random.choice([1, 2, 3, 4])
    if ram == 1:
        gaussianBlur = GaussianBlur(img)
        output_img.append(gaussianBlur)
    elif ram == 2:
        averageBlur = AverageBlur(img)
        output_img.append(averageBlur)
    elif ram == 3:
        gaussianNoise = GaussianNoise(img)
        output_img.append(gaussianNoise)
    elif ram == 4:
        saltPepperNoise = SaltPepperNoise(img)
        output_img.append(saltPepperNoise)
    return output_img


def GaussianBlur(image, size=3):
    Gauss = cv2.GaussianBlur(image, (size, size), 0)
    return Gauss


def AverageBlur(image, size=3):
    kernel = np.ones((size, size), np.float32) / (size * size)
    averaged_image = cv2.filter2D(image, -1, kernel)
    return averaged_image


def GaussianNoise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss.astype(np.uint8)
    return noisy


def SaltPepperNoise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = (255, 255, 255)
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = (0, 0, 0)
    return noisy
