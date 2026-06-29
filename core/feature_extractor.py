import cv2
import numpy as np

# FEATURE EXTRACTOR
# This file has one job: take an outfit image and turn its colors into numbers.
# These numbers are called a "feature vector".
# Later, the ML model will use these numbers to learn what you like.

WARM_HUE_LOW_1= 0
WARM_HUE_HIGH_1= 30
WARM_HUE_LOW_2= 150
WARM_HUE_HIGH_2= 180

def extract_features(image):


    """
    Takes one outfit image and returns a feature vector (a list of numbers).
 
    The feature vector contains 5 numbers:
      1. mean_hue       → the average color of the outfit (0-180)
      2. hue_std        → how much the colors vary (low = one solid color, high = many colors)
      3. mean_saturation→ how strong/vivid the colors are (0=grey, 255=very colorful)
      4. mean_brightness→ how bright the outfit is (0=dark, 255=very bright)
      5. warm_cool_ratio→ what percentage of pixels are warm colors (0.0 to 1.0)
                          0.0 = fully cool outfit (blues/greens)
                          1.0 = fully warm outfit (reds/oranges)
 
    INPUT:  image → a BGRA outfit image (the PNG we loaded with OpenCV)
    OUTPUT: a list of 5 numbers → [mean_hue, hue_std, mean_sat, mean_bright, warm_cool]
    """

    
    if image.shape[2] ==4:
        alpha_channel = image[:,:,3]
        mask= alpha_channel > 10

    else:
        mask= np.ones(image.shape[:2], dtype=bool)  


    bgr_image= image[:, :, :3]
    hsv_image= cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    hue_channel= hsv_image[:, :, 0]
    saturation_channel= hsv_image[:, :, 1]
    brightness_channel= hsv_image[:, :, 2]


    hue_values= hue_channel[mask]
    saturation_values= saturation_channel[mask]
    brightness_values= brightness_channel[mask] 

    if len(hue_values) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    

    mean_hue= float(np.mean(hue_values))
    hue_std= float(np.std(hue_values))
    mean_saturation=float(np.mean(saturation_values))
    mean_brightness= float(np.mean(brightness_values))

    warm_mask_1 = (hue_values >= WARM_HUE_LOW_1) & (hue_values <= WARM_HUE_HIGH_1)
    warm_mask_2 = (hue_values >= WARM_HUE_LOW_2) & (hue_values <= WARM_HUE_HIGH_2)

    total_warm_pixels= np.sum(warm_mask_1 | warm_mask_2)
    total_pixels= len(hue_values)

    warm_cool_ratio= float(total_warm_pixels/total_pixels) if total_pixels > 0 else 0.0

    return [mean_hue, hue_std, mean_saturation, mean_brightness, warm_cool_ratio]


def get_combination_vector(top_features, bot_features):

    return top_features + bot_features

