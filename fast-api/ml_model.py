import cv2
import numpy as np
from sklearn.cluster import KMeans

def read_image(file):
    image_bytes = np.frombuffer(file.file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    return image

def apply_mask(image, mask):
    """
    Keeps only user-selected pixels
    """
    masked = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked[mask > 0]
    return pixels

def rgb_to_lab(pixels):
    pixels = np.uint8(pixels.reshape(-1, 1, 3))
    lab_pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB)
    return lab_pixels.reshape(-1, 3)

def get_dominant_color(lab_pixels, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(lab_pixels)

    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = labels[np.argmax(counts)]

    dominant_color = kmeans.cluster_centers_[dominant_cluster]
    return dominant_color

def detect_undertone(lab_color):
    L, a, b = lab_color

    if a > 10 and b > 10:
        return "Warm"
    elif b < 5:
        return "Cool"
    else:
        return "Neutral"

def recommend_palette(undertone):
    palettes = {
        "Warm": ["Olive", "Mustard", "Coral", "Brown"],
        "Cool": ["Navy Blue", "Emerald", "Lavender", "Grey"],
        "Neutral": ["Black", "White", "Teal", "Soft Pink"]
    }
    return palettes.get(undertone, [])

def process_selected_pixels(image_file, mask_file, region_type):
    image = read_image(image_file)
    mask = read_image(mask_file)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    selected_pixels = apply_mask(image, mask)

    lab_pixels = rgb_to_lab(selected_pixels)
    dominant_color = get_dominant_color(lab_pixels)
    undertone = detect_undertone(dominant_color)
    palette = recommend_palette(undertone)

    return {
        "region": region_type,
        "dominant_lab_color": dominant_color.tolist(),
        "undertone": undertone,
        "recommended_colors": palette,
        "explanation": f"{undertone} undertones are enhanced by these colors."
    }
