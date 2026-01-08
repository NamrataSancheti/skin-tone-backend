import cv2
import numpy as np
from sklearn.cluster import KMeans


def read_image(file):
    file.file.seek(0)
    image_bytes = np.frombuffer(file.file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file")

    return image


def apply_mask(image, mask):
    masked = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked[mask > 0]
    return pixels


def rgb_to_lab(pixels):
    pixels = np.uint8(pixels.reshape(-1, 1, 3))
    lab_pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB)
    return lab_pixels.reshape(-1, 3)


def get_dominant_color(lab_pixels, k=5):
    if len(lab_pixels) < k:
        k = max(1, len(lab_pixels) // 2)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(lab_pixels)

    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = labels[np.argmax(counts)]

    return kmeans.cluster_centers_[dominant_cluster]


def detect_undertone(lab_color):
    _, a, b = lab_color

    if a >= 20 and b >= 15:
        return "Warm"
    elif a < 20 and b < 15:
        return "Cool"
    else:
        return "Neutral"


def lab_to_rgb_hex(lab_color):
    lab = np.uint8([[lab_color]])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)[0][0]
    return '#%02x%02x%02x' % tuple(rgb)


def recommend_palette(undertone):
    palettes = {
        "Warm": ["Olive", "Mustard", "Coral", "Brown", "Terracotta", "Peach", "Gold"],
        "Cool": ["Navy Blue", "Emerald", "Lavender", "Grey", "Turquoise", "Silver", "Plum"],
        "Neutral": ["Black", "White", "Teal", "Soft Pink", "Beige", "Charcoal", "Mint"]
    }
    return palettes.get(undertone, [])


def name_to_hex(name):
    color_map = {
        "Olive": "#808000",
        "Mustard": "#FFDB58",
        "Coral": "#FF7F50",
        "Brown": "#A52A2A",
        "Terracotta": "#E2725B",
        "Peach": "#FFE5B4",
        "Gold": "#FFD700",
        "Navy Blue": "#000080",
        "Emerald": "#50C878",
        "Lavender": "#E6E6FA",
        "Grey": "#808080",
        "Turquoise": "#40E0D0",
        "Silver": "#C0C0C0",
        "Plum": "#8E4585",
        "Black": "#000000",
        "White": "#FFFFFF",
        "Teal": "#008080",
        "Soft Pink": "#FFB6C1",
        "Beige": "#F5F5DC",
        "Charcoal": "#36454F",
        "Mint": "#98FF98"
    }
    return color_map.get(name, "#000000")


def process_selected_pixels(image_file, mask_file, region_type):
    image = read_image(image_file)
    mask = read_image(mask_file)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    selected_pixels = apply_mask(image, mask)
    if len(selected_pixels) == 0:
        return {"error": "No pixels selected!"}

    lab_pixels = rgb_to_lab(selected_pixels)
    dominant_color = get_dominant_color(lab_pixels, k=5)

    undertone = detect_undertone(dominant_color)
    palette_names = recommend_palette(undertone)
    palette_hex = [name_to_hex(c) for c in palette_names]

    representative_color = lab_to_rgb_hex(dominant_color)

    return {
        "region": region_type,
        "dominant_lab_color": dominant_color.tolist(),
        "representative_color": representative_color,
        "undertone": undertone,
        "recommended_colors": [
            {"name": n, "hex": h} for n, h in zip(palette_names, palette_hex)
        ],
        "explanation": f"{undertone} undertones are enhanced by these colors."
    }
