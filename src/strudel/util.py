import json
import os
import re

import numpy as np
from PIL import Image


def load_json(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(filename: str):
    with open(filename, "r") as json_file:
        return [json.loads(line) for line in json_file]


def save_json(filename: str, data: dict):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_jsonl(filename: str, data: list):
    with open(filename, "a", encoding="utf-8") as f:
        for sample in data:
            f.write(f"{json.dumps(sample, ensure_ascii=False)}\n")


def load_text(filename: str):
    with open(filename) as text_file:
        return text_file.read()


def save_text(filename: str, data: str):
    with open(filename, "w") as text_file:
        text_file.write(data)


def save_bytes(filename: str, data: str):
    with open(filename, "wb") as binary_file:
        binary_file.write(data)


def check_dirs(path: str):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))


def insert_line(path: str, line: str):
    with open(path, "r") as file:
        content = file.read()

    with open(path, "w") as file:
        file.write(line + content)


def remove_files(path: str, endings: list = [""]):
    for ext in endings:
        try:
            if ext != "":
                os.remove(f"{path}.{ext}")
            else:
                os.remove(path)
        except:
            pass


def remove_files_dir(path: str, endings: list):
    for f in os.listdir(path):
        for ext in endings:
            if f.endswith(f".{ext}"):
                os.remove(os.path.join(path, f))


def images_are_similar(img_path1, img_path2, tolerance=5):
    img1 = Image.open(img_path1).convert("RGBA")
    img2 = Image.open(img_path2).convert("RGBA")

    # Get the smaller common size
    common_size = (min(img1.size[0], img2.size[0]), min(img1.size[1], img2.size[1]))  # width  # height

    # Resize both to the smaller size
    img1_resized = img1.resize(common_size)
    img2_resized = img2.resize(common_size)

    # Convert to numpy arrays
    arr1 = np.array(img1_resized)
    arr2 = np.array(img2_resized)

    # Compare with tolerance
    return np.allclose(arr1, arr2, atol=tolerance)


def is_image_single_color(img_path):
    try:
        img = Image.open(img_path).convert("RGBA")
        pixels = np.array(img)
        # Flatten all pixel values and check if all are the same
        return np.all(pixels == pixels[0, 0])
    except Exception as e:
        print(f"Error opening image: {e}")
        return False


def classify_image_black_or_white(image, threshold=0.9, black_level=50, white_level=205):
    """
    Classify an image as mostly black, mostly white, or neither.

    Parameters:
        image_path (str): Path to the image file.
        threshold (float): Minimum fraction of pixels required to classify as black/white (0-1).
        black_level (int): Max grayscale value to consider a pixel as black (0-255).
        white_level (int): Min grayscale value to consider a pixel as white (0-255).

    Returns:
        str: "mostly black", "mostly white", or "neither"
    """
    # Open image and convert to grayscale
    # img = Image.open(image_path).convert("L")
    img = image.convert("L")
    pixels = np.array(img)

    total_pixels = pixels.size
    black_pixels = np.sum(pixels <= black_level)
    white_pixels = np.sum(pixels >= white_level)

    frac_black = black_pixels / total_pixels
    frac_white = white_pixels / total_pixels

    if frac_black >= threshold:
        return "black"
    elif frac_white >= threshold:
        return "white"
    else:
        return None


def is_image_mainly_black(image_path, threshold=5, black_ratio=0.55, alpha_threshold=10):
    """
    Check if the image is mainly black.

    Parameters:
    - image_path: path to the image file
    - threshold: pixel values below this are considered black (0-255)
    - black_ratio: minimum ratio of black pixels to consider the image as mainly black

    Returns:
    - True if image is mainly black, False otherwise
    """
    img = Image.open(image_path).convert("RGBA")
    np_img = np.array(img)

    # Split RGBA
    r, g, b, a = np_img[:, :, 0], np_img[:, :, 1], np_img[:, :, 2], np_img[:, :, 3]

    # Create a mask for "visible" pixels
    visible_mask = a >= alpha_threshold

    # Check if RGB values are all below threshold
    black_mask = (r < threshold) & (g < threshold) & (b < threshold)

    # Apply visibility filter
    black_visible_pixels = np.sum(black_mask & visible_mask)
    total_visible_pixels = np.sum(visible_mask)

    if total_visible_pixels == 0:
        return False  # no visible content → not black

    return (black_visible_pixels / total_visible_pixels) >= black_ratio


def is_image_valid(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()  # Checks if image is not corrupted
        return True
    except Exception as e:
        print(f"Image is broken: {e}")
        return False


def resize_png_preserve_aspect(image_path, max_width, max_height, keep_transparency=True):
    """
    Resize a PNG image while preserving the aspect ratio.

    Parameters:
    - input_path: Path to the input PNG image
    - output_path: Path to save the resized image
    - max_width, max_height: Bounding box for resized image
    - keep_transparency: If False, background will be white
    """
    img = Image.open(image_path).convert("RGBA")

    # Resize while keeping aspect ratio
    img.thumbnail((max_width, max_height), Image.LANCZOS)

    if keep_transparency:
        img.save(image_path, format="PNG")
    else:
        # Create white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Paste using alpha channel as mask
        background.save(image_path, format="PNG")


def extract_part(text, term_1, term_2, return_empty, remove_first_line=False, reverse=False):
    text_result = "" if return_empty else text
    offset = len(term_1)
    start_code = text.find(term_1) if not reverse else text.rfind(term_1)

    if start_code != -1:
        if term_2 != "":
            end_code = text.find(term_2, start_code + offset)  # if not reverse else text.rfind(term_2, start_code+offset)
            if end_code != -1:
                text_result = text[start_code + offset : end_code]
                if remove_first_line:
                    first_line = text_result.split("\n")[0].strip()
                    if "{" not in first_line and len(first_line) < 10:
                        text_result = "\n".join(text_result.split("\n")[1:])
            else:
                if remove_first_line:
                    # text_result = "" # if code is incomplete, return empty string
                    text_result = text[start_code + offset :]
                    first_line = text_result.split("\n")[0].strip()
                    if "{" not in first_line and len(first_line) < 10:
                        text_result = "\n".join(text_result.split("\n")[1:])
                else:
                    text_result = text[start_code + offset :]
        else:
            text_result = text[start_code + offset :]

    return text_result.strip()


def check_reasoning(text):
    if len(text.split("</think>")) > 1:
        return text.split("</think>")[1].strip()
    else:
        return text


def check_reasoning_code(text):
    if len(text.split("</think_code>")) > 1:
        return text.split("</think_code>")[1].strip()
    else:
        return text


def replace_bpmndi(content: str) -> str:
    pattern = r"<bpmndi:BPMNDiagram.*?</bpmndi:BPMNDiagram>"
    replacement = '<bpmndi:BPMNDiagram id="BPMNDiagram_1"></bpmndi:BPMNDiagram>'

    return re.sub(pattern, replacement, content, flags=re.DOTALL)
