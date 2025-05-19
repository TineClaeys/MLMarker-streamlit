from PIL import Image
import os

# Reference image
ref_image_path = "./Markverse/Coding Mark.png"
ref_img = Image.open(ref_image_path)
target_size = ref_img.size  # (width, height)

# Folders
input_folder = "./Markverse"
output_folder = "./Markverse/cropped_images"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        width, height = img.size
        target_width, target_height = target_size

        # Only crop if image is larger than the target
        if width >= target_width and height >= target_height:
            left = (width - target_width) // 2
            top = (height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            cropped = img.crop((left, top, right, bottom))
        else:
            # Optionally pad instead of skipping or resizing
            cropped = img

        cropped.save(os.path.join(output_folder, filename))
