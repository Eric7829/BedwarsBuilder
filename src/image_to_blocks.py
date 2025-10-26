import cv2
import numpy as np
import os
from skimage import color
# run with python src/image_to_blocks.py
# Block color palette with indexes
import cv2
import numpy as np
import os
#run with python src/itb_without_CIEDE200.py
# Block color palette with indexes
BLOCKS = {
    1: ("stone_brick", "#66686d"),
    2: ("obsidian", "#05070b"),
    3: ("wool_white", "#c7c7c9"),
    4: ("andesite", "#696870"),
    5: ("clay_red", "#c07a74"),
    6: ("coral_block", "#698bc9"),
    7: ("wood_plank_oak", "#b8986f"),
    8: ("wood_plank_birch", "#cfbdae"),
    9: ("wood_plank_spruce", "#aa8b6b"),
    10: ("diamond_block", "#d0e7e1"),
    11: ("sand", "#dad7c2"),
    12: ("purple_lucky_block", "#cc46e4"),
    13: ("wool_red", "#bc332f"),
    14: ("wool_green", "#0cb747"),
    15: ("wool_yellow", "#c9b113"),
    16: ("wool_blue", "#432eba"),
    17: ("wool_cyan", "#64b8bd"),
    18: ("wool_pink", "#d593c2"),
    19: ("wool_orange", "#d88513"),
    20: ("wool_purple", "#a22cbb"),
    21: ("blastproof_ceramic", "#c99779"),
    22: ("clay_black", "#161719"),
    23: ("clay_light_green", "#abc762"),
    24: ("clay_tan", "#b6947c"),
    25: ("clay_white", "#dbdbdb"),
    26: ("lucky_block", "#e5d246"),
    27: ("diorite", "#c4c5c7"),
    28: ("clay_dark_brown", "#765448"),
    29: ("clay_blue", "#546097"),
    30: ("ice", "#cadde7"),
    31: ("clay_dark_green", "#6eae64"),
    32: ("green_concrete", "#5a8967"),
    33: ("clay_purple", "#8e5399"),
    34: ("marble_pillar", "#ecdbce"),
    35: ("clay", "#b6adbf"),
    36: ("marble", "#f3e9e0"),
    37: ("iron_block", "#f5efed"),
    38: ("sandstone_smooth", "#e9d09d"),
    39: ("red_sand", "#e4a35d"),
}


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_lab_batch(rgb_array):
    """Convert RGB array to LAB color space (vectorized)."""
    # Normalize RGB to [0, 1]
    rgb_normalized = rgb_array.astype(np.float32) / 255.0
    # Convert to LAB
    lab = color.rgb2lab(rgb_normalized)
    return lab

# Pre-compute block RGB and LAB values for faster lookup
BLOCK_RGB_CACHE = {block_id: hex_to_rgb(hex_color) for block_id, (name, hex_color) in BLOCKS.items()}
# Convert all block colors to LAB at once
block_rgb_array = np.array([BLOCK_RGB_CACHE[i] for i in sorted(BLOCK_RGB_CACHE.keys())])
block_lab_array = rgb_to_lab_batch(block_rgb_array.reshape(1, -1, 3)).reshape(-1, 3)
BLOCK_IDS = np.array(sorted(BLOCK_RGB_CACHE.keys()))

def find_closest_blocks_batch(pixel_lab_array):
    """Find the closest block for each pixel using fast CIEDE2000-like approximation."""
    # pixel_lab_array shape: (height, width, 3)
    height, width = pixel_lab_array.shape[:2]
    
    # Reshape for batch processing
    pixels_flat = pixel_lab_array.reshape(-1, 3)  # (num_pixels, 3)
    
    num_pixels = pixels_flat.shape[0]
    num_blocks = block_lab_array.shape[0]
    
    closest_blocks = np.zeros(num_pixels, dtype=int)
    
    # Process in chunks to avoid memory issues
    chunk_size = 500
    for chunk_start in range(0, num_pixels, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_pixels)
        chunk_pixels = pixels_flat[chunk_start:chunk_end]  # (chunk_size, 3)
        
        if chunk_start % (chunk_size * 10) == 0:
            print(f"Processing pixels {chunk_start}/{num_pixels}...")
        
        # Fast vectorized distance calculation using broadcasting
        # chunk_pixels: (chunk_size, 3), block_lab_array: (num_blocks, 3)
        # Result: (chunk_size, num_blocks)
        diff = chunk_pixels[:, np.newaxis, :] - block_lab_array[np.newaxis, :, :]  # (chunk_size, num_blocks, 3)
        
        # Use Euclidean distance in LAB space (much faster approximation)
        # LAB space is perceptually uniform, so this is a reasonable approximation to CIEDE2000
        distances = np.sqrt(np.sum(diff ** 2, axis=2))  # (chunk_size, num_blocks)
        
        # Find closest block for each pixel in chunk
        closest_blocks[chunk_start:chunk_end] = BLOCK_IDS[np.argmin(distances, axis=1)]
    
    return closest_blocks.reshape(height, width)

def resize_image_to_fixed_dimensions(img, width=512, height=512):
    """Resize the image to fixed dimensions."""
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def convert_image_to_blocks(input_path, output_path):
    """Convert image to 2D array of block indexes."""
    # Read the image
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return None
    
    # Resize image to 512x512
    img = resize_image_to_fixed_dimensions(img, width=512, height=512)
    
    # Convert from BGR to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, _ = img_rgb.shape
    print(f"Processing {width}x{height} image...")
    
    # Convert entire image to LAB color space at once
    print("Converting image to LAB color space...")
    img_lab = rgb_to_lab_batch(img_rgb)
    
    # Find closest blocks for all pixels at once
    print("Finding closest blocks for all pixels...")
    block_array = find_closest_blocks_batch(img_lab)
    
    # Save as Lua table
    with open(output_path, 'w') as f:
        f.write("return {\n")
        for y in range(height):
            row_str = "    {" + ", ".join(str(block_array[y, x]) for x in range(width)) + "}"
            if y < height - 1:
                row_str += ","
            f.write(row_str + "\n")
        f.write("}\n")
    
    print(f"Conversion complete! Saved to {output_path}")
    return block_array

if __name__ == "__main__":

    # Allow reading both PNG and JPG/JPEG files
    input_image = "src/input.png"  # Change this to "src/input.jpg" or "src/input.jpeg" if needed
    output_file = "src/output.lua"
    
    # Check file extension
    if not input_image.lower().endswith(('.png', '.jpg', '.jpeg')):
        print("Error: Unsupported file format. Please use PNG, JPG, or JPEG.")
        exit()

    result = convert_image_to_blocks(input_image, output_file)
    
    if result is not None:
        print(f"\nOutput is a {result.shape[0]}x{result.shape[1]} array")
        print(f"Block usage statistics:")
        unique, counts = np.unique(result, return_counts=True)
        for block_id, count in zip(unique, counts):
            block_name = BLOCKS[block_id][0]
            print(f"  {block_name} (ID {block_id}): {count} pixels")
