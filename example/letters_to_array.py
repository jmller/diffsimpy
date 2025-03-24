from PIL import Image, ImageDraw, ImageFont
import numpy as np

def generate_letter_matrix(letter, font_path=None):
    # Create a new blank image (larger than 8x8 to render the letter)
    img = Image.new('1', (8, 8), 0)  # '1' for binary pixels (black and white)

    # Load a default font
    if font_path == None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path)

    # Initialize drawing context
    draw = ImageDraw.Draw(img)

    # Draw the letter on the image (centered)
    draw.text((0, -2), letter, font=font, fill=1)

    # Resize the image to 8x8 if necessary
    img = img.resize((8, 8))

    # Convert the image to a numpy array
    matrix = np.array(img)

    # Convert numpy array (0/255) to binary (0/1)
    binary_matrix = (matrix > 0).astype(int)

    return binary_matrix