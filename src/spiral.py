import math

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def spiral_image(
    image: Image.Image, scale_factor: float, twist_factor: float, direction: int = 1
) -> Image.Image:
    """
    Applies a spiral effect to the given image using the PIL library.

    Parameters:
    - image: The input PIL.Image.Image to be processed.
    - scale_factor: The factor by which to scale the image before applying the twist.
    - twist_factor: Determines the strength of the twisting effect.
    - direction: A flag indicating whether the twist should be clockwise (1)
                 or counter-clockwise (-1).

    Returns:
    - A new PIL.Image.Image object with the spiral effect applied.
    """
    # Step 1: Upscale the image
    upscaled_size = (
        int(image.width * scale_factor),
        int(image.height * scale_factor),
    )
    upscaled_image = image.resize(upscaled_size, Image.BICUBIC)

    # Step 2: Calculate the center of the upscaled image
    center_x, center_y = upscaled_size[0] / 2, upscaled_size[1] / 2

    # Calculate the maximum distance to a corner (farthest pixel)
    max_distance = math.sqrt(center_x**2 + center_y**2)

    # Step 3: Create a new image of the same size for the output
    output_image = Image.new("RGB", upscaled_size)

    # Prepare for pixel access
    upscaled_pixels = upscaled_image.load()
    output_pixels = output_image.load()

    # Step 4: Apply the spiral effect
    for x in range(upscaled_size[0]):
        for y in range(upscaled_size[1]):
            # Calculate the distance from the center
            dx, dy = x - center_x, y - center_y
            distance = math.sqrt(dx**2 + dy**2)

            # Calculate the angle and adjust by the twist factor
            angle = math.atan2(dy, dx) + (
                (distance / max_distance) * 2 * math.pi * twist_factor * direction
            )

            # Calculate the new position based on the angle and distance
            new_x = int(center_x + math.cos(angle) * distance)
            new_y = int(center_y + math.sin(angle) * distance)

            # Set the pixel if it's within bounds
            if 0 <= new_x < upscaled_size[0] and 0 <= new_y < upscaled_size[1]:
                output_pixels[x, y] = upscaled_pixels[new_x, new_y]

    # Step 5: Downscale the image to the original size
    downscaled_image = output_image.resize(image.size, Image.BICUBIC)

    return downscaled_image


def display_spiral_grid(image_path: str, scales: list, twists: list):
    """
    Generates a grid of images processed with different scale and twist factors and displays them.

    Parameters:
    - image_path: The path to the input image.
    - scales: A list of scale factors to apply.
    - twists: A list of twist factors to apply.
    """
    # Load the image
    image = Image.open(image_path)

    # Create a subplot grid
    fig, axes = plt.subplots(len(scales), len(twists), figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, scale in enumerate(scales):
        for j, twist in enumerate(twists):
            # Apply the spiral effect
            spiraled_image = spiral_image(image, scale, twist)

            # Display the image
            ax = axes[i, j]
            ax.imshow(np.array(spiraled_image))
            ax.set_title(f"Scale: {scale}, Twist: {twist}")
            ax.axis("off")

    plt.show()


if __name__ == "__main__":
    scales = [0.1, 0.5, 2, 3]
    twists = [0.1, 0.5, 1]

    image_path = "HD-beautiful-nature-background.jpg"
    display_spiral_grid(image_path, scales, twists)
