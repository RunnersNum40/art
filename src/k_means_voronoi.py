import numpy as np
from typing import Tuple, Optional
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import colour


def rgb_to_lab(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an RGB image to Lab color space.

    Args:
    - image: Numpy array representing the image in the RGB color space.

    Returns:
    - Numpy array representing the image in the Lab color space.
    """
    # Convert to Oklab color space
    image = colour.RGB_to_XYZ(image, colourspace="sRGB")
    image = colour.XYZ_to_Oklab(image)
    # Normalize to [0, 1]
    mins, maxs = np.min(image, axis=(0, 1)), np.max(image, axis=(0, 1))
    image = (image - mins) / (maxs - mins)
    return image, mins, maxs


def lab_to_rgb(image: np.ndarray, mins: np.ndarray, maxes: np.ndarray) -> np.ndarray:
    """
    Convert an Lab image array back to RGB color space.

    Args:
    - image: Numpy array representing the image in the Lab color space.
    - mins: Minimum values for each channel.
    - maxes: Maximum values for each channel.

    Returns:
    - Numpy array representing the image in the RGB color space.
    """
    # Denormalize from [0, 1]
    image = image * (maxes - mins) + mins
    # Convert back to RGB color space
    image = colour.Oklab_to_XYZ(image)
    image = colour.XYZ_to_RGB(image, colourspace="sRGB")
    return image


def cluster_image(
    image: Image.Image,
    n_clusters: int,
    color_importance: float = 1.0,
    random_state: Optional[int] = None,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster the input image in the OKLab color space and return centroids with their positions.

    Args:
    - image: Input PIL Image object.
    - n_clusters: Number of clusters for k-means.
    - color_importance: Weight for the color channels in clustering.
    - random_state: Random seed for reproducibility.
    - batch_size: Number of samples to use in each batch for MiniBatchKMeans.

    Returns:
    - Tuple of the image shape, the fitted KMeans cluster centers (including their original positions), and their pixel positions.
    """
    oklab, color_mins, color_maxes = rgb_to_lab(np.array(image))
    original_shape = oklab.shape
    pixels = oklab.reshape(-1, 3) * color_importance
    xy = np.indices(dimensions=(original_shape[0], original_shape[1])).reshape(2, -1).T
    # Normalize positions
    xy = xy / np.max(xy, axis=0)

    pixels_with_xy = np.concatenate([pixels, xy], axis=1).astype(np.float32)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=random_state, batch_size=batch_size
    ).fit(pixels_with_xy)

    centroids = kmeans.cluster_centers_
    centroids[:, :3] /= color_importance

    return centroids, color_mins, color_maxes


def create_image_from_position_clusters(
    centroids: np.ndarray,
    shape: Tuple[int, int],
    color_mins: np.ndarray,
    color_maxes: np.ndarray,
    scaling_factor: int = 2,
) -> Image.Image:
    """
    Create an image by finding the closest centroid based on pixel position and using its color.

    Supports scaling the generation size before resizing to the output shape to reduce aliasing.

    Args:
    - centroids: Cluster centers including their pixel positions.
    - shape: Shape of the output image.
    - color_mins: Minimum values for each color channel.
    - color_maxes: Maximum values for each color channel.
    - scaling_factor: Multiplier on the origial generation size before resizing to output

    Returns:
    - New PIL Image object reconstructed from the clusters.
    """
    generation_shape = tuple(dim * scaling_factor for dim in shape[::-1])

    # Extract positions and colors from centroids
    centroid_positions = centroids[:, 3:5]
    centroid_colors = centroids[:, :3]

    # Create a grid of all pixel positions and normalize
    xy = (
        np.indices(dimensions=(generation_shape[0], generation_shape[1]))
        .reshape(2, -1)
        .T
    )
    xy = xy / np.max(xy, axis=0)

    # Use KD-Tree for efficient nearest centroid finding
    nearest = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    nearest.fit(centroid_positions)
    _, indices = nearest.kneighbors(xy)
    closest_centroid_colors = centroid_colors[indices.flatten()]

    # Reconstruct the image
    new_image_array = closest_centroid_colors.reshape(generation_shape[:2] + (3,))
    new_image = Image.fromarray(
        lab_to_rgb(new_image_array, color_mins, color_maxes).astype(np.uint8)
    )

    return new_image.resize(shape, Image.BICUBIC)


def segment_image(
    image: Image.Image,
    n_clusters: int,
    processing_shape: Tuple[int, int] = (960, 540),
    output_shape: Tuple[int, int] = (1920, 1080),
) -> Image.Image:
    """
    Wrapper function to segment an image by clustering based on pixel positions
    and color, then creating a new image by assigning colors based on the closest centroid in position space.

    The end result is a voronoi-like segmentation of the image with greater detail in regions of high contrast.

    Args:
    - image: PIL Image object.
    - n_clusters: Number of clusters for k-means.
    - processing_shape: Shape to resize the image to before clustering.
    - output_shape: Shape to resize the final image to.

    Returns:
    - A new PIL Image object with each pixel colored by the color of the nearest centroid in position space.
    """
    image = image.resize(processing_shape, Image.BICUBIC)
    centroids, color_mins, color_maxes = cluster_image(
        image, n_clusters, color_importance=100
    )
    new_image = create_image_from_position_clusters(
        centroids, output_shape, color_mins, color_maxes
    )
    return new_image


if __name__ == "__main__":
    # image_path = "../examples/k_means_voronoi/landscape.jpg"
    # image = Image.open(image_path)
    #
    # for n_clusters in [16, 128, 512, 2048]:
    #     save_path = f"../examples/k_means_voronoi/segmented_{n_clusters}_landscape.jpg"
    #     segmented_image = segment_image(image, n_clusters)
    #     segmented_image.save(save_path)

    image_path = "../images/originals/lake.jpg"
    image = Image.open(image_path)

    segmented_image = segment_image(image, 515)
    segmented_image.show()
