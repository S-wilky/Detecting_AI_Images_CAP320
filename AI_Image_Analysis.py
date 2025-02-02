import cv2
import numpy as np

# Load and process the image
# Image path needs to be replaced by the image location on your system
def load_image(image_path: str):
    """
    Load an image and convert it to RGB format.

    :param image_path: Path to the image file
    :return: Image in RGB format
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at the specified path.")

    # Convert image to RGB as OpenCV loads in BGR by default
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Detect surface artifacts using edge detection
def detect_surface_artifacts(image):
    """
    Detect abrupt transitions in an image using Canny edge detection.
    Surface artifacts in AI-generated images often present unusual edges.

    :param image: Image in RGB format
    :return: Number of detected edge pixels
    """
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)
    return np.sum(edges > 0)


# Check for anatomical inconsistencies (e.g., disproportionate limbs)
def analyze_anatomy(image):
    """
    Prototype function for detecting anatomical irregularities using contours.

    :param image: Image in RGB format
    :return: Number of detected contours
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply binary threshold
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours)


# Analyze background consistency
def analyze_background(image):
    """
    Detect uniform color regions in the background using k-means clustering.

    :param image: Image in RGB format
    :return: Variance of cluster centers
    """
    # Reshape image into a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria for k-means clustering and number of clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3  # Number of clusters

    _, _, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Calculate variance of color clusters as a background consistency metric
    return np.var(centers)


# Main function to process and analyze the image
def main(image_path):
    # Load image
    image = load_image(image_path)

    # Detect surface artifacts
    surface_artifact_score = detect_surface_artifacts(image)

    # Analyze anatomy
    anatomy_score = analyze_anatomy(image)

    # Analyze background
    background_score = analyze_background(image)

    # Basic heuristic to classify image as AI-generated or not
    # (This is a naive thresholding approach for demonstration purposes)
    if surface_artifact_score > 100000 or anatomy_score > 500 or background_score > 1000:
        print("The image is likely AI-generated.")
    else:
        print("The image is likely human-generated.")



if __name__ == "__main__":
    # Replace with your image path
    image_path = input("Input the path to your image:")
    try:
        main(image_path)
    except ValueError as e:
        print(f"Error: {e}")
