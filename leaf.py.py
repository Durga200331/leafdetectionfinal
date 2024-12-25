import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = "D:\leafdetection\imgnw6.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load image. Check the file path.")
else:
    # Step 1: Original Image (Convert BGR to RGB)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2: Convert to HSV for color thresholding
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 3: Define the green color range and create a mask for green
    lower_green = np.array([60, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_before_morph = cv2.inRange(hsv, lower_green, upper_green)

    # Step 4: Apply morphological operations to clean up the mask
    kernel = np.ones((9,9), np.uint8)
    mask_after_morph = cv2.morphologyEx(mask_before_morph, cv2.MORPH_OPEN, kernel)
    mask_after_morph = cv2.morphologyEx(mask_after_morph, cv2.MORPH_OPEN, kernel)
    

    # Step 5: Remove the soil background by applying the mask to the original image
    plant_only_image = cv2.bitwise_and(original_image, original_image, mask=mask_after_morph)

    # Step 6: Convert the plant-only image to grayscale
    grayscale_plant_image = cv2.cvtColor(plant_only_image, cv2.COLOR_RGB2GRAY)

    # Step 7: Apply edge detection (using Canny edge detector)
    edges = cv2.Canny(grayscale_plant_image, 50, 200)

    # Step 8: Apply morphological operations to fill gaps in edges
    kernel = np.ones((7,7), np.uint8)  # Smaller kernel for edge gap filling
    edges_morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Step 9: Apply thresholding (binary threshold) to enhance edges
    _, thresholded_edges = cv2.threshold(edges_morphed, 127, 255, cv2.THRESH_BINARY)

    # Step 10: Find contours based on the processed edges
    contours, _ = cv2.findContours(thresholded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 11: Draw contours on the original image
    leaf_count = 0
    output_image = image.copy()
    bounding_boxes = []  # List to store bounding box coordinates

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Threshold to filter small objects (non-leaves)
            leaf_count += 1
            # Draw bounding boxes for each valid contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            bounding_boxes.append((x, y, w, h))

    # Step 12: Plot all stages in one figure
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    axs[0, 0].imshow(original_image)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(plant_only_image)
    axs[0, 1].set_title('Plant Only (Soil Removed)')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(grayscale_plant_image, cmap='gray')
    axs[0, 2].set_title('Grayscale Plant Image')
    axs[0, 2].axis('off')

    axs[1, 0].imshow(edges, cmap='gray')
    axs[1, 0].set_title('Edge Detection (Canny)')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(edges_morphed, cmap='gray')
    axs[1, 1].set_title('Edges After Morphological Operations')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(output_image)
    axs[1, 2].set_title('Contour Detection on Original Image')
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    # Print the number of contours detected
    print(f"Number of contours detected: {leaf_count}")

    # Step 13: Draw each valid contour in separate boxes on a new page
    fig, axs = plt.subplots(1, len(bounding_boxes), figsize=(15, 5))
    
    if len(bounding_boxes) == 1:
        axs = [axs]  # If there's only one leaf, make axs iterable

    # Resize the leaf images to a fixed size
    target_size = (100, 100)  # Resize to 100x100 pixels for uniform size
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        leaf_image = image[y:y+h, x:x+w]
        resized_leaf_image = cv2.resize(leaf_image, target_size)  # Resize to uniform size
        axs[i].imshow(cv2.cvtColor(resized_leaf_image, cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"Leaf {i+1}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
