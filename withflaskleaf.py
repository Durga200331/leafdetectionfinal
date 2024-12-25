from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for frontend-backend communication

# Folder to store uploaded and processed images
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/process_image', methods=['POST'])
def process_image():
    # Step 1: Check if an image is uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    # Step 2: Save the uploaded image
    file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Step 3: Read the uploaded image
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Step 4: Process the image for leaf detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([60, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations to clean the mask
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply the mask to the original image
    plant_only_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert to grayscale and detect edges
    grayscale = cv2.cvtColor(plant_only_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 50, 200)
    edges_morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(edges_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_count = 0
    bounding_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Filter small objects
            leaf_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # Step 5: Save the processed image
    output_path = os.path.join(PROCESSED_FOLDER, 'output.jpg')
    cv2.imwrite(output_path, image)

    # Save individual bounding box images
    bounding_box_paths = []
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        leaf_image = image[y:y + h, x:x + w]
        box_path = os.path.join(PROCESSED_FOLDER, f'leaf_{i + 1}.jpg')
        cv2.imwrite(box_path, leaf_image)
        bounding_box_paths.append(box_path)

    # Step 6: Return results as JSON
    return jsonify({
        'leaf_count': leaf_count,
        'output_image': f'/download/output.jpg',
        'bounding_boxes': [f'/download/{os.path.basename(path)}' for path in bounding_box_paths]
    })


@app.route('/download/<filename>')
def download_file(filename):
    # Serve processed images dynamically
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    return send_file(file_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
