from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import io
from PIL import Image

app = Flask(__name__)

# Path to the stored original image
ORIGINAL_IMG_PATH = 'static/pan1.jpg'

# Function to detect if the given card is a PAN card
def is_pan_card(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    keywords = ['Income Tax Department', 'Permanent Account Number', 'INCOME TAX DEPARTMENT', 'GOVT. OF INDIA']
    for keyword in keywords:
        if keyword.lower() in text.lower():
            return True
    return False

# Function to detect tampering and calculate the percentage of tampered area
def detect_tampering(original_img, test_img):
    original_img_resized = cv2.resize(original_img, (test_img.shape[1], test_img.shape[0]))
    original_gray = cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    diff_img = cv2.absdiff(original_gray, test_gray)
    _, thresh_img = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)
    tampered_area = np.sum(thresh_img > 0)
    total_area = thresh_img.size
    tampered_percentage = (tampered_area / total_area) * 100
    return tampered_percentage, thresh_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'test' not in request.files:
        return "Please upload a test image."

    test_file = request.files['test']
    test_img = cv2.imdecode(np.fromstring(test_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Load the original image from the repository
    original_img = cv2.imread(ORIGINAL_IMG_PATH)

    if not is_pan_card(test_img):
        return "The provided image is not a PAN card."

    tampered_percentage, tampered_img = detect_tampering(original_img, test_img)
    contours, _ = cv2.findContours(tampered_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    test_img_contours = test_img.copy()
    cv2.drawContours(test_img_contours, contours, -1, (0, 255, 0), 2)

    # Save the tampered image with contours
    is_success, buffer = cv2.imencode(".jpg", test_img_contours)
    io_buf = io.BytesIO(buffer)
    return send_file(io_buf, mimetype='image/jpeg', as_attachment=True, download_name='tampered_image.jpg')

if __name__ == '__main__':
    app.run(debug=True)
