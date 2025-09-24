# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

## Program:
```
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load your image
img = cv2.imread("k.jpg")

# Check if image loading was successful
if img is None:
    print("Error: Could not load image k.jpg")
else:
    # Load sunglasses PNG (make sure it has transparent background)
    sunglasses = cv2.imread("gl.jpg", cv2.IMREAD_UNCHANGED)

    # Convert white pixels to transparent
    if sunglasses.shape[2] == 3: # if no alpha channel
        # Convert to grayscale
        gray_sunglasses = cv2.cvtColor(sunglasses, cv2.COLOR_BGR2GRAY)
        # Create alpha channel from white pixels
        _, alpha = cv2.threshold(gray_sunglasses, 250, 255, cv2.THRESH_BINARY_INV)
        # Merge alpha channel
        b, g, r = cv2.split(sunglasses)
        sunglasses = cv2.merge((b, g, r, alpha))

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        # Resize sunglasses to fit the face width
        sung_w = w
        scale = sung_w / sunglasses.shape[1]
        sung_h = int(sunglasses.shape[0] * scale)
        resized_sung = cv2.resize(sunglasses, (sung_w, sung_h))

        # Get ROI (region of interest) above the eyes
        y_offset = y + int(h/4)   # adjust placement
        x_offset = x
        y1, y2 = y_offset, y_offset + resized_sung.shape[0]
        x1, x2 = x_offset, x_offset + resized_sung.shape[1]

        # Ensure it doesn't go outside image
        if y2 > img.shape[0]:
            y2 = img.shape[0]
        if x2 > img.shape[1]:
            x2 = img.shape[1]

        # Check if sunglasses image has an alpha channel
        if resized_sung.shape[2] == 4:
            # Split sunglasses into RGB and alpha
            sung_rgb = resized_sung[:, :, :3]
            sung_alpha = resized_sung[:, :, 3] / 255.0

            # Overlay sunglasses on the image with alpha blending
            for c in range(3):
                img[y1:y2, x1:x2, c] = (sung_alpha * sung_rgb[:, :, c] +
                                       (1 - sung_alpha) * img[y1:y2, x1:x2, c])
        else:
            # If no alpha channel, just overlay the BGR image
            img[y1:y2, x1:x2] = resized_sung[:, :, :3]


    # Save or show result
    cv2_imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```
# Output:

<img width="426" height="570" alt="image" src="https://github.com/user-attachments/assets/9e435a75-9156-4792-b745-c33833415517" />


