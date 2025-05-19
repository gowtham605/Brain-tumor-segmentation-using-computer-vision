import cv2
import numpy as np

# Load the image
image = cv2.imread('brain_tumor.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply thresholding to create a binary image
_, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours in the binary image
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through the contours and filter based on area
for contour in contours:
    area = cv2.contourArea(contour)
    # Set a threshold for the tumor area
    if area > 1000:
        # Create a mask for the tumor region
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Extract the tumor region from the original image
        tumor = cv2.bitwise_and(image, image, mask=mask)

        # Display the segmented tumor
        cv2.imshow('Tumor', tumor)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
