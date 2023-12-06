import cv2
import os
import shutil

def create_sketch(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    
    inverted = cv2.bitwise_not(blurred)

    
    _, sketch = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    
    cv2.imwrite(output_path, sketch)

def clear_and_create_sketches(input_folder, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"sketch_{filename}")

            
            create_sketch(input_path, output_path)
            print(f"Sketch created: {output_path}")

    print("Sketch images created and saved to the 'Sketches' folder.")

input_folder = "data/cars_train/cars_train"
output_folder = "data/Sketches"

clear_and_create_sketches(input_folder, output_folder)
