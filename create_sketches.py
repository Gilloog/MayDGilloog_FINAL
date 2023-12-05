import cv2
import os


def create_sketch(input_path, output_path):
    
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

   
    inverted = cv2.bitwise_not(blurred)

   
    _, sketch = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

   
    cv2.imwrite(output_path, sketch)


input_folder = "data/cars_test/cars_test"
output_folder = "Sketches"


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"sketch_{filename}")
        
        create_sketch(input_path, output_path)

print("Sketch images created and saved to the 'Sketches' folder.")


