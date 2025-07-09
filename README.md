Iranian License Plate Recognition (YOLOv8 & CNN)
This project implements an end-to-end system for Iranian license plate recognition (ALPR). It leverages a two-stage approach:

License Plate Detection and Character Segmentation using YOLOv8 models.

Character Recognition (OCR) using a Convolutional Neural Network (CNN).

The pipeline is designed to first locate license plates in an image, then segment individual characters from the detected plates, and finally, recognize each character to reconstruct the full license plate number.

Features
Accurate Plate Detection: Utilizes YOLOv8 for robust identification of license plates in diverse images.

Precise Character Segmentation: YOLOv8 is also employed to accurately isolate individual characters on the detected plates.

High-Performance OCR: A custom CNN model is trained to recognize Persian characters and digits with high accuracy.

Full Pipeline Implementation: Provides a complete workflow from image input to recognized license plate text.

Persian Character Support: Specifically designed to handle the unique characters found on Iranian license plates.

Project Structure
The provided code snippets demonstrate the key steps involved:

Dataset Download: Downloads train.zip, val.zip, and test.zip datasets from Google Drive using gdown.

Dataset Preparation (License Plate Detection):

Extracts downloaded zip files.

Converts XML annotations (VOC format) to YOLO format (.txt files) for license plate bounding boxes.

Organizes data into dataset/images/{train, val} and dataset/labels/{train, val} directories.

Generates data.yaml for YOLOv8 training.

License Plate Detector Training:

Initializes and trains a YOLOv8 model (yolov8n.pt or last.pt for resuming) on the prepared license plate dataset.

Configuration for training includes epochs, image size, batch size, and saving periods.

License Plate Detector Evaluation:

Evaluates the trained YOLOv8 plate detector on a test set, reporting mAP, precision, and recall.

Dataset Preparation (Character Segmentation):

Similar to plate detection, but focuses on preparing data for character segmentation.

Converts XML annotations for individual characters to YOLO format.

Organizes data for character segmentation training.

Character Segmenter Training:

Trains another YOLOv8 model for character segmentation.

Character Segmenter Evaluation:

Evaluates the character segmenter.

OCR Dataset Preparation (Cutting):

Crops individual characters from the images based on XML annotations.

Converts cropped images to grayscale.

Organizes characters into class-specific folders (cropped_datasettt/{train, validation, test}/<character_label>).

OCR Model Training:

Defines and compiles a CNN model using TensorFlow/Keras.

Trains the CNN model on the cropped character images.

OCR Model Evaluation:

Evaluates the trained CNN model on the test set, reporting accuracy.

Full Pipeline Implementation:

Loads the trained YOLOv8 plate detector, YOLOv8 character segmenter, and CNN OCR model.

Defines a function recognize_plates_from_image that orchestrates the entire process:

Detects plates.

Segments characters from plates.

Recognizes characters using the OCR model.

Includes a function draw_farsi_text to correctly render Persian text on images for visualization.

Sample Demonstrations:

Includes code to run the full pipeline on sample images and save the results with recognized text and bounding boxes.

Getting Started
To run this project, follow these steps:

Clone the repository:

Bash

git clone <repository_url>
cd <repository_name>
Install dependencies:
The project requires gdown, patool, ultralytics, opencv-python, easyocr, tensorflow, Pillow, arabic-reshaper, and python-bidi. You can install them using pip:

Bash

pip install gdown patool ultralytics opencv-python tensorflow pillow arabic-reshaper python-bidi
Note: easyocr might not be strictly necessary for the core pipeline as presented, but it's listed in your original pip install.

Download the datasets:
The provided Python code includes gdown.download commands. When running the notebook or script, these will automatically download the necessary zipped datasets for plate detection and character segmentation/OCR from the specified Google Drive links.

Run the scripts/notebooks:
Execute the Python code in sequence. The provided code snippets should be run in a suitable environment (e.g., a Jupyter notebook, Google Colab) where GPU access is available for efficient training of the YOLO and CNN models.

The project will perform the following actions:

Download and extract datasets.

Prepare data for YOLO and CNN models.

Train YOLOv8 models for plate detection and character segmentation.

Train a CNN model for OCR.

Evaluate all models.

Demonstrate the end-to-end pipeline on sample images.

Models
plate_detection_model_yolo.pt: Trained YOLOv8 model for detecting license plates.

segmentation_model_yolo.pt: Trained YOLOv8 model for segmenting characters within a license plate.

ocr_model_cnn.h5: Trained CNN model for optical character recognition of individual Persian characters and digits.

These model files are expected to be available in the /content/ directory (as per the provided code paths) or adjusted to your local paths after training.

Results
The runs directory (or /content/runs in the Colab context) will contain training logs and weights for the YOLO models. The 100_samples directory will contain images with recognized license plates and bounding boxes, demonstrating the end-to-end pipeline's output.
