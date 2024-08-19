import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import tempfile
import streamlit as st

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Set the path to the installed Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize BLIP processor and model for auto-captioning
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to('cpu')

# Mock YOLOv5 model (Replace with actual model loading if possible)
class MockYOLOv5:
    def predict(self, image, size=640):
        # Mocking prediction results
        return type('Results', (object,), {'xyxy': [np.array([[50, 50, 200, 200, 0.9, 1]])]})()

yolov5_model = MockYOLOv5()

# Function to generate captions using BLIP
def generate_caption(image):
    try:
        inputs = caption_processor(images=image, return_tensors="pt").to('cpu')
        with torch.no_grad():
            caption_ids = caption_model.generate(**inputs, max_new_tokens=50)
        caption = caption_processor.decode(caption_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logging.error(f"Error generating caption: {e}")
        return "Captioning failed"

# Function to segment an image into objects using YOLOv5 (Mocked)
def segment_image(image_path):
    logging.info(f"Segmenting image: {image_path}")
    if not os.path.exists(image_path):
        logging.error(f"Image file does not exist: {image_path}")
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        results = yolov5_model.predict(image, size=640)  # Perform inference
        boxes = results.xyxy[0].numpy()  # Extract boxes

        logging.info(f"Segmentation completed: {len(boxes)} boxes detected.")
        return boxes, image
    except Exception as e:
        logging.error(f"Error in segmentation: {e}")
        raise

# Function to extract objects from bounding boxes
def extract_objects(boxes, image):
    objects = []
    for i, box in enumerate(boxes):
        try:
            if len(box) < 6:
                logging.warning(f"Skipping invalid box with data: {box}")
                continue

            # Unpack bounding box and detection data
            x1, y1, x2, y2, conf, class_id = box[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to integers

            # Extract object image
            object_image = np.array(image)[y1:y2, x1:x2]
            object_image_pil = Image.fromarray(object_image)
            filename = f"object_{i}.png"
            object_image_pil.save(filename)

            objects.append({
                'object_id': i,
                'filename': filename,
                'bounding_box': (x1, y1, x2, y2),
                'confidence': conf,
                'class_id': class_id
            })
        except Exception as e:
            logging.error(f"Error processing box {i}: {e}")

    logging.info(f"Extracted {len(objects)} objects.")
    return objects

# Function to identify objects using auto-captioning
def identify_objects(objects):
    descriptions = []
    for obj in objects:
        try:
            image = Image.open(obj['filename'])
            caption = generate_caption(image)
            descriptions.append({
                'object_id': obj['object_id'],
                'description': caption
            })
            logging.info(f"Caption generated for object {obj['object_id']}: {caption}")
        except Exception as e:
            logging.error(f"Error in captioning object {obj['object_id']}: {e}")
            descriptions.append({
                'object_id': obj['object_id'],
                'description': "Captioning failed"
            })

    return descriptions

# Function to extract text from objects using OCR
def extract_text_from_objects(objects):
    text_data = []
    for obj in objects:
        try:
            if not os.path.exists(obj['filename']):
                raise FileNotFoundError(f"Image file does not exist: {obj['filename']}")

            # Check for file readability
            image = Image.open(obj['filename'])
            if image is None:
                raise ValueError(f"Image file not readable: {obj['filename']}")

            # Convert to OpenCV image
            image_cv = np.array(image)
            text = pytesseract.image_to_string(image_cv)
            text_data.append({
                'object_id': obj['object_id'],
                'text': text
            })
            logging.info(f"Text extracted for object {obj['object_id']}: {text}")
        except Exception as e:
            logging.error(f"Error extracting text from object {obj['object_id']}: {e}")
            text_data.append({
                'object_id': obj['object_id'],
                'text': "OCR failed"
            })

    return text_data

# Function to summarize attributes of the objects
def summarize_attributes(objects, descriptions, text_data):
    summary = []
    for obj in objects:
        description = next((item['description'] for item in descriptions if item['object_id'] == obj['object_id']), None)
        text = next((item['text'] for item in text_data if item['object_id'] == obj['object_id']), None)

        summary.append({
            'object_id': obj['object_id'],
            'bounding_box': obj['bounding_box'],
            'confidence': obj['confidence'],
            'class_id': obj['class_id'],
            'description': description,
            'text': text
        })

    return summary

# Function to generate output image and CSV summaries
def generate_output(image_path, summary, objects):
    # Open and process the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Define colors for bounding boxes
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "lime", "pink"]

    # Try to load custom font
    font_size = 20
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        logging.warning("Custom font 'arialbd.ttf' not found. Using default font.")

    # Draw bounding boxes and captions on the image
    for obj in summary:
        x1, y1, x2, y2 = obj['bounding_box']
        color = colors[obj['object_id'] % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - font_size - 5), obj['description'], fill=color, font=font)

    # Save the annotated image
    output_image_path = tempfile.mktemp(suffix=".png")
    image.save(output_image_path)

    # Create DataFrame for summary
    df_summary = pd.DataFrame(summary)
    output_csv_path = tempfile.mktemp(suffix=".csv")
    df_summary.to_csv(output_csv_path, index=False)
    
    # Create DataFrame for objects
    df_objects = pd.DataFrame(objects)
    objects_csv_path = tempfile.mktemp(suffix=".csv")
    df_objects.to_csv(objects_csv_path, index=False)

    logging.info(f"Output saved as '{output_image_path}', '{output_csv_path}', and '{objects_csv_path}'.")

    return output_image_path, output_csv_path, objects_csv_path

# Function to execute the full pipeline with parallel processing
def run_pipeline(image_path):
    try:
        boxes, image = segment_image(image_path)
        objects = extract_objects(boxes, image)

        # Run identification and text extraction in parallel
        with ThreadPoolExecutor() as executor:
            descriptions_future = executor.submit(identify_objects, objects)
            text_data_future = executor.submit(extract_text_from_objects, objects)
            
            descriptions = descriptions_future.result()
            text_data = text_data_future.result()

        summary = summarize_attributes(objects, descriptions, text_data)
        return generate_output(image_path, summary, objects)
    except Exception as e:
        logging.error(f"Error in pipeline: {e}")
        raise

# Streamlit interface
def main():
    st.title("AI Pipeline for Image Analysis")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.image(temp_file_path, caption='Uploaded Image', use_column_width=True)

        if st.button("Run Pipeline"):
            try:
                output_image_path, output_csv_path, objects_csv_path = run_pipeline(temp_file_path)

                # Display the output image with annotations
                st.image(output_image_path, caption='Output Image with Annotations', use_column_width=True)

                # Display summary CSV
                summary_df = pd.read_csv(output_csv_path)
                st.write("Summary of Detected Objects:")
                st.dataframe(summary_df)

                # Display extracted objects CSV
                objects_df = pd.read_csv(objects_csv_path)
                st.write("Extracted Objects:")
                st.dataframe(objects_df)

                # Download buttons for CSV files
                with open(output_csv_path, "rb") as file:
                    st.download_button(
                        label="Download Summary CSV",
                        data=file,
                        file_name="summary.csv",
                        mime="text/csv"
                    )
                
                with open(objects_csv_path, "rb") as file:
                    st.download_button(
                        label="Download Extracted Objects CSV",
                        data=file,
                        file_name="object_id.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error: {e}")
                logging.error(f"Error running pipeline: {e}")

if __name__ == "__main__":
    main()
