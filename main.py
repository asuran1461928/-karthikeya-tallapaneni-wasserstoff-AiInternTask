import torch
from PIL import Image, ImageDraw, ImageFont
import tempfile
import logging
import os
import pytesseract
import streamlit as st
from yolov5 import YOLOv5
from transformers import BlipProcessor, BlipForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Set the path to the installed Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize YOLOv5 model from the official repository
model_path = 'yolov5s.pt'  # Path to your YOLOv5 model weights
yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='local')

# Initialize BLIP processor and model for auto-captioning
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to('cpu')

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

# Function to segment an image into objects using YOLOv5
def segment_image(image_path):
    logging.info(f"Segmenting image: {image_path}")
    if not os.path.exists(image_path):
        logging.error(f"Image file does not exist: {image_path}")
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        results = yolov5_model(image)  # Perform inference
        boxes = results.xyxy[0].cpu().numpy()  # Extract boxes

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

            x1, y1, x2, y2, conf, class_id = box[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to integers

            object_image = np.array(image)[y1:y2, x1:x2]
            object_image_pil = Image.fromarray(object_image)

            # Save to a temporary file
            temp_filename = tempfile.mktemp(suffix=".png")
            object_image_pil.save(temp_filename)

            objects.append({
                'object_id': i,
                'filename': temp_filename,
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
            # Use PIL to open the image file
            image = Image.open(obj['filename'])
            # Convert image to grayscale for better OCR results (optional)
            image = image.convert('L')
            # Perform OCR with pytesseract
            text = pytesseract.image_to_string(image)
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
            'text': text,
            'filename': obj['filename']
        })

    return summary

# Function to generate output image and display results
def generate_output(image_path, summary):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "lime", "pink"]

    font_size = 20
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        logging.warning("Custom font 'arialbd.ttf' not found. Using default font.")

    for obj in summary:
        x1, y1, x2, y2 = obj['bounding_box']
        color = colors[obj['object_id'] % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - font_size - 5), obj['description'], fill=color, font=font)

    output_image_path = tempfile.mktemp(suffix=".png")
    image.save(output_image_path)

    logging.info(f"Output image saved as '{output_image_path}'.")

    return output_image_path

# Function to execute the full pipeline with parallel processing
def run_pipeline(image_path):
    try:
        boxes, image = segment_image(image_path)

        with ThreadPoolExecutor() as executor:
            objects_future = executor.submit(extract_objects, boxes, image)
            objects = objects_future.result()

            descriptions_future = executor.submit(identify_objects, objects)
            text_data_future = executor.submit(extract_text_from_objects, objects)

            descriptions = descriptions_future.result()
            text_data = text_data_future.result()

        summary = summarize_attributes(objects, descriptions, text_data)
        output_image_path = generate_output(image_path, summary)
        logging.info("Pipeline completed successfully.")
        return output_image_path, summary
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return None, None

# Streamlit app integration
def main():
    st.title("Image Analysis App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, uploaded_file.name)
        image.save(image_path)

        # Run the AI pipeline on the uploaded image
        output_image_path, summary = run_pipeline(image_path)

        # Display the results
        if output_image_path:
            st.image(output_image_path, caption='Output Image with Annotations', use_column_width=True)

            # Display detailed information
            st.subheader("Object Details")
            for obj in summary:
                st.write(f"**Object ID**: {obj['object_id']}")
                st.write(f"**Bounding Box**: {obj['bounding_box']}")
                st.write(f"**Confidence**: {obj['confidence']:.2f}")
                st.write(f"**Class ID**: {obj['class_id']}")
                st.write(f"**Description**: {obj['description']}")
                st.write(f"**Text**: {obj['text']}")
                st.image(obj['filename'], caption=f"Object {obj['object_id']} Image", use_column_width=True)
                st.write("---")
        else:
            st.write("Error: Output files not found.")

if __name__ == "__main__":
    main()
