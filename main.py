import streamlit as st
import os
import pandas as pd
import tempfile
import logging
from PIL import Image, ImageDraw, ImageFont
from yolov5 import YOLOv5
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize YOLOv5 model
model_path = 'yolov5s.pt'  # Path to your YOLOv5 model weights
yolov5_model = YOLOv5(model_path, device='cpu')  # Use 'cpu' or 'cuda'

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

            x1, y1, x2, y2, conf, class_id = box[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to integers

            object_image = image.crop((x1, y1, x2, y2))

            # Save to a temporary file
            temp_filename = tempfile.mktemp(suffix=".png")
            object_image.save(temp_filename)

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

# Function to summarize attributes of the objects
def summarize_attributes(objects, descriptions):
    summary = []
    for obj in objects:
        description = next((item['description'] for item in descriptions if item['object_id'] == obj['object_id']), None)

        summary.append({
            'object_id': obj['object_id'],
            'bounding_box': obj['bounding_box'],
            'confidence': obj['confidence'],
            'class_id': obj['class_id'],
            'description': description
        })

    return summary

# Function to generate output image and CSV summary
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

    segmented_images = []
    for obj in summary:
        x1, y1, x2, y2 = obj['bounding_box']
        color = colors[obj['object_id'] % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - font_size - 5), obj['description'], fill=color, font=font)

        # Save each segmented object image with label
        object_image_path = tempfile.mktemp(suffix=f"_{obj['object_id']}.png")
        image.crop((x1, y1, x2, y2)).save(object_image_path)
        segmented_images.append({
            'object_id': obj['object_id'],
            'image_path': object_image_path,
            'description': obj['description']
        })

    output_image_path = tempfile.mktemp(suffix=".png")
    image.save(output_image_path)

    df = pd.DataFrame(summary)
    output_csv_path = tempfile.mktemp(suffix=".csv")
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Output saved as '{output_image_path}' and '{output_csv_path}'.")

    return output_image_path, output_csv_path, segmented_images

# Function to execute the full pipeline
def run_pipeline(image_path):
    try:
        boxes, image = segment_image(image_path)

        objects = extract_objects(boxes, image)
        descriptions = identify_objects(objects)

        summary = summarize_attributes(objects, descriptions)
        output_image_path, output_csv_path, segmented_images = generate_output(image_path, summary)
        logging.info("Pipeline completed successfully.")
        return output_image_path, output_csv_path, segmented_images
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return None, None, None

# Streamlit app integration
def main():
    st.title("AI Pipeline for Image Segmentation and Object Identification")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Running the pipeline...")

        try:
            # Save the uploaded image to a named temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_image_path = temp_file.name
                image.save(temp_image_path)

            # Run the AI pipeline on the uploaded image
            output_image_path, output_csv_path, segmented_images = run_pipeline(temp_image_path)

            # Display results
            if output_image_path and output_csv_path:
                st.image(output_image_path, caption='Output Image with Annotations', use_column_width=True)

                # Display segmented images with labels
                st.subheader("Segmented Objects")
                for seg_img in segmented_images:
                    col1, col2 = st.columns(2)
                    col1.image(seg_img['image_path'], caption=seg_img['description'], use_column_width=True)
                    col2.write(f"**Description**: {seg_img['description']}")

                # Display CSV summary
                summary_df = pd.read_csv(output_csv_path)
                st.write("Summary of Detected Objects:")
                st.dataframe(summary_df)

                # Provide download link for CSV
                with open(output_csv_path, "rb") as file:
                    st.download_button(
                        label="Download Summary CSV",
                        data=file,
                        file_name="summary.csv",
                        mime="text/csv"
                    )
            else:
                st.write("Error: Output files not found.")

        except OSError as e:
            st.error(f"Failed to save the image: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
