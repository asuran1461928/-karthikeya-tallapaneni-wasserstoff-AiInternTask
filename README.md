# AI Pipeline for Image Segmentation and Object Identification

This project is a Streamlit application that allows users to upload an image, segment it using YOLOv5, and identify objects within the image using BLIP (Bootstrapping Language-Image Pre-training). The app outputs the segmented objects with captions, annotated images, and a CSV file summarizing the detected objects.

## Features

- **Image Segmentation**: Segment objects within an image using YOLOv5.
- **Object Identification**: Automatically generate captions for segmented objects using BLIP.
- **Results Display**: View the segmented objects, annotations, and download a CSV summary.
- **User-Friendly Interface**: Streamlit provides a clean and intuitive interface for interacting with the app.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, manually install the necessary libraries:

```bash
pip install streamlit pandas pillow yolov5 transformers torch
```

### Download the YOLOv5 Model

Ensure you have the YOLOv5 model weights file (`yolov5s.pt`). If not, you can download it from the official [YOLOv5 repository](https://github.com/ultralytics/yolov5) or use the following command:

```bash
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt -P .
```

### Running the App

After installing the dependencies, you can start the Streamlit app by running:

```bash
streamlit run app.py
```

Replace `app.py` with the name of your Python file if different.

## Usage

1. **Upload an Image**: Click on "Choose an image..." and upload a `.jpg`, `.jpeg`, or `.png` file.
2. **View Results**: The app will display the uploaded image, the output image with annotations, segmented objects, and a summary of detected objects.
3. **Download CSV**: Download the CSV file containing the details of the detected objects.

## Troubleshooting

- **OSError**: If you encounter issues saving images, ensure that your environment has write permissions in the directory where the temporary files are created.
- **Font Errors**: If the custom font (`arialbd.ttf`) is not found, the app will fall back to a default font.

## Logging

The app logs its activity at the `INFO` level. If you encounter issues, check the logs for more detailed error messages.

## Future Enhancements

- **GPU Support**: Modify the app to use GPU acceleration for faster processing.
- **Additional Models**: Add support for other object detection and captioning models.
- **Customization**: Allow users to adjust model parameters directly from the app.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [BLIP](https://github.com/salesforce/BLIP) by Salesforce Research
- [Streamlit](https://streamlit.io) for the user interface framework

---
