# Gradio Demo Application

This project is a Gradio-based web application that allows users to upload images and receive predictions from a fine-tuned CLIP model. Below are the details on how to set up and run the application.

## Project Structure

```
gradio-demo-app
├── src
│   ├── app.py               # Main entry point for the application
│   ├── dataset_utils.py     # Utility functions for dataset handling
│   ├── clip_model_utils.py   # Functions for loading and processing the CLIP model
│   └── types
│       └── index.py         # Type definitions used in the application
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd gradio-demo-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the application, execute the following command:

```
python src/app.py
```

Once the application is running, you can access it in your web browser at `http://localhost:7860`.

## Usage

1. Upload an image of a traffic sign.
2. Click the "Submit" button to receive a prediction.
3. The predicted label will be displayed below the image.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.