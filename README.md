# CLIP Traffic Sign Classification

This repository provides a complete framework for fine-tuning and evaluating Contrastive Language-Image Pre-Training (CLIP) models on the task of traffic sign classification. It includes scripts for training, evaluation, and an interactive web demo built with Gradio. The project supports models from both Hugging Face `transformers` and `open_clip` libraries.

## 🌟 Features

  - **Fine-tuning:** Train CLIP models on custom traffic sign datasets.
  - **Multi-Backend Support:** Seamlessly switch between Hugging Face (`openai/clip-*`) and OpenCLIP (`ViT-B-16`, etc.) models.
  - **Comprehensive Training:** Features include detailed logging (TensorBoard and text files), early stopping, and periodic model saving.
  - **Evaluation:** Generate detailed classification reports and confusion matrices for model performance analysis.
  - **Interactive Demo:** An easy-to-use Gradio web UI to perform both closed-set (predefined traffic signs) and open-vocabulary classification with your trained models.
  - **Dataset Support:** Pre-configured for two datasets:
      - A GTSRB-like dataset from Kaggle.
      - The Taiwan Traffic Sign Recognition Benchmark (TTSRB).

## ⚙️ Setup and Installation

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd CLIP-traffic-sign-classification
```

### 2\. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

Install all the required Python packages using the main `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4\. Download Datasets

The training and evaluation scripts expect the datasets to be in the `./data/` directory.

#### a) Kaggle Traffic Sign Dataset (for `traffic_Data`)

This dataset is used for general traffic sign training.
The dataset can be download from: https://www.kaggle.com/datasets/tuanai/traffic-signs-dataset

1.  **Setup Kaggle API:** Make sure you have your `kaggle.json` API token in `~/.kaggle/kaggle.json`.

2.  **Download and Unzip:** The following commands will download the dataset and place it in the correct directory.

    ```bash
    # Configure Kaggle API credentials
    mkdir -p ~/.kaggle
    # cp /path/to/your/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json

    # Download and extract the dataset
    kaggle datasets download -d ahemateja19bec1025/traffic-sign-dataset-classification
    unzip traffic-sign-dataset-classification.zip -d ./data/
    ```

    This will create a `./data/traffic_Data` directory.

#### b) Taiwan Traffic Sign Recognition Benchmark (for `ttsrb`)

This dataset is used for evaluation in `test_on_ttsrb.py` and can also be used for training.
The dataset can be download from: https://github.com/exodustw/Taiwan-Traffic-Sign-Recognition-Benchmark


Your final `./data` directory should look like this:

```
data/
├── traffic_Data/
│   ├── DATA/
│   ├── TEST/
│   └── labels.csv
│   └── ...
└── ttsrb/
    ├── train/
    ├── train_plus/
    ├── test/
    └── ...
```

## 🚀 Usage

### 1\. Training a Model

Use the `train.py` script to fine-tune a CLIP model. You can specify the model architecture, dataset, and training hyperparameters.

**Example Command:**

```bash
python train.py \
    --model_name "openai/clip-vit-base-patch32" \
    --dataset_name "traffic_Data" \
    --output_dir "./results/ViT_B_patch32_traffic_data" \
    --batch_size 128 \
    --num_epochs 50 \
    --learning_rate 5e-7 \
    --patience 5
```

  - `--model_name`: Choose a model from Hugging Face (e.g., `"openai/clip-vit-base-patch16"`) or OpenCLIP (e.g., `"ViT-B-16"`).
  - `--dataset_name`: Specify the dataset to use (`"traffic_Data"` or `"ttsrb"`).
  - `--output_dir`: Directory to save checkpoints, logs, and TensorBoard data.

### 2\. Monitoring Training with TensorBoard

You can visualize training metrics like loss and validation accuracy using TensorBoard.

```bash
tensorboard --logdir ./results/ViT_B_patch32_traffic_data/tensorboard
```

You will see plots for loss, accuracy, and a visualization of the image-text similarity matrix at each validation step.

### 3\. Evaluating a Model

After training, use `test_on_ttsrb.py` to evaluate your model's performance on the TTSRB dataset. This script generates a classification report and a confusion matrix.

**Example Command:**

```bash
python test_on_ttsrb.py \
    --model_name "openai/clip-vit-base-patch16" \
    --model_path "./results/ViT_B_patch16_ttsrb/best_model.pt" \
    --output_dir "./results/ViT_B_patch16_ttsrb/evaluation"
```

This will save `evaluation_report_ttsrb.txt` and `confusion_matrix_ttsrb.png` in the specified output directory.

### 4\. Zero-Shot Inference

The `hw1.ipynb` notebook provides a simple example of how to load a fine-tuned model and perform zero-shot inference on a few test images, visualizing the results directly in the notebook.

## 🖥️ Gradio Demo Application

This project includes an interactive Gradio application to explore your trained models in a user-friendly interface.

### Running the App

1.  **Install Gradio requirements:**

    ```bash
    pip install -r gradio-demo-app/requirements.txt
    ```

2.  **Run the application from the project root directory:**

    ```bash
    python gradio-demo-app/src/app.py
    ```

    The application will launch on a local server (e.g., `http://127.0.0.1:7860`).

### Demo Features

The demo has two main tabs:

  - **Traffic Sign Classification (Closed-Set):** Upload an image or select one from the test set to classify it against all known traffic sign classes. It displays the top prediction and a list of confidence scores.
  - **Open-Vocabulary Explorer:** Upload any image and provide your own custom, comma-separated labels (e.g., "stop sign, yield sign, speed limit sign"). The model will classify the image among the labels you provided, showcasing its zero-shot capabilities.
