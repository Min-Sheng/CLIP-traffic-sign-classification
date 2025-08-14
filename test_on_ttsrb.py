import matplotlib.font_manager as fm
from matplotlib import rcParams

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
prop = fm.FontProperties(fname=font_path)
family_name = prop.get_name()

rcParams['font.family'] = family_name
rcParams['axes.unicode_minus'] = False

import argparse
import sys
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from rich.progress import Progress
from loguru import logger

from dataset_utils import TrafficSignDataset, val_transform
from clip_model_utils import load_model_and_processor


DATASET = "ttsrb"


def test(model_name, model_path, output_dir):
    # Setup logging
    log_txt_path = Path(output_dir) / "test_log.txt"
    logger.remove()
    logger.add(str(log_txt_path), format="{time} {level} {message}", level="INFO", enqueue=True, mode='w')
    logger.add(sys.stdout, format="{time} {level} {message}", level="INFO", enqueue=True)

    logger.info("Starting testing...")
    logger.info(f"Model: {model_name}")
    logger.info(f"Model Weights Path: {model_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load Model and Processor
    try:
        model, processor, backend = load_model_and_processor(model_name, device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Load test data
    data_root = Path("./data") / DATASET
    test_data_dir = data_root / 'train'

    test_image_paths = []
    test_labels = []

    for class_folder in sorted(list(test_data_dir.iterdir())):
        if class_folder.is_dir():
            for file_path in class_folder.iterdir():
                if file_path.suffix == '.png':
                    test_image_paths.append(str(file_path))
                    test_labels.append(class_folder.stem)
    
    class_names = sorted(np.unique(test_labels))
    class_ids = np.arange(len(class_names))
    label_dict = dict(zip(class_ids, class_names))
    reversed_label_dict = {v: k for k, v in label_dict.items()}
    test_class_ids = [reversed_label_dict[test_label] for test_label in test_labels]
    test_df = pd.DataFrame({'Path': test_image_paths, 'ClassId': test_class_ids})

    test_df['ClassName'] = test_df['ClassId'].map(label_dict) # for prompting
    test_dataset = TrafficSignDataset(test_df, transform=val_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    logger.info(f"Dataset: {DATASET} are loaded with {len(test_dataset)} images.")

    # Prepare Text Prompts (the "Zero-Shot Head")
    all_text_prompts = [f"a photo of {name} traffic sign" for name in class_names]

    # Perform Inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # Pre-compute text features once
        logger.info("Pre-computing text features for all classes...")
        text_inputs = processor(text=all_text_prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        
        if backend == 'huggingface':
            all_text_features = model.get_text_features(**text_inputs)
        else: # openclip
            all_text_features = model.encode_text(text_inputs)
        all_text_features /= all_text_features.norm(dim=-1, keepdim=True)
        
        # Get the learned temperature
        logit_scale = model.logit_scale.exp()
        
        logger.info("Running inference on the test set...")
        with Progress() as progress:
            task = progress.add_task("[cyan]Evaluating...", total=len(test_dataloader))
            for images, labels, _ in test_dataloader:
                images = images.to(device)
                
                # Get image features
                if backend == 'huggingface':
                    image_features = model.get_image_features(pixel_values=images)
                else: # openclip
                    image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity logits (scaled)
                logits = (image_features @ all_text_features.T) * logit_scale
                
                # Get predictions
                preds = logits.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                progress.update(task, advance=1)

    # Calculate and Save Metrics
    logger.info("Calculating metrics...")
    
    # Map numeric predictions back to class names for the report
    pred_class_indices = [class_ids[i] for i in all_preds]

    # Overall Accuracy
    accuracy = accuracy_score(all_labels, pred_class_indices)
    logger.info(f"Overall Accuracy: {accuracy:.4f}")

    # Classification Report (Precision, Recall, F1-Score)
    report = classification_report(
        all_labels, 
        pred_class_indices, 
        target_names=class_names,
        digits=4
    )
    logger.info("Classification Report:\n" + report)
    
    report_path = Path(output_dir) / f"evaluation_report_{DATASET}.txt"
    with open(report_path, 'w') as f:
        f.write(f"Evaluation Report for model: {model_name}\n")
        f.write(f"Weights: {model_path}\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    logger.info(f"Classification report saved to {report_path}")

    # Generate and Save Confusion Matrix
    logger.info("Generating confusion matrix...")
    cm = confusion_matrix(all_labels, pred_class_indices, labels=class_ids)
    
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12}, cbar=False) # Annotations with small font size
    
    ax.set_xlabel('Predicted Labels', fontsize=14)
    ax.set_ylabel('True Labels', fontsize=14)
    ax.set_title(f'Overall Accuracy: {accuracy:.4f}', fontsize=18)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = Path(output_dir) / f"confusion_matrix_{DATASET}.png"
    plt.savefig(cm_path, dpi=300)
    logger.info(f"Confusion matrix plot saved to {cm_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned CLIP model.")
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True,
        help='Name of the base CLIP model (e.g., "openai/clip-vit-base-patch16").'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='Path to the fine-tuned .pt model weights file.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str,
        required=True,
        help='Directory to save evaluation results.'
    )
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    test(
        model_name=args.model_name,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
