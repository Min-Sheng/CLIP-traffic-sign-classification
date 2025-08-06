import argparse
import matplotlib.pyplot as plt
import random
from pathlib import Path
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
from PIL import Image
from loguru import logger
from rich.progress import Progress

from dataset_utils import TrafficSignDataset, train_transform, val_transform, unnormalize
from visualize_utils import TensorboardVisualizer

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

MODEL_NAME = "openai/clip-vit-base-patch32"
DATA_ROOT = './data/traffic_Data'
LABEL_CSV = './data/corrected_labels.csv'


def main(
    model_name,
    batch_size,
    num_epochs,
    learning_rate,
    output_dir,
    save_every=5,
    patience=5,
    eval_every=1,
):
     # TensorBoard visualizer
    visualizer = TensorboardVisualizer(log_dir=str(Path(output_dir) / "tensorboard"))

    # txt log file
    log_txt_path = Path(output_dir) / "train_log.txt"
    logger.remove()  # remove default console logger
    logger.add(str(log_txt_path), format="{time} {level} {message}", level="INFO", enqueue=True, mode='w')
    logger.add(sys.stdout, format="{time} {level} {message}", level="INFO", enqueue=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)

    logger.info(f"CLIP model is loaded on {device}")
   
    data_root = './data/traffic_Data'
    train_data_dir = Path(data_root) / 'DATA'

    train_image_paths = []
    train_labels = []

    for class_folder in sorted(list(train_data_dir.iterdir())):
        if class_folder.is_dir():
            for file_path in class_folder.iterdir():
                if file_path.suffix == '.png':
                    train_image_paths.append(str(file_path))
                    train_labels.append(int(class_folder.stem))

    df = pd.DataFrame({'Path': train_image_paths, 'ClassId': train_labels})

    label_map = pd.read_csv(LABEL_CSV)
    label_dict = dict(zip(label_map['ClassId'], label_map['Name']))

    df['ClassName'] = df['ClassId'].map(label_dict) # for prompting
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['ClassId'], random_state=seed)

    train_dataset = TrafficSignDataset(train_df, transform=train_transform)
    val_dataset = TrafficSignDataset(val_df, transform=val_transform)
    logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    loss_img_fn = nn.CrossEntropyLoss()
    loss_txt_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9,0.98),
        eps=1e-6,
        weight_decay=0.2,
    ) # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    # Early stopping & model saving
    best_acc = 0.0
    patience = 5
    patience_counter = 0

    logger.info(f"Starting training with batch_size={batch_size}, num_epochs={num_epochs}, learning_rate={learning_rate}, output_dir={output_dir}")

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_step(model, processor, train_dataloader, optimizer, loss_img_fn, loss_txt_fn, device)

        # eval every n epoch
        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            val_loss, val_acc, similarity_matrix, class_names = val_step(
                model, processor, val_dataloader, loss_img_fn, loss_txt_fn, device, label_dict
            )

            # TensorBoard logger
            visualizer.log_metrics(train_loss, val_loss, val_acc, epoch)
            
            sampled_images = sample_images_per_class(val_dataset, label_dict)
            visualizer.log_similarity_matrix(similarity_matrix, class_names, epoch, sampled_images=sampled_images)

            # txt logger
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save the best model
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                save_path = Path(output_dir) / "best_model.pt"
                torch.save(model.state_dict(), save_path)
                logger.info(f"Best model saved with accuracy: {best_acc:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Early stopping counter: {patience_counter}/{patience}")

            # Early stopping
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

        # Save periodically
        if (epoch + 1) % save_every == 0:
            periodic_save_path = Path(output_dir) / f"model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), periodic_save_path)
            logger.info(f"Model saved at epoch {epoch+1} to {periodic_save_path}")

    visualizer.close()


def train_step(model, processor, dataloader, optimizer, loss_img_fn, loss_txt_fn, device):
    total_loss = 0
    model.train()
    with Progress() as progress:
        task = progress.add_task("[cyan]Training...", total=len(dataloader))
        for batch in dataloader:
            images, _, text_prompts = batch
            inputs = processor(
                text=text_prompts, return_tensors='pt', padding=True, truncation=True,
            ).to(device)
            inputs['pixel_values'] = images.to(device)
            outputs = model(**inputs)

            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            batch_size = images.size(0)
            ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

            loss_img = loss_img_fn(logits_per_image, ground_truth)
            loss_txt = loss_txt_fn(logits_per_text, ground_truth)
            loss = (loss_img + loss_txt) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.update(task, advance=1)

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Train Loss: {avg_loss:.4f}")
    return avg_loss


def val_step(model, processor, dataloader, loss_img_fn, loss_txt_fn, device, label_dict):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    class_ids_sorted = sorted(label_dict.keys())
    class_names = [label_dict[class_id] for class_id in class_ids_sorted]

    all_text_prompts = [f"A photo of a {name} traffic sign" for name in class_names]
    all_text_inputs = processor(
        text=all_text_prompts, return_tensors='pt', padding=True, truncation=True,
    ).to(device)

    class_image_feature_sum = {class_id: None for class_id in class_ids_sorted}
    class_image_feature_count = {class_id: 0 for class_id in class_ids_sorted}
    with torch.no_grad():
        all_text_features = model.get_text_features(**all_text_inputs)
        all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)

        with Progress() as progress:
            task = progress.add_task("[magenta]Validating...", total=len(dataloader))
            for batch in dataloader:
                images, labels, text_prompts = batch
                inputs = processor(
                    text=text_prompts, return_tensors='pt', padding=True, truncation=True,
                ).to(device)
                inputs['pixel_values'] = images.to(device)

                outputs = model(**inputs)

                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text

                batch_size = images.size(0)
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

                loss_img = loss_img_fn(logits_per_image, ground_truth)
                loss_txt = loss_txt_fn(logits_per_text, ground_truth)
                loss = (loss_img + loss_txt) / 2

                total_loss += loss.item()

                image_features = model.get_image_features(pixel_values=inputs.pixel_values)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                similarity_logits = image_features @ all_text_features.T
                preds = similarity_logits.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

                for feature, label in zip(image_features, labels):
                    class_id = int(label.item())
                    if class_image_feature_sum[class_id] is None:
                        class_image_feature_sum[class_id] = feature
                    else:
                        class_image_feature_sum[class_id] += feature
                    class_image_feature_count[class_id] += 1

                progress.update(task, advance=1)

        # Calculate mean features
        class_mean_image_features_dict = {}
        for class_id in class_ids_sorted:
            if class_image_feature_count[class_id] > 0:
                mean_feature = class_image_feature_sum[class_id] / class_image_feature_count[class_id]
                mean_feature = mean_feature / mean_feature.norm(dim=-1, keepdim=True)
                class_mean_image_features_dict[class_id] = mean_feature
            else:
                class_mean_image_features_dict[class_id] = torch.zeros_like(next(iter(class_image_feature_sum.values()))).to(device)

        all_mean_image_features = []
        for class_id in sorted(class_mean_image_features_dict.keys()):
            feature = class_mean_image_features_dict[class_id] # (num_classes, hidden_dim)
            all_mean_image_features.append(feature)
        all_mean_image_features = torch.stack(all_mean_image_features, dim=0)  # (num_classes, hidden_dim)

        similarity_matrix = all_text_features @ all_mean_image_features.T  # (num_classes, num_classes)

    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    return avg_loss, acc, similarity_matrix.cpu().numpy(), class_names


def sample_images_per_class(dataset, label_dict, n=1):
    class_ids_sorted = sorted(label_dict.keys())  # e.g., [0, 1, 2, 3, ..., 57]
    class_to_indices = {class_id: [] for class_id in class_ids_sorted}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        class_id = int(sample['label']) if isinstance(sample, dict) else int(sample[1])
        class_to_indices[class_id].append(idx)

    sampled_images = {class_id: None for class_id in class_ids_sorted}
    for class_id in class_ids_sorted:
        indices = class_to_indices[class_id]
        chosen = random.sample(indices, min(n, len(indices)))
        images = []
        if len(chosen) == 0:
            image = Image.new('RGB', (224, 224), color='white')  # Placeholder if no images
            images.append(image)
        else:
            for idx in chosen:
                sample = dataset[idx]
                image = sample[0]
                if isinstance(image, torch.Tensor):
                    image = to_pil_image(unnormalize(image.squeeze()).clamp(0, 1))
                images.append(image)
        sampled_images[class_id] = images
    return sampled_images


def save_args(args, output_dir):
    args_path = Path(output_dir) / "args.txt"
    with open(args_path, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CLIP model on traffic sign dataset")
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='Name of the CLIP model to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=5e-7, help='Learning rate for the optimizer')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the model and results')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every n epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate every n epochs')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_args(args, args.output_dir)

    main(args.model_name, args.batch_size, args.num_epochs, args.learning_rate, args.output_dir, args.save_every, args.eval_every)
