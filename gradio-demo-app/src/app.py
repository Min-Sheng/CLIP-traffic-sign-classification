import sys
import gradio as gr
import torch
from PIL import Image
import pandas as pd
from pathlib import Path
from dataset_utils import val_transform
from clip_model_utils import load_model_and_processor


# Automatically detect the number of available GPUs
if not torch.cuda.is_available():
    print("Warning: CUDA not detected. All models will run on CPU.", file=sys.stderr)
    num_gpus = 0
else:
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} available GPU(s).")


# List of available models and their paths
MODEL_CONFIGS = {
    "openai/clip-vit-base-patch32:finetune-on-traffic-sign": "/home/vincentwu/clip_hw/HW1/results/ViT_B_patch32/best_model.pt",
    "openai/clip-vit-base-patch16:finetune-on-traffic-sign": "/home/vincentwu/clip_hw/HW1/results/ViT_B_patch16/best_model.pt",
    "openai/clip-vit-base-patch32:finetune-on-ttsrb": "/home/vincentwu/clip_hw/HW1/results/ViT_B_patch32_ttsrb/best_model.pt",
    "openai/clip-vit-base-patch16:finetune-on-ttsrb": "/home/vincentwu/clip_hw/HW1/results/ViT_B_patch16_ttsrb/best_model.pt",
}

# Labels for the closed-set (traffic sign) task
label_csv = './data/traffic_Data/corrected_labels.csv'
label_map = pd.read_csv(label_csv)
label_dict = dict(zip(label_map['ClassId'], label_map['Name']))
class_ids_sorted = sorted(label_dict.keys())
class_names = [label_dict[class_id] for class_id in class_ids_sorted]
closed_set_prompts = [f"a photo of {name} traffic sign" for name in class_names]

# Test Images for the closed-set task
test_data_dir = Path('./data/traffic_Data/TEST')
test_image_paths = sorted([str(p) for p in test_data_dir.glob("*.png")])

def get_text_features(prompts, model, processor, backend, device):
    """Computes and returns normalized text features for a given list of prompts."""
    text_inputs = processor(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        if backend == 'huggingface':
            text_features = model.get_text_features(**text_inputs)
        else:
            text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

# Preload all models and cache text features for the CLOSED-SET task
print("Starting to load models and pre-compute text features...")
loaded_models = {}
for i, (model_name, model_path) in enumerate(MODEL_CONFIGS.items()):
    device = f"cuda:{i % num_gpus}" if num_gpus > 0 else "cpu"
    print(f"Loading model '{model_name}' onto device '{device}'...")
    model, processor, backend = load_model_and_processor(model_name.split(':')[0], device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Pre-computing closed-set text features for '{model_name}'...")
    cached_closed_set_features = get_text_features(closed_set_prompts, model, processor, backend, device)
    loaded_models[model_name] = (model, processor, backend, device, cached_closed_set_features)
print("All models and text features loaded successfully!")


# --- Backend Functions for Gradio ---
def run_closed_set_prediction(image, model_name):
    """
    Performs closed-set prediction and returns both the top label and all confidence scores.
    """
    # Unpack model and cached text features for all traffic sign classes
    model, _, backend, device, text_features = loaded_models[model_name]

    image = image.convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get image features
        if backend == 'huggingface':
            image_features = model.get_image_features(pixel_values=image_tensor)
        else:
            image_features = model.encode_image(image_tensor)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Scale logits using the model's learned temperature
        logit_scale = model.logit_scale.exp()
        logits = (image_features @ text_features.T) * logit_scale

        # Get probabilities using softmax
        probs = logits.softmax(dim=-1).squeeze()

        # Find top prediction
        pred_id = probs.argmax().item()
        pred_label = class_names[pred_id]
        
        # Format all scores into a dictionary for the gr.Label component
        confidences = {name: prob.item() for name, prob in zip(class_names, probs)}

    return pred_label, confidences


def run_open_vocab_prediction(image, model_name, custom_labels_str, prompt_template):
    """
    Handles prediction for the open-vocabulary task with custom user labels.
    """
    if image is None:
        return { "error": "Please upload an image." }, None
    if not custom_labels_str:
        return { "error": "Please enter at least one custom label." }, None

    model, processor, backend, device, _ = loaded_models[model_name]
    custom_labels = [label.strip() for label in custom_labels_str.split(',') if label.strip()]
    if not custom_labels:
        return { "error": "Please enter valid, comma-separated labels." }, None

    custom_prompts = [prompt_template.format(label) for label in custom_labels]
    text_features = get_text_features(custom_prompts, model, processor, backend, device)
    
    image = image.convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        if backend == 'huggingface':
            image_features = model.get_image_features(pixel_values=image_tensor)
        else:
            image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        similarity = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
        scores = similarity[0].cpu().numpy()

    top_result = {custom_labels[i]: float(scores[i]) for i in range(len(custom_labels))}
    barplot_data = pd.DataFrame({"Class": custom_labels, "Confidence": scores})
    return top_result, barplot_data


# --- Gradio UI with Two Tabs ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CLIP Model Explorer")
    model_selector = gr.Dropdown(
        choices=list(MODEL_CONFIGS.keys()), 
        value="openai/clip-vit-base-patch16", 
        label="Select Model (Applies to all tabs)"
    )
    
    with gr.Tabs():
        # --- TAB 1: Closed-Set Traffic Sign Classification ---
        with gr.TabItem("Traffic Sign Classification"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1. Choose an Image")
                    with gr.Tabs():
                        with gr.TabItem("Select from Test Set"):
                            closed_set_selector = gr.Dropdown(choices=test_image_paths, label="Select Test Image")
                        with gr.TabItem("Upload Image"):
                            closed_set_uploader = gr.Image(type="pil", label="Upload Your Image")
                    closed_set_button = gr.Button("Classify Traffic Sign", variant="primary")
                with gr.Column():
                    gr.Markdown("### 2. View Results")
                    closed_set_preview = gr.Image(type="pil", label="Image Preview")
                    closed_set_label = gr.Textbox(label="Top Prediction")
                    closed_set_scores = gr.Label(label="Confidence Scores", num_top_classes=5)

            def update_traffic_preview_from_dropdown(image_path):
                if image_path:
                    image = Image.open(image_path).convert("RGB")
                    return image.resize((512, 512)), None, "", None # Clear preview, uploader, label, and scores
                return None, None, "", None

            def update_traffic_preview_from_upload(uploaded_image):
                if uploaded_image:
                    image = uploaded_image.convert("RGB")
                    return image.resize((512, 512)), None, "", None # Clear preview, dropdown, label, and scores
                return None, None, "", None

            closed_set_selector.change(
                fn=update_traffic_preview_from_dropdown,
                inputs=closed_set_selector,
                outputs=[closed_set_preview, closed_set_uploader, closed_set_label, closed_set_scores]
            )
            
            closed_set_uploader.change(
                fn=update_traffic_preview_from_upload,
                inputs=closed_set_uploader,
                outputs=[closed_set_preview, closed_set_selector, closed_set_label, closed_set_scores]
            )

            def handle_closed_set_predict(model, dropdown_path, uploaded_image):
                if uploaded_image is not None: image = uploaded_image
                elif dropdown_path is not None: image = Image.open(dropdown_path)
                else: return "Please select or upload an image!", None
                
                # The backend function now returns two values
                prediction, scores = run_closed_set_prediction(image, model)
                return prediction, scores
            
            closed_set_button.click(
                fn=handle_closed_set_predict,
                inputs=[model_selector, closed_set_selector, closed_set_uploader],
                outputs=[closed_set_label, closed_set_scores]
            )

        # --- TAB 2: Open-Vocabulary Explorer ---
        with gr.TabItem("Open-Vocabulary Explorer"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1. Upload an Image")
                    open_vocab_uploader = gr.Image(type="pil")
                with gr.Column():
                    gr.Markdown("### 2. Define Your Classes")
                    open_vocab_labels = gr.Textbox(
                        lines=3,
                        label="Enter Class Names (comma-separated)",
                        placeholder="e.g., Slow down, Stop, Exit, Yield"
                    )
                    with gr.Accordion("Advanced Prompting Options", open=False):
                        open_vocab_template = gr.Textbox(
                            label="Prompt Template",
                            value="a photo of a {} traffic sign",
                            info="Use {} as a placeholder for the class names above."
                        )
                    open_vocab_button = gr.Button(
                        "Classify with Custom Labels", 
                        variant="primary",
                        interactive=False
                    )
            
            gr.Markdown("--- \n ### 3. Analyze Results")
            with gr.Row():
                open_vocab_top_result = gr.Label(label="Top Prediction")
                open_vocab_scores = gr.BarPlot(label="Confidence Scores", x="Class", y="Confidence", y_lim=[0,1])

            def update_button_state(image, labels_text):
                if image is not None and labels_text and labels_text.strip():
                    return gr.Button(interactive=True)
                else:
                    return gr.Button(interactive=False)

            open_vocab_labels.input(
                fn=update_button_state,
                inputs=[open_vocab_uploader, open_vocab_labels],
                outputs=open_vocab_button
            )
            open_vocab_uploader.change(
                fn=update_button_state,
                inputs=[open_vocab_uploader, open_vocab_labels],
                outputs=open_vocab_button
            )
            open_vocab_button.click(
                fn=run_open_vocab_prediction,
                inputs=[open_vocab_uploader, model_selector, open_vocab_labels, open_vocab_template],
                outputs=[open_vocab_top_result, open_vocab_scores]
            )

# Launch the Gradio App
demo.launch(share=True)