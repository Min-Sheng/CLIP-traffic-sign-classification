
import open_clip
from transformers import CLIPProcessor, CLIPModel


class GenericCLIPProcessor:
    def __init__(self, tokenizer, transform):
        self.tokenizer = tokenizer
        self.transform = transform

    def __call__(self, text=None, images=None, *args, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text)

        if images is not None:
            pixel_values = self.transform(images)

        if text is not None and images is not None:
            encoding["pixel_values"] = pixel_values
            return encoding

        elif text is not None:
            return encoding

        else:
            return pixel_values


def load_model_and_processor(model_name: str, device: str):

    if model_name.startswith("openai/clip-"):
        backend = 'huggingface'
        # Use HuggingFace CLIP (ViT-based)
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model.to(device)
        return model, processor, backend

    else:
        backend = 'open_clip'
        # Use OpenCLIP (ResNet-based)
        print(f"Using OpenCLIP model: {model_name}")
        model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained='openai')
        tokenizer = open_clip.get_tokenizer(model_name)
        processor = GenericCLIPProcessor(tokenizer, transform)

        model.to(device)
        return model, processor, backend