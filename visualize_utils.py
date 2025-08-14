import matplotlib.font_manager as fm
from matplotlib import rcParams

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
prop = fm.FontProperties(fname=font_path)
family_name = prop.get_name()

rcParams['font.family'] = family_name
rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np
import io
from torch.utils.tensorboard import SummaryWriter


class TensorboardVisualizer:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, train_loss, val_loss, val_acc, epoch):
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("Accuracy/val", val_acc, epoch)

    def log_similarity_matrix(self, similarity, texts, epoch, sampled_images=None):
        plt.figure(figsize=(20, 14))
        plt.imshow(similarity, vmin=0.1, vmax=0.3)
        plt.yticks(range(len(texts)), texts, fontsize=6)
        plt.xticks([])
        plt.title("Cosine similarity between text and image features", size=20)

        if sampled_images is not None:
            for i, images in sampled_images.items():
                image = images[0] if isinstance(images, list) else images
                plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)
        
        plt.xlim([-0.5, len(texts) - 0.5])
        plt.ylim([len(texts) + 0.5, -2])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = np.array(plt.imread(buf))
        plt.close()
        # TensorBoard expects shape (C, H, W)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        img = np.transpose(img, (2, 0, 1))
        self.writer.add_image("Cosine_similarity", img, epoch)
        buf.close()

    def close(self):
        self.writer.close()