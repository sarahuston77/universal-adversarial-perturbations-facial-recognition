import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_source = "../test_utk_dataset/1_0_0_20161219140623097.jpg.chip.jpg"
perturbation_file_source = "../universal.npy"

# Prep images
img_pil = Image.open(image_source).resize((224, 224))
img = np.array(img_pil).astype(np.float32) / 255.0

# prep perturbation vecctor/delta
delta = np.load(perturbation_file_source).astype(np.float32)
delta = delta[0]

if delta.ndim == 3 and delta.shape[0] in [1, 3]:
    delta = np.transpose(delta, (1, 2, 0))

# perturb the image
perturbed = np.clip(img + delta, 0, 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
max_abs = np.max(np.abs(delta))
vis = (delta / max_abs + 1) / 2
plt.imshow(vis)
plt.title("UAP")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(perturbed)
plt.title("Perturbed")
plt.axis("off")

plt.savefig("uap_comparison.png", bbox_inches='tight', pad_inches=0)

plt.show()