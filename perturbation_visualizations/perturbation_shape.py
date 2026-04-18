import os
import numpy as np
import matplotlib.pyplot as plt

perturbations_dir = "../perturbations"

for fname in sorted(os.listdir(perturbations_dir)):
    if not fname.endswith(".npy"):
        continue

    name = os.path.splitext(fname)[0]
    delta = np.load(os.path.join(perturbations_dir, fname)).astype(np.float32)

    # Remove batch dimension if present, so (1, 224, 224, 3) → (224, 224, 3)
    if delta.ndim == 4:
        delta = delta[0]

    # Normalize to [0, 1] for imshow: map [-max_abs, +max_abs] → [0, 1]
    max_abs = np.max(np.abs(delta))
    vis = (delta / max_abs + 1) / 2

    plt.figure()
    plt.imshow(vis)
    plt.axis("off")
    plt.title(name)

    out_path = f"uap_visual_{name}.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved {out_path}")