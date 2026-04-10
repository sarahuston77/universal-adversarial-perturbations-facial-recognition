import numpy as np
import matplotlib.pyplot as plt

# load perturbation
delta = np.load("../universal.npy").astype(np.float32)
# Remove batch dimensions, so (1, 224, 224, 3) → (224, 224, 3)
delta = delta[0]

# Find largest mangitude value in the perturbation, this will be used to normalize values consistently
# Example: delta range: [-0.039, +0.038]→ max_abs = 0.039
max_abs = np.max(np.abs(delta))
# delta / max_abs -> Normalize all values to a value between [-1, 1]
# vis = (vis + 1) / 3, shift to [0, 1]. So [-1, 1] → [0, 1]
# This is because matplotlib.imshow() expects values in [0,1] for floats
vis = (delta / max_abs + 1) / 2

# Treat vis as a RGB image where each pixel is now a value in the interval [0,1] from the shift above
plt.imshow(vis)
plt.axis("off")

plt.savefig("uap_visual.png", bbox_inches='tight', pad_inches=0)
plt.show()