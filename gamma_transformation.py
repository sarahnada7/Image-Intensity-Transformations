import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# POWER-LAW (GAMMA) TRANSFORMATION
# =============================================================================
# Formula:  s = c * r ^ gamma
#
#   r     = input pixel value  (normalized to 0-1 range)
#   s     = output pixel value (normalized to 0-1 range)
#   c     = constant (we use c = 1)
#   gamma = controls the shape of the transformation curve
#
# IMPORTANT NOTE:
#   Before applying the formula, we MUST normalize the image from [0,255] to [0,1]
#   by dividing by 255. After applying, we restore back to [0,255] by multiplying by 255.
#   This is because the power formula only works correctly in the [0,1] range.
#   Example: if r=200 and gamma=2 → (200)^2 = 40000 (out of range!)
#            but normalized: (200/255)^2 = 0.615 → * 255 = 156 (correct)
#
# EFFECT OF GAMMA:
#   gamma < 1  → curve bends UP   → dark regions get brighter (dark enhancement)
#   gamma = 1  → straight line    → no change (identity)
#   gamma > 1  → curve bends DOWN → bright regions get darker (light enhancement)
# =============================================================================

# Load image as grayscale
img = cv2.imread("vi.jpg", cv2.IMREAD_GRAYSCALE)

# =============================================================================
# GAMMA TRANSFORMATION FUNCTION
# =============================================================================
def gamma_transform(image, gamma, c=1):

    # STEP 1 — Normalize from [0, 255] to [0, 1]
    # We divide by 255.0 to get decimal values between 0 and 1
    normalized = image.astype(np.float32) / 255.0

    # STEP 2 — Apply gamma formula: s = c * r^gamma
    transformed = c * np.power(normalized, gamma)

    # STEP 3 — Restore from [0, 1] back to [0, 255]
    # Multiply by 255 and clip to make sure values stay in valid range
    restored = np.clip(transformed * 255, 0, 255).astype(np.uint8)

    return restored

# =============================================================================
# APPLY 3 DIFFERENT GAMMA VALUES
# =============================================================================

# gamma < 1 : dark regions are enhanced (image gets brighter)
gamma_low    = gamma_transform(img, gamma=0.4)

# gamma = 1 : no change at all — same as original image
gamma_one    = gamma_transform(img, gamma=1.0)

# gamma > 1 : light regions are enhanced (image gets darker)
gamma_high   = gamma_transform(img, gamma=2.5)

# =============================================================================
# DISPLAY ALL RESULTS
# =============================================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Power-Law (Gamma) Transformation  —  s = c * r^gamma  (c=1)",
             fontsize=13, fontweight="bold")

axes[0].imshow(img,        cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(gamma_low,  cmap="gray", vmin=0, vmax=255)
axes[1].set_title("gamma = 0.4\n(dark regions enhanced\nimage brighter)")
axes[1].axis("off")

axes[2].imshow(gamma_one,  cmap="gray", vmin=0, vmax=255)
axes[2].set_title("gamma = 1.0\n(linear, no change\nsame as original)")
axes[2].axis("off")

axes[3].imshow(gamma_high, cmap="gray", vmin=0, vmax=255)
axes[3].set_title("gamma = 2.5\n(light regions enhanced\nimage darker)")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("gamma_transformation.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# TRANSFORMATION CURVES (s vs r)
# =============================================================================
# Shows how each gamma value maps input r to output s.
# All curves meet at (0,0) and (1,1) — the endpoints never change.
r = np.linspace(0, 1, 256)   # input values from 0 to 1 (normalized range)

plt.figure(figsize=(7, 5))
plt.plot(r, r,            color="gray",   linewidth=1.5, linestyle="--", label="gamma = 1.0  (identity)")
plt.plot(r, r ** 0.4,     color="green",  linewidth=2.5, label="gamma = 0.5  (dark boost)")
plt.plot(r, r ** 2.5,     color="red",    linewidth=2.5, label="gamma = 4  (light boost)")

plt.xlabel("Input intensity  r  (normalized 0-1)")
plt.ylabel("Output intensity  s  (normalized 0-1)")
plt.title("Gamma Transformation Curves", fontsize=13, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gamma_curves.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved: gamma_transformation.png  and  gamma_curves.png")