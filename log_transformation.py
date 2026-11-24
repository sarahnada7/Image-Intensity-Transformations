import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# LOGARITHMIC INTENSITY TRANSFORMATION
# =============================================================================
# Formula:  s = c * log10(1 + r)
#
#   r = input pixel value  (0 to 255)
#   s = output pixel value (0 to 255)
#   c = scaling constant   = 255 / log10(256)
#
# WHY log10(1 + r) and not just log10(r)?
#   Because log10(0) = -infinity, which breaks the math.
#   Adding 1 ensures the minimum input is log10(1) = 0  (safe).
#
# WHAT does it do?
#   - Dark pixels (low r)   get pushed UP significantly
#   - Bright pixels (high r) get compressed (pushed up less)
#   - Result: dark details become visible, image looks brighter overall
#
# WHY c = 255 / log10(256)?
#   We want the maximum input (r=255) to map to exactly 255:
#   s = c * log10(1 + 255) = c * log10(256) = 255
#   So: c = 255 / log10(256)
# =============================================================================

# Load image as grayscale
img = cv2.imread("victor.jpg", cv2.IMREAD_GRAYSCALE)

# =============================================================================
# STEP 1 — Compute the constant c
# =============================================================================
# c ensures the output stays in the 0-255 range
c = 255 / np.log10(256)
print(f"Constant c = {c:.4f}")

# =============================================================================
# STEP 2 — Apply the log transformation
# =============================================================================
# Convert to float32 first so the math works without integer overflow
# Then apply: s = c * log10(1 + r)
log_img = c * np.log10(1 + img.astype(np.float32))

# Clip to [0, 255] just in case of floating point errors, then convert back to uint8
log_img = np.clip(log_img, 0, 255).astype(np.uint8)

# =============================================================================
# STEP 3 — Display original vs log transformed
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Logarithmic Transformation  —  s = c * log10(1 + r)", 
             fontsize=13, fontweight="bold")

axes[0].imshow(img,     cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original\n(dark image)")
axes[0].axis("off")

axes[1].imshow(log_img, cmap="gray", vmin=0, vmax=255)
axes[1].set_title(f"Log Transformed\ns = {c:.1f} * log10(1 + r)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("log_transformation.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# STEP 4 — Show the transformation curve (s vs r)
# =============================================================================
# This curve shows HOW the formula maps each input r to output s.
# The curve bends upward → low values get boosted more than high values.
r = np.arange(0, 256)
s = c * np.log10(1 + r)

plt.figure(figsize=(7, 5))
plt.plot(r, r, color="gray",  linewidth=1.5, linestyle="--", label="Identity s = r (reference)")
plt.plot(r, s, color="blue",  linewidth=2.5, label="Log  s = c * log10(1 + r)")
plt.xlabel("Input intensity  r  (0-255)")
plt.ylabel("Output intensity  s  (0-255)")
plt.title("Log Transformation Curve", fontsize=13, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("log_curve.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved: log_transformation.png  and  log_curve.png")