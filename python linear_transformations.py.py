import cv2
import matplotlib.pyplot as plt

# =============================================================================
# BASIC LINEAR INTENSITY TRANSFORMATIONS
# =============================================================================
# Intensity transformation changes the pixel values of an image.
# The general formula is:  s = T(r)
#   r = input pixel value  (original image)
#   s = output pixel value (transformed image)
# =============================================================================

# Load the image directly as grayscale
# Each pixel has one value: 0 (black) to 255 (white)
img = cv2.imread("kate.jpg", cv2.IMREAD_GRAYSCALE)

# =============================================================================
# TRANSFORMATION 1 — Identity:  s = r
# =============================================================================
# The output equals the input — no change at all.
# Every pixel keeps its exact original value.
# This is the "do nothing" baseline transformation.
identity = img.copy()    # s = r

# =============================================================================
# TRANSFORMATION 2 — Negative:  s = 255 - r
# =============================================================================
# Inverts every pixel value:
#   - Bright pixels (e.g. r=200) become dark  (s=55)
#   - Dark  pixels (e.g. r=30)  become bright (s=225)
# This is like a photo negative.
# Useful for: revealing hidden detail in dark areas of an image.
negative = 255 - img     # s = 255 - r   (same as s = L-1-r where L=256)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Linear Intensity Transformations", fontsize=14, fontweight="bold")

# Original image
axes[0].imshow(img,      cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original\n(grayscale input)")
axes[0].axis("off")

# Identity result
axes[1].imshow(identity, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Identity\ns = r\n(no change)")
axes[1].axis("off")

# Negative result
axes[2].imshow(negative, cmap="gray", vmin=0, vmax=255)
axes[2].set_title("Negative\ns = 255 - r\n(inverted brightness)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("linear_transformations.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: linear_transformations.png")