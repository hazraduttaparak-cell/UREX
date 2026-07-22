#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import bigfish 
import bigfish.stack as stack
import bigfish.multistack as multistack
import bigfish.detection as detection
print("Big-FISH version: {0}".format(bigfish.__version__))
import matplotlib.pyplot as plt 
import numpy as np


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'widget')
import matplotlib.pyplot as plt


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


path_input = "D:/REX Paper/Image analysis/Unexpanded/1_unexp"
path_output = "D:/REX Paper/Image analysis/Unexpanded/1_unexp"


# In[ ]:


rna = stack.read_image(os.path.join(path_input, "MAX_1_unexp_RNA.tif"))
print("rna stack")
print("\r shape: {0}".format(rna.shape))
print("\r dtype: {0}".format(rna.dtype), "\n")


# In[ ]:


rna_2d = rna[2]  # Select z-slice 2
print("smfish channel (one z-slice)")
print("\r shape: {0}".format(rna_2d.shape))
print("\r dtype: {0}".format(rna_2d.dtype))


# In[ ]:


print("smfish")
print("\r min: {0}".format(rna_2d.min()))
print("\r max: {0}".format(rna_2d.max()), "\n")

rna_2d_rescaled = stack.rescale(rna_2d, channel_to_stretch=None)
print("smfish rescaled")
print("\r min: {0}".format(rna_2d_rescaled.min()))
print("\r max: {0}".format(rna_2d_rescaled.max()), "\n")

rna_2d_stretched = stack.rescale(rna_2d, channel_to_stretch=0)
print("smfish stretched")
print("\r min: {0}".format(rna_2d_stretched.min()))
print("\r max: {0}".format(rna_2d_stretched.max()))


# In[ ]:


import bigfish.plot as plot

images = [rna_2d, rna_2d_rescaled, rna_2d_stretched]
titles = ["original image", "rescaled", "stretched"]
plot.plot_images(images, titles=titles)


# In[ ]:


# 8-bit
rna_8bit = stack.cast_img_uint8(rna)
print("dtype: {0}".format(rna_8bit.dtype))
print("\r min: {0}".format(rna_8bit.min()))
print("\r max: {0}".format(rna_8bit.max()), "\n")

# 16-bit
print("dtype: {0}".format(rna.dtype))
print("\r min: {0}".format(rna.min()))
print("\r max: {0}".format(rna.max()), "\n")

# 32-bit
rna_32bit = stack.cast_img_float32(rna)
print("dtype: {0}".format(rna_32bit.dtype))
print("\r min: {0}".format(rna_32bit.min()))
print("\r max: {0}".format(rna_32bit.max()), "\n")

# 64-bit
rna_64bit = stack.cast_img_float64(rna)
print("dtype: {0}".format(rna_64bit.dtype))
print("\r min: {0}".format(rna_64bit.min()))
print("\r max: {0}".format(rna_64bit.max()))


# In[ ]:


rna_2d_mean = stack.mean_filter(rna_2d, kernel_shape="square", kernel_size=30)
rna_2d_median = stack.median_filter(rna_2d, kernel_shape="square", kernel_size=30)
rna_2d_min = stack.minimum_filter(rna_2d, kernel_shape="square", kernel_size=30)
rna_2d_max = stack.maximum_filter(rna_2d, kernel_shape="square", kernel_size=30)


# In[ ]:


rna_2d_gaussian = stack.gaussian_filter(rna_2d, sigma=5)


# In[ ]:


images = [rna_2d, rna_2d_mean, rna_2d_median, rna_2d_min, rna_2d_max, rna_2d_gaussian]
titles = ["original image", "mean filter", "median filter", "minimum filter", "maximum filter", "gaussian filter"]
plot.plot_images(images, rescale=True, titles=titles)


# In[ ]:


rna_bool = rna_2d > 90
rna_dilated = stack.dilation_filter(rna_bool, kernel_shape="square", kernel_size=30)
rna_eroded = stack.erosion_filter(rna_bool, kernel_shape="square", kernel_size=30)


# In[ ]:


images = [rna_bool, rna_dilated, rna_eroded]
titles = ["masked image", "binary dilation", "binary erosion"]
plot.plot_images(images, rescale=True, titles=titles)


# In[ ]:


images = [rna_2d]
titles = ["gaussian filter"]
plot.plot_images(images, rescale=True, titles=titles)


# In[ ]:


rna_log = stack.log_filter(rna_2d, sigma=3)
rna_background_mean = stack.remove_background_mean(rna_2d, kernel_shape="square", kernel_size=31)
rna_background_gaussian = stack.remove_background_gaussian(rna_2d, sigma=3)


# In[ ]:


images = [rna_log, rna_background_gaussian, rna_background_mean]
titles = ["LoG filter", "remove gaussian background", "remove mean background"]
plot.plot_images(images, contrast=True, titles=titles)


# In[ ]:


rna_mip = stack.maximum_projection(rna)


# In[ ]:


rna_blurred = rna.copy()
rna_blurred[-10:, ...] = stack.gaussian_filter(rna_blurred[-10:, ...], sigma=5) * 1.5
rna_blurred_mip = stack.maximum_projection(rna_blurred)


# In[ ]:


images = [rna_mip, rna_blurred_mip]
titles = ["FISH channel (MIP)", "blurred FISH channel (MIP)"]
plot.plot_images(images, titles=titles, framesize=(10, 5), contrast=True)


# In[ ]:


focus = stack.compute_focus(rna, neighborhood_size=31)
focus_blurred = stack.compute_focus(rna_blurred, neighborhood_size=31)
measures = [focus.mean(axis=(1, 2)), focus_blurred.mean(axis=(1, 2))]
plot.plot_sharpness(measures, labels=["original", "blurred"], 
                    title="Sharpness measurae over smFISH channel")


# In[ ]:


nb_to_keep = rna.shape[0] - 2
z_indices_to_keep = stack.get_in_focus_indices(focus, proportion=nb_to_keep)
print("In-focus z-slices in the original image:", z_indices_to_keep)
z_indices_to_keep_blurred = stack.get_in_focus_indices(focus_blurred, proportion=nb_to_keep)
print("In-focus z-slices in the blurred image:", z_indices_to_keep_blurred)


# In[ ]:


in_focus_image = stack.in_focus_selection(rna, focus, proportion=nb_to_keep)
in_focus_image_mip = stack.maximum_projection(in_focus_image)


# In[ ]:


in_focus_image_blurred = stack.in_focus_selection(rna_blurred, focus_blurred, proportion=nb_to_keep)
in_focus_image_blurred_mip = stack.maximum_projection(in_focus_image_blurred)


# In[ ]:


images = [rna_mip, in_focus_image_mip]
titles = ["FISH channel (MIP)", "FISH channel (in-focus selection + MIP)"]
plot.plot_images(images, titles=titles, framesize=(10, 5), contrast=True)


# In[ ]:


#Spot detection for expanded and unexpanded individual agglomerates

# without tight clustering and sensitive pass and reduction of threshold




import os
import numpy as np
import bigfish.stack as stack
import bigfish.detection as detection
import pandas as pd
import matplotlib.pyplot as plt

# =============== USER PARAMETERS ===============
# Provide your image here (already loaded)
# rna = stack.read_image("your_image_path.tif")  # Uncomment if needed

output_dir = r"D:/REX Paper/Image analysis/Expanded/Sections/5_exp/Individual agglomerates n=5/53_exp"
voxel_size = (800, 103, 103)  # (z, y, x) in nanometers

# Bumped spot radius for better detection
spot_radius = (810, 113, 113)  # nm

os.makedirs(output_dir, exist_ok=True)

# ===================== SPOT DETECTION PIPELINE =====================

# 1. Auto-threshold detection (raw image)
spots, thr = detection.detect_spots(
    images=rna,
    voxel_size=voxel_size,
    spot_radius=spot_radius,
    return_threshold=True
)
print(f"Auto-threshold detected spots: {spots.shape[0]}, threshold={thr}")


# 5. Decompose Dense Regions
spots_decomposed = detection.decompose_dense(
    image=rna,
    spots=spots,
    voxel_size=voxel_size,
    spot_radius=spot_radius,
    alpha=0.7, beta=1, gamma=5
)[0]
print(f"Spots after decomposition: {spots_decomposed.shape[0]}")

# 6. Subpixel Gaussian Fitting
spots_subpixel = detection.fit_subpixel(
    image=rna,
    spots=spots_decomposed,
    voxel_size=voxel_size,
    spot_radius=spot_radius
)
print(f"Spots after subpixel fitting: {spots_subpixel.shape[0]}")

# 7. Calculate mean intensity (3x3x3 neighborhood)
Intensity = np.zeros(spots_subpixel.shape[0])
for i, (z, y, x) in enumerate(spots_subpixel[:, :3]):
    z, y, x = int(round(z)), int(round(y)), int(round(x))
    z_slice = slice(max(0, z-1), min(rna.shape[0], z+2))
    y_slice = slice(max(0, y-1), min(rna.shape[1], y+2))
    x_slice = slice(max(0, x-1), min(rna.shape[2], x+2))
    neighborhood = rna[z_slice, y_slice, x_slice]
    Intensity[i] = neighborhood.mean()

# ===================== EXPORT RESULTS =====================
# Spot positions
spot_df = pd.DataFrame(spots_subpixel[:, :3], columns=['z', 'y', 'x'])
positions_path = os.path.join(output_dir, "spots_positions_subpixel.csv")
spot_df.to_csv(positions_path, index=False)
print(f"Spot positions saved to: {positions_path}")

# Mean intensities
intensity_df = pd.DataFrame({'mean_intensity': Intensity})
intensity_path = os.path.join(output_dir, "spot_mean_intensities_subpixel.csv")
intensity_df.to_csv(intensity_path, index=False)
print(f"Mean intensities saved to: {intensity_path}")

# ===================== PLOT HISTOGRAM =====================
plt.figure(figsize=(10, 6))
plt.hist(Intensity, bins=50, color='teal', alpha=0.7, edgecolor='black')
plt.title('Distribution of Spot Mean Intensities (Subpixel Fitting)', fontsize=14)
plt.xlabel('Mean Intensity (3x3x3 neighborhood)', fontsize=12)
plt.ylabel('Number of Spots', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)
median = np.median(Intensity)
plt.axvline(median, color='red', linestyle='--', linewidth=1.5, label=f'Median: {median:.1f}')
plt.legend()
plot_path = os.path.join(output_dir, "intensity_distribution_subpixel.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Histogram saved to: {plot_path}")

# ===================== OPTIONAL: PRINT EXAMPLES =====================
print("\nFirst 5 spot positions (subpixel):\n", spots_subpixel[:5, :3])
print("First 5 mean intensities:\n", Intensity[:5])


# In[ ]:


print(spots_subpixel[:5, :3])  # After subpixel fitting (should be floats)


# In[ ]:


# --- Generate the two-panel plot ---
plot.plot_detection(rna_mip, spots_subpixel, contrast=True)

