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


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'qt')


# In[ ]:


from matplotlib import pyplot as plt


# In[ ]:


path_input = r"D:/REX Paper/Image analysis/Unexpanded/8_unexp"
path_output = r"D:/REX Paper/Image analysis/Unexpanded/8_unexp"


# In[ ]:


rna = stack.read_image(os.path.join(path_input, "8_unexp_RNA_MIP.tif"))
print("rna stack")
print("\r shape: {0}".format(rna.shape))
print("\r dtype: {0}".format(rna.dtype), "\n")


# In[ ]:


print("smfish")
print("\r min: {0}".format(rna.min()))
print("\r max: {0}".format(rna.max()), "\n")

rna_rescaled = stack.rescale(rna, channel_to_stretch=None)
print("smfish rescaled")
print("\r min: {0}".format(rna_rescaled.min()))
print("\r max: {0}".format(rna_rescaled.max()), "\n")

rna_stretched = stack.rescale(rna, channel_to_stretch=0)
print("smfish stretched")
print("\r min: {0}".format(rna_stretched.min()))
print("\r max: {0}".format(rna_stretched.max()))


# In[ ]:


import bigfish.plot as plot

images = [rna, rna_rescaled, rna_stretched]
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


rna_2d_mean = stack.mean_filter(rna, kernel_shape="square", kernel_size=30)
rna_2d_median = stack.median_filter(rna, kernel_shape="square", kernel_size=30)
rna_2d_min = stack.minimum_filter(rna, kernel_shape="square", kernel_size=30)
rna_2d_max = stack.maximum_filter(rna, kernel_shape="square", kernel_size=30)


# In[ ]:


rna_2d_gaussian = stack.gaussian_filter(rna, sigma=5)


# In[ ]:


images = [rna, rna_2d_mean, rna_2d_median, rna_2d_min, rna_2d_max, rna_2d_gaussian]
titles = ["original image", "mean filter", "median filter", "minimum filter", "maximum filter", "gaussian filter"]
plot.plot_images(images, rescale=True, titles=titles)


# In[ ]:


rna_bool = rna > 90
rna_dilated = stack.dilation_filter(rna_bool, kernel_shape="square", kernel_size=30)
rna_eroded = stack.erosion_filter(rna_bool, kernel_shape="square", kernel_size=30)


# In[ ]:


images = [rna_bool, rna_dilated, rna_eroded]
titles = ["masked image", "binary dilation", "binary erosion"]
plot.plot_images(images, rescale=True, titles=titles)


# In[ ]:


images = [rna]
titles = ["gaussian filter"]
plot.plot_images(images, rescale=True, titles=titles)


# In[ ]:


rna_log = stack.log_filter(rna, sigma=3)
rna_background_mean = stack.remove_background_mean(rna, kernel_shape="square", kernel_size=31)
rna_background_gaussian = stack.remove_background_gaussian(rna, sigma=3)


# In[ ]:


images = [rna_log, rna_background_gaussian, rna_background_mean]
titles = ["LoG filter", "remove gaussian background", "remove mean background"]
plot.plot_images(images, contrast=True, titles=titles)


# In[ ]:


rna_blurred = rna.copy()
rna_blurred[-10:, ...] = stack.gaussian_filter(rna_blurred[-10:, ...], sigma=5) * 1.5


# In[ ]:


images = [rna_blurred]
titles = ["FISH channel (MIP)", "blurred FISH channel (MIP)"]
plot.plot_images(images, titles=titles, framesize=(10, 5), contrast=True)


# In[ ]:


import os
import numpy as np
import bigfish.stack as stack
import bigfish.detection as detection
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============== USER PARAMETERS ===============
# Example: load your 2D image (uncomment if needed)
# rna = stack.read_image("your_image_path.tif")  # should be (Y, X)

output_dir = r"D:/REX Paper/Image analysis/Unexpanded/8_unexp"
voxel_size = (103, 103)  # (y, x) in nanometers for 2D
spot_radius = (113, 113)  # nm, approximate Gaussian kernel size for 2D

os.makedirs(output_dir, exist_ok=True)

# ===================== SPOT DETECTION PIPELINE =====================

# 1. Auto-threshold detection (raw image)
spots_auto, thr = detection.detect_spots(
    images=rna,
    voxel_size=voxel_size,
    spot_radius=spot_radius,
    return_threshold=True
)
print(f"Auto-threshold detected spots: {spots_auto.shape[0]}, threshold={thr}")

# 2. More sensitive pass (same threshold)
spots_sens, _ = detection.detect_spots(
    images=rna,
    voxel_size=voxel_size,
    spot_radius=spot_radius,
    threshold=thr * 0.5,
    return_threshold=True
)
print(f"Sensitive pass detected spots: {spots_sens.shape[0]}")

# 3. Tight clustering (nearby spots)
log_kernel_size = tuple(r / v for r, v in zip(spot_radius, voxel_size))
spots_tight, _ = detection.detect_spots(
    images=rna,
    log_kernel_size=log_kernel_size,
    minimum_distance=(3, 3),
    threshold=thr * 0.5,
    return_threshold=True
)
print(f"Tight clustering detected spots: {spots_tight.shape[0]}")

# 4. Combine multi-scale / multi-pass detections
all_spots = np.vstack([spots_auto, spots_sens, spots_tight])
all_spots_int = np.round(all_spots[:, :2]).astype(int)
_, uniq_idx = np.unique(all_spots_int, axis=0, return_index=True)
spots_union = all_spots[np.sort(uniq_idx)]
print(f"Total spots after union: {spots_union.shape[0]}")

# ===================== SUBPIXEL FITTING =====================
spots_subpixel_list = []
mini_batch_size = 100

for i in tqdm(range(0, spots_union.shape[0], mini_batch_size),
              desc="Subpixel fitting (mini-batches)", unit="mini-batch"):
    batch = spots_union[i:i + mini_batch_size]
    try:
        sub_mini = detection.fit_subpixel(
            image=rna,
            spots=batch,
            voxel_size=voxel_size,
            spot_radius=spot_radius
        )
        sub_mini = sub_mini[~np.isnan(sub_mini).any(axis=1)]
        if sub_mini.size > 0:
            spots_subpixel_list.append(sub_mini)
    except Exception:
        recovered = []
        for spot in batch:
            try:
                sub = detection.fit_subpixel(
                    image=rna,
                    spots=np.array([spot]),
                    voxel_size=voxel_size,
                    spot_radius=spot_radius
                )
                if not np.isnan(sub).any():
                    recovered.append(sub[0])
            except Exception:
                continue
        if recovered:
            spots_subpixel_list.append(np.array(recovered))

if spots_subpixel_list:
    spots_subpixel = np.vstack(spots_subpixel_list)
else:
    spots_subpixel = np.empty((0, 2))

print(f"Spots after subpixel fitting: {spots_subpixel.shape[0]}")

# ===================== INTENSITY CALCULATION =====================
Intensity = np.zeros(spots_subpixel.shape[0])
for i, (y, x) in enumerate(spots_subpixel[:, :2]):
    y, x = int(round(y)), int(round(x))
    y_slice = slice(max(0, y - 1), min(rna.shape[0], y + 2))
    x_slice = slice(max(0, x - 1), min(rna.shape[1], x + 2))
    neighborhood = rna[y_slice, x_slice]
    Intensity[i] = neighborhood.mean()

# ===================== EXPORT RESULTS =====================
spot_df = pd.DataFrame(spots_subpixel[:, :2], columns=['y', 'x'])
positions_path = os.path.join(output_dir, "spots_positions_subpixel_2D.csv")
spot_df.to_csv(positions_path, index=False)
print(f"Spot positions saved to: {positions_path}")

intensity_df = pd.DataFrame({'mean_intensity': Intensity})
intensity_path = os.path.join(output_dir, "spot_mean_intensities_subpixel_2D.csv")
intensity_df.to_csv(intensity_path, index=False)
print(f"Mean intensities saved to: {intensity_path}")

# ===================== PLOT HISTOGRAM =====================
plt.figure(figsize=(10, 6))
plt.hist(Intensity, bins=50, color='teal', alpha=0.7, edgecolor='black')
plt.title('Distribution of Spot Mean Intensities (2D Subpixel Fitting)', fontsize=14)
plt.xlabel('Mean Intensity (3x3 neighborhood)', fontsize=12)
plt.ylabel('Number of Spots', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)
median = np.median(Intensity)
plt.axvline(median, color='red', linestyle='--', linewidth=1.5, label=f'Median: {median:.1f}')
plt.legend()
plot_path = os.path.join(output_dir, "intensity_distribution_subpixel_2D.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Histogram saved to: {plot_path}")

# ===================== OPTIONAL: PRINT EXAMPLES =====================
print("\nFirst 5 spot positions (subpixel):\n", spots_subpixel[:5, :2])
print("First 5 mean intensities:\n", Intensity[:5])


# In[ ]:


# --- Generate the two-panel plot ---
plot.plot_detection(rna, spots_subpixel, contrast=True)


# In[ ]:


import os
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.filters import threshold_local

def measure_spot_shapes_adaptive_2d(
    rna, 
    spots_subpixel, 
    output_dir, 
    voxel_size=(103, 103),   # (y, x) in nanometers for 2D
    local_window=7,
    block_size=7,
    offset=0,
    method='mean',
): 

    """
    Measure area and axes of spots in a 2D image using adaptive/local thresholding.
    All measurements are reported in nanometer units; no expansion factor normalization is applied.
    """

    os.makedirs(output_dir, exist_ok=True)
    results = []
    half_win = local_window // 2
    y_dim, x_dim = rna.shape

    print(f"Measuring shapes for {spots_subpixel.shape[0]} spots using adaptive thresholding (2D)...")

    for idx, (y, x) in enumerate(spots_subpixel[:, :2]):
        y, x = int(round(y)), int(round(x))

        # Skip invalid coordinates
        if y < 0 or y >= y_dim or x < 0 or x >= x_dim:
            print(f"Skipping spot {idx} due to invalid coordinates (y={y}, x={x})")
            continue

        # Extract local window around the spot
        y_start = max(0, y - half_win)
        y_end = min(y + half_win + 1, y_dim)
        x_start = max(0, x - half_win)
        x_end = min(x + half_win + 1, x_dim)

        local_img = rna[y_start:y_end, x_start:x_end]

        if local_img.size == 0:
            print(f"Skipping spot {idx} due to empty local image window.")
            continue

        # Adaptive threshold
        try:
            bs = min(block_size, min(local_img.shape))
            if bs % 2 == 0:
                bs -= 1
            if bs < 3:
                bs = 3
            local_thresh = threshold_local(local_img, block_size=bs, method=method, offset=offset)
            mask = local_img > local_thresh
        except Exception as e:
            print(f"Adaptive thresholding failed for spot {idx}: {e}")
            continue

        # Label regions and extract properties
        labeled = label(mask)
        props = regionprops(labeled)

        if not props:
            print(f"No region found for spot {idx} at (y={y}, x={x})")
            continue

        # Pick the region closest to the spot center
        center = np.array([y - y_start, x - x_start])
        min_dist = float('inf')
        best_prop = None
        for prop in props:
            dist = np.linalg.norm(np.array(prop.centroid) - center)
            if dist < min_dist:
                min_dist = dist
                best_prop = prop

        # Measure area and axes (in nanometers)
        area_px = best_prop.area
        area_nm2 = area_px * voxel_size[0] * voxel_size[1]
        major_axis_nm = best_prop.major_axis_length * voxel_size[0]
        minor_axis_nm = best_prop.minor_axis_length * voxel_size[1]

        results.append({
            'y': y,
            'x': x,
            'area_nm2': area_nm2,
            'major_axis_nm': round(major_axis_nm, 3),
            'minor_axis_nm': round(minor_axis_nm, 3),
        })

    # Save results
    if not results:
        print("No valid spots found for measurement.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "spot_shape_measurements_adaptive_2D.csv")
    df.to_csv(csv_path, index=False)
    print(f"Spot shape measurements saved to: {csv_path}")

    return df


# ==========================
# Example usage
# ==========================
if __name__ == "__main__":
    # rna = ... (2D numpy array)
    # spots_subpixel = ... (Nx2 numpy array from Big-FISH subpixel fitting)

    output_dir = r"D:/REX Paper/Image analysis/Unexpanded/8_unexp"
    voxel_size = (103, 103)  # (y, x) in nm

    df_shapes = measure_spot_shapes_adaptive_2d(
        rna=rna,
        spots_subpixel=spots_subpixel,
        output_dir=output_dir,
        voxel_size=voxel_size,
        local_window=7,
        block_size=7,
        offset=0,
        method='mean',  # or 'gaussian'
    )

    if not df_shapes.empty:
        print("\nSpot shape measurement summary (adaptive thresholding 2D):")
        print(f"Total spots measured: {len(df_shapes)}")
        print(f"Median area (nm²): {df_shapes['area_nm2'].median():.2f}")
        print(f"Median major axis (nm): {df_shapes['major_axis_nm'].median():.2f}")
        print(f"Median minor axis (nm): {df_shapes['minor_axis_nm'].median():.2f}")
    else:
        print("No spot measurements were produced.")


# In[ ]:


import numpy as np
import os
import pandas as pd

def save_spot_data_csv_ordered_clean_2D(
    spots_subpixel, 
    Intensity, 
    output_dir, 
    decimals=2
):
    """
    Save 2D spot data (y, x) with intensity and computed copy numbers
    into a clean, comma-separated CSV file.
    Works for 2D smFISH or similar microscopy data.
    """
    # --- Convert inputs to arrays ---
    spots = np.asarray(spots_subpixel)
    intensity = np.asarray(Intensity)

    # --- Check for valid data ---
    if len(spots) != len(intensity):
        min_length = min(len(spots), len(intensity))
        print(f"⚠️ Mismatched array lengths. Truncating to {min_length} spots.")
        spots = spots[:min_length]
        intensity = intensity[:min_length]

    if len(intensity) == 0:
        raise ValueError("❌ No spots detected — intensity array is empty.")

    # --- Handle NaN or negative values ---
    intensity = np.nan_to_num(intensity, nan=0.0)
    intensity[intensity < 0] = 0.0

    # --- Compute copy numbers ---
    nonzero_intensity = intensity[intensity > 0]
    if len(nonzero_intensity) == 0:
        print("⚠️ All intensities are zero — assigning uniform copy number = 1")
        copy_numbers = np.ones_like(intensity)
    else:
        lowest_intensity = np.min(nonzero_intensity)
        safe_intensity = np.where(intensity == 0, lowest_intensity * 0.5, intensity)
        copy_numbers = safe_intensity / lowest_intensity

    # --- Assign ranks and tags ---
    order = np.argsort(copy_numbers)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(copy_numbers) + 1)
    copy_number_tags = [f"Copy Number {rank}" for rank in ranks]

    # --- Prepare DataFrame for saving ---
    df = pd.DataFrame({
        'y': np.round(spots[:, 0], decimals),
        'x': np.round(spots[:, 1], decimals),
        'intensity': np.round(intensity, decimals),
        'copy_number': np.round(copy_numbers, decimals),
        'copy_number_tag': copy_number_tags
    })

    # --- Sort for readability ---
    df['rank'] = ranks
    df = df.sort_values('rank').drop('rank', axis=1)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "spots_data_with_copy_numbers_2D.csv")
    df.to_csv(output_file, index=False)

    print(f"✅ Saved {len(df)} spots to:\n{output_file}")
    print("\nExample rows:")
    print(df.head(5))

    return df


# ==============================================================
# Example usage (for your current workflow)
# ==============================================================

# Compute mean intensities of detected spots from a 2D image
Intensity = np.zeros(spots_subpixel.shape[0])
for i, (y, x) in enumerate(spots_subpixel[:, :2]):
    y, x = int(round(y)), int(round(x))
    y_slice = slice(max(0, y - 1), min(rna.shape[0], y + 2))
    x_slice = slice(max(0, x - 1), min(rna.shape[1], x + 2))
    neighborhood = rna[y_slice, x_slice]
    Intensity[i] = neighborhood.mean()

# Save everything
output_dir = r"D:/REX Paper/Image analysis/Unexpanded/8_unexp"
df = save_spot_data_csv_ordered_clean_2D(spots_subpixel, Intensity, output_dir)


# In[ ]:


# --- Imports ---
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from skimage import io, exposure
from matplotlib.widgets import Slider

# --- Enable interactive mode ---
plt.ion()

# --- Load your data ---
df = pd.read_csv(r"D:/REX Paper/Image analysis/Unexpanded/5_unexp/spots_data_with_copy_numbers_2D.csv")

# --- Check required columns ---
required = {"x", "y", "copy_number"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["copy_number"] = pd.to_numeric(df["copy_number"], errors="coerce")

# --- Categorize copy numbers ---
def categorize(c):
    if pd.isna(c) or c == 0:
        return "NA"
    elif c >= 51:
        return "≥51 mRNAs"
    elif 21 <= c <= 50:
        return "21-50 mRNAs"
    elif 6 <= c <= 20:
        return "6-20 mRNAs"
    else:
        return "1-5 mRNAs"

df["category"] = df["copy_number"].apply(categorize)

# --- Define colors ---
color_list = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]
colors = {
    "1-5 mRNAs": color_list[0],
    "6-20 mRNAs": color_list[1],
    "21-50 mRNAs": color_list[2],
    "≥51 mRNAs": color_list[3],
}

point_size = 20
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", color_list)
norm = mcolors.Normalize(vmin=df["copy_number"].min(), vmax=df["copy_number"].max())

# --- Load and process the raw image ---
image_path = r"D:/REX Paper/Image analysis/Unexpanded/5_unexp/5_unexp_RNA_MIP_rep.tif"
img = io.imread(image_path).astype(float)
img_norm = exposure.rescale_intensity(img, in_range='image', out_range=(0, 1))
img_inverted = 1 - img_norm  # white signals on black background
img_height, img_width = img_inverted.shape

# --- Create figure with 3 panels ---
fig, (ax_overlay, ax_spots, ax_dist) = plt.subplots(
    3, 1, figsize=(7, 13), sharex=False,
    gridspec_kw={'height_ratios': [1.2, 1.2, 1.5]}
)

# ===============================================================
# 1️⃣ Panel 1: Raw image + spots overlay
# ===============================================================
overlay_img = ax_overlay.imshow(img_inverted, cmap='gray', origin='upper', vmin=0, vmax=1)
overlay_scatter = ax_overlay.scatter(
    df["x"], df["y"], s=20,
    c=df["copy_number"], cmap=cmap, norm=norm,
    alpha=0.6, edgecolors='none'
)
ax_overlay.set_xlim(0, img_width)
ax_overlay.set_ylim(img_height, 0)
ax_overlay.set_aspect(img_height / img_width)
ax_overlay.set_title("Raw Image with RNA Spots Overlay (White on Black)", pad=8)
ax_overlay.set_ylabel("Y")

# ===============================================================
# 2️⃣ Panel 2: Spots-only panel
# ===============================================================
ax_spots.scatter(
    df["x"], df["y"], s=20,
    c=df["copy_number"], cmap=cmap, norm=norm,
    alpha=0.6, edgecolors='none'
)
ax_spots.set_xlim(0, img_width)
ax_spots.set_ylim(img_height, 0)
ax_spots.set_aspect(img_height / img_width)
ax_spots.set_title("RNA Spots Only (Spatial Distribution)", pad=8)
ax_spots.set_ylabel("Y")
ax_spots.set_facecolor("#F9F9F9")
ax_spots.grid(alpha=0.3)

# ===============================================================
# 3️⃣ Panel 3: Copy number variation along X-axis
# ===============================================================
ax_dist.scatter(df["x"], df["copy_number"], s=point_size,
                c=df["copy_number"], cmap=cmap, norm=norm,
                alpha=0.8, edgecolors='none')
ax_dist.set_xlabel("X")
ax_dist.set_ylabel("Copy Number")
ax_dist.set_title("Copy Number Variation along X-axis", pad=8)
ax_dist.grid(alpha=0.3)
ax_dist.set_facecolor("#F2F2F2")
ax_dist.set_xlim(0, img_width)

# ===============================================================
# 🎨 Legend & Colorbar
# ===============================================================
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=cat,
           markerfacecolor=col, markersize=9)
    for cat, col in colors.items()
]
ax_spots.legend(handles=legend_elements, title="Copy Number Range",
                frameon=True, facecolor="white", loc="center left",
                bbox_to_anchor=(1.12, 0.5))

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax_overlay, ax_spots, ax_dist], fraction=0.025, pad=0.15)
cbar.set_label("Copy Number Intensity")

# Move colorbar upward
pos = cbar.ax.get_position()
cbar.ax.set_position([pos.x0, pos.y0 + 0.35, pos.width, pos.height * 0.55])

plt.tight_layout(rect=[0, 0, 0.8, 1])

# ===============================================================
# 🔄 Synced Zoom (debounced)
# ===============================================================
zoom_lock = False
zoom_sync_axes = [ax_overlay, ax_spots]

def on_zoom(ax):
    global zoom_lock
    if zoom_lock:
        return
    if ax in zoom_sync_axes:
        zoom_lock = True
        try:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            for other_ax in zoom_sync_axes:
                if other_ax != ax:
                    other_ax.set_xlim(xlim)
                    other_ax.set_ylim(ylim)
            fig.canvas.draw_idle()
        finally:
            zoom_lock = False

for ax in zoom_sync_axes:
    ax.callbacks.connect('xlim_changed', on_zoom)
    ax.callbacks.connect('ylim_changed', on_zoom)

# ===============================================================
# 🧭 Dynamic Range Sliders (vmin / vmax)
# ===============================================================
init_vmin = 0.0
init_vmax = 1.0

ax_vmin = plt.axes([0.15, 0.08, 0.65, 0.03])
ax_vmax = plt.axes([0.15, 0.04, 0.65, 0.03])

slider_vmin = Slider(ax_vmin, 'Min Intensity', 0, 0.99, valinit=init_vmin)
slider_vmax = Slider(ax_vmax, 'Max Intensity', 0.01, 1.0, valinit=init_vmax)

def update_dynamic_range(val):
    vmin = slider_vmin.val
    vmax = slider_vmax.val
    if vmin >= vmax:
        return
    overlay_img.set_clim(vmin, vmax)
    fig.canvas.draw_idle()

slider_vmin.on_changed(update_dynamic_range)
slider_vmax.on_changed(update_dynamic_range)

# ===============================================================
# 💾 Save & Keyboard Shortcuts
# ===============================================================
save_dir = r"D:/REX Paper/Image analysis/Unexpanded/5_unexp"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "spots_with_image_and_distribution.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

zoom_counter = 0

def on_key(event):
    global zoom_counter
    if event.key == 's':
        zoom_counter += 1
        zoom_save_path = os.path.join(save_dir, f"spots_zoomed_{zoom_counter}.png")
        plt.savefig(zoom_save_path, dpi=300, bbox_inches="tight")
        print(f"💾 Zoomed region {zoom_counter} saved: {zoom_save_path}")
    elif event.key == 'r':
        for ax in zoom_sync_axes:
            ax.set_xlim(0, img_width)
            ax.set_ylim(img_height, 0)
        fig.canvas.draw_idle()
        print("🔄 Zoom reset.")

fig.canvas.mpl_connect('key_press_event', on_key)

print("\n🟢 Instructions:")
print(" - Use the toolbar’s magnifying glass 🧐 to zoom or the hand ✋ to pan.")
print(" - Press 's' to save current zoomed view.")
print(" - Press 'r' to reset zoom.")
print(" - Adjust dynamic range (vmin/vmax) with sliders below the figure.\n")

plt.show(block=True)
print(f"✅ Main figure saved: {save_path}")




# modified by Jeet
# ===============================================================
# 📊 Histogram of Copy Number Categories (1–5, 6–20, 21–50, ≥51)
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Define categories ---
def categorize_copy_number(n):
    if n <= 5:
        return "1–5"
    elif 6 <= n <= 20:
        return "6–20"
    elif 21 <= n <= 50:
        return "21–50"
    else:
        return "≥51"

df["copy_category"] = df["copy_number"].apply(categorize_copy_number)

# --- Count spots in each category ---
category_counts = df["copy_category"].value_counts().reindex(["1–5","6–20","21–50","≥51"], fill_value=0)

# --- Plot ---
fig_cat, ax_cat = plt.subplots(figsize=(7, 5))

bars = ax_cat.bar(category_counts.index, category_counts.values, edgecolor='black')

ax_cat.set_xlabel("Copy Number Category", fontsize=12)
ax_cat.set_ylabel("Number of Spots", fontsize=12)
ax_cat.set_title("RNA Spot Counts by Copy Number Category", fontsize=14)
ax_cat.grid(axis='y', alpha=0.3)

# --- Annotate each bar with value ---
for bar in bars:
    height = bar.get_height()
    ax_cat.text(bar.get_x() + bar.get_width()/2, height + 1,
                f"{int(height)}", ha='center', va='bottom')

# --- Save figure ---
save_path = os.path.join(save_dir, "RNA_spot_copy_category_histogram.png")
fig_cat.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"📁 Category histogram saved to: {save_path}")
