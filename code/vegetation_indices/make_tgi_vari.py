import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# load image
image_path = 'ortho.png'
rgb_image = plt.imread(image_path)

if rgb_image.dtype == np.uint8:
    rgb_image = rgb_image.astype(np.float32) / 255.0

height, width, _ = rgb_image.shape
print(f"Image loaded: {height}x{width}")

# compute vegetation indices
def compute_vis_on_chunk(chunk):
    R = chunk[:, :, 0]
    G = chunk[:, :, 1]
    B = chunk[:, :, 2]

    # vari
    denom_vari = G + R - B
    denom_vari[denom_vari == 0] = 1e-6
    vari = (G - R) / denom_vari

    # tgi
    max_rgb = np.maximum(np.maximum(R, G), B)
    max_rgb[max_rgb == 0] = 1e-6
    tgi = (G - 0.39 * R - 0.61 * B) / max_rgb

    return vari, tgi

# process in chunks
num_chunks = 4
chunk_width = width // num_chunks
vari_full = np.zeros((height, width), dtype=np.float32)
tgi_full = np.zeros((height, width), dtype=np.float32)

for i in range(num_chunks):
    start_col = i * chunk_width
    end_col = width if i == num_chunks - 1 else (i + 1) * chunk_width
    chunk = rgb_image[:, start_col:end_col, :]
    vari_chunk, tgi_chunk = compute_vis_on_chunk(chunk)
    vari_full[:, start_col:end_col] = vari_chunk
    tgi_full[:, start_col:end_col] = tgi_chunk
    print(f"Processed chunk {i+1}/{num_chunks}")

# sample for histograms
sample_size = int(0.15 * height * width)
indices = np.random.choice(height * width, sample_size, replace=False)
vari_sample = vari_full.ravel()[indices]
tgi_sample = tgi_full.ravel()[indices]

# histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(vari_sample, bins=50, color='green', alpha=0.7)
plt.title('VARI Frequency Histogram')
plt.xlabel('VARI Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.xlim(-0.2, 0.8)

plt.subplot(1, 2, 2)
plt.hist(tgi_sample, bins=50, color='green', alpha=0.7)
plt.title('TGI Frequency Histogram')
plt.xlabel('TGI Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.xlim(-0.2, 0.8)

plt.tight_layout()
plt.savefig('vi_histograms.png', dpi=300)
plt.show()

# vari: soil -> dry -> moderate veg -> lush green
cmap_vari = LinearSegmentedColormap.from_list(
    "vari_final",
    ["#4b3b2a", "#a89060", "#b7d57a", "#2e8b57", "#006400"]
)

# tgi: darker soil tones, less saturated greens
cmap_tgi = LinearSegmentedColormap.from_list(
    "tgi_final",
    ["#4f4f4f", "#8b8b6f", "#a6c86d", "#1e8f3a"]
)

def normalize_vi(vi, vmin, vmax):
    vi_norm = (vi - vmin) / (vmax - vmin)
    return np.clip(vi_norm, 0, 1)

# for rice fields (empirical tuning)
vari_norm = normalize_vi(vari_full, vmin=-0.05, vmax=0.4)
tgi_norm = normalize_vi(tgi_full, vmin=0.0, vmax=0.5)

# save and visualize
plt.imsave('vari_map.png', vari_norm, cmap=cmap_vari)
plt.imsave('tgi_map.png', tgi_norm, cmap=cmap_tgi)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(vari_norm, cmap=cmap_vari)
plt.title('VARI Map (Rebalanced Green)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(tgi_norm, cmap=cmap_tgi)
plt.title('TGI Map (Duller Background)')
plt.axis('off')

plt.tight_layout()
plt.show()