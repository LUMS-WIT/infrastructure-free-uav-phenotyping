import torch
from models import Model
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
from torchvision.transforms import functional
from scipy.ndimage import maximum_filter, label, find_objects
from scipy.spatial import KDTree
import os


# peak detection (unchanged)
def get_peaks(density_map, threshold=0.001):
    neighborhood = maximum_filter(density_map, size=7)
    peaks = (density_map == neighborhood) & (density_map > threshold)
    labeled, _ = label(peaks)
    slices = find_objects(labeled)
    points = [
        (int((dy.start + dy.stop - 1) / 2), int((dx.start + dx.stop - 1) / 2))
        for dy, dx in slices
    ]
    return points  # (y, x) in density_map coords


# simple non-max / merge across tiles
# points: list of dicts with keys: x, y, base_sigma
# merges points closer than radius pixels (global coords)
def nms_merge_points(points, radius=25):
    if not points:
        return []
    pts = np.array([[p['x'], p['y']] for p in points], dtype=np.float32)
    tree = KDTree(pts)
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    merged = []
    for i in range(n):
        if visited[i]:
            continue
        idxs = tree.query_ball_point(pts[i], r=radius)
        # merge cluster by averaging coords and base_sigma
        xs = [points[j]['x'] for j in idxs]
        ys = [points[j]['y'] for j in idxs]
        sigs = [points[j]['base_sigma'] for j in idxs]
        merged.append({
            'x': float(np.mean(xs)),
            'y': float(np.mean(ys)),
            'base_sigma': float(np.mean(sigs))
        })
        visited[idxs] = True
    return merged


# cosine weight window for blending overlaps
def make_cosine_weight(h, w):
    # 1d raised cosine in each axis, outer product
    y = np.linspace(0, np.pi, h, dtype=np.float32)
    x = np.linspace(0, np.pi, w, dtype=np.float32)
    wy = (1 - np.cos(y)) / 2.0
    wx = (1 - np.cos(x)) / 2.0
    w2d = np.outer(wy, wx)
    # normalize to [0,1], avoid zero in the middle being > others; this already peaks at 1
    w2d /= w2d.max() if w2d.max() > 0 else 1.0
    return w2d


# sliding window over large image
def sliding_windows(W, H, tile, overlap):
    step = tile - overlap
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))
    # ensure last tile touches the far edge
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, min(tile, W - x0), min(tile, H - y0)


if __name__ == "__main__":
    ORTHO_PATH = "ortho.png"
    OUT_DIR = "final_results/"
    os.makedirs(OUT_DIR, exist_ok=True)

    TILE = 1024          # tile size in pixels
    OVERLAP = 128        # overlap to reduce edge artifacts
    PEAK_THRESH = 0.004
    BORDER_MARGIN = 10   # optional global border filter
    GLOBAL_PADDING = 10  # pixels to pad around the orthomosaic where no counting/estimation is done
    MERGE_RADIUS = 30    # nms radius (pixels) to merge duplicate peaks across tile boundaries

    # model input sizes
    model_w, model_h = 320, 320
    size_model_w, size_model_h = 400, 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    model.load_state_dict(torch.load('model_weights/checkpoint_counting.pth', map_location=device))
    model.eval()

    model_size = Model(gap=True).to(device)
    model_size.load_state_dict(torch.load('model_weights/checkpoint_size_estimation.pth', map_location=device))
    model_size.eval()

    # load orthomosaic
    big_img = Image.open(ORTHO_PATH).convert("RGB")
    W, H = big_img.size
    print(f"Orthomosaic size: {W} x {H}")

    # prepare stitched heatmap accumulators
    heat_sum = np.zeros((H, W), dtype=np.float32)
    heat_wsum = np.zeros((H, W), dtype=np.float32)

    # collect global detections
    global_points = []  # list of dicts: {'x','y','base_sigma'}

    # prebuild cosine weight for a full tile; cropped per actual (w,h) tile
    base_weight = make_cosine_weight(TILE, TILE)

    # process tiles
    for x0, y0, tw, th in sliding_windows(W, H, TILE, OVERLAP):
        tile_img = big_img.crop((x0, y0, x0 + tw, y0 + th))

        # prepare tensors for count model (320x320)
        tile_resized = tile_img.resize((model_w, model_h), Image.BILINEAR)
        t = functional.to_tensor(tile_resized)
        t = functional.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        t = t.unsqueeze(0).to(device)

        # prepare tensors for size model (400x400)
        tile_resized_size = tile_img.resize((size_model_w, size_model_h), Image.BILINEAR)
        ts = functional.to_tensor(tile_resized_size)
        ts = functional.normalize(ts, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ts = ts.unsqueeze(0).to(device)

        with torch.no_grad():
            output, _ = model(t)
            density_map = output.squeeze().cpu().numpy()

            pre_dis = model_size(ts)
            pre_dis = float(pre_dis.item())

        # scale predicted size back to tile's original scale
        scale_factor_size = ((tw / size_model_w) + (th / size_model_h)) / 2.0
        tile_sigma = pre_dis * scale_factor_size

        # peaks on density map
        points_dm = get_peaks(density_map, threshold=PEAK_THRESH)
        dm_h, dm_w = density_map.shape

        # map peaks to tile coords
        scale_x = tw / dm_w
        scale_y = th / dm_h
        mapped_points = [(int(x * scale_x), int(y * scale_y)) for (y, x) in points_dm]

        # filter border inside tile (optional)
        tile_points = [
            (x, y) for (x, y) in mapped_points
            if BORDER_MARGIN <= x < tw - BORDER_MARGIN and BORDER_MARGIN <= y < th - BORDER_MARGIN
        ]

        # exclude points in the global padding region (absolute coordinates)
        tile_points = [
            (x, y) for (x, y) in tile_points
            if (GLOBAL_PADDING <= (x0 + x) < (W - GLOBAL_PADDING)) and (GLOBAL_PADDING <= (y0 + y) < (H - GLOBAL_PADDING))
        ]

        # add to global list (offset by tile origin)
        for (x, y) in tile_points:
            global_points.append({
                'x': x0 + x,
                'y': y0 + y,
                'base_sigma': tile_sigma
            })

        dm_img = Image.fromarray(density_map.astype(np.float32))
        dm_img = dm_img.resize((tw, th), Image.BILINEAR)
        dm = np.array(dm_img, dtype=np.float32)

        # crop base_weight to current (th, tw)
        w2d = base_weight[:th, :tw]
        # blend additively
        heat_sum[y0:y0+th, x0:x0+tw] += dm * w2d
        heat_wsum[y0:y0+th, x0:x0+tw] += w2d

    # normalize stitched heatmap (avoid divide-by-zero)
    mask = heat_wsum > 1e-6
    stitched_heat = np.zeros_like(heat_sum, dtype=np.float32)
    stitched_heat[mask] = heat_sum[mask] / heat_wsum[mask]

    # merge duplicate points across tile borders
    merged_points = nms_merge_points(global_points, radius=MERGE_RADIUS)
    print(f"Detections before merge: {len(global_points)} | after merge: {len(merged_points)}")

    # compute adaptive box sizes globally
    if len(merged_points) > 0:
        coords = np.array([[p['x'], p['y']] for p in merged_points], dtype=np.float32)
        tree = KDTree(coords)
        num_points = len(coords)
        k = min(num_points, 3)
        if k > 1:
            distances, _ = tree.query(coords, k=num_points if num_points > 1 else 1)
            local_mean = 0.8 * np.mean(distances[:, 1:k], axis=1) * 2
        else:
            local_mean = np.full(num_points, np.inf, dtype=np.float32)

        plant_sizes = []
        for i, p in enumerate(merged_points):
            dm = local_mean[i]
            base = p['base_sigma']
            dis = 0.5 * base + 0.5 * (dm if np.isfinite(dm) else base)
            plant_sizes.append(dis)
    else:
        plant_sizes = []

    # global color thresholds (quantiles)
    if len(plant_sizes) > 1:
        lower_q = float(np.percentile(plant_sizes, 25))
        upper_q = float(np.percentile(plant_sizes, 75))
        avg_sigma = float(np.mean([p['base_sigma'] for p in merged_points])) if merged_points else 0.0
    elif len(plant_sizes) == 1:
        lower_q = upper_q = plant_sizes[0]
        avg_sigma = merged_points[0]['base_sigma']
    else:
        lower_q = upper_q = 0.0
        avg_sigma = 0.0

    # render outputs
    from PIL import ImageDraw, ImageFont
    # 1) global points overlay (Pillow, lossless)
    overlay_img = big_img.copy()
    draw = ImageDraw.Draw(overlay_img, "RGBA")
    for p in merged_points:
        x, y = p['x'], p['y']
        r = 5
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,128))
    # draw total count at top left
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except Exception:
        font = None
    count_text = f"Total Count: {len(merged_points)}"
    draw.text((20, 20), count_text, fill=(255,255,255,255), font=font)
    out_points = os.path.join(OUT_DIR, "Counting.png")
    overlay_img.save(out_points, format=big_img.format if big_img.format else "PNG")
    print(f"Saved: {out_points}")

    # 2) global annotated sizes (Pillow)
    annot_img = big_img.copy()
    draw = ImageDraw.Draw(annot_img, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = None
    for (p, dis) in zip(merged_points, plant_sizes):
        x, y = p['x'], p['y']
        rect = [x - dis/2, y - dis/2, x + dis/2, y + dis/2]
        draw.rectangle(rect, outline=(255,0,0,255), width=1)
        text = f'{dis:.1f}'
        text_xy = (x, y)
        draw.text(text_xy, text, fill=(255,255,0,255), font=font, anchor="mm")
    # draw average size at top left
    if plant_sizes:
        avg_size = np.mean(plant_sizes)
        avg_text = f"Average Size: {avg_size:.2f}"
        draw.text((20, 20), avg_text, fill=(255,255,255,255), font=font)
    out_annot = os.path.join(OUT_DIR, "Size Estimation (Annotated).png")
    annot_img.save(out_annot, format=big_img.format if big_img.format else "PNG")
    print(f"Saved: {out_annot}")

    # 3) global color-coded sizes (Pillow)
    color_img = big_img.copy()
    draw = ImageDraw.Draw(color_img, "RGBA")
    for (p, dis) in zip(merged_points, plant_sizes):
        x, y = p['x'], p['y']
        if dis < lower_q:
            color = (255,0,0,255)      # red
        elif dis > upper_q:
            color = (255,255,0,255)   # yellow
        else:
            color = (255,165,0,255)  # orange
        rect = [x - dis/2, y - dis/2, x + dis/2, y + dis/2]
        draw.rectangle(rect, outline=color, width=1)
        draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
    out_color = os.path.join(OUT_DIR, "Size Estimation (Colored).png")
    color_img.save(out_color, format=big_img.format if big_img.format else "PNG")
    print(f"Saved: {out_color}")

    # 4) global heatmap (Pillow)
    # normalize and convert heatmap to RGB using a colormap
    import matplotlib
    norm_heat = (stitched_heat - np.nanmin(stitched_heat)) / (np.nanmax(stitched_heat) - np.nanmin(stitched_heat) + 1e-8)
    cmap = matplotlib.colormaps['jet']
    heat_rgba = (cmap(norm_heat) * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgba).convert("RGB")
    out_heat = os.path.join(OUT_DIR, "Counting Heatmap.png")
    heat_img.save(out_heat, format=big_img.format if big_img.format else "PNG")
    print(f"Saved: {out_heat}")