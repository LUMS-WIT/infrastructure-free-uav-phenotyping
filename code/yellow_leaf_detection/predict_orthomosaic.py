
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm
import pickle
from pathlib import Path
from skimage import color
from PIL import ImageCms
from segment_anything import sam_model_registry, SamPredictor

np.random.seed(0)

XGBOOST_THRESHOLD = 0.9998

# hsv filter for senescent pixels
def hsv_filter_senescent(image_bgr, mask, s_thresh=50, v_thresh=160):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)
    dark_pixels = ((s < s_thresh) | (v < v_thresh))
    mask_clean = mask.copy()
    mask_clean[(mask == 2) & dark_pixels] = 0
    return mask_clean

# resize image to max dimension
def resize_max_dim(img, max_dim=1024):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    return img, scale

# compute excess green index
def excess_green(rgb):
    r = rgb[...,0].astype(np.float32)
    g = rgb[...,1].astype(np.float32)
    b = rgb[...,2].astype(np.float32)
    exg = 2*g - r - b
    exg -= exg.min()
    denom = exg.max() if exg.max() > 0 else 1.0
    return exg / denom

# sample points spread across mask
def sample_points_spread(mask, k):
    H, W = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0,2), dtype=np.int32)
    pts = []
    grid = int(np.sqrt(k)) + 1
    for gy in range(grid):
        for gx in range(grid):
            y0 = int(gy * H / grid); y1 = int((gy+1) * H / grid)
            x0 = int(gx * W / grid); x1 = int((gx+1) * W / grid)
            in_cell = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
            idx = np.where(in_cell)[0]
            if len(idx) > 0:
                j = np.random.choice(idx)
                pts.append([xs[j], ys[j]])
            if len(pts) >= k:
                break
        if len(pts) >= k:
            break
    if len(pts) < k:
        remaining = min(k - len(pts), len(xs))
        rand_idx = np.random.choice(len(xs), size=remaining, replace=False)
        for j in rand_idx:
            pts.append([xs[j], ys[j]])
    return np.array(pts, dtype=np.int32)

# choose best mask from candidates
def choose_best_mask(masks, pos_pts, neg_pts):
    best_idx, best_score = 0, -1e9
    H, W = masks[0].shape
    for i, m in enumerate(masks):
        m_bool = m.astype(bool)
        pos_hit = int(m_bool[pos_pts[:,1], pos_pts[:,0]].sum()) if len(pos_pts) else 0
        neg_hit = int(m_bool[neg_pts[:,1], neg_pts[:,0]].sum()) if len(neg_pts) else 0
        area_ratio = m_bool.mean()
        score = pos_hit - 2*neg_hit
        if area_ratio > 0.9:
            score -= 10
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx

# postprocess mask to remove small components
def postprocess_mask(mask, min_area=500):
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels <= 1:
        return mask
    areas = np.bincount(labels.ravel())
    keep = np.where(areas >= min_area)[0]
    keep[keep == 0] = -1
    keep_mask = np.isin(labels, keep)
    return keep_mask.astype(np.uint8)

# generate plant mask using SAM
def generate_plant_mask(image_rgb, sam_ckpt, min_component=500, max_dim=1024):
    small, scale = resize_max_dim(image_rgb, max_dim)
    exg = excess_green(small)
    q60 = np.quantile(exg, 0.95)  # higher = stricter, e.g. 0.90–0.98
    q25 = np.quantile(exg, 0.20)  # lower = stricter, e.g. 0.10–0.30
    pos_cand = exg > q60
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    S = hsv[...,1].astype(np.float32) / 255.0
    B = small[...,2].astype(np.float32)
    neg_cand = (exg < q25) | (S < 0.18) | (B > np.quantile(B, 0.70))  # less strict
    # sample more points for better coverage
    pos_pts = sample_points_spread(pos_cand, 80)
    neg_pts = sample_points_spread(neg_cand, 24)
    # add a few corner negatives to discourage global fill
    Hs, Ws = small.shape[:2]
    corner_pad = 8
    corner_pts = np.array([[corner_pad, corner_pad], [Ws - corner_pad, corner_pad], [corner_pad, Hs - corner_pad], [Ws - corner_pad, Hs - corner_pad]], dtype=np.int32)
    neg_pts = np.vstack([neg_pts, corner_pts])
    # run SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(small)
    pts = np.vstack([pos_pts, neg_pts]).astype(np.float32)
    labels = np.concatenate([np.ones(len(pos_pts), dtype=np.int32), np.zeros(len(neg_pts), dtype=np.int32)])
    masks, scores, logits = predictor.predict(point_coords=pts, point_labels=labels, multimask_output=True)
    # pick best mask
    best = choose_best_mask(masks, pos_pts, neg_pts)
    mask_small = masks[best].astype(np.uint8)
    # postprocess
    mask_small = postprocess_mask(mask_small, min_component)
    # upscale
    H, W = image_rgb.shape[:2]
    if scale != 1.0:
        plant_mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        plant_mask = mask_small
    return plant_mask

model_path = "SegVeg_data/model_scikit"
model_SVM = pickle.load(Path(model_path).open("rb"))

# load the original model (from published SegVeg paper) but optimized through XGBoost (same data, but reduced computational time)
model_path = "SegVeg_data/XGBoost"
model_XG = pickle.load(Path(model_path).open("rb"))
new_attrs = ['grow_policy', 'max_bin', 'eval_metric', 'callbacks', 'early_stopping_rounds', 'max_cat_to_onehot', 'max_leaves', 'sampling_method', 'enable_categorical', 'feature_types', 'max_cat_threshold', 'predictor']
for attr in new_attrs:
    setattr(model_XG, attr, None)

# load the original model (from published SegVeg paper) but improved thanks to Jonas Anderegg et al. work.
# no more confusion between chlorosis-yellow and necrosis-brown
model_path = "SegVeg_data/Necrosis.pkl"
model_Necrosis = pickle.load(Path(model_path).open("rb"))

# get features from rectangle_fields.py
def get_features(image):
    pil_image = Image.fromarray(image)

    hsv = np.array(pil_image.convert(mode='HSV'))
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = np.array(ImageCms.applyTransform(pil_image, rgb2lab))
    ycbcr = np.array(pil_image.convert(mode='YCbCr'))
    Labb = color.rgb2lab(image)
    r = image[:, :,0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    h = (hsv[:, :, 0].astype(np.float32) * 360) / 255
    s = (hsv[:, :, 1]) / 2.55
    a = Labb[:, :, 1]
    bb = Lab[:, :, 2]
    ge =  np.mean([r,g,b], axis = 0)

    CMYlist = [1 - r / 255, 1 - g / 255, 1 - b / 255]
    CMYlist = np.array([np.min(idx) for idx in zip(*CMYlist)])
    m = ((1 - g / 255 - CMYlist ) / (1 - CMYlist )) * 100
    ye = ((1 - b / 255 - CMYlist ) / (1 - CMYlist )) * 100

    cb = ycbcr[:, :, 1]
    cr = ycbcr[:, :, 2]

    i = (0.596 * r - 0.275 * g - 0.321 * b)
    q = (0.212 * r - 0.523 * g + 0.311 * b)

    model_input = np.stack((r,g,b,h,s,a,bb,ge,m,ye,cb,cr,i,q), axis=2).squeeze(0)
    model_input = np.nan_to_num(model_input) # handle black pixels (avoiding Input contains NaN, infinity or a value too large for dtype('float64') error)

    return model_input

# use the same prediction_XG_SVM as in rectangle_fields.py (with get_features)
def prediction_XG_SVM(image, model, threshold, contrasted, mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    if mask is None:
        mask = np.ones((height, width))
    image = image.reshape((width*height), 1, 3)
    mask = mask.reshape((width*height), 1)
    yellow_green_mask = np.zeros(image.shape[:-1])
    vegetation_pixels = mask > 0
    image = image[None, vegetation_pixels]
    featured_image = get_features(image)
    probas = model.predict_proba(featured_image)[:,1]
    y_pred = (probas >= threshold).astype(int)
    yellow_green_mask[vegetation_pixels] = y_pred
    mask[(mask == 1) & (yellow_green_mask != 1)] = 2
    mask = mask.reshape((height, width))
    prob_map = np.zeros_like(yellow_green_mask, dtype=np.float32)
    prob_map[vegetation_pixels] = probas
    prob_map = prob_map.reshape((height, width))
    return mask, prob_map

# visualize necrosis and yellow leaf detection
def visualisation_Necrosis(rgb_image: np.ndarray, yg_mask: np.ndarray, stg) -> np.ndarray:
    """
    take rgb image and yellow_green_mask and apply color
    """
    image_copy = rgb_image.copy()

    # works because BGR
    if stg == "SegVeg_Necrosis":
        image_copy[yg_mask == 0] = (0, 0, 0)         # background = black
        image_copy[yg_mask == 1] = (34, 70, 34)      # green = dark dull green (BGR)
        image_copy[yg_mask == 3] = (0, 255, 255)     # chlorosis = bright yellow (BGR)
        image_copy[yg_mask == 2] = (0, 0, 255)       # necrosis = bright red
    if stg == "SegVeg_XG_SVM":
        image_copy[yg_mask == 0] = (0, 0, 0)         # background = black
        image_copy[yg_mask == 1] = (34, 70, 34)      # green = dark dull green (BGR)
        image_copy[yg_mask == 2] = (0, 255, 255)     # unhealthy (senescent) = bright yellow (BGR)

    visualisation = cv2.addWeighted(rgb_image, 0.4, image_copy, 0.6, 0)

    index_class = np.unique(yg_mask)
    color_vizu = []
    # update legend colors: 0=black, 1=dark dull green, 2=red, 3=bright yellow
    id = [(0, 0, 0), (34, 70, 34), (0, 0, 255), (0, 255, 255)]
    for c in index_class:
        color_vizu.append(id[int(c)])

    return visualisation, index_class, color_vizu

# split big image into tiles with overlap
def tile_image(img, tile_size=1024, overlap=64):
    h, w = img.shape[:2]
    tiles = []
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = x, min(x + tile_size, w)
            tile = img[y1:y2, x1:x2]
            tiles.append(((y1, y2, x1, x2), tile))
    return tiles, (h, w)

# stitch tiled predictions back into full-size array
def stitch_tiles(tiles, shape):
    H, W = shape
    out = np.zeros((H, W), dtype=np.uint8)
    for (y1, y2, x1, x2), tile_mask in tiles:
        out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], tile_mask)
    return out

# main segmentation pipeline for orthomosaic
def segveg_on_orthomosaic(orthomosaic_path, sam_ckpt, model_XG, out_dir, tile_size=1024, overlap=64):
    os.makedirs(out_dir, exist_ok=True)

    pad_px = 200  # you can change this value as needed
    orig_img = cv2.imread(orthomosaic_path, cv2.IMREAD_COLOR)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    H, W = orig_img.shape[:2]

    # step 1: tile orthomosaic
    tiles, shape = tile_image(orig_img, tile_size=tile_size, overlap=overlap)

    stitched_masks = []
    stitched_probs = []

    for (y1, y2, x1, x2), tile in tqdm(tiles, desc="Processing tiles"):
        # plant mask
        plant_mask = generate_plant_mask(tile, sam_ckpt)

        # SegVeg classification
        yg_mask, prob_map = prediction_XG_SVM(
            cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
            model_XG,
            threshold=0.8,
            contrasted=1,
            mask=plant_mask
        )
        yg_mask = hsv_filter_senescent(tile, yg_mask)

        stitched_masks.append(((y1, y2, x1, x2), yg_mask))
        stitched_probs.append(((y1, y2, x1, x2), (prob_map*255).astype(np.uint8)))

    # step 2: stitch results back
    full_mask = stitch_tiles(stitched_masks, shape)

    # step 3: save overlay visualization only (no .tif files)
    base_name = os.path.splitext(os.path.basename(orthomosaic_path))[0]
    visu, _, _ = visualisation_Necrosis(orig_img, full_mask, "SegVeg_XG_SVM")
    overlay_path = os.path.join(out_dir, f"Disease Detection.png")
    Image.fromarray(cv2.cvtColor(visu, cv2.COLOR_BGR2RGB)).save(overlay_path, format="PNG")
    print(f"Saved overlay to {overlay_path}")

segveg_on_orthomosaic(
    orthomosaic_path="orthomosaic.png",
    sam_ckpt="sam_vit_b.pth",
    model_XG=model_XG,
    out_dir="results/",
    tile_size=2048,
    overlap=256
)
