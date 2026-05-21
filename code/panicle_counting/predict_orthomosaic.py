import cv2
from pathlib import Path
import subprocess
import sys
import shutil
import argparse
import tempfile
import numpy as np
import time

def create_sliding_windows(image_path, window_size=1024, stride=512, output_dir="temp_windows"):

    # read the input image
    image = cv2.imread(image_path)

    img_h, img_w = image.shape[:2]
    print(f"Image size: {img_w}x{img_h}")
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    windows_info = []
    count = 0

    # slide window over the image
    for y in range(0, img_h - window_size + 1, stride):
        for x in range(0, img_w - window_size + 1, stride):
            window = image[y:y + window_size, x:x + window_size]
            fname = f"win_{count:05d}_{x}_{y}.jpg"
            cv2.imwrite(str(out_dir / fname), window)
            windows_info.append({
                "filename": fname,
                "x": x, "y": y,
                "width": window_size,
                "height": window_size,
                "window_id": count
            })
            count += 1
            if count % 50 == 0:
                print(f"  Created {count} windows:")
    print(f"Total windows: {count}")
    return windows_info, out_dir


def run_yolo_detection(windows_dir, weights_path, conf_threshold=0.25, device="0"):
    # run YOLOv5 detect.py on all tiles
    output_dir = Path("temp_detections")
    cmd = [
        sys.executable, "yolov5/detect.py",
        "--weights", weights_path,
        "--source", str(windows_dir),
        "--project", str(output_dir.parent),
        "--name", output_dir.name,
        "--conf-thres", str(conf_threshold),
        "--save-txt", "--save-conf",
        "--device", device, "--exist-ok"
    ]
    print("Running YOLOv5 detection:")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_dir
    except subprocess.CalledProcessError as e:
        print("[ERROR] YOLOv5 detection failed:", e.stderr)
        return None


def parse_yolo_detections(det_dir, windows_info):
    # convert YOLO window coordinates to global coordinates
    labels_dir = det_dir / "labels"
    if not labels_dir.exists():
        print("[WARN] No label files produced.")
        return []
    all_dets = []
    # parse each window's detection file
    for w in windows_info:
        lf = labels_dir / f"{Path(w['filename']).stem}.txt"
        if not lf.exists():
            continue
        with open(lf) as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue
                cls = int(p[0])
                x_c, y_c, w_n, h_n = map(float, p[1:5])
                conf = float(p[5]) if len(p) > 5 else 1.0
                win_size = w['width']
                x1 = int(w['x'] + (x_c - w_n / 2) * win_size)
                y1 = int(w['y'] + (y_c - h_n / 2) * win_size)
                x2 = int(w['x'] + (x_c + w_n / 2) * win_size)
                y2 = int(w['y'] + (y_c + h_n / 2) * win_size)
                all_dets.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": conf, "class": cls,
                    "area": (x2 - x1) * (y2 - y1)
                })
    print(f"Parsed {len(all_dets)} detections.")
    return all_dets


def iou(a, b):
    # calculate intersection over union for two boxes
    x1 = max(a['x1'], b['x1'])
    y1 = max(a['y1'], b['y1'])
    x2 = min(a['x2'], b['x2'])
    y2 = min(a['y2'], b['y2'])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = a['area'] + b['area'] - inter
    return inter / union if union else 0.0


def improved_nms(dets, iou_thresh=0.3):
    # perform non-maximum suppression on detection results
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d['confidence'], reverse=True)
    keep = []
    suppressed = [False] * len(dets)
    for i, d in enumerate(dets):
        if suppressed[i]:
            continue
        keep.append(d)
        for j in range(i + 1, len(dets)):
            if suppressed[j]:
                continue
            if iou(d, dets[j]) > iou_thresh:
                suppressed[j] = True
    print(f"NMS kept {len(keep)} / {len(dets)} boxes")
    return keep


def save_results(image_path, detections, out_dir):
    # save the detection results as an image with boxes
    img = cv2.imread(image_path)
    if img is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    overlay = img.copy()
    for d in detections:
        cv2.rectangle(overlay,
                      (d['x1'], d['y1']), (d['x2'], d['y2']),
                      (0, 0, 255), 2)
    out_path = out_dir / (Path(image_path).stem + "_detected.png")
    cv2.imwrite(str(out_path), overlay)
    print(f"Saved final image: {out_path}")


def process_image(image_path, weights, output, win_size, stride, conf, iou_t, device):
    # process a single image: sliding window, detection, NMS, and save
    with tempfile.TemporaryDirectory() as tmp:
        win_info, win_dir = create_sliding_windows(image_path, win_size, stride, tmp)
        if not win_info:
            return
        det_dir = run_yolo_detection(win_dir, weights, conf, device)
        if not det_dir:
            return
        dets = parse_yolo_detections(det_dir, win_info)
        if not dets:
            print("No detections found.")
            return
        final = improved_nms(dets, iou_t)
        save_results(image_path, final, output)
        # cleanup YOLO results
        time.sleep(0.1)
        shutil.rmtree(det_dir, ignore_errors=True)


def main():
    # main entry point for CLI usage
    ap = argparse.ArgumentParser(description="Sliding-window YOLO on very large images.")
    ap.add_argument("--input", default='orthomosaic.png', help="Path to single image or folder.")
    ap.add_argument("--weights", default='model_weights.pt', help="YOLOv5 weights.")
    ap.add_argument("--output", default="ortho_results", help="Output directory.")
    ap.add_argument("--window-size", type=int, default=1024, help="Tile size (default 1024 recommended for huge mosaics).")
    ap.add_argument("--stride", type=int, default=512, help="Stride (default 512 ≈50%% overlap).")
    ap.add_argument("--conf", type=float, default=0.7, help="Confidence threshold.")
    ap.add_argument("--iou", type=float, default=0.4, help="NMS IoU threshold.")
    ap.add_argument("--device", default="cpu", help="YOLO device (0 for GPU, cpu for CPU).")
    args = ap.parse_args()

    if not Path("yolov5/detect.py").exists():
        print("[ERROR] yolov5/detect.py not found.")
        return
    if not Path(args.weights).exists():
        print(f"[ERROR] Weights file {args.weights} not found.")
        return

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    if input_path.is_file():
        process_image(str(input_path), args.weights, output_dir,
                      args.window_size, args.stride, args.conf, args.iou, args.device)
    else:
        images = [p for p in input_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        print(f"Found {len(images)} images.")
        for idx, img in enumerate(images, 1):
            print(f"[{idx}/{len(images)}] Processing {img.name}")
            process_image(str(img), args.weights, output_dir,
                          args.window_size, args.stride, args.conf, args.iou, args.device)
    print("Done.")


if __name__ == "__main__":
    main()
