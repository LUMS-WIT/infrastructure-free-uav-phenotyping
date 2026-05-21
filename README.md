# Infrastructure-Free UAV Phenotyping Dataset

> **Paper Status**: Accepted to *Computers and Electronics in Agriculture*
> 
> This repository provides code, data, and documentation to support reproducibility and review of our work.

For questions or requests for additional data, please contact the corresponding author at 27100016@lums.edu.pk.

Acknowledgements: We thank the Life Sciences Department at LUMS for granting access to the field sites used in this study.

## Overview

This repository contains the complete dataset and source code for our paper "UAV-Based Crop Reconstruction and Trait Estimation Using RGB Imagery Without External Geospatial Infrastructure", under review in *Computers and Electronics in Agriculture*.

We present a low-cost, infrastructure-free UAV phenotyping workflow that enables reliable crop monitoring using only RGB imagery from consumer-grade drones, without Ground Control Points (GCPs) or RTK-GPS. The dataset includes weekly UAV flights over rice fields throughout the 2025 growing season, with phenotypic analysis spanning early, mid, and late growth stages.

### Key Features

- **Consumer-grade UAV Platform**: DJI Mini 4 Pro (RGB camera only)
- **No External Infrastructure**: No GCPs or RTK-GPS required
- **Complete Growing Season**: 13 weekly flight missions from August to October 2025
- **Multi-stage Phenotyping**: Plant counting, health monitoring, panicle detection, and height estimation
- **High Resolution**: Orthomosaics at ~1.85 mm/pixel GSD (6m altitude)
- **DEM Products**: Digital Elevation Models at ~5.8 mm/pixel GSD (10m altitude) with RANSAC-based correction
- **Ground Truth Data**: Manual canopy height measurements and plant/panicle count validation

### Dataset Highlights

| Metric | Value |
|--------|-------|
| Plant Counting Accuracy | 94.95% |
| Panicle Detection Accuracy | 91.59% |
| Height Estimation (Pearson R) | 0.876 |
| Height MAE | 0.177 m |
| Field Coverage | 2 plots × 30m × 10m |
| Total Flights | 13 dual-altitude missions |

---

## Repository Structure

```
infrastructure-free-uav-phenotyping/
│
├── data/                         # UAV-derived products organized by date (limited public release)
│   ├── 2025-08-13/               # early stage: plant counting & sizing
|   ├── 2025-08-22/
│   ├── 2025-08-27/               # mid stage: yellow leaf detection
│   ├── 2025-08-28/               # mid stage: yellow leaf detection
│   ├── 2025-09-04/               # first corrected DEM acquisition
│   ├── 2025-09-05/               
│   ├── 2025-09-09/               
│   ├── 2025-09-12/               # first ground truth height measurements
│   ├── 2025-09-17/               # panicle detection begins
│   ├── 2025-09-19/               
│   ├── 2025-09-26/               
│   ├── 2025-10-01/               
│   └── 2025-10-03/
│
├── code/                         # processing pipeline and models
│   ├── dem_processing/           # RANSAC-based DEM flattening
│   ├── plant_counting/           # RiceNet implementation
│   ├── panicle_counting/         # YOLOv5 fine-tuned model
│   ├── yellow_leaf_detection/    # SAM + SegVeg pipeline
│   └── vegetation_indices/       # VARI and TGI computation
```

---

## Dataset Description

### Weekly Data Organization

Each dated folder contains available data products. Note: due to restrictions from the Life Sciences Department, the public repository contains full datasets only for the following dates:

- 2025-08-13 — Plant counting & size estimation (full orthomosaics)
- 2025-08-27 — Yellow leaf detection (full orthomosaics and indices)

All other dated folders include a limited public release: 10 randomly sampled RGB frames from different field locations (for use in examples and evaluation scripts). If you require additional data or access to the full datasets, please contact Hussain at 27100016@lums.edu.pk and we will coordinate access where permissible.

**Analysis Results:**

| Date Range | Analysis Type | Folder |
|------------|--------------|---------|
| 2025-08-13 | Plant Counting & Size Estimation | `plant_counting_&_size_estimation/` |
| 2025-08-27 | Yellow Leaf Detection (SAM + SegVeg) | `yellow_leaf_detection/` |
| 2025-09-17 onwards | Panicle Detection (YOLOv5) | `panicle_counting/` |

---

## Processing Pipeline

### 1. Orthomosaic Generation (Agisoft Metashape)

```bash
# Flight altitude: 6m
# GSD: ~1.85 mm/pixel
# Frame extraction: Every 6th frame from 4K video @ 24fps
# Overlap: 70-80% forward and lateral
```

**Key Settings:**
- Photo alignment: High accuracy
- Dense cloud: Mild depth filtering
- No Ground Control Points (GCPs)
- No RTK-GPS corrections

### 2. DEM Generation & Correction

```bash
# Flight altitude: 10m
# GSD: ~5.8 mm/pixel
# Flight pattern: Follow-course multi-S-shaped
```

**Artifact Mitigation:**
1. Multi-S-shaped flight pattern reduces doming effects
2. RANSAC-based plane fitting removes residual linear trends
3. Automated processing (no manual intervention required)

See `code/dem_post_processing/ransac_flatten.py` for implementation.

### 3. Phenotypic Analysis

**Early Stage (Week 1-2):** Plant counting using RiceNet density regression
- Input: 1024×1024 pixel tiles with 128px overlap
- Output: Plant locations and pixel-area estimates
- Accuracy: 94.95% agreement with manual counts

**Mid Stage (Week 3-4):** Health monitoring using SAM + SegVeg
- Stage 1: SAM generates plant masks (isolates from flooded background)
- Stage 2: SegVeg classifies healthy vs. stressed vegetation
- Input: 2048×2048 pixel tiles with 256px overlap

**Late Stage (Week 5+):** Panicle detection using fine-tuned YOLOv5
- Input: 1024×1024 pixel tiles with 50% overlap
- Confidence threshold: 0.7, NMS IoU: 0.4
- Accuracy: 91.59% average detection agreement

---

## Prerequisites

```bash
# Python 3.8+
pip install numpy scipy rasterio opencv-python torch torchvision
pip install segment-anything timm pandas matplotlib seaborn
```

---

## Model Weights & Dependencies

### Models Included in This Repository

All model weights provided here were trained by the authors:

| Model | Task | Location | Size | Notes |
|-------|------|----------|------|-------|
| YOLOv5 (fine-tuned) | Rice panicle detection | `code/panicle_counting/weights/rice_panicle_yolov5.pt` | ~90MB | Fine-tuned on UAV rice panicle dataset |
| RiceNet (self-trained) | Plant counting | `code/plant_counting/weights/checkpoint_counting.pt` | ~80MB | Trained on public rice dataset |
| RiceNet (self-trained) | Size estimation | `code/plant_counting/weights/checkpoint_size_estimation.pt` | ~62MB | Trained on public rice dataset |

**Architecture Reference**: RiceNet architecture from [Bai et al., 2023](https://doi.org/10.34133/plantphenomics.0020)  

### Models to Download Separately

Due to file size and licensing, please download these models from their official sources:

**SAM ViT-B (Segmentation)**
```bash
# Download from Meta AI (375MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# Place in: code/yellow_leaf_detection/weights/sam_vit_b.pth
```
Reference: [Kirillov et al., 2023](https://arxiv.org/abs/2304.02643)

**SegVeg (Vegetation Classification)**
```bash
# Download SegVeg data from https://github.com/mserouar/SegVeg/releases/tag/v1.1.0
# Follow setup instructions in their repository
```
Reference: [Madec et al., 2023](https://doi.org/10.34133/2022/9803570)


## Hardware & Software

### UAV Platform
- **Model**: DJI Mini 4 Pro
- **Weight**: 249g (sub-250g category)
- **Sensor**: 1/1.3" CMOS, 48MP
- **Flight time**: ~34 minutes per battery

### Flight Planning
- **Software**: WaypointMap, Litchi
- **Altitude**: 6m (orthomosaics), 10m (DEMs)
- **Speed**: 1 m/s

### Processing Software
- **Photogrammetry**: Agisoft Metashape 1.8.5

---

## Authors & Contact

**Muhammad Hussain Habib Chaudhry** (Corresponding Author)  
Department of Computer Science, LUMS  
Email: 27100016@lums.edu.pk

**Muhammad Ibrahim Rana**  
Center for Water Informatics & Technology, LUMS  
Email: m_rana@lums.edu.pk

**Hassan Jaleel**  
Center for Water Informatics & Technology, LUMS  
Department of Electrical Engineering, LUMS  
Email: hassan.jaleel@lums.edu.pk

**Institution**: Lahore University of Management Sciences (LUMS), Pakistan  
**Website**: https://wit.lums.edu.pk

---

## License

This repository utilizes a dual-license structure to distinguish between the software pipeline and the dataset:

* **Software/Code:** All source code, processing scripts, and model implementations are licensed under the [MIT License](LICENSE).
* **Data:** All raw UAV imagery, processed orthomosaics, Digital Elevation Models (DEMs), and ground-truth annotations are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA). 

## Citation

If you use the data or code in this repository, please cite both the dataset and the associated manuscript:

**Dataset Citation:**
> Chaudhry, M. H. H., Rana, M. I., & Jaleel, H. (2026). Infrastructure-Free UAV Phenotyping Dataset [Dataset]. GitHub. https://github.com/LUMS-WIT/infrastructure-free-uav-phenotyping

**Paper Citation:**
> Chaudhry, M. H. H., Rana, M. I., & Jaleel, H. (2026). UAV-Based Crop Reconstruction and Trait Estimation Using RGB Imagery Without External Geospatial Infrastructure. *Computers and Electronics in Agriculture*. Submitted.
