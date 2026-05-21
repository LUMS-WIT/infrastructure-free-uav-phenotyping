import numpy as np
import rasterio
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt
from pathlib import Path

def simple_edge_mask(dem_array, edge_threshold=-200):
    # mask out pixels below the edge threshold
    valid_mask = dem_array > edge_threshold
    print(f"simple masking: excluded {np.sum(~valid_mask)} pixels below threshold {edge_threshold}")
    print(f"keeping {np.sum(valid_mask)} pixels ({100*np.sum(valid_mask)/dem_array.size:.1f}%)")
    return valid_mask

def load_and_validate_dems(dem_paths, edge_threshold=-200):
    dems = []
    profiles = []
    valid_masks = []

    print("loading DEMs:")
    shapes = []
    for path in dem_paths:
        print(f"  - {path}")
        with rasterio.open(path) as src:
            dem = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            dems.append(dem)
            profiles.append(profile)
            shapes.append(dem.shape)
            # create individual mask for each DEM
            mask = simple_edge_mask(dem, edge_threshold)
            valid_masks.append(mask)

    # find smallest common shape among DEMs
    min_rows = min([s[0] for s in shapes])
    min_cols = min([s[1] for s in shapes])
    if len(set(shapes)) > 1:
        print(f"warning: DEMs have different shapes, cropping all to ({min_rows}, {min_cols})")
    # crop all DEMs and masks to smallest shape
    dems = [dem[:min_rows, :min_cols] for dem in dems]
    valid_masks = [mask[:min_rows, :min_cols] for mask in valid_masks]

    # create combined mask: only pixels valid in all DEMs
    combined_mask = np.logical_and.reduce(valid_masks)
    print(f"\ncombined valid pixels: {np.sum(combined_mask)} ({100*np.sum(combined_mask)/combined_mask.size:.1f}%)")

    return dems, profiles, combined_mask

def fit_reference_trend_surface(reference_dem, valid_mask, polynomial_degree=2):
    # fit a polynomial trend surface to the reference DEM using valid pixels
    rows, cols = reference_dem.shape
    x_coords = np.arange(cols)
    y_coords = np.arange(rows)
    X, Y = np.meshgrid(x_coords, y_coords)

    # extract valid points for fitting
    x_valid = X[valid_mask]
    y_valid = Y[valid_mask]
    z_valid = reference_dem[valid_mask]

    print(f"fitting degree-{polynomial_degree} reference surface to {len(z_valid)} points:")

    # fit polynomial features
    coordinates = np.column_stack([x_valid, y_valid])
    poly_features = PolynomialFeatures(degree=polynomial_degree, include_bias=True)
    X_poly = poly_features.fit_transform(coordinates)

    # use RANSAC for robust regression
    base_model = LinearRegression()
    model = RANSACRegressor(base_model, random_state=42)
    model.fit(X_poly, z_valid)

    return model, poly_features, X, Y

def apply_consistent_correction(dem, model, poly_features, X, Y, valid_mask, reference_mean, correction_strength=1.0, original_nodata=None):
    # apply the trend surface correction to the DEM
    rows, cols = dem.shape

    # generate trend surface using the reference model
    all_coordinates = np.column_stack([X.flatten(), Y.flatten()])
    X_poly_all = poly_features.transform(all_coordinates)
    trend_surface = model.predict(X_poly_all).reshape(rows, cols)

    # apply correction
    correction = trend_surface * correction_strength
    corrected_dem = dem - correction

    # restore invalid areas if needed
    if original_nodata is not None:
        corrected_dem[~valid_mask] = original_nodata

    # normalize to reference mean for consistent comparison
    current_mean = np.mean(corrected_dem[valid_mask])
    offset = reference_mean - current_mean
    corrected_dem[valid_mask] += offset

    print(f"applied offset: {offset:.3f} to match reference mean")

    return corrected_dem, correction

def process_multiple_dems_consistently(dem_paths, output_dir=None,
                                     edge_threshold=-200,
                                     correction_strength=1.0,
                                     polynomial_degree=2,
                                     reference_index=0):
    # process multiple DEMs to apply consistent flattening and normalization
    if output_dir is None:
        output_dir = Path(dem_paths[0]).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
    # load all DEMs
    dems, profiles, combined_mask = load_and_validate_dems(dem_paths, edge_threshold)

    if np.sum(combined_mask) < 100:
        print("error: too few valid pixels in combined mask, try lowering edge_threshold")
        return None

    # use reference DEM to fit the trend surface
    reference_dem = dems[reference_index]
    model, poly_features, X, Y = fit_reference_trend_surface(
        reference_dem, combined_mask, polynomial_degree
    )

    # calculate reference statistics
    reference_mean = np.mean(reference_dem[combined_mask])
    print(f"reference DEM mean elevation: {reference_mean:.3f}")

    # process all DEMs with the same correction
    corrected_dems = []
    output_paths = []

    # first, apply consistent correction to all DEMs and collect all valid pixels
    all_valid_pixels = []
    temp_corrected = []
    for i, (dem_path, dem, profile) in enumerate(zip(dem_paths, dems, profiles)):
        print(f"\nprocessing DEM {i+1}: {Path(dem_path).name}")
        corrected_dem, correction = apply_consistent_correction(
            dem, model, poly_features, X, Y, combined_mask,
            reference_mean, correction_strength, profile.get('nodata')
        )
        temp_corrected.append((corrected_dem, dem_path, profile))
        all_valid_pixels.append(corrected_dem[combined_mask])

    # compute global 0.5 percentile (ground) and max from all valid pixels
    all_valid_pixels = np.concatenate(all_valid_pixels)
    global_ground = np.nanpercentile(all_valid_pixels, 0.5)
    global_max = np.nanmax(all_valid_pixels)
    scale = 5.0 / (global_max - global_ground) if global_max > global_ground else 1.0

    # now normalize all DEMs using global ground/max
    corrected_dems = []
    output_paths = []
    for corrected_dem, dem_path, profile in temp_corrected:
        norm_dem = np.full_like(corrected_dem, np.nan)
        norm_dem[combined_mask] = (corrected_dem[combined_mask] - global_ground) * scale
        input_path = Path(dem_path)
        output_path = output_dir / f"{input_path.stem}_flattened{input_path.suffix}"
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(norm_dem, 1)
        corrected_dems.append(norm_dem)
        output_paths.append(output_path)
        valid_corrected = norm_dem[combined_mask]
        print(f"normalized range: {np.nanmin(valid_corrected):.2f} to {np.nanmax(valid_corrected):.2f}")
        print(f"ground (2nd percentile) value: {np.nanpercentile(valid_corrected,2):.3f}")
        print(f"saved: {output_path}")

    return corrected_dems, output_paths, combined_mask, dems

def compare_dems_visualization(original_dems, corrected_dems, dem_names, valid_mask, save_path=None):
    # create comprehensive comparison visualization showing before/after effects
    n_dems = len(original_dems)
    
    # create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))

    # calculate global color scale limits for fair comparison
    all_original_valid = np.concatenate([dem[valid_mask] for dem in original_dems])
    all_corrected_valid = np.concatenate([dem[valid_mask] for dem in corrected_dems if not np.all(np.isnan(dem[valid_mask]))])

    orig_vmin, orig_vmax = np.nanpercentile(all_original_valid, [2, 98])
    corr_vmin, corr_vmax = np.nanpercentile(all_corrected_valid, [2, 98])

    # 1. original DEMs
    for i in range(n_dems):
        ax = plt.subplot(4, n_dems, i + 1)
        masked_original = original_dems[i].copy()
        masked_original[~valid_mask] = np.nan
        im = ax.imshow(masked_original, cmap='terrain', vmin=orig_vmin, vmax=orig_vmax)
        ax.set_title(f'Original: {dem_names[i]}', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 2. corrected DEMs
    for i in range(n_dems):
        ax = plt.subplot(4, n_dems, n_dems + i + 1)
        masked_corrected = corrected_dems[i].copy()
        masked_corrected[~valid_mask] = np.nan
        im = ax.imshow(masked_corrected, cmap='terrain', vmin=corr_vmin, vmax=corr_vmax)
        ax.set_title(f'Flattened: {dem_names[i]}', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 3. cross-sectional profiles (horizontal)
    ax_profile_h = plt.subplot(4, 2, 5)
    center_row = original_dems[0].shape[0] // 2
    x_range = np.arange(original_dems[0].shape[1])
    
    for i in range(n_dems):
        ax_profile_h.plot(x_range, original_dems[i][center_row, :], 
                         label=f'{dem_names[i]} (original)', linestyle='--', alpha=0.8, linewidth=2)
        ax_profile_h.plot(x_range, corrected_dems[i][center_row, :], 
                         label=f'{dem_names[i]} (flattened)', linewidth=2)
    
    ax_profile_h.set_xlabel('Column Index')
    ax_profile_h.set_ylabel('Elevation')
    ax_profile_h.set_title('Horizontal Cross-section (Center Row)')
    ax_profile_h.legend()
    ax_profile_h.grid(True, alpha=0.3)
    
    # 4. cross-sectional profiles (vertical)
    ax_profile_v = plt.subplot(4, 2, 6)
    center_col = original_dems[0].shape[1] // 2
    y_range = np.arange(original_dems[0].shape[0])
    
    for i in range(n_dems):
        ax_profile_v.plot(y_range, original_dems[i][:, center_col], 
                         label=f'{dem_names[i]} (original)', linestyle='--', alpha=0.8, linewidth=2)
        ax_profile_v.plot(y_range, corrected_dems[i][:, center_col], 
                         label=f'{dem_names[i]} (flattened)', linewidth=2)
    
    ax_profile_v.set_xlabel('Row Index')
    ax_profile_v.set_ylabel('Elevation')
    ax_profile_v.set_title('Vertical Cross-section (Center Column)')
    ax_profile_v.legend()
    ax_profile_v.grid(True, alpha=0.3)
    
    # 5. statistics comparison
    ax_stats = plt.subplot(4, 2, 7)
    
    stats_data = []
    labels = []
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i in range(n_dems):
        orig_valid = original_dems[i][valid_mask]
        corr_valid = corrected_dems[i][valid_mask]
        
        # Remove NaN values
        orig_valid = orig_valid[~np.isnan(orig_valid)]
        corr_valid = corr_valid[~np.isnan(corr_valid)]
        
        orig_std = np.std(orig_valid)
        corr_std = np.std(corr_valid)
        
        stats_data.extend([orig_std, corr_std])
        labels.extend([f'{dem_names[i]}\n(Original)', f'{dem_names[i]}\n(Flattened)'])
    
    bars = ax_stats.bar(range(len(stats_data)), stats_data, 
                        color=[colors[i//2] for i in range(len(stats_data))],
                        alpha=0.8)
    # Optionally set individual alpha values per bar
    for i, bar in enumerate(bars):
        bar.set_alpha(0.5 if i%2==0 else 1.0)
    
    ax_stats.set_xlabel('DEM')
    ax_stats.set_ylabel('Standard Deviation')
    ax_stats.set_title('Elevation Variability (Before vs After)')
    ax_stats.set_xticks(range(len(labels)))
    ax_stats.set_xticklabels(labels, rotation=45, ha='right')
    ax_stats.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, stats_data):
        height = bar.get_height()
        ax_stats.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                     f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 6. elevation histograms
    ax_hist = plt.subplot(4, 2, 8)
    
    for i in range(n_dems):
        orig_valid = original_dems[i][valid_mask]
        corr_valid = corrected_dems[i][valid_mask]
        
        # Remove NaN values
        orig_valid = orig_valid[~np.isnan(orig_valid)]
        corr_valid = corr_valid[~np.isnan(corr_valid)]
        
        ax_hist.hist(orig_valid, bins=50, alpha=0.4, 
                    label=f'{dem_names[i]} (original)', color=colors[i], density=True)
        ax_hist.hist(corr_valid, bins=50, alpha=0.7, 
                    label=f'{dem_names[i]} (flattened)', color=colors[i], 
                    density=True, histtype='step', linewidth=2)
    
    ax_hist.set_xlabel('Elevation')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Elevation Distribution')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"visualization saved to: {save_path}")

    plt.show()

    return fig

if __name__ == "__main__":
    # example DEM file list
    dem_files = [
        "dem1.tif",
        "dem2.tif",
        "dem3.tif"
    ]

    print(f"processing {len(dem_files)} DEMs")

    # process DEMs consistently
    result = process_multiple_dems_consistently(
        dem_files,
        edge_threshold=-500,      # adjust based on your edge artifacts
        correction_strength=1.0,   # full correction
        polynomial_degree=2,       # try degree 4 for better curve fitting
        reference_index=0          # use first DEM as reference
    )

    if result is not None:
        corrected_dems, output_paths, valid_mask, original_dems = result

        # create comprehensive visualization
        dem_names = [Path(p).stem for p in dem_files]

        print("\ncreating comprehensive before/after visualization:")
        fig1 = compare_dems_visualization(
            original_dems, corrected_dems, dem_names, valid_mask,
        )

        print("Correction summary statistics:")

        for i, name in enumerate(dem_names):
            orig_valid = original_dems[i][valid_mask]
            corr_valid = corrected_dems[i][valid_mask]

            # remove NaN values
            orig_valid = orig_valid[~np.isnan(orig_valid)]
            corr_valid = corr_valid[~np.isnan(corr_valid)]

            if len(orig_valid) > 0 and len(corr_valid) > 0:
                orig_std = np.std(orig_valid)
                corr_std = np.std(corr_valid)
                reduction = ((orig_std - corr_std) / orig_std) * 100

                print(f"\n{name}:")
                print(f"  original std dev: {orig_std:.3f} m")
                print(f"  corrected std dev: {corr_std:.3f} m")
                print(f"  variability reduction: {reduction:.1f}%")