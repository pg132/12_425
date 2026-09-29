import h5py
import numpy as np


# ============================================================
# Configuration
# ============================================================

path = "/Users/picogilman/Desktop/MIT/8.290/coding/sector36_time_flux.mat"

n_pixels = 1000
TIME_OFFSET = 2459200


# ============================================================
# Load data
# ============================================================

with h5py.File(path, "r") as f:
    time = f["time"][:].squeeze()
    flux = f["flux"]


    n_sensors, n_times = flux.shape

    print(f"Total sensors: {n_sensors}")
    print(f"Total time points: {n_times}")

    # First 100 pixels
    pixel_flux = flux[:n_pixels, :]

    # Convert to floating point
    pixel_flux = pixel_flux.astype(float)

    # Treat zero and non-finite values as missing
    pixel_flux[(pixel_flux == 0) | ~np.isfinite(pixel_flux)] = np.nan

    # ========================================================
    # Define valid time ranges
    # ========================================================

    adjusted_time = time - TIME_OFFSET

    time_valid = (
        ((adjusted_time >= 83) & (adjusted_time <= 92.5))
        | (adjusted_time >= 96)
    )

    

    print(
        f"Using {np.sum(time_valid)} of {n_times} time points "
        f"for normalization and drop detection."
    )


    # ========================================================
    # Normalize each pixel by its own mean
    # over the valid time ranges
    # ========================================================

    # ========================================================
    # Pixel baseline: mean of the middle 80% of valid values
    # ========================================================

    pixel_mean = np.full(n_pixels, np.nan)

    for i in range(n_pixels):
        values = pixel_flux[i, time_valid]
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        values = np.sort(values)

        trim = int(0.10 * len(values))

        if 2 * trim >= len(values):
            pixel_mean[i] = np.nanmean(values)
        else:
            middle = values[trim:len(values) - trim]
            pixel_mean[i] = np.mean(middle)


    # ========================================================
    # Toss out pixels whose mean flux is outside [100, 10000]
    # ========================================================

    pixel_valid = (
        np.isfinite(pixel_mean)
        & (pixel_mean >= 100)
        & (pixel_mean <= 1e4)
    )

    # Keep track of the original pixel numbers
    original_pixel_indices = np.arange(n_pixels)[pixel_valid]

    pixel_flux = pixel_flux[pixel_valid]
    pixel_mean = pixel_mean[pixel_valid]

    n_pixels = len(pixel_mean)

    print(f"Keeping {n_pixels} pixels after mean-flux filtering.")


    # ========================================================
    # Normalize each pixel by its own mean
    # ========================================================

    pixel_normalized = pixel_flux / pixel_mean[:, None]


    # ========================================================
    # Compute sensor average at each time
    # ========================================================

    sensor_average = np.full(n_times, np.nan)

    sensor_average[time_valid] = np.nanmean(
        pixel_normalized[:, time_valid],
        axis=0
    )


    # ========================================================
    # Sensor shift relative to maximum valid sensor average
    # ========================================================

    sensor_max = np.nanmax(sensor_average)

    sensor_shift = sensor_average / sensor_max


    # ========================================================
    # Correct each pixel for sensor-wide changes
    # ========================================================

    corrected = pixel_normalized / sensor_shift[None, :]


    # ========================================================
    # Find downward deviations
    # ========================================================

    drops = []

    for i in range(n_pixels):
        valid = (
            np.isfinite(corrected[i])
            & time_valid
        )

        drop = 1.0 - corrected[i]

        # Only keep downward deviations
        valid &= drop > 0

        indices = np.where(valid)[0]

        for j in indices:
            drops.append(
                (
                    i,
                    j,
                    time[j],
                    pixel_flux[i, j],
                    pixel_mean[i],
                    pixel_normalized[i, j],
                    sensor_average[j],
                    sensor_shift[j],
                    corrected[i, j],
                    drop[j],
                )
            )


# ============================================================
# Convert to structured NumPy array
# ============================================================

drops = np.array(
    drops,
    dtype=[
        ("pixel", int),
        ("time_index", int),
        ("time", float),
        ("raw_flux", float),
        ("pixel_mean", float),
        ("pixel_normalized", float),
        ("sensor_average", float),
        ("sensor_shift", float),
        ("corrected_flux", float),
        ("drop", float),
    ],
)


# ============================================================
# Sort by largest drop
# ============================================================

drops = np.sort(drops, order="drop")[::-1]


# ============================================================
# Output
# ============================================================

print("\nLargest drops:")

print(
    f"{'Pixel':>5}  "
    f"{'Time index':>10}  "
    f"{'BJD-2459200':>12}  "
    f"{'Raw flux':>10}  "
    f"{'Pixel mean':>11}  "
    f"{'Pixel norm':>11}  "
    f"{'Sensor avg':>11}  "
    f"{'Sensor shift':>12}  "
    f"{'Corrected':>10}  "
    f"{'Drop':>9}"
)

print("-" * 125)

for row in drops[:100]:
    print(
        f"{row['pixel']:5d}  "
        f"{row['time_index']:10d}  "
        f"{row['time'] - TIME_OFFSET:12.5f}  "
        f"{row['raw_flux']:10.3f}  "
        f"{row['pixel_mean']:11.3f}  "
        f"{row['pixel_normalized']:11.6f}  "
        f"{row['sensor_average']:11.6f}  "
        f"{row['sensor_shift']:12.6f}  "
        f"{row['corrected_flux']:10.6f}  "
        f"{row['drop']:8.4%}"
    )


