import h5py
import numpy as np
import matplotlib.pyplot as plt

path = "/Users/picogilman/Desktop/MIT/8.290/coding/sector36_time_flux.mat"

with h5py.File(path, "r") as f:
    time = f["time"][:].squeeze()
    flux = f["flux"]

    n_sensors, n_times = flux.shape

    # Pick 100 random sensors
    rng = np.random.default_rng(44)
    sensor_indices = np.sort(
        rng.choice(n_sensors, size=100, replace=False)
    )

    average_flux = np.empty(n_times)

    for j in range(n_times):

        # Print progress every 100 time steps
        if j % 100 == 0:
            print(f"Processing time step {j}/{n_times}")

        # Get the 100 selected sensors at this time
        values = flux[sensor_indices, j]

        # Remove zeros and NaNs/infinities
        values = values[(values != 0) & np.isfinite(values)]

        if len(values) == 0:
            average_flux[j] = np.nan
            continue

        # Sort the remaining values
        values = np.sort(values)

        # Throw out bottom 25% and top 25%
        lower = len(values) // 4
        upper = len(values) - lower

        # Average the middle 50%
        average_flux[j] = np.mean(values[lower:upper])

# Convert time to BJD - 2459200
time_plot = time - 2459200

# Clamp plotted flux to 0–3000
average_flux_plot = np.clip(average_flux, 0, 3000)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(time_plot, average_flux_plot, ".", markersize=2)

plt.xlabel("BJD − 2459200")
plt.ylabel("Mean flux")
plt.title("Average flux of 100 randomly selected sensors")

plt.ylim(0, 3000)

plt.tight_layout()
plt.show()