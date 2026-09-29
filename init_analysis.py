import h5py
import numpy as np
import matplotlib.pyplot as plt


path = "/Users/picogilman/Desktop/MIT/8.290/coding/sector36_time_flux.mat"

with h5py.File(path, "r") as f:
    print(list(f.keys()))

with h5py.File(path, "r") as f:

    print("Datasets:")
    for name in ["TIC_ID", "flux", "time"]:
        d = f[name]
        print(f"  {name}: shape={d.shape}, dtype={d.dtype}")

    # Read the time array
    time = f["time"][:].squeeze()

    print("\nTime:")
    print("  first 10:", time[:10])
    print("  last 10:", time[-10:])
    print("  range:", time.min(), "to", time.max())
    print("  number of points:", len(time))

    # Read just ONE light curve
    flux = f["flux"][0, :]

    print("\nFirst light curve:")
    print("  first 10:", flux[:10])
    print("  min:", np.nanmin(flux))
    print("  max:", np.nanmax(flux))
    print("  mean:", np.nanmean(flux))

    time_plot = time - 2450000

    plt.plot(time_plot, flux, ".", markersize=2)
    plt.xlabel("BJD − 2450000")
    plt.ylabel("Flux")
    plt.title("First light curve")
    plt.show()