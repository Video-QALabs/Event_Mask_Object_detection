import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def read_h5_events(filepath):
    """Read events from HDF5 file produced by v2e."""
    print("Reading events from H5...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"H5 file not found: {filepath}")

    with h5py.File(filepath, 'r') as f:
        # First, let's examine the file structure in detail
        print("\nH5 file structure:")
        print("==================")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Type: {obj.dtype}")
                print(f"  Attributes: {list(obj.attrs.keys())}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
                print(f"  Attributes: {list(obj.attrs.keys())}")
        f.visititems(print_structure)
        print("==================\n")

        # After printing structure, pause execution
        input("Press Enter to continue with data reading...")
        
        # Now we can examine the events dataset
        events_dataset = f['events']
        print(f"\nEvents dataset type: {type(events_dataset)}")
        print(f"Events dataset shape: {events_dataset.shape}")
        print(f"Events dataset dtype: {events_dataset.dtype}")
        
        # We'll pause here to examine the output before proceeding
        input("Press Enter to continue with data extraction...")
        
        # Read all events and split into arrays based on v2e format [t,x,y,p]
        events_data = events_dataset[:]  # Shape: (N, 4)
        t = events_data[:, 0]  # timestamp
        x = events_data[:, 1]  # x coordinate
        y = events_data[:, 2]  # y coordinate
        p = events_data[:, 3]  # polarity
        
        print("\nEvent statistics:")
        print(f"X range: {x.min()}-{x.max()}")
        print(f"Y range: {y.min()}-{y.max()}")
        print(f"Polarity range: {p.min()}-{p.max()}")
        print(f"Time range: {t.min()}-{t.max()} (duration: {(t[-1]-t[0])/1e6:.3f} s)")
        
    print(f"\nLoaded {len(x)} events")
    return x, y, p, t


def events_to_frames(x, y, p, t, frame_times=None, resolution=(346, 260)):
    """Convert events into frames by binning events."""
    width, height = resolution
    frames = []

    if frame_times is None:
        # Create frames with 33ms duration (approx. 30 fps)
        duration_us = 33000  # 33ms in microseconds
        frame_times = np.arange(t[0], t[-1], duration_us)
        print(f"\nGenerating frames:")
        print(f"  Frame duration: 33ms")
        print(f"  Number of frames: {len(frame_times)}")

    for i in range(len(frame_times) - 1):
        t_start, t_end = frame_times[i], frame_times[i + 1]
        mask = (t >= t_start) & (t < t_end)

        img = np.zeros((height, width, 3), dtype=np.uint8)

        if np.any(mask):
            xs, ys, ps = x[mask], y[mask], p[mask]

            # Positive events → Blue
            img[ys[ps == 1], xs[ps == 1]] = [0, 150, 255]

            # Negative events → White
            img[ys[ps == 0], xs[ps == 0]] = [255, 255, 255]

        frames.append(img)

    return frames


def visualize_frames(frames):
    """Animate frames using matplotlib."""
    if not frames:
        print("No frames to visualize.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    img_plot = ax.imshow(frames[0])
    ax.set_title("Event Frame 0")

    def update(idx):
        img_plot.set_array(frames[idx])
        ax.set_title(f"Event Frame {idx}")
        return img_plot,

    ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
    plt.show()


if __name__ == "__main__":
    h5_path = "../v2e-output/traffic_scenev.h5"
    frame_times_path = "../v2e-output/dvs_preview-frame_times.txt"

    try:
        frame_times = np.loadtxt(frame_times_path)
        print(f"Loaded {len(frame_times)} frame timestamps")
    except (FileNotFoundError, IOError):
        print("No frame_times.txt found, using fixed 33ms bins.")
        frame_times = None  # events_to_frames will handle this case
    frame_times_path = "../v2e-output/dvs-video-frame_times.txt"  # optional

    print("Reading events from H5...")
    x, y, p, t = read_h5_events(h5_path)

    if os.path.exists(frame_times_path):
        frame_times = np.loadtxt(frame_times_path)
        print(f"Loaded {len(frame_times)} frame times.")
    else:
        frame_times = None
        print("No frame_times.txt found, using fixed 33ms bins.")

    print(f"Loaded {len(t)} events, generating frames...")
    frames = events_to_frames(x, y, p, t, frame_times, resolution=(346, 260))
    visualize_frames(frames)





