import os,sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/usr/lib/python3/dist-packages")

import hdf5plugin
os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGIN_PATH

from metavision_core.event_io import RawReader

from dbscan import run_dbscan_on_image, overlay_clusters
import cv2
import argparse

os.environ["MV_HAL_PLUGIN_PATH"] = "/lib/metavision/hal/plugins"
# raw_filepath= "/mnt/SSD_1TB/EventBoard/REC3/cam2/rec.raw"
DELTA_T = 100000  # in microseconds

STREAM_H, STREAM_W = 720, 1280

def event_batch_generator(reader, delta_t=DELTA_T, max_batches=None):
    yielded = 0
    while not reader.is_done():
        events = reader.load_delta_t(delta_t)
        if events.size:
            yield events
            yielded += 1
            if max_batches is not None and yielded >= max_batches:
                break

def visualize_batches_3D(events):
    _state = globals().setdefault("_ev3d_state", {})

    def _on_key(ev):
        if ev.key == "n":
            _state["proceed"] = True
        elif ev.key == "q":
            plt.close(_state["fig"])

    # Randomly discard 50% of event tuples in this batch
    n = events.size

    keep_n = n//5  # discard floor(n/5), keep ceil(n/5)
    keep_idx = np.random.permutation(n)[:keep_n]
    events = events[keep_idx]
        
    x = events['x'].astype('u2')
    y = events['y'].astype('u2')
    t = events['t'].astype('i8')
    p = events['p'].astype('i2')

    
    colors = np.where(p.astype(bool), "red", "blue")
    if "fig" not in _state:
        _state["fig"] = plt.figure(figsize=(16, 9), dpi=120)
        _state["fig"].canvas.mpl_connect("key_press_event", _on_key)
        _state["ax"] = _state["fig"].add_axes([0.02, 0.04, 0.96, 0.94], projection='3d')
        _state["ax"].xaxis.labelpad = 6
        _state["ax"].yaxis.labelpad = 6
        _state["ax"].zaxis.labelpad = 6
        # _state["ax"] = _state["fig"].add_subplot(projection='3d')

    _state["ax"].scatter(t, x, y, c=colors, s=1, marker='o', depthshade=False)

    _state["ax"].set_xlabel("t (us)")
    _state["ax"].set_ylabel("x")
    _state["ax"].set_zlabel("y")    

    _state["ax"].set_title(f"Batch of {int(events.size)} events - press 'n' for next, 'q' to quit")

    plt.draw()
    _state["proceed"] = False
    while True:
        if not plt.fignum_exists(_state["fig"].number):
            sys.exit(0)
        if _state.get("proceed"):
            _state["ax"].cla()
            break
        plt.pause(0.05)

def visualize_batches_frame(events, to_plot=False):
    state = globals().setdefault("_ev2d_state", {})

    def _on_key(ev):
        if ev.key == "n":
            state["proceed"] = True
        elif ev.key == "q":
            plt.close(state["fig"])

    frame = np.zeros((STREAM_H, STREAM_W, 3), dtype=np.uint8)
    x = events["x"]
    y = events["y"]
    p = events["p"].astype(bool)

    frame[y[p], x[p]] = (255, 0, 0)
    frame[y[~p], x[~p]] = (0, 0, 255)

    if to_plot and ("fig" not in state or not plt.fignum_exists(state["fig"].number)):
        state["fig"] = plt.figure(figsize=(16, 9), dpi=120)
        state["fig"].canvas.mpl_connect("key_press_event", _on_key)
        state["ax"] = state["fig"].add_axes([0.02, 0.04, 0.96, 0.94])
        state["im"] = state["ax"].imshow(frame)
    elif to_plot:
        state["im"].set_data(frame)
    
    if to_plot:
        state["ax"].set_title(f"Batch of {int(events.size)} events - press 'n' for next, 'q' to quit")
        plt.draw()    
        while True:
            if not plt.fignum_exists(state["fig"].number):
                sys.exit(0)
            if state.get("proceed"):
                break
            plt.pause(0.05)
    else:
        state["proceed"] = True
        return frame

PLOT_CLUSTERS = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an event RAW file and write output frames.")
    parser.add_argument("rawfile", help="Path to the input .raw file")
    parser.add_argument("outdir", help="Directory to save output frames")
    args = parser.parse_args()

    raw_filepath = args.rawfile

    os.makedirs(args.outdir, exist_ok=True)

    reader = RawReader(raw_filepath)    
    for fidx, events in enumerate(event_batch_generator(reader)):
        # print(f"Processing frame {fidx}")
        # visualize_batches_3D(events)
        frame= visualize_batches_frame(events)        

        # Run DBSCAN on the frame
        _, clustered_img, cluster_stats = run_dbscan_on_image(frame, eps=10.0, min_samples=30, th=32, morph_open=0)
        # print(f"Detected {len(cluster_stats)} clusters in the frame")

        vis, stats= overlay_clusters(frame, clustered_img, cluster_stats, draw_boxes=True)
        
        if not PLOT_CLUSTERS:
            cv2.imwrite(f"{args.outdir}/frame_{fidx:05d}.png", vis)
            # print(f"Saved frame_{fidx:05d}.png with {len(cluster_stats)} clusters")
            if fidx and fidx % 100 == 0:
                print(f"Processed {fidx} frames, last had {len(cluster_stats)} clusters")
            continue
        
        plt.figure(figsize=(16, 9), dpi=120)
        plt.imshow(vis)
        plt.title(f"DBSCAN Clusters - {len(cluster_stats)} clusters detected - press 'n' for next, 'q' to quit")

        fig = plt.gcf()
        _wait_state = {"go": False}
        _wait_state["quit"] = False

        def _on_space(ev):
            if ev.key in (" ", "space"):
                _wait_state["go"] = True
            if ev.key == "q":
                plt.close(fig)
                _wait_state["quit"] = True

        fig.canvas.mpl_connect("key_press_event", _on_space)

        while True:
            if not plt.fignum_exists(fig.number):
                break
            if _wait_state["go"]:
                plt.close(fig)
                break
            if _wait_state.get("quit"):
                sys.exit(0)
            plt.pause(0.05)
                
        # print(type(events), events.size)
        # print(np.asarray(events).shape)
        # break



