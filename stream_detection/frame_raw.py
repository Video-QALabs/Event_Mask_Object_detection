import os,sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("/usr/lib/python3/dist-packages")

import hdf5plugin
os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGIN_PATH

from metavision_core.event_io import RawReader

os.environ["MV_HAL_PLUGIN_PATH"] = "/lib/metavision/hal/plugins"
raw_filepath= "/mnt/SSD_1TB/EventBoard/REC3/cam2/rec.raw"
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

def visualize_batches_frame(events):
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

    if "fig" not in state or not plt.fignum_exists(state["fig"].number):
        state["fig"] = plt.figure(figsize=(16, 9), dpi=120)
        state["fig"].canvas.mpl_connect("key_press_event", _on_key)
        state["ax"] = state["fig"].add_axes([0.02, 0.04, 0.96, 0.94])
        state["im"] = state["ax"].imshow(frame)
    else:
        state["im"].set_data(frame)

    state["ax"].set_title(f"Batch of {int(events.size)} events - press 'n' for next, 'q' to quit")

    plt.draw()
    state["proceed"] = False
    while True:
        if not plt.fignum_exists(state["fig"].number):
            sys.exit(0)
        if state.get("proceed"):
            break
        plt.pause(0.05)


if __name__ == "__main__":
    reader = RawReader(raw_filepath)
    for events in event_batch_generator(reader):
        # visualize_batches_3D(events)

        visualize_batches_frame(events)

        # print(type(events), events.size)
        # print(np.asarray(events).shape)
        # break



