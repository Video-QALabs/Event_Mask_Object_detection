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
DELTA_T = 100000    # in microseconds

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

PLOT_CLUSTERS = True

def show_clusters_interactive(vis_img, num_clusters):
    """Show or update a persistent window for cluster visualization.
    Press space to advance (update image in same window). Press 'q' to quit.
    """
    state = globals().setdefault("_clusters_state", {})

    def _on_key(ev):
        if ev.key in (" ", "space"):
            state["proceed"] = True
        elif ev.key == "q":
            # Close window and exit entirely
            if "fig" in state and plt.fignum_exists(state["fig"].number):
                plt.close(state["fig"])
            sys.exit(0)

    # Create persistent figure on first call, otherwise update image data
    if "fig" not in state or not plt.fignum_exists(state["fig"].number):
        state["fig"] = plt.figure(figsize=(16, 9), dpi=120)
        state["fig"].canvas.mpl_connect("key_press_event", _on_key)
        state["ax"] = state["fig"].add_axes([0.02, 0.04, 0.96, 0.94])
        state["im"] = state["ax"].imshow(vis_img)
    else:
        state["im"].set_data(vis_img)

    state["ax"].set_title(f"DBSCAN Clusters - {int(num_clusters)} clusters detected - press 'space' for next, 'q' to quit")
    plt.draw()

    state["proceed"] = False
    while True:
        if not plt.fignum_exists(state["fig"].number):
            # Window closed by user; exit to stop the loop
            sys.exit(0)
        if state.get("proceed"):
            break
        plt.pause(0.05)


# --- Simple centroid-based tracker to persist cluster IDs across batches ---
_tracker_state = {
    "tracks": {},   # id -> {centroid: (x,y), bbox: (x,y,w,h), last_seen: frame_idx}
    "next_id": 0,
}


def _euclid2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def reassign_labels_by_centroid(stats, frame_idx, max_centroid_dist=50.0):
    """Assign stable IDs to current clusters by nearest previous centroid.

    Returns list of tuples (track_id, stat) and updates global _tracker_state.
    """
    global _tracker_state
    tracks = _tracker_state["tracks"]
    used_prev = set()
    labeled = []

    max_d2 = max_centroid_dist * max_centroid_dist

    # Greedy nearest matching to previous tracks
    for st in stats:
        c = st.centroid  # (cx, cy)
        best_id = None
        best_d2 = None
        for tid, tinfo in tracks.items():
            if tid in used_prev:
                continue
            d2 = _euclid2(c, tinfo["centroid"])
            if d2 <= max_d2 and (best_d2 is None or d2 < best_d2):
                best_d2 = d2
                best_id = tid

        if best_id is None:
            # Create new track
            tid = _tracker_state["next_id"]
            _tracker_state["next_id"] += 1
        else:
            tid = best_id
            used_prev.add(tid)

        # Update track info
        tracks[tid] = {"centroid": c, "bbox": st.bbox, "last_seen": frame_idx, "size": st.size}
        labeled.append((tid, st))

    # Optionally prune stale tracks (not seen for N frames)
    stale_horizon = 30
    to_del = [tid for tid, tinfo in tracks.items() if frame_idx - tinfo.get("last_seen", frame_idx) > stale_horizon]
    for tid in to_del:
        tracks.pop(tid, None)

    return labeled


def draw_labeled_boxes(img, labeled_stats):
    vis = img.copy()
    for tid, st in labeled_stats:
        x, y, w_box, h_box = st.bbox
        cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (0, 255, 0), 1)
        cx, cy = int(st.centroid[0]), int(st.centroid[1])
        cv2.circle(vis, (cx, cy), 2, (0, 255, 255), -1)
        cv2.putText(vis, f"id={tid} n={st.size}", (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, lineType=cv2.LINE_AA)
    return vis

def merge_overlapping_clusters(stats):
    """Merge clusters whose bounding boxes overlap into combined clusters.

    Returns a new list of ClusterStat with merged bbox, weighted centroid, and summed size.
    """
    def _bbox_xyxy(b):
        x, y, w, h = b
        return x, y, x + w - 1, y + h - 1


    def _boxes_overlap(b1, b2):
        x1, y1, x2, y2 = _bbox_xyxy(b1)
        X1, Y1, X2, Y2 = _bbox_xyxy(b2)
        return not (x2 < X1 or X2 < x1 or y2 < Y1 or Y2 < y1)

    if not stats:
        return stats

    try:
        from dbscan import ClusterStat
    except Exception:
        ClusterStat = None  # type: ignore

    n = len(stats)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_overlap(stats[i].bbox, stats[j].bbox):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        groups.append(comp)

    merged = []
    for comp in groups:
        if len(comp) == 1:
            merged.append(stats[comp[0]])
            continue
        # Merge component
        total_size = sum(stats[k].size for k in comp)
        if total_size == 0:
            total_size = 1
        cx = sum(stats[k].centroid[0] * stats[k].size for k in comp) / total_size
        cy = sum(stats[k].centroid[1] * stats[k].size for k in comp) / total_size
        # Union bbox
        xs1, ys1, xs2, ys2 = [], [], [], []
        for k in comp:
            x1, y1, x2, y2 = _bbox_xyxy(stats[k].bbox)
            xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
        ux1, uy1, ux2, uy2 = min(xs1), min(ys1), max(xs2), max(ys2)
        ubbox = (int(ux1), int(uy1), int(ux2 - ux1 + 1), int(uy2 - uy1 + 1))
        # Choose a representative label (min of group) if dataclass is available
        rep_label = min(stats[k].label for k in comp)
        if ClusterStat is not None:
            merged.append(ClusterStat(label=int(rep_label), size=int(total_size), centroid=(float(cx), float(cy)), bbox=ubbox))
        else:
            # Fallback to a dict-like object if dataclass is not importable
            merged.append(type("_Cluster", (), {"label": int(rep_label), "size": int(total_size), "centroid": (float(cx), float(cy)), "bbox": ubbox})())

    return merged


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
        frame = visualize_batches_frame(events)

        # Run DBSCAN on the frame
        frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), sigmaX=0, sigmaY=0)
        mask, pts, labels = run_dbscan_on_image(frame_blur, eps=10.0, min_samples=30, th=32, morph_open=0)
        # Compute stats from points/labels
        from dbscan import compute_cluster_stats  # local import from same folder
        stats = compute_cluster_stats(labels, pts)
        # Merge overlapping clusters
        stats = merge_overlapping_clusters(stats)
        # Reassign stable IDs
        labeled_stats = reassign_labels_by_centroid(stats, fidx, max_centroid_dist=20.0)
        print(f"Detected {len(labeled_stats)} clusters in the frame")
        # Draw with stable IDs
        vis = draw_labeled_boxes(frame, labeled_stats)

        if not PLOT_CLUSTERS:
            cv2.imwrite(f"{args.outdir}/frame_{fidx:05d}.png", vis)
            if fidx and fidx % 100 == 0:
                print(f"Processed {fidx} frames, last had {len(labeled_stats)} clusters")
            continue

        # Show/update a persistent window instead of creating a new one
        show_clusters_interactive(vis, len(labeled_stats))

        # print(type(events), events.size)
        # print(np.asarray(events).shape)
        # break



