import h5py
import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN

class EventClusterTracker:
    def __init__(self):
        """
        Pure event-based tracker using DBSCAN to identify dense motion clusters.
        """
        # Current tracking state
        self.tracked_clusters = {}
        self.next_id = 0
        
        # DBSCAN parameters optimized for dense motion clustering
        self.dbscan_eps = 12.0  # Pixels within 12px are in same cluster
        self.dbscan_min_samples = 200  # Need at least 30 event pixels for valid cluster
        self.density_threshold = 500  # Only keep clusters with at least 40 event pixels
        self.max_track_distance = 50  # Maximum distance for frame-to-frame tracking
        self.track_timeout = 15  # Frames before losing track
        
        print("Event Cluster Tracker initialized")
    
    def cluster_dense_event_pixels(self, event_mask, frame_idx):
        """
        Use DBSCAN to find DENSE clusters of event pixels representing moving objects.
        """
        # Get coordinates of ALL event pixels
        y_coords, x_coords = np.where(event_mask > 0)
        
        if len(x_coords) < self.dbscan_min_samples:
            return []
        
        print(f"Frame {frame_idx}: Processing {len(x_coords)} event pixels")
        
        # Prepare event pixel coordinates for DBSCAN
        event_coordinates = np.column_stack([x_coords, y_coords])
        
        # Apply DBSCAN on pixel coordinates
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        cluster_labels = dbscan.fit_predict(event_coordinates)
        
        # Process clusters and keep only dense ones
        unique_labels = np.unique(cluster_labels)
        dense_clusters = []
        
        noise_count = np.sum(cluster_labels == -1)
        total_clustered = np.sum(cluster_labels != -1)
        valid_clusters = len([l for l in unique_labels if l != -1])
        
        print(f"  Found {valid_clusters} raw clusters, {total_clustered} clustered pixels, {noise_count} noise")
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise
                continue
            
            # Get pixels in this cluster
            cluster_mask = (cluster_labels == cluster_id)
            cluster_x = x_coords[cluster_mask]
            cluster_y = y_coords[cluster_mask]
            cluster_size = len(cluster_x)
            
            # DENSITY FILTER: Only keep sufficiently dense clusters
            if cluster_size < self.density_threshold:
                continue
            
            # Calculate cluster properties
            centroid_x = np.mean(cluster_x)
            centroid_y = np.mean(cluster_y)
            
            # Calculate cluster spread for radius estimation
            distances = np.sqrt((cluster_x - centroid_x)**2 + (cluster_y - centroid_y)**2)
            max_distance = np.max(distances)
            std_distance = np.std(distances)
            
            # Adaptive radius based on cluster spread
            radius = max(15, min(std_distance * 2.5 + 10, 80))
            
            # Calculate density metrics
            cluster_area = np.pi * (max_distance ** 2) if max_distance > 0 else 1
            density = cluster_size / cluster_area
            
            cluster_info = {
                'id': cluster_id,
                'centroid': [centroid_x, centroid_y],
                'radius': radius,
                'pixel_count': cluster_size,
                'density': density,
                'max_spread': max_distance,
                'std_spread': std_distance,
                'pixel_coords': (cluster_x, cluster_y),
                'frame': frame_idx
            }
            
            dense_clusters.append(cluster_info)
            print(f"    Dense Cluster: {cluster_size} pixels at ({centroid_x:.1f},{centroid_y:.1f}), "
                  f"density={density:.2f}, radius={radius:.1f}")
        
        print(f"  Kept {len(dense_clusters)} clusters with ≥60 pixels (dense motion only)")
        return dense_clusters
    
    def track_clusters_across_frames(self, current_clusters, frame_idx):
        """
        Track dense clusters across frames using centroid matching.
        """
        # Update existing tracks and find matches
        matched_tracks = set()
        new_assignments = []
        
        for cluster in current_clusters:
            cluster_centroid = cluster['centroid']
            best_track_id = None
            min_distance = float('inf')
            
            # Find closest existing track
            for track_id, track_info in self.tracked_clusters.items():
                if not track_info['active']:
                    continue
                
                last_position = track_info['path'][-1]
                distance = np.sqrt((cluster_centroid[0] - last_position[0])**2 + 
                                 (cluster_centroid[1] - last_position[1])**2)
                
                if distance < min_distance and distance < self.max_track_distance:
                    min_distance = distance
                    best_track_id = track_id
            
            # Assign cluster to track
            if best_track_id is not None:
                # Update existing track
                track = self.tracked_clusters[best_track_id]
                track['path'].append(cluster_centroid)
                track['current_cluster'] = cluster
                track['last_seen'] = 0
                track['frames_tracked'] += 1
                matched_tracks.add(best_track_id)
                new_assignments.append((best_track_id, cluster, min_distance))
            else:
                # Create new track
                new_track_id = self.next_id
                self.next_id += 1
                self.tracked_clusters[new_track_id] = {
                    'path': [cluster_centroid],
                    'current_cluster': cluster,
                    'first_seen': frame_idx,
                    'last_seen': 0,
                    'frames_tracked': 1,
                    'active': True,
                    'cluster_history': [cluster]
                }
                matched_tracks.add(new_track_id)
                new_assignments.append((new_track_id, cluster, 0))
                print(f"  NEW TRACK #{new_track_id} started at ({cluster_centroid[0]:.1f},{cluster_centroid[1]:.1f})")
        
        # Update unmatched tracks (increment last_seen)
        tracks_to_deactivate = []
        for track_id, track_info in self.tracked_clusters.items():
            if track_id not in matched_tracks and track_info['active']:
                track_info['last_seen'] += 1
                if track_info['last_seen'] > self.track_timeout:
                    tracks_to_deactivate.append(track_id)
                    track_info['active'] = False
                    print(f"  LOST TRACK #{track_id} (timeout)")
        
        return new_assignments
    
    def draw_dense_clusters_and_tracks(self, rgb_frame, event_mask, dense_clusters, frame_idx):
        """
        Visualize:
        1. All event pixels as light green overlay
        2. Dense clusters as green circles with pixel counts
        3. Tracked objects as colored circles with track IDs and paths
        """
        output_frame = rgb_frame.copy()
        
        # 1. Overlay ALL event pixels in light green
        event_pixels = (event_mask > 0)
        if np.any(event_pixels):
            overlay = output_frame.copy()
            overlay[event_pixels] = [0, 255, 0]  # Green
            cv2.addWeighted(overlay, 0.2, output_frame, 0.8, 0, output_frame)
        
        # 2. Draw dense cluster circles (green for single, cyan for merged)
        for cluster in dense_clusters:
            center = tuple(map(int, cluster['centroid']))
            radius = int(cluster['radius'])
            
            # Draw cluster boundary
            cv2.circle(output_frame, center, radius, (0, 255, 0), 2)
            
            # Show cluster info
            pixel_count = cluster['pixel_count']
            density = cluster['density']
            cv2.putText(output_frame, f"C:{pixel_count}", 
                       (center[0] - 25, center[1] - radius - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(output_frame, f"D:{density:.1f}", 
                       (center[0] - 25, center[1] - radius - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 3. Draw tracked objects with unique colors and paths
        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
                 (0, 255, 255), (128, 0, 255), (255, 128, 0), (128, 255, 0)]
        
        active_tracks = [tid for tid, track in self.tracked_clusters.items() if track['active']]
        
        for i, track_id in enumerate(active_tracks):
            track = self.tracked_clusters[track_id]
            color = colors[i % len(colors)]
            
            # Draw track path
            if len(track['path']) > 1:
                for j in range(1, len(track['path'])):
                    pt1 = tuple(map(int, track['path'][j-1]))
                    pt2 = tuple(map(int, track['path'][j]))
                    cv2.line(output_frame, pt1, pt2, color, 2)
            
            # Draw current position
            if len(track['path']) > 0:
                current_pos = track['path'][-1]
                center = tuple(map(int, current_pos))
                
                # Get radius from current cluster if available
                if track['current_cluster'] is not None:
                    radius = int(track['current_cluster']['radius'])
                else:
                    radius = 25
                
                # Draw track circle
                cv2.circle(output_frame, center, radius, color, 3)
                
                # Draw track ID and info
                cv2.putText(output_frame, f"T#{track_id}", 
                           (center[0] - 20, center[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                frames_tracked = track['frames_tracked']
                cv2.putText(output_frame, f"F:{frames_tracked}", 
                           (center[0] - 15, center[1] + radius + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 4. Frame statistics
        total_event_pixels = np.sum(event_pixels)
        active_track_count = len(active_tracks)
        info_text = f"Frame {frame_idx} | Events: {total_event_pixels} | Dense Clusters: {len(dense_clusters)} | Tracks: {active_track_count}"
        cv2.putText(output_frame, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output_frame

def read_h5_events(filepath):
    """Read events from HDF5 file."""
    print("Reading events from H5 file...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"H5 file not found: {filepath}")

    with h5py.File(filepath, 'r') as f:
        events_data = f['events'][:]
        t = events_data[:, 0]  # timestamp
        x = events_data[:, 1]  # x coordinate  
        y = events_data[:, 2]  # y coordinate
        p = events_data[:, 3]  # polarity
        
    print(f"Loaded {len(t)} events")
    return x, y, p, t

def events_to_masks(x, y, p, t, frame_times, resolution=(346, 260), threshold=1):
    """Convert events to binary masks for each frame."""
    width, height = resolution
    masks = []

    print(f"Event time range: {t.min():.0f} to {t.max():.0f} microseconds")
    print(f"Frame time range: {frame_times[0]:.0f} to {frame_times[-1]:.0f} microseconds")

    frame_times = frame_times.ravel().astype(np.int64)

    for i in range(len(frame_times) - 1):
        t_start = frame_times[i]
        t_end = frame_times[i + 1]
        
        # Find events in time window
        event_mask = (t >= t_start) & (t < t_end)
        binary_mask = np.zeros((height, width), dtype=np.uint8)

        if np.any(event_mask):
            frame_x, frame_y = x[event_mask], y[event_mask]
            
            # Count events per pixel
            for xi, yi in zip(frame_x, frame_y):
                if 0 <= yi < height and 0 <= xi < width:
                    binary_mask[yi, xi] += 1

            # Apply threshold
            binary_mask = (binary_mask >= threshold).astype(np.uint8)

        if i % 100 == 0:
            print(f"Processing frame {i}/{len(frame_times)-1}, events: {np.sum(event_mask)}")

        masks.append(binary_mask)

    return masks

def create_event_cluster_tracking_video(input_video_path, event_masks, output_path):
    """
    Create video showing pure event-based DBSCAN clustering and tracking:
    - Light green overlay for all event pixels
    - Green circles for dense clusters with pixel counts
    - Colored circles and paths for tracked motion clusters
    """
    # Open RGB input video for background
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open RGB video: {input_video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(event_masks))

    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")

    # Initialize tracker
    tracker = EventClusterTracker()

    # Processing statistics
    total_dense_clusters = 0
    total_tracks_created = 0

    print("Processing frames for dense event clustering...")
    frame_idx = 0
    
    while cap.isOpened() and frame_idx < len(event_masks):
        ret, rgb_frame = cap.read()
        if not ret:
            break

        # Get corresponding event mask
        event_mask = event_masks[frame_idx]
        
        # Resize event mask to match RGB frame if needed
        if event_mask.shape[:2] != (height, width):
            event_mask = cv2.resize(event_mask, (width, height), 
                                  interpolation=cv2.INTER_NEAREST)

        # Find dense clusters of event pixels using DBSCAN
        dense_clusters = tracker.cluster_dense_event_pixels(event_mask, frame_idx)
        total_dense_clusters += len(dense_clusters)

        # Track clusters across frames
        track_assignments = tracker.track_clusters_across_frames(dense_clusters, frame_idx)

        # Count new tracks created this frame
        new_tracks_this_frame = len([a for a in track_assignments if tracker.tracked_clusters[a[0]]['frames_tracked'] == 1])
        total_tracks_created += new_tracks_this_frame

        # Create visualization
        output_frame = tracker.draw_dense_clusters_and_tracks(rgb_frame, event_mask, dense_clusters, frame_idx)

        # Write frame
        out.write(output_frame)
        frame_idx += 1

        if frame_idx % 25 == 0:
            active_tracks = len([t for t in tracker.tracked_clusters.values() if t['active']])
            print(f"Frame {frame_idx}/{total_frames}: {len(dense_clusters)} dense clusters, {active_tracks} active tracks")

    # Cleanup
    cap.release()
    out.release()

    # Final statistics
    active_tracks = len([t for t in tracker.tracked_clusters.values() if t['active']])
    total_tracks = len(tracker.tracked_clusters)
    
    print(f"\nProcessing complete!")
    print(f"Frames processed: {frame_idx}")
    print(f"Total dense clusters found: {total_dense_clusters}")
    print(f"Total tracks created: {total_tracks}")
    print(f"Final active tracks: {active_tracks}")
    print(f"Average clusters per frame: {total_dense_clusters/frame_idx:.1f}")
    print(f"Output saved: {output_path}")

if __name__ == "__main__":
    # File paths
    input_video = "../sample_videos_to_test/clip_002.mp4"  # RGB input video (for background)
    h5_path = "../v2e-output/traffic_scenev.h5"  # Event data
    output_path = "../v2e-output/event_cluster_tracking.mp4"
    
    print("File paths:")
    print(f"RGB input video: {os.path.abspath(input_video)}")
    print(f"H5 events file: {os.path.abspath(h5_path)}")
    print(f"Output video: {os.path.abspath(output_path)}")

    # Read event data
    x, y, p, t = read_h5_events(h5_path)

    # Load frame timestamps
    frame_times_path = "../v2e1-output/dvs_preview-frame_times.txt"
    try:
        print(f"\nLoading frame times: {os.path.abspath(frame_times_path)}")
        frame_times = np.loadtxt(frame_times_path)
        
        if frame_times.ndim > 1:
            timestamps = frame_times[:, 1] if frame_times.shape[1] > 1 else frame_times[:, 0]
        else:
            timestamps = frame_times
            
        if np.max(timestamps) < 1000:
            timestamps = timestamps * 1e6  # Convert to microseconds
            
        frame_times = timestamps
        print(f"Loaded {len(frame_times)} timestamps")
        
    except (FileNotFoundError, IOError) as e:
        print(f"Using fallback timing: {str(e)}")
        frame_interval = 33000  # 30 FPS
        frame_times = np.arange(t[0], t[-1], frame_interval)

    # Generate event masks
    print("\nGenerating event masks...")
    event_masks = events_to_masks(x, y, p, t, frame_times, threshold=1)

    # Create tracking video
    print("\nCreating Event Cluster Tracking video...")
    create_event_cluster_tracking_video(input_video, event_masks, output_path)