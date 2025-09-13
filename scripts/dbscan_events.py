import h5py
import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from ultralytics import RTDETR
from scipy import stats
import matplotlib.pyplot as plt
# Initialize YOLO model
yolo_model = RTDETR("rtdetr-l.pt")

class SACAAugmentedEventTracker:
    def __init__(self):
        """
        Enhanced event tracker using SACA algorithm with YOLO semantic labeling for moving objects.
        """
        # Current tracking state
        self.tracked_clusters = {}
        self.next_id = 0
        
        # Object detection results - now updated every 15 frames
        self.current_detections = []
        self.detection_boxes = []  # [(x1, y1, x2, y2, class_name, confidence)]
        self.last_detection_frame = -1
        self.detection_interval = 15  # Run YOLO every 15 frames
        
        # Motion-relevant class filter
        self.motion_classes = {'person', 'car', 'bus', 'truck', 'motorcycle', 'bicycle', 'train'}
        
        # SACA parameters
        self.saca_C = 3  # Minimum cluster size parameter
        self.use_center = False  # Use nearest neighbor instead of cluster center for noise reassignment
        
        # DBSCAN fallback parameters
        self.dbscan_eps = 15.0
        self.dbscan_min_samples = 50
        self.density_threshold = 100
        
        # Tracking parameters
        self.max_track_distance = 50
        self.track_timeout = 15
        
        # Semantic matching parameters
        self.confidence_threshold = 0.5
        
        print("SACA-Augmented Event Tracker with YOLO integration initialized")
        print(f"Tracking motion classes: {self.motion_classes}")
        print(f"YOLO detection interval: every {self.detection_interval} frames")
    
    def update_detections(self, detections, frame_idx):
        """Update object detection results from current frame, filtering for motion-relevant classes."""
        self.current_detections = detections
        self.detection_boxes = []
        self.last_detection_frame = frame_idx
        
        print(f"\nProcessing {len(detections)} detections for frame {frame_idx}:")
        motion_count = 0
        for i, (x1, y1, x2, y2, class_name, confidence) in enumerate(detections):
            if confidence >= self.confidence_threshold:
                if class_name.lower() in self.motion_classes:
                    self.detection_boxes.append((x1, y1, x2, y2, class_name, confidence))
                    motion_count += 1
                    print(f"  MOTION Detection {i}: {class_name} ({confidence:.2f}) at [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
                else:
                    print(f"  Skipped (static): {class_name} ({confidence:.2f})")
        
        print(f"Kept {motion_count} motion-relevant detections from {len(detections)} total")
    
    def should_run_detection(self, frame_idx):
        """Check if we should run YOLO detection on this frame."""
        return (frame_idx % self.detection_interval == 0) or (frame_idx == 0)
    
    def get_detection_age(self, frame_idx):
        """Get how many frames old the current detections are."""
        if self.last_detection_frame == -1:
            return float('inf')
        return frame_idx - self.last_detection_frame
    
    def modified_z_score_filter(self, data, threshold=3.5):
        """Remove outliers using modified Z-score with NaN handling."""
        if len(data) == 0:
            return np.array([], dtype=int)
        
        # Remove any NaN or infinite values
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return np.array([], dtype=int)
        
        median = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median))
        
        if mad == 0:  # All values are the same
            return np.array([], dtype=int)
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        return np.where(outlier_mask)[0]
    
    def saca_clustering(self, event_coordinates, frame_idx):
        """
        Implement SACA clustering algorithm with robust error handling.
        """
        N = len(event_coordinates)
        if N < 10:
            return []
        
        print(f"Frame {frame_idx}: Running SACA on {N} event pixels")
        
        # Remove duplicate points
        unique_coords, unique_indices = np.unique(event_coordinates, axis=0, return_index=True)
        if len(unique_coords) < N:
            print(f"  Removed {N - len(unique_coords)} duplicate points")
            event_coordinates = unique_coords
            N = len(event_coordinates)
        
        if N < 10:
            return []
        
        # Initialize
        cluster_number = np.zeros(N, dtype=int)
        noise_list = []
        non_noise_list = []
        neighbor_cands = [[] for _ in range(N)]
        
        # Calculate Distance Matrix with robust handling
        try:
            dist_matrix = pairwise_distances(event_coordinates, metric='euclidean')
        except Exception as e:
            print(f"  Distance matrix failed: {e}, using DBSCAN")
            return self.dbscan_fallback(event_coordinates, frame_idx)
        
        # Find minimum distances with robust handling
        # Create a mask for the diagonal and set those to infinity
        diag_mask = np.eye(N, dtype=bool)
        dist_matrix_masked = dist_matrix.copy()
        dist_matrix_masked[diag_mask] = np.inf
        
        # Find minimum distances (excluding self)
        mins = np.min(dist_matrix_masked, axis=1)
        
        # Check for valid distances
        if not np.all(np.isfinite(mins)) or np.any(mins <= 0) or np.all(mins == 0):
            print(f"  Invalid distances detected, using DBSCAN fallback")
            return self.dbscan_fallback(event_coordinates, frame_idx)
        
        # Remove outliers
        try:
            outlier_idx = self.modified_z_score_filter(mins)
            valid_mins = np.delete(mins, outlier_idx)
            
            if len(valid_mins) == 0:
                valid_mins = mins
        except Exception as e:
            print(f"  Outlier filtering failed: {e}")
            valid_mins = mins
        
        # Calculate SACA parameters
        sigma_opt = np.min(valid_mins) if len(valid_mins) > 0 else 1.0
        L = np.max(valid_mins) if len(valid_mins) > 0 else 10.0
        
        # Ensure sigma_opt is not zero or too small
        if sigma_opt <= 0 or not np.isfinite(sigma_opt):
            sigma_opt = 1.0
        
        if L <= sigma_opt or not np.isfinite(L):
            L = sigma_opt * 10.0
        
        T = max(1, int((L / sigma_opt) / 2)) if sigma_opt > 0 else 10
        threshold_distance = sigma_opt * T / 2
        
        print(f"  SACA params: sigma_opt={sigma_opt:.2f}, L={L:.2f}, T={T}, thresh={threshold_distance:.2f}")
        
        # Separate points into dense and sparse areas
        for i in range(N):
            neighbors = np.where(dist_matrix[i] <= threshold_distance)[0]
            neighbors = neighbors[neighbors != i]  # Remove self
            neighbor_cands[i] = neighbors.tolist()
            
            if len(neighbors) <= self.saca_C:
                cluster_number[i] = -1  # Mark as noise
                noise_list.append(i)
            else:
                non_noise_list.append(i)
        
        if len(non_noise_list) == 0:
            print(f"  No dense regions found, using DBSCAN fallback")
            return self.dbscan_fallback(event_coordinates, frame_idx)
        
        print(f"  Found {len(non_noise_list)} core points, {len(noise_list)} noise points")
        
        # Label core shapes (connected components)
        c = 0
        while True:
            unassigned = [i for i in non_noise_list if cluster_number[i] == 0]
            if not unassigned:
                break
            
            c += 1
            s = unassigned[0]
            Q = [s]
            
            while Q:
                i = Q.pop(0)
                cluster_number[i] = c
                
                for j in neighbor_cands[i]:
                    if cluster_number[j] == 0:
                        cluster_number[j] = c
                        Q.append(j)
        
        # Re-assign noise samples
        for i in noise_list:
            if len(non_noise_list) > 0:
                non_noise_distances = dist_matrix[i, non_noise_list]
                if len(non_noise_distances) > 0:
                    nearest_idx = non_noise_list[np.argmin(non_noise_distances)]
                    cluster_number[i] = cluster_number[nearest_idx]
        
        # Convert to cluster format
        clusters = []
        unique_labels = np.unique(cluster_number)
        unique_labels = unique_labels[unique_labels > 0]  # Remove noise
        
        for cluster_id in unique_labels:
            cluster_mask = (cluster_number == cluster_id)
            cluster_points = event_coordinates[cluster_mask]
            
            if len(cluster_points) < self.density_threshold:
                continue
            
            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            radius = max(15, min(np.std(distances) * 2.5 + 10, 80))
            
            cluster_info = {
                'id': cluster_id,
                'centroid': centroid.tolist(),
                'radius': radius,
                'pixel_count': len(cluster_points),
                'density': len(cluster_points) / (np.pi * radius**2),
                'max_spread': np.max(distances),
                'std_spread': np.std(distances),
                'pixel_coords': (cluster_points[:, 0], cluster_points[:, 1]),
                'frame': frame_idx
            }
            
            clusters.append(cluster_info)
        
        print(f"  SACA found {len(clusters)} dense clusters")
        return clusters
    
    def dbscan_fallback(self, event_coordinates, frame_idx):
        """Fallback to DBSCAN if SACA fails."""
        print(f"  Using DBSCAN fallback")
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        cluster_labels = dbscan.fit_predict(event_coordinates)
        
        clusters = []
        unique_labels = np.unique(cluster_labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_mask = (cluster_labels == cluster_id)
            cluster_points = event_coordinates[cluster_mask]
            
            if len(cluster_points) < self.density_threshold:
                continue
            
            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            radius = max(15, min(np.std(distances) * 2.5 + 10, 80))
            
            cluster_info = {
                'id': cluster_id,
                'centroid': centroid.tolist(),
                'radius': radius,
                'pixel_count': len(cluster_points),
                'density': len(cluster_points) / (np.pi * radius**2),
                'max_spread': np.max(distances),
                'std_spread': np.std(distances),
                'pixel_coords': (cluster_points[:, 0], cluster_points[:, 1]),
                'frame': frame_idx
            }
            
            clusters.append(cluster_info)
        
        return clusters
    
    def match_cluster_to_detection(self, cluster, frame_idx):
        """Match cluster to YOLO detection for semantic labeling (motion objects only)."""
        if not self.detection_boxes:
            return None, None, 0.0

        cx, cy = cluster['centroid']
        best_match = None
        best_conf = 0
        detection_age = self.get_detection_age(frame_idx)

        # Apply confidence penalty based on detection age
        age_penalty = max(0.1, 1.0 - (detection_age * 0.05))  # Reduce confidence as detections get older

        for x1, y1, x2, y2, class_name, conf in self.detection_boxes:
            # Only consider motion-relevant classes
            if class_name.lower() in self.motion_classes:
                if x1 <= cx <= x2 and y1 <= cy <= y2:  # centroid inside box
                    adjusted_conf = conf * age_penalty
                    if adjusted_conf > best_conf:
                        best_conf = adjusted_conf
                        best_match = (class_name, conf)

        if best_match:
            return best_match[0], best_match[1], age_penalty
        return None, None, 0.0
    
    def cluster_dense_event_pixels(self, event_mask, frame_idx):
        """Use SACA to find dense clusters with semantic labeling for motion objects."""
        y_coords, x_coords = np.where(event_mask > 0)
        
        if len(x_coords) < 50:
            return []
        
        event_coordinates = np.column_stack([x_coords, y_coords])
        
        # Try SACA first, fallback to DBSCAN if needed
        try:
            dense_clusters = self.saca_clustering(event_coordinates, frame_idx)
        except Exception as e:
            print(f"  SACA failed: {str(e)}, using DBSCAN fallback")
            dense_clusters = self.dbscan_fallback(event_coordinates, frame_idx)
        
        # Add semantic labeling to clusters (motion objects only)
        motion_clusters = []
        detection_age = self.get_detection_age(frame_idx)
        
        for cluster in dense_clusters:
            class_name, confidence, overlap = self.match_cluster_to_detection(cluster, frame_idx)
            cluster['class_name'] = class_name
            cluster['detection_confidence'] = confidence
            cluster['detection_overlap'] = overlap
            cluster['detection_age'] = detection_age
            
            # Only keep clusters that match motion objects or are large enough to be potentially moving
            if class_name or cluster['pixel_count'] > 200:  # Keep unmatched but large clusters
                motion_clusters.append(cluster)
                age_info = f" (age: {detection_age})" if detection_age < float('inf') else " (no detections)"
                semantic_info = f" -> {class_name} ({confidence:.2f}){age_info}" if class_name else " -> MOTION?"
                print(f"    Cluster: {cluster['pixel_count']} pixels at ({cluster['centroid'][0]:.1f},{cluster['centroid'][1]:.1f}){semantic_info}")
        
        print(f"  Kept {len(motion_clusters)}/{len(dense_clusters)} clusters (motion-relevant)")
        return motion_clusters
    
    def track_clusters_across_frames(self, current_clusters, frame_idx):
        """Track clusters across frames with semantic consistency."""
        matched_tracks = set()
        new_assignments = []
        
        for cluster in current_clusters:
            cluster_centroid = cluster['centroid']
            cluster_class = cluster['class_name']
            
            best_track_id = None
            min_distance = float('inf')
            
            # Find closest existing track with matching semantics
            for track_id, track_info in self.tracked_clusters.items():
                if not track_info['active']:
                    continue
                
                # Check semantic consistency for motion objects
                track_class = track_info.get('class_name')
                if track_class and cluster_class and track_class != cluster_class:
                    continue
                
                last_position = track_info['path'][-1]
                distance = np.sqrt((cluster_centroid[0] - last_position[0])**2 + 
                                 (cluster_centroid[1] - last_position[1])**2)
                
                if distance < min_distance and distance < self.max_track_distance:
                    min_distance = distance
                    best_track_id = track_id
            
            # Assign cluster to track
            if best_track_id is not None:
                track = self.tracked_clusters[best_track_id]
                track['path'].append(cluster_centroid)
                track['current_cluster'] = cluster
                track['last_seen'] = 0
                track['frames_tracked'] += 1
                
                # Update class info if we have better/newer detection
                if cluster_class and (not track.get('class_name') or 
                                    cluster.get('detection_age', float('inf')) < track.get('last_detection_age', float('inf'))):
                    track['class_name'] = cluster_class
                    track['detection_confidence'] = cluster['detection_confidence']
                    track['last_detection_age'] = cluster.get('detection_age', float('inf'))
                
                matched_tracks.add(best_track_id)
                new_assignments.append((best_track_id, cluster, min_distance))
            else:
                # Create new track for motion objects
                new_track_id = self.next_id
                self.next_id += 1
                self.tracked_clusters[new_track_id] = {
                    'path': [cluster_centroid],
                    'current_cluster': cluster,
                    'first_seen': frame_idx,
                    'last_seen': 0,
                    'frames_tracked': 1,
                    'active': True,
                    'cluster_history': [cluster],
                    'class_name': cluster_class,
                    'detection_confidence': cluster.get('detection_confidence', 0.0),
                    'last_detection_age': cluster.get('detection_age', float('inf')),
                    'clustering_method': 'SACA'
                }
                matched_tracks.add(new_track_id)
                new_assignments.append((new_track_id, cluster, 0))
                
                class_info = f" ({cluster_class})" if cluster_class else " (MOTION?)"
                print(f"  NEW MOTION TRACK #{new_track_id}{class_info} at ({cluster_centroid[0]:.1f},{cluster_centroid[1]:.1f})")
        
        # Update unmatched tracks
        for track_id, track_info in self.tracked_clusters.items():
            if track_id not in matched_tracks and track_info['active']:
                track_info['last_seen'] += 1
                if track_info['last_seen'] > self.track_timeout:
                    track_info['active'] = False
                    class_info = f" ({track_info.get('class_name', 'MOTION')})"
                    print(f"  LOST MOTION TRACK #{track_id}{class_info} (timeout)")
        
        return new_assignments
    
    def draw_semantic_clusters_and_tracks(self, rgb_frame, event_mask, dense_clusters, frame_idx):
        """Visualize SACA clusters with semantic labels and tracking for motion objects."""
        output_frame = rgb_frame.copy()
        detection_age = self.get_detection_age(frame_idx)
        
        # 1. Draw current detection boxes for motion objects (when available)
        if detection_age < self.detection_interval:  # Only show recent detections
            for x1, y1, x2, y2, class_name, confidence in self.detection_boxes:
                # Motion objects get bright cyan boxes
                alpha = max(0.3, 1.0 - (detection_age * 0.05))  # Fade with age
                color = (0, 255, 255)
                thickness = max(1, int(3 * alpha))
                
                cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                label = f"MOTION: {class_name}: {confidence:.2f} (age:{detection_age})"
                cv2.putText(output_frame, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 2. Overlay event pixels in light green with higher opacity for visibility
        event_pixels = (event_mask > 0)
        if np.any(event_pixels):
            overlay = output_frame.copy()
            overlay[event_pixels] = [50, 255, 50]  # Brighter light green
            cv2.addWeighted(overlay, 0.3, output_frame, 0.7, 0, output_frame)  # 30% opacity
        
        # 3. Draw SACA clusters with motion-specific coloring
        for cluster in dense_clusters:
            center = tuple(map(int, cluster['centroid']))
            radius = int(cluster['radius'])
            class_name = cluster['class_name']
            detection_age = cluster.get('detection_age', float('inf'))
            
            # Color based on motion object type and detection freshness
            if class_name == 'person':
                color = (0, 255, 0)  # Green for person
                label_color = (0, 255, 0)
            elif class_name in ['car', 'truck', 'bus']:
                color = (255, 0, 0)  # Blue for vehicles
                label_color = (255, 0, 0)
            elif class_name:
                color = (0, 150, 255)  # Orange for other motion objects
                label_color = (0, 150, 255)
            else:
                color = (0, 255, 255)  # Cyan for unidentified motion
                label_color = (0, 255, 255)
            
            # Adjust intensity based on detection age
            if detection_age < float('inf'):
                intensity = max(0.5, 1.0 - (detection_age * 0.03))
                color = tuple(int(c * intensity) for c in color)
                label_color = tuple(int(c * intensity) for c in label_color)
            
            # Draw cluster boundary with thicker lines for motion objects
            cv2.circle(output_frame, center, radius, color, 3)
            
            # Enhanced labels for motion objects
            class_label = class_name if class_name else "MOTION?"
            age_label = f" (age:{detection_age})" if detection_age < float('inf') else ""
            cv2.putText(output_frame, f"SACA: {class_label}{age_label}", 
                       (center[0] - 50, center[1] - radius - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
            cv2.putText(output_frame, f"Events:{cluster['pixel_count']}", 
                       (center[0] - 35, center[1] - radius - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        
        # 4. Draw motion tracks with enhanced visualization
        motion_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)]
        
        active_tracks = [tid for tid, track in self.tracked_clusters.items() if track['active']]
        
        for i, track_id in enumerate(active_tracks):
            track = self.tracked_clusters[track_id]
            color = motion_colors[i % len(motion_colors)]
            
            # Draw motion path with thicker lines
            if len(track['path']) > 1:
                for j in range(1, len(track['path'])):
                    pt1 = tuple(map(int, track['path'][j-1]))
                    pt2 = tuple(map(int, track['path'][j]))
                    cv2.line(output_frame, pt1, pt2, color, 3)
            
            # Draw current position for motion objects
            if len(track['path']) > 0:
                current_pos = track['path'][-1]
                center = tuple(map(int, current_pos))
                
                radius = int(track['current_cluster']['radius']) if track['current_cluster'] else 25
                
                # Draw motion track circle with enhanced styling
                cv2.circle(output_frame, center, radius, color, 4)
                
                # Enhanced track labels for motion
                class_name = track.get('class_name', 'MOTION')
                track_age = track.get('last_detection_age', float('inf'))
                age_info = f" (age:{track_age})" if track_age < float('inf') else ""
                track_label = f"T#{track_id}: {class_name}{age_info}"
                
                # Add motion indicator
                cv2.putText(output_frame, "MOTION", 
                           (center[0] - 30, center[1] - radius - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(output_frame, track_label, 
                           (center[0] - 50, center[1] - radius - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 5. Enhanced frame statistics for motion tracking
        total_event_pixels = np.sum(event_pixels)
        active_track_count = len(active_tracks)
        labeled_clusters = len([c for c in dense_clusters if c['class_name']])
        motion_objects = len(set([track.get('class_name') for track in self.tracked_clusters.values() 
                                if track['active'] and track.get('class_name')]))
        
        # Detection status indicator
        detection_status = f"Fresh" if detection_age == 0 else f"Age:{detection_age}" if detection_age < float('inf') else "No Detections"
        
        info_text = f"MOTION TRACKING - Frame {frame_idx} | Events: {total_event_pixels} | Motion Clusters: {len(dense_clusters)} ({labeled_clusters} labeled) | Active Tracks: {active_track_count} | Object Types: {motion_objects} | YOLO: {detection_status}"
        
        # Draw info background for better visibility
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(output_frame, (5, 5), (text_size[0] + 15, 30), (0, 0, 0), -1)
        cv2.putText(output_frame, info_text, (10, 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return output_frame

def run_object_detection(rgb_frame, frame_idx, save_path=None):
    """Run YOLO detection on the given frame and save visualization with motion focus."""
    results = yolo_model(rgb_frame, verbose=False)
    detections = []
    motion_classes = {'person', 'car', 'bus', 'truck', 'motorcycle', 'bicycle', 'train'}
    
    # Create a copy for visualization if saving
    if save_path:
        vis_frame = rgb_frame.copy()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = yolo_model.names[cls_id]
            detections.append((x1, y1, x2, y2, class_name, conf))

            # Enhanced visualization for motion objects (only if saving)
            if save_path:
                if class_name.lower() in motion_classes:
                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
                    cv2.putText(vis_frame, f"MOTION: {class_name} {conf:.2f}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # Draw static objects with thinner gray lines
                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 1)
                    cv2.putText(vis_frame, f"STATIC: {class_name} {conf:.2f}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    if save_path:
        cv2.imwrite(save_path, vis_frame)
        print(f"Saved frame {frame_idx} YOLO detections (motion focus) to {save_path}")
    
    return detections

def create_saca_semantic_tracking_video(input_video_path, event_masks, output_path):
    """Create video with SACA clustering and YOLO semantic labeling for moving objects."""
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

    # Initialize SACA tracker for motion objects
    tracker = SACAAugmentedEventTracker()

    print("Processing frames with SACA clustering and YOLO motion object tracking...")
    print(f"YOLO detection will run every {tracker.detection_interval} frames...")
    
    frame_idx = 0
    detection_frames = []
    
    while cap.isOpened() and frame_idx < len(event_masks):
        ret, rgb_frame = cap.read()
        if not ret:
            break

        # Run YOLO detection every 15th frame (and first frame)
        if tracker.should_run_detection(frame_idx):
            save_path = f"saca_motion_frame_{frame_idx}.jpg" if frame_idx % 45 == 0 else None  # Save every 45th frame image
            detections = run_object_detection(rgb_frame, frame_idx, save_path)
            tracker.update_detections(detections, frame_idx)
            detection_frames.append(frame_idx)
            print(f"*** YOLO DETECTION RUN ON FRAME {frame_idx} ***")

        # Get corresponding event mask
        event_mask = event_masks[frame_idx]
        
        # Resize event mask if needed
        if event_mask.shape[:2] != (height, width):
            event_mask = cv2.resize(event_mask, (width, height), 
                                  interpolation=cv2.INTER_NEAREST)

        # Find dense clusters using SACA with motion object focus
        dense_clusters = tracker.cluster_dense_event_pixels(event_mask, frame_idx)

        # Track clusters across frames with motion-specific logic
        track_assignments = tracker.track_clusters_across_frames(dense_clusters, frame_idx)

        # Create visualization with motion object emphasis
        output_frame = tracker.draw_semantic_clusters_and_tracks(rgb_frame, event_mask, 
                                                               dense_clusters, frame_idx)

        # Write frame
        out.write(output_frame)
        frame_idx += 1

        if frame_idx % 25 == 0:
            active_tracks = len([t for t in tracker.tracked_clusters.values() if t['active']])
            motion_tracks = len([t for t in tracker.tracked_clusters.values() 
                               if t['active'] and t.get('class_name')])
            detection_age = tracker.get_detection_age(frame_idx)
            print(f"Frame {frame_idx}/{total_frames}: {len(dense_clusters)} motion clusters, "
                  f"{active_tracks} tracks ({motion_tracks} identified), detection age: {detection_age}")

    # Cleanup
    cap.release()
    out.release()

    # Final statistics
    active_tracks = len([t for t in tracker.tracked_clusters.values() if t['active']])
    motion_identified = len([t for t in tracker.tracked_clusters.values() 
                           if t.get('class_name')])
    saca_tracks = len([t for t in tracker.tracked_clusters.values() 
                      if t.get('clustering_method') == 'SACA'])
    
    print(f"\nSACA + YOLO Motion Object Tracking Complete!")
    print(f"YOLO detections run on frames: {detection_frames}")
    print(f"Total detection runs: {len(detection_frames)}")
    print(f"Total motion tracks created: {len(tracker.tracked_clusters)}")
    print(f"SACA-based tracks: {saca_tracks}")
    print(f"Motion objects identified: {motion_identified}")
    print(f"Final active motion tracks: {active_tracks}")
    print(f"Output saved: {output_path}")


def read_h5_events(filepath):
        """Read events from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            events_data = f['events'][:]
            t = events_data[:, 0]  # timestamp
            x = events_data[:, 1]  # x coordinate
            y = events_data[:, 2]  # y coordinate
            p = events_data[:, 3]  # polarity
        return x, y, p, t

def create_event_masks_from_h5(events_filepath, video_width, video_height, 
                              frame_rate=30, total_frames=None, accumulation_time=33.33):
    """
    Create event masks from HDF5 events file with temporal accumulation.
    
    Args:
        events_filepath: Path to HDF5 events file
        video_width: Width of output frames
        video_height: Height of output frames  
        frame_rate: Video frame rate (fps)
        total_frames: Maximum number of frames to process
        accumulation_time: Time window for accumulating events (ms)
    
    Returns:
        List of event masks (numpy arrays)
    """
    print(f"Loading events from: {events_filepath}")
    x, y, p, t = read_h5_events(events_filepath)
    
    print(f"Loaded {len(x)} events")
    print(f"Time range: {t[0]:.2f} to {t[-1]:.2f} microseconds")
    print(f"Spatial range: x=[{x.min()},{x.max()}], y=[{y.min()},{y.max()}]")
    
    # Convert time to milliseconds
    t_ms = t / 1000.0
    duration_ms = t_ms[-1] - t_ms[0]
    
    # Calculate frame parameters
    frame_duration_ms = 1000.0 / frame_rate
    if total_frames is None:
        total_frames = int(duration_ms / frame_duration_ms) + 1
    
    print(f"Creating {total_frames} frames with {frame_duration_ms:.2f}ms per frame")
    print(f"Event accumulation window: {accumulation_time:.2f}ms")
    
    event_masks = []
    start_time = t_ms[0]
    
    for frame_idx in range(total_frames):
        # Calculate time window for this frame
        frame_start = start_time + frame_idx * frame_duration_ms
        frame_end = frame_start + accumulation_time
        
        # Find events in this time window
        time_mask = (t_ms >= frame_start) & (t_ms < frame_end)
        frame_events = np.sum(time_mask)
        
        # Create event mask
        event_mask = np.zeros((video_height, video_width), dtype=np.uint8)
        
        if frame_events > 0:
            frame_x = x[time_mask]
            frame_y = y[time_mask]
            
            # Clip coordinates to frame bounds
            frame_x = np.clip(frame_x, 0, video_width - 1)
            frame_y = np.clip(frame_y, 0, video_height - 1)
            
            # Accumulate events at each pixel
            for px, py in zip(frame_x, frame_y):
                event_mask[int(py), int(px)] = min(255, event_mask[int(py), int(px)] + 1)
        
        event_masks.append(event_mask)
        
        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx}/{total_frames}: {frame_events} events")
    
    print(f"Created {len(event_masks)} event masks")
    return event_masks


def visualize_tracking_statistics(tracker, save_path="tracking_stats.png"):
    """Create visualization of tracking statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Track duration histogram
    durations = [track['frames_tracked'] for track in tracker.tracked_clusters.values()]
    if durations:
        axes[0, 0].hist(durations, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Track Duration Distribution')
        axes[0, 0].set_xlabel('Frames Tracked')
        axes[0, 0].set_ylabel('Number of Tracks')
    
    # Object class distribution
    classes = [track.get('class_name', 'Unknown') for track in tracker.tracked_clusters.values() 
               if track.get('class_name')]
    if classes:
        from collections import Counter
        class_counts = Counter(classes)
        axes[0, 1].bar(class_counts.keys(), class_counts.values())
        axes[0, 1].set_title('Detected Object Classes')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Track path lengths
    path_lengths = [len(track['path']) for track in tracker.tracked_clusters.values()]
    if path_lengths:
        axes[1, 0].hist(path_lengths, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Track Path Length Distribution')
        axes[1, 0].set_xlabel('Path Length')
        axes[1, 0].set_ylabel('Number of Tracks')
    
    # Detection confidence distribution
    confidences = [track.get('detection_confidence', 0) for track in tracker.tracked_clusters.values()
                   if track.get('detection_confidence', 0) > 0]
    if confidences:
        axes[1, 1].hist(confidences, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Detection Confidence Distribution')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Number of Detections')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tracking statistics saved to: {save_path}")


def export_tracking_results(tracker, output_path="tracking_results.txt"):
    """Export detailed tracking results to text file."""
    with open(output_path, 'w') as f:
        f.write("=== SACA + YOLO Motion Object Tracking Results ===\n\n")
        
        # Summary statistics
        total_tracks = len(tracker.tracked_clusters)
        active_tracks = len([t for t in tracker.tracked_clusters.values() if t['active']])
        identified_tracks = len([t for t in tracker.tracked_clusters.values() 
                                if t.get('class_name')])
        
        f.write(f"Total tracks created: {total_tracks}\n")
        f.write(f"Currently active tracks: {active_tracks}\n")
        f.write(f"Tracks with object identification: {identified_tracks}\n\n")
        
        # Object class summary
        classes = [track.get('class_name') for track in tracker.tracked_clusters.values() 
                   if track.get('class_name')]
        if classes:
            from collections import Counter
            class_counts = Counter(classes)
            f.write("Object Classes Detected:\n")
            for class_name, count in sorted(class_counts.items()):
                f.write(f"  {class_name}: {count} tracks\n")
            f.write("\n")
        
        # Detailed track information
        f.write("=== Detailed Track Information ===\n\n")
        
        for track_id, track in tracker.tracked_clusters.items():
            f.write(f"Track #{track_id}:\n")
            f.write(f"  Class: {track.get('class_name', 'Unknown')}\n")
            f.write(f"  Status: {'Active' if track['active'] else 'Inactive'}\n")
            f.write(f"  Frames tracked: {track['frames_tracked']}\n")
            f.write(f"  Path length: {len(track['path'])}\n")
            f.write(f"  First seen: Frame {track['first_seen']}\n")
            f.write(f"  Detection confidence: {track.get('detection_confidence', 0):.3f}\n")
            f.write(f"  Clustering method: {track.get('clustering_method', 'Unknown')}\n")
            
            # Path coordinates (first and last few points)
            path = track['path']
            if len(path) > 0:
                f.write(f"  Start position: ({path[0][0]:.1f}, {path[0][1]:.1f})\n")
                f.write(f"  End position: ({path[-1][0]:.1f}, {path[-1][1]:.1f})\n")
                
                # Calculate total distance traveled
                total_distance = 0
                for i in range(1, len(path)):
                    dx = path[i][0] - path[i-1][0]
                    dy = path[i][1] - path[i-1][1]
                    total_distance += np.sqrt(dx*dx + dy*dy)
                f.write(f"  Total distance traveled: {total_distance:.1f} pixels\n")
            
            f.write("\n")
    
    print(f"Detailed tracking results exported to: {output_path}")


def main():
    """Main execution function."""
    # Configuration
    RGB_VIDEO_PATH = "path/to/your/rgb_video.mp4"  # Update this path
    EVENTS_H5_PATH = "path/to/your/events.h5"      # Update this path
    OUTPUT_VIDEO_PATH = "saca_motion_tracking_output.mp4"
    
    # Video parameters
    VIDEO_WIDTH = 640   # Update based on your video
    VIDEO_HEIGHT = 480  # Update based on your video
    FRAME_RATE = 30
    MAX_FRAMES = None   # Process all frames, or set a limit like 500
    
    try:
        print("=== SACA + YOLO Motion Object Tracking System ===\n")
        
        # Step 1: Create event masks from HDF5 file
        print("Step 1: Creating event masks from HDF5 events...")
        event_masks = create_event_masks_from_h5(
            EVENTS_H5_PATH, 
            VIDEO_WIDTH, 
            VIDEO_HEIGHT,
            frame_rate=FRAME_RATE,
            total_frames=MAX_FRAMES,
            accumulation_time=33.33  # ~30 FPS accumulation window
        )
        
        if len(event_masks) == 0:
            print("ERROR: No event masks created!")
            return
        
        # Step 2: Process video with SACA clustering and YOLO tracking
        print("\nStep 2: Processing video with SACA + YOLO motion tracking...")
        create_saca_semantic_tracking_video(
            RGB_VIDEO_PATH,
            event_masks,
            OUTPUT_VIDEO_PATH
        )
        
        print(f"\n✓ Processing complete!")
        print(f"✓ Output video saved: {OUTPUT_VIDEO_PATH}")
        print(f"✓ Individual frame samples saved as: saca_motion_frame_*.jpg")
        
        # Optional: Generate additional analysis
        print("\nGenerating additional analysis files...")
        
        # Note: You'd need to modify this to access the tracker instance
        # from create_saca_semantic_tracking_video if you want these features
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def process_batch_videos(video_folder, events_folder, output_folder):
    """Process multiple video files in batch."""
    import glob
    
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        events_path = os.path.join(events_folder, f"{video_name}.h5")
        output_path = os.path.join(output_folder, f"{video_name}_saca_tracking.mp4")
        
        if not os.path.exists(events_path):
            print(f"WARNING: Events file not found for {video_name}, skipping...")
            continue
        
        print(f"\n=== Processing {video_name} ===")
        
        try:
            # You'll need to adjust these parameters based on your videos
            event_masks = create_event_masks_from_h5(
                events_path, 640, 480, frame_rate=30, max_frames=1000
            )
            
            create_saca_semantic_tracking_video(
                video_path, event_masks, output_path
            )
            
            print(f"✓ Completed {video_name}")
            
        except Exception as e:
            print(f"ERROR processing {video_name}: {str(e)}")
            continue
