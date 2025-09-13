import h5py
import numpy as np
import cv2
import os
from sklearn.cluster import OPTICS
from scipy.spatial.distance import cdist
from ultralytics import YOLO
import json

class EventClusterTracker:
    def __init__(self, yolo_model_path='yolov8n.pt'):
        """
        Event-based tracker using OPTICS to handle varying density clusters with overlap merging.
        Integrates YOLO for object detection and labeling.
        """
        # Current tracking state
        self.tracked_clusters = {}
        self.next_id = 0
        
        # OPTICS parameters optimized for varying density event clustering
        self.optics_min_samples = 300  # Need at least 300 event pixels for valid cluster
        self.optics_max_eps = 15.0     # Maximum distance for cluster formation
        self.density_threshold = 500   # Only keep clusters with at least 500 event pixels
        self.overlap_threshold = 0.5   # 50% overlap merges clusters
        self.max_track_distance = 100  # Maximum distance for frame-to-frame tracking
        self.track_timeout = 15        # Frames before losing track
        
        # YOLO detection parameters
        self.yolo_detection_interval = 15  # Run YOLO every 15th frame
        self.yolo_confidence_threshold = 0.5  # Minimum confidence for detections
        self.object_overlap_threshold = 0.3   # Minimum overlap to associate YOLO detection with track
        
        # Initialize YOLO model
        self.yolo_model = None
        if os.path.exists(yolo_model_path):
            try:
                self.yolo_model = YOLO(yolo_model_path)
                print(f"YOLO model loaded: {yolo_model_path}")
            except Exception as e:
                print(f"Warning: Could not load YOLO model: {e}")
        else:
            print(f"YOLO model file not found: {yolo_model_path}. Continuing without YOLO detection.")
        
        # Statistics tracking
        self.detection_stats = {
            'total_detections': 0,
            'successful_associations': 0,
            'unmatched_detections': 0,
            'labeled_tracks': 0
        }
        
        print("Event Cluster Tracker initialized with OPTICS, overlap merging, and optional YOLO detection")
    
    def calculate_bbox_overlap(self, bbox1, bbox2):
        """
        Calculate IoU (Intersection over Union) between two bounding boxes.
        bbox format: [x1, y1, x2, y2]
        """
        # Calculate intersection area
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def cluster_to_bbox(self, cluster):
        """
        Convert cluster to bounding box format [x1, y1, x2, y2].
        """
        x_coords, y_coords = cluster['pixel_coords']
        x1, x2 = np.min(x_coords), np.max(x_coords)
        y1, y2 = np.min(y_coords), np.max(y_coords)
        
        # Add some padding based on cluster radius
        padding = max(5, int(cluster['radius'] * 0.2))
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = x2 + padding
        y2 = y2 + padding
        
        return [x1, y1, x2, y2]
    
    def run_yolo_detection(self, rgb_frame, frame_idx):
        """
        Run YOLO detection on the current frame and return detections.
        """
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(rgb_frame, conf=self.yolo_confidence_threshold, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name,
                            'frame': frame_idx
                        }
                        detections.append(detection)
            
            self.detection_stats['total_detections'] += len(detections)
            if detections:
                print(f"    YOLO detected {len(detections)} objects: {[d['class_name'] for d in detections]}")
            return detections
            
        except Exception as e:
            print(f"    YOLO detection error: {e}")
            return []
    
    def associate_detections_with_tracks(self, detections, current_clusters, frame_idx):
        """
        Associate YOLO detections with tracked clusters based on maximum overlap.
        """
        associations = []
        
        # Convert clusters to bounding boxes
        cluster_bboxes = []
        for cluster in current_clusters:
            bbox = self.cluster_to_bbox(cluster)
            cluster_bboxes.append(bbox)
        
        # Find best matches between detections and clusters
        for detection in detections:
            det_bbox = detection['bbox']
            best_cluster_idx = None
            max_overlap = 0.0
            
            for i, cluster_bbox in enumerate(cluster_bboxes):
                overlap = self.calculate_bbox_overlap(det_bbox, cluster_bbox)
                if overlap > max_overlap and overlap >= self.object_overlap_threshold:
                    max_overlap = overlap
                    best_cluster_idx = i
            
            if best_cluster_idx is not None:
                cluster = current_clusters[best_cluster_idx]
                association = {
                    'detection': detection,
                    'cluster': cluster,
                    'overlap': max_overlap,
                    'frame': frame_idx
                }
                associations.append(association)
                self.detection_stats['successful_associations'] += 1
                print(f"    Associated {detection['class_name']} (conf={detection['confidence']:.2f}) with cluster at ({cluster['centroid'][0]:.1f},{cluster['centroid'][1]:.1f}), overlap={max_overlap:.2f}")
            else:
                self.detection_stats['unmatched_detections'] += 1
                print(f"    Unmatched detection: {detection['class_name']} (conf={detection['confidence']:.2f})")
        
        return associations
    
    def update_track_labels(self, associations, track_assignments):
        """
        Update track labels based on YOLO detection associations.
        """
        # Create mapping from clusters to track IDs
        cluster_to_track = {}
        for track_id, cluster, distance in track_assignments:
            cluster_to_track[id(cluster)] = track_id
        
        # Update tracks with detection labels
        for assoc in associations:
            cluster = assoc['cluster']
            detection = assoc['detection']
            cluster_id = id(cluster)
            
            if cluster_id in cluster_to_track:
                track_id = cluster_to_track[cluster_id]
                track = self.tracked_clusters[track_id]
                
                # Initialize labels if not present
                if 'labels' not in track:
                    track['labels'] = {}
                    track['label_history'] = []
                    track['best_label'] = None
                    track['best_confidence'] = 0.0
                
                # Update label counts
                class_name = detection['class_name']
                if class_name not in track['labels']:
                    track['labels'][class_name] = {'count': 0, 'total_confidence': 0.0, 'max_confidence': 0.0}
                
                track['labels'][class_name]['count'] += 1
                track['labels'][class_name]['total_confidence'] += detection['confidence']
                track['labels'][class_name]['max_confidence'] = max(
                    track['labels'][class_name]['max_confidence'], 
                    detection['confidence']
                )
                
                # Add to label history
                track['label_history'].append({
                    'frame': assoc['frame'],
                    'class_name': class_name,
                    'confidence': detection['confidence'],
                    'overlap': assoc['overlap']
                })
                
                # Update best label (most frequent, with tie-breaking by confidence)
                best_class = max(track['labels'].keys(), 
                               key=lambda x: (track['labels'][x]['count'], 
                                            track['labels'][x]['max_confidence']))
                
                if track['best_label'] != best_class:
                    track['best_label'] = best_class
                    track['best_confidence'] = track['labels'][best_class]['max_confidence']
                    print(f"    Track #{track_id} labeled as '{best_class}' (conf={track['best_confidence']:.2f})")
    
    def calculate_cluster_overlap(self, cluster1, cluster2):
        """
        Calculate overlap ratio between two clusters based on pixel coordinates.
        Returns the fraction of overlap relative to the smaller cluster.
        """
        # Get pixel coordinates for both clusters
        x1, y1 = cluster1['pixel_coords']
        x2, y2 = cluster2['pixel_coords']
        
        # Convert to sets of (x,y) tuples for efficient intersection
        pixels1 = set(zip(x1, y1))
        pixels2 = set(zip(x2, y2))
        
        # Calculate intersection
        intersection = pixels1.intersection(pixels2)
        overlap_count = len(intersection)
        
        # Calculate overlap ratio relative to smaller cluster
        min_size = min(len(pixels1), len(pixels2))
        if min_size == 0:
            return 0.0
            
        overlap_ratio = overlap_count / min_size
        return overlap_ratio
    
    def merge_overlapping_clusters(self, clusters, frame_idx):
        """
        Merge clusters that have significant overlap (>50% by default).
        """
        if len(clusters) <= 1:
            return clusters
            
        # Track which clusters have been merged
        merged_indices = set()
        merged_clusters = []
        
        for i, cluster1 in enumerate(clusters):
            if i in merged_indices:
                continue
                
            # Start with current cluster
            merged_cluster = cluster1.copy()
            clusters_to_merge = [cluster1]
            merged_indices.add(i)
            
            # Check overlap with remaining clusters
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                if j in merged_indices:
                    continue
                    
                overlap_ratio = self.calculate_cluster_overlap(cluster1, cluster2)
                
                if overlap_ratio >= self.overlap_threshold:
                    clusters_to_merge.append(cluster2)
                    merged_indices.add(j)
                    print(f"    Merging clusters with {overlap_ratio:.2f} overlap")
            
            # If we found clusters to merge, combine them
            if len(clusters_to_merge) > 1:
                merged_cluster = self.combine_clusters(clusters_to_merge, frame_idx)
                print(f"    Created merged cluster with {merged_cluster['pixel_count']} pixels")
            
            merged_clusters.append(merged_cluster)
        
        return merged_clusters
    
    def combine_clusters(self, clusters, frame_idx):
        """
        Combine multiple clusters into one merged cluster.
        """
        # Combine all pixel coordinates
        all_x = []
        all_y = []
        total_pixels = 0
        
        for cluster in clusters:
            x_coords, y_coords = cluster['pixel_coords']
            all_x.extend(x_coords)
            all_y.extend(y_coords)
            total_pixels += cluster['pixel_count']
        
        all_x = np.array(all_x)
        all_y = np.array(all_y)
        
        # Calculate new centroid
        centroid_x = np.mean(all_x)
        centroid_y = np.mean(all_y)
        
        # Calculate new spread metrics
        distances = np.sqrt((all_x - centroid_x)**2 + (all_y - centroid_y)**2)
        max_distance = np.max(distances)
        std_distance = np.std(distances)
        
        # Adaptive radius for merged cluster
        radius = max(15, min(std_distance * 2.5 + 10, 100))
        
        # Calculate density
        cluster_area = np.pi * (max_distance ** 2) if max_distance > 0 else 1
        density = total_pixels / cluster_area
        
        merged_cluster = {
            'id': -1,  # Will be assigned by OPTICS
            'centroid': [centroid_x, centroid_y],
            'radius': radius,
            'pixel_count': total_pixels,
            'density': density,
            'max_spread': max_distance,
            'std_spread': std_distance,
            'pixel_coords': (all_x, all_y),
            'frame': frame_idx,
            'is_merged': True,
            'merged_from': len(clusters)
        }
        
        return merged_cluster
    
    def cluster_dense_event_pixels(self, event_mask, frame_idx):
        """
        Use OPTICS to find clusters of event pixels with varying densities.
        """
        # Get coordinates of ALL event pixels
        y_coords, x_coords = np.where(event_mask > 0)
        
        if len(x_coords) < self.optics_min_samples:
            return []
        
        print(f"Frame {frame_idx}: Processing {len(x_coords)} event pixels")
        
        # Prepare event pixel coordinates for OPTICS
        event_coordinates = np.column_stack([x_coords, y_coords])
        
        # Apply OPTICS clustering - better for varying density
        optics = OPTICS(min_samples=self.optics_min_samples, 
                       max_eps=self.optics_max_eps,
                       cluster_method='xi')
        cluster_labels = optics.fit_predict(event_coordinates)
        
        # Process clusters and keep only dense ones
        unique_labels = np.unique(cluster_labels)
        dense_clusters = []
        
        noise_count = np.sum(cluster_labels == -1)
        total_clustered = np.sum(cluster_labels != -1)
        valid_clusters = len([l for l in unique_labels if l != -1])
        
        print(f"  Found {valid_clusters} OPTICS clusters, {total_clustered} clustered pixels, {noise_count} noise")
        
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
                'frame': frame_idx,
                'is_merged': False,
                'merged_from': 1
            }
            
            dense_clusters.append(cluster_info)
            print(f"    Dense Cluster: {cluster_size} pixels at ({centroid_x:.1f},{centroid_y:.1f}), "
                  f"density={density:.2f}, radius={radius:.1f}")
        
        print(f"  Found {len(dense_clusters)} dense clusters before overlap merging")
        
        # Merge overlapping clusters
        merged_clusters = self.merge_overlapping_clusters(dense_clusters, frame_idx)
        
        print(f"  Final result: {len(merged_clusters)} clusters after overlap merging")
        return merged_clusters
    
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
                merged_info = f" (merged from {cluster['merged_from']})" if cluster['is_merged'] else ""
                print(f"  NEW TRACK #{new_track_id} started at ({cluster_centroid[0]:.1f},{cluster_centroid[1]:.1f}){merged_info}")
        
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
    
    def draw_dense_clusters_and_tracks(self, rgb_frame, event_mask, dense_clusters, frame_idx, detections=None):
        """
        Visualize:
        1. All event pixels as light green overlay
        2. Dense clusters as green circles (cyan for merged clusters)
        3. Tracked objects as colored circles with track IDs, paths, and labels
        4. YOLO detections as purple bounding boxes
        """
        output_frame = rgb_frame.copy()
        
        # 1. Overlay ALL event pixels in light green
        event_pixels = (event_mask > 0)
        if np.any(event_pixels):
            overlay = output_frame.copy()
            overlay[event_pixels] = [0, 255, 0]  # Green
            cv2.addWeighted(overlay, 0.2, output_frame, 0.8, 0, output_frame)
        
        # 4. Draw YOLO detections first (so they appear behind tracks)
        if detections:
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # Draw purple bounding box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
                cv2.putText(output_frame, f"{class_name} {confidence:.2f}", 
                           (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1)
        
        # 2. Draw dense cluster circles (green for single, cyan for merged)
        for cluster in dense_clusters:
            center = tuple(map(int, cluster['centroid']))
            radius = int(cluster['radius'])
            
            # Different colors for merged vs single clusters
            if cluster.get('is_merged', False):
                color = (255, 255, 0)  # Cyan for merged
                thickness = 3
            else:
                color = (0, 255, 0)    # Green for single
                thickness = 2
            
            # Draw cluster boundary
            cv2.circle(output_frame, center, radius, color, thickness)
            
            # Show cluster info
            pixel_count = cluster['pixel_count']
            density = cluster['density']
            merged_from = cluster.get('merged_from', 1)
            
            cv2.putText(output_frame, f"C:{pixel_count}", 
                       (center[0] - 25, center[1] - radius - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(output_frame, f"D:{density:.1f}", 
                       (center[0] - 25, center[1] - radius - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            if merged_from > 1:
                cv2.putText(output_frame, f"M:{merged_from}", 
                           (center[0] - 25, center[1] - radius - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 3. Draw tracked objects with unique colors, paths, and labels
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
                
                # Draw best label if available
                if 'best_label' in track and track['best_label'] is not None:
                    label_text = f"{track['best_label']} ({track['best_confidence']:.2f})"
                    cv2.putText(output_frame, label_text, 
                               (center[0] - 30, center[1] + radius + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 5. Frame statistics including YOLO info
        total_event_pixels = np.sum(event_pixels)
        active_track_count = len(active_tracks)
        merged_count = len([c for c in dense_clusters if c.get('is_merged', False)])
        labeled_tracks = len([t for t in self.tracked_clusters.values() if t.get('best_label') is not None])
        detection_count = len(detections) if detections else 0
        
        info_text = f"Frame {frame_idx} | Events: {total_event_pixels} | Clusters: {len(dense_clusters)} | Merged: {merged_count} | Tracks: {active_track_count} | Labeled: {labeled_tracks} | YOLO: {detection_count}"
        cv2.putText(output_frame, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return output_frame

    def calculate_comprehensive_metrics(self, total_frames, yolo_frames, total_clusters, merged_clusters):
        """Calculate comprehensive tracking and detection metrics."""
        metrics = {
            'processing': {
                'total_frames': total_frames,
                'yolo_detection_frames': yolo_frames,
                'yolo_detection_rate': yolo_frames / total_frames if total_frames > 0 else 0.0
            },
            'clustering': {
                'total_clusters_found': total_clusters,
                'merged_clusters': merged_clusters,
                'merge_rate': merged_clusters / total_clusters if total_clusters > 0 else 0.0,
                'avg_clusters_per_frame': total_clusters / total_frames if total_frames > 0 else 0.0
            },
            'tracking': {
                'total_tracks_created': len(self.tracked_clusters),
                'active_tracks': len([t for t in self.tracked_clusters.values() if t['active']]),
                'labeled_tracks': len([t for t in self.tracked_clusters.values() if t.get('best_label') is not None])
            },
            'detection': {
                'total_yolo_detections': self.detection_stats['total_detections'],
                'successful_associations': self.detection_stats['successful_associations'],
                'unmatched_detections': self.detection_stats['unmatched_detections'],
                'association_rate': (self.detection_stats['successful_associations'] / 
                                   self.detection_stats['total_detections'] 
                                   if self.detection_stats['total_detections'] > 0 else 0.0)
            }
        }
        
        # Calculate labeling rate
        total_tracks = metrics['tracking']['total_tracks_created']
        labeled_tracks = metrics['tracking']['labeled_tracks']
        metrics['tracking']['labeling_rate'] = labeled_tracks / total_tracks if total_tracks > 0 else 0.0
        
        return metrics

    def print_detection_statistics(self, metrics=None):
        """Print comprehensive detection and tracking statistics."""
        if metrics is None:
            # Create basic metrics if not provided
            total_tracks = len(self.tracked_clusters)
            labeled_tracks = len([t for t in self.tracked_clusters.values() if t.get('best_label') is not None])
            
            metrics = {
                'detection': self.detection_stats.copy(),
                'tracking': {
                    'total_tracks_created': total_tracks,
                    'labeled_tracks': labeled_tracks,
                    'labeling_rate': labeled_tracks / total_tracks if total_tracks > 0 else 0.0
                }
            }
        
        print("\n" + "="*80)
        print("DETECTION AND TRACKING STATISTICS")
        print("="*80)
        
        # Processing stats
        if 'processing' in metrics:
            proc = metrics['processing']
            print(f"Processing:")
            print(f"  Total frames: {proc['total_frames']}")
            print(f"  YOLO detection frames: {proc['yolo_detection_frames']}")
            print(f"  YOLO detection rate: {proc['yolo_detection_rate']*100:.1f}%")
        
        # Clustering stats  
        if 'clustering' in metrics:
            clust = metrics['clustering']
            print(f"\nClustering:")
            print(f"  Total clusters found: {clust['total_clusters_found']}")
            print(f"  Merged clusters: {clust['merged_clusters']}")
            print(f"  Merge rate: {clust['merge_rate']*100:.1f}%")
            print(f"  Average clusters per frame: {clust['avg_clusters_per_frame']:.1f}")
        
        # Detection stats
        if 'detection' in metrics:
            det = metrics['detection']
            print(f"\nYOLO Detections:")
            print(f"  Total detections: {det['total_yolo_detections']}")
            print(f"  Successfully associated: {det['successful_associations']}")
            print(f"  Unmatched detections: {det['unmatched_detections']}")
            print(f"  Association rate: {det['association_rate']*100:.1f}%")
        
        # Track labeling stats
        if 'tracking' in metrics:
            track = metrics['tracking']
            print(f"\nTrack Labeling:")
            print(f"  Total tracks created: {track['total_tracks_created']}")
            print(f"  Labeled tracks: {track['labeled_tracks']}")
            if 'active_tracks' in track:
                print(f"  Currently active tracks: {track['active_tracks']}")
            print(f"  Labeling rate: {track['labeling_rate']*100:.1f}%")
        
        # Label distribution
        label_counts = {}
        for track in self.tracked_clusters.values():
            if track.get('best_label'):
                label = track['best_label']
                label_counts[label] = label_counts.get(label, 0) + 1
        
        if label_counts:
            print(f"\nLabel Distribution:")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {label}: {count} tracks")
        
        print("="*80)


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

def create_event_cluster_tracking_video(input_video_path, event_masks, output_path, yolo_model_path='yolov8n.pt'):
    """
    Create video showing OPTICS-based clustering with overlap merging and YOLO detection:
    - Light green overlay for all event pixels
    - Green circles for single clusters, cyan for merged clusters
    - Colored circles and paths for tracked motion clusters with labels
    - Purple bounding boxes for YOLO detections
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

    # Initialize tracker with YOLO
    tracker = EventClusterTracker(yolo_model_path)

    # Processing statistics
    total_dense_clusters = 0
    total_merged_clusters = 0
    total_tracks_created = 0
    yolo_detection_frames = 0

    print("Processing frames for OPTICS event clustering with YOLO object detection...")
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

        # Find dense clusters of event pixels using OPTICS with overlap merging
        dense_clusters = tracker.cluster_dense_event_pixels(event_mask, frame_idx)
        total_dense_clusters += len(dense_clusters)
        total_merged_clusters += len([c for c in dense_clusters if c.get('is_merged', False)])

        # Track clusters across frames
        track_assignments = tracker.track_clusters_across_frames(dense_clusters, frame_idx)

        # Run YOLO detection every 15th frame
        detections = []
        if frame_idx % tracker.yolo_detection_interval == 0:
            detections = tracker.run_yolo_detection(rgb_frame, frame_idx)
            yolo_detection_frames += 1
            
            # Associate detections with tracked clusters
            if detections and dense_clusters:
                associations = tracker.associate_detections_with_tracks(detections, dense_clusters, frame_idx)
                tracker.update_track_labels(associations, track_assignments)

        # Count new tracks created this frame
        new_tracks_this_frame = len([a for a in track_assignments if tracker.tracked_clusters[a[0]]['frames_tracked'] == 1])
        total_tracks_created += new_tracks_this_frame

        # Create visualization
        output_frame = tracker.draw_dense_clusters_and_tracks(rgb_frame, event_mask, dense_clusters, frame_idx, detections)

        # Write frame
        out.write(output_frame)
        frame_idx += 1

        if frame_idx % 25 == 0:
            active_tracks = len([t for t in tracker.tracked_clusters.values() if t['active']])
            merged_count = len([c for c in dense_clusters if c.get('is_merged', False)])
            labeled_tracks = len([t for t in tracker.tracked_clusters.values() if t.get('best_label') is not None])
            print(f"Frame {frame_idx}/{total_frames}: {len(dense_clusters)} clusters ({merged_count} merged), {active_tracks} active tracks, {labeled_tracks} labeled")

    # Cleanup
    cap.release()
    out.release()

    # Final comprehensive statistics
    active_tracks = len([t for t in tracker.tracked_clusters.values() if t['active']])
    total_tracks = len(tracker.tracked_clusters)
    
    print(f"\nProcessing complete!")
    print(f"Frames processed: {frame_idx}")
    print(f"YOLO detection frames: {yolo_detection_frames}")
    print(f"Total dense clusters found: {total_dense_clusters}")
    print(f"Total merged clusters: {total_merged_clusters}")
    print(f"Total tracks created: {total_tracks}")
    print(f"Final active tracks: {active_tracks}")
    print(f"Average clusters per frame: {total_dense_clusters/frame_idx:.1f}")
    print(f"Merge rate: {total_merged_clusters/total_dense_clusters*100:.1f}%")
    
    # Print detailed detection statistics with comprehensive metrics
    metrics = tracker.calculate_comprehensive_metrics(
        frame_idx, yolo_detection_frames, total_dense_clusters, total_merged_clusters
    )
    tracker.print_detection_statistics(metrics)
    
    # Save metrics to file for analysis
    metrics_file = output_path.replace('.mp4', '_metrics.json')
    with open(metrics_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = {}
        for category, values in metrics.items():
            serializable_metrics[category] = {}
            for key, value in values.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[category][key] = value.item()
                else:
                    serializable_metrics[category][key] = value
        
        json.dump(serializable_metrics, f, indent=2)
    print(f"Detailed metrics saved to: {metrics_file}")
    
    print(f"Output saved: {output_path}")

if __name__ == "__main__":
    # File paths
    input_video = "../sample_videos_to_test/clip_002.mp4"  # RGB input video (for background)
    h5_path = "../v2e-output/traffic_scenev.h5"  # Event data
    output_path = "../v2e-output/event_cluster_tracking_with_yolo.mp4"
    yolo_model_path = "yolov8n.pt"  # YOLO model path
    
    print("File paths:")
    print(f"RGB input video: {os.path.abspath(input_video)}")
    print(f"H5 events file: {os.path.abspath(h5_path)}")
    print(f"YOLO model: {yolo_model_path}")
    print(f"Output video: {os.path.abspath(output_path)}")

    # Read event data
    x, y, p, t = read_h5_events(h5_path)

    # Load frame timestamps
    frame_times_path = "../v2e-output/dvs_preview-frame_times.txt"  # Fixed path
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

    # Create tracking video with YOLO detection
    print("\nCreating Event Cluster Tracking video with OPTICS, overlap merging, and YOLO detection...")
    create_event_cluster_tracking_video(input_video, event_masks, output_path, yolo_model_path)