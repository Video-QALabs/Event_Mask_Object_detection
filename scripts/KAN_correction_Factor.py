import h5py
import numpy as np
import cv2
import os
from sklearn.cluster import OPTICS
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from sklearn.preprocessing import StandardScaler
from efficient_kan import KAN
import warnings
import traceback
from collections import deque, defaultdict
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

class RobustTrajectoryKAN(nn.Module):
    """
    Robust KAN-based trajectory predictor using efficient-kan library
    """
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=2, num_layers=4):
        super(RobustTrajectoryKAN, self).__init__()
        
        # Use efficient-kan KAN model
        # layers_hidden defines the full architecture: [input_dim, hidden_dim, ..., hidden_dim, output_dim]
        layers_hidden = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=10,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0
        )
        
        # Set input dimension manually since efficient-kan determines it from first forward pass
        self.input_dim = input_dim
        
        # Additional uncertainty estimation head
        self.uncertainty_head = nn.Linear(input_dim, output_dim)  # Use input_dim instead of hidden_dim
        self.dropout = nn.Dropout(0.2)
        
        # Store dimensions for uncertainty calculation
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
    def forward(self, x, return_uncertainty=False):
        # Forward pass through efficient-kan
        output = self.kan(x)
        
        if return_uncertainty:
            # For uncertainty estimation, we'll use the full input features
            batch_size = x.shape[0]
            
            # Create uncertainty based on input feature variance and model confidence
            input_std = torch.std(x, dim=1, keepdim=True)
            base_uncertainty = torch.sigmoid(input_std).repeat(1, self.output_dim)
            
            # Use full input for learned uncertainty (uncertainty_head expects input_dim features)
            learned_uncertainty = torch.sigmoid(self.uncertainty_head(x))
            
            # Combine base and learned uncertainty
            uncertainty = 0.7 * base_uncertainty + 0.3 * learned_uncertainty
            uncertainty = torch.clamp(uncertainty, 0.1, 0.9)  # Keep uncertainty in reasonable range
            
            return output, uncertainty
        
        return output

class EventClusterTracker:
    def __init__(self):
        """
        Event-based tracker using OPTICS to handle varying density clusters with overlap merging.
        """
        # Current tracking state
        self.tracked_clusters = {}
        self.next_id = 0
        
        # OPTICS parameters optimized for varying density event clustering
        self.optics_min_samples = 300  # Need at least 200 event pixels for valid cluster
        self.optics_max_eps = 15.0     # Maximum distance for cluster formation
        self.density_threshold = 500   # Only keep clusters with at least 500 event pixels
        self.overlap_threshold = 0.5   # 30% overlap merges clusters
        self.max_track_distance = 100   # Maximum distance for frame-to-frame tracking
        self.track_timeout = 15        # Frames before losing track
        
        # Initialize GPU device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Event Cluster Tracker initialized with OPTICS and overlap merging on {self.device}")
        
        # Initialize Robust KAN for trajectory prediction
        self.kan_predictor = RobustTrajectoryKAN(input_dim=8, hidden_dim=128, output_dim=2).to(self.device)
        self.kan_optimizer = optim.AdamW(self.kan_predictor.parameters(), lr=0.001, weight_decay=1e-5)
        self.kan_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.kan_optimizer, T_max=100)
        self.prediction_horizon = 50  # Predict 50 frames into the future
        
        # Robust optimization parameters
        self.adaptation_rate = 0.1
        self.uncertainty_threshold = 0.3
        self.min_confidence = 0.7
        
        # Training state tracking
        self.kan_trained = False
        self.training_data_count = 0
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.kan_train_count = 0
        self.last_frame_clusters = {}

    def create_prediction_snapshot(self, rgb_frame, frame_idx, output_dir):
        """
        Creates a snapshot image with all current predicted trajectories drawn on it.
        """
        snapshot_frame = rgb_frame.copy()

        for track_id, track in self.tracked_clusters.items():
            if 'predicted_path' in track and len(track['predicted_path']) > 0:
                future_path = track['predicted_path']
                uncertainty = track.get('prediction_uncertainty', [0] * len(future_path))

                # Draw the predicted path
                for i in range(len(future_path) - 1):
                    cv2.line(snapshot_frame, tuple(future_path[i].astype(int)), tuple(future_path[i+1].astype(int)), (255, 0, 255), 2)

                # Draw uncertainty markers
                for i, point in enumerate(future_path):
                    radius = int(uncertainty[i] * 5) # Scale uncertainty for visibility
                    cv2.circle(snapshot_frame, tuple(point.astype(int)), radius, (0, 255, 255), 1)

        # Save the snapshot
        output_filename = os.path.join(output_dir, f"prediction_snapshot_frame_{frame_idx}.png")
        cv2.imwrite(output_filename, snapshot_frame)
        print(f"Saved prediction snapshot: {output_filename}")
        return snapshot_frame


    def cubic_bspline_basis(self, t, i, k, knots):
        """
        Calculate cubic B-spline basis function using Cox-de Boor recursion formula.
        """
        if k == 0:
            return 1.0 if knots[i] <= t < knots[i+1] else 0.0
        
        # Handle division by zero
        left_denom = knots[i+k] - knots[i]
        right_denom = knots[i+k+1] - knots[i+1]
        
        left_term = 0.0
        if left_denom != 0:
            left_term = (t - knots[i]) / left_denom * self.cubic_bspline_basis(t, i, k-1, knots)
        
        right_term = 0.0
        if right_denom != 0:
            right_term = (knots[i+k+1] - t) / right_denom * self.cubic_bspline_basis(t, i+1, k-1, knots)
        
        return left_term + right_term
    
    def calculate_cubic_bspline_path(self, control_points, num_samples=50):
        """
        Calculate smooth cubic B-spline curve through the control points (trajectory).
        """
        if len(control_points) < 4:
            return control_points  # Not enough points for cubic B-spline
        
        n = len(control_points)
        k = 3  # Cubic B-spline (degree 3)
        
        # Create knot vector (clamped)
        knots = []
        # Clamp start (repeat first knot k+1 times)
        for _ in range(k + 1):
            knots.append(0.0)
        # Internal knots
        for i in range(1, n - k):
            knots.append(float(i))
        # Clamp end (repeat last knot k+1 times)
        for _ in range(k + 1):
            knots.append(float(n - k))
        
        # Generate curve points
        curve_points = []
        t_start = knots[k]
        t_end = knots[n]
        
        for sample in range(num_samples + 1):
            t = t_start + (t_end - t_start) * sample / num_samples
            
            # Ensure t is within valid range
            t = max(t_start, min(t_end - 1e-10, t))
            
            x, y = 0.0, 0.0
            
            # Calculate curve point using basis functions
            for i in range(n):
                basis = self.cubic_bspline_basis(t, i, k, knots)
                x += control_points[i][0] * basis
                y += control_points[i][1] * basis
            
            curve_points.append([x, y])
        
        return curve_points
    
    def prepare_trajectory_features(self, path):
        """
        Prepare robust features for KAN trajectory prediction from path history
        """
        if len(path) < 4:
            return None
        
        # Extract last 4 positions for better feature extraction
        recent_positions = path[-4:]
        
        # Calculate velocities, accelerations, and jerk
        pos = recent_positions
        vel = [[pos[i+1][0] - pos[i][0], pos[i+1][1] - pos[i][1]] for i in range(3)]
        acc = [[vel[i+1][0] - vel[i][0], vel[i+1][1] - vel[i][1]] for i in range(2)]
        jerk = [acc[1][0] - acc[0][0], acc[1][1] - acc[0][1]]
        
        # Current velocity and acceleration
        current_vel = vel[-1]
        current_acc = acc[-1]
        
        # Speed and direction features
        speed = np.sqrt(current_vel[0]**2 + current_vel[1]**2)
        direction = np.arctan2(current_vel[1], current_vel[0])
        
        # Features: [current_x, current_y, vel_x, vel_y, acc_x, acc_y, speed, direction]
        features = [pos[-1][0], pos[-1][1], current_vel[0], current_vel[1], 
                   current_acc[0], current_acc[1], speed, direction]
        return np.array(features, dtype=np.float32)
    
    def predict_linear_trajectory(self, track):
        """
        Simple linear extrapolation fallback when KAN is not trained yet
        """
        if len(track['path']) < 3:
            return [], []
        
        # Get last few positions
        recent_positions = track['path'][-3:]
        
        # Calculate average velocity
        velocities = []
        for i in range(1, len(recent_positions)):
            vel_x = recent_positions[i][0] - recent_positions[i-1][0]
            vel_y = recent_positions[i][1] - recent_positions[i-1][1]
            velocities.append([vel_x, vel_y])
        
        # Average velocity
        avg_vel_x = np.mean([v[0] for v in velocities])
        avg_vel_y = np.mean([v[1] for v in velocities])
        
        # Predict future points using linear extrapolation
        predicted_points = []
        uncertainties = []
        current_pos = recent_positions[-1]
        
        for step in range(min(self.prediction_horizon, 20)):  # Limit linear prediction
            next_x = current_pos[0] + avg_vel_x * (step + 1)
            next_y = current_pos[1] + avg_vel_y * (step + 1)
            predicted_points.append([next_x, next_y])
            # High uncertainty for linear prediction
            uncertainties.append([0.8, 0.8])
        
        return predicted_points, uncertainties
    
    def predict_future_trajectory(self, track):
        """
        Use robust KAN to predict future trajectory points with uncertainty estimation
        """
        # If KAN not trained yet, use simple linear extrapolation
        if not self.kan_trained:
            return self.predict_linear_trajectory(track)
            
        # Use KAN prediction if trained
            
        features = self.prepare_trajectory_features(track['path'])
        if features is None:
            return [], []
        
        # Scale features before prediction
        features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
        
        # Convert to GPU tensor
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
        
        predicted_points = []
        uncertainties = []
        current_features = features_tensor.clone()
        
        # Set model to evaluation mode for inference
        self.kan_predictor.eval()
        
        # Predict multiple steps into the future
        with torch.no_grad():
            for step in range(self.prediction_horizon):
                # Predict next position with uncertainty
                pred_pos_scaled, uncertainty = self.kan_predictor(current_features, return_uncertainty=True)
                
                # Inverse transform predicted position to original scale
                pred_pos = self.target_scaler.inverse_transform(pred_pos_scaled.cpu().numpy())
                
                # Apply uncertainty-based adaptive weighting
                uncertainty_weight = 1.0 - uncertainty.mean().cpu().numpy()
                if uncertainty_weight < self.min_confidence:
                    # High uncertainty, reduce prediction confidence
                    pred_pos = pred_pos * uncertainty_weight + self.target_scaler.inverse_transform(current_features[:, :2].cpu().numpy()) * (1 - uncertainty_weight)
                
                predicted_points.append(pred_pos[0])
                uncertainties.append(uncertainty.cpu().numpy()[0])
                
                # Robust feature update with momentum
                current_pos_scaled = current_features[:, :2].cpu().numpy()
                current_pos = self.target_scaler.inverse_transform(current_pos_scaled)[0]
                
                new_x, new_y = pred_pos[0, 0], pred_pos[0, 1]
                
                # Calculate new motion features in original scale
                new_vel_x, new_vel_y = new_x - current_pos[0], new_y - current_pos[1]
                new_speed = np.sqrt(new_vel_x**2 + new_vel_y**2)
                new_direction = np.arctan2(new_vel_y, new_vel_x)
                
                # Create new feature vector and scale it
                new_features = np.array([new_x, new_y, new_vel_x, new_vel_y, 
                                         current_features[0, 4].item(), current_features[0, 5].item(), 
                                         new_speed, new_direction], dtype=np.float32)
                new_features_scaled = self.feature_scaler.transform(new_features.reshape(1, -1))
                
                # Adaptive feature update based on uncertainty
                alpha = self.adaptation_rate * uncertainty_weight
                current_features = torch.tensor(alpha * new_features_scaled + (1 - alpha) * current_features.cpu().numpy(), dtype=torch.float32).to(self.device)
        
        return predicted_points, uncertainties
    
    def train_kan_predictor(self, current_clusters, force_retrain=False):
        """
        Trains the KAN predictor with the history of tracked clusters.
        Includes a self-correction mechanism to retrain if predictions are poor.
        """
        if not self.use_kan:
            return

        print(f"Starting KAN training cycle {self.kan_train_count + 1}...")

        # --- Main Training ---
        self._perform_training(current_clusters)
        self.kan_train_count += 1
        self.kan_trained = True
        print("Initial training complete.")

        # --- Self-Correction Mechanism (after first 2 training cycles) ---
        if self.kan_train_count > 2 and not force_retrain:
            print("Performing self-correction validation...")
            needs_retraining = False
            corrected_data = []

            for track_id, track in self.tracked_clusters.items():
                if len(track['positions']) < self.min_track_len_for_pred:
                    continue

                # 1. Predict the next point based on the current model
                history = np.array(track['positions'])
                features = self._extract_features(history)
                
                if features.shape[0] == 0:
                    continue

                last_feature = self.feature_scaler.transform(features[-1, :].reshape(1, -1))
                last_feature_tensor = torch.tensor(last_feature, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    predicted_mean, predicted_std = self.kan_predictor(last_feature_tensor)
                
                predicted_target = self.target_scaler.inverse_transform(predicted_mean.cpu().numpy())
                predicted_next_pos = history[-1, :2] + predicted_target[0, :2] # Predict delta and add to last pos

                # 2. Check against ground truth (the actual next point in history)
                ground_truth_pos = history[-1, :2]
                
                # Find the actual cluster this track was matched with in the current frame
                actual_next_pos = None
                for c in current_clusters:
                    if c['track_id'] == track_id:
                        actual_next_pos = c['center']
                        break
                
                if actual_next_pos is not None:
                    ground_truth_pos = actual_next_pos


                distance = np.linalg.norm(predicted_next_pos - ground_truth_pos)
                confidence_radius = 25  # pixels

                # 3. If prediction is outside the confidence zone, flag for retraining
                if distance > confidence_radius:
                    print(f"Track {track_id}: Bad prediction. Distance: {distance:.2f} > {confidence_radius}. Flagging for retrain.")
                    needs_retraining = True
                    
                    # Create a corrected data point
                    # Feature: The one that led to the bad prediction
                    # Target: The vector that would have led to the correct prediction
                    corrected_target = ground_truth_pos - history[-2, :2]
                    corrected_data.append((features[-1, :], corrected_target))

            # 4. If any track had a bad prediction, perform retraining with corrected data
            if needs_retraining:
                print("Retraining with corrected data...")
                self._perform_training(current_clusters, extra_data=corrected_data)
                print("Self-correction retraining complete.")


    def _perform_training(self, current_clusters, extra_data=None):
        """Helper method to perform a single training pass."""
        features_list = []
        targets_list = []

        for track_id, track in self.tracked_clusters.items():
            if len(track['positions']) < self.min_track_len_for_pred:
                continue
            
            history = np.array(track['positions'])
            features = self._extract_features(history)
            
            # Targets are the deltas from one position to the next
            targets = history[1:, :2] - history[:-1, :2]

            if features.shape[0] > 0 and targets.shape[0] > 0:
                # Ensure features and targets align
                min_len = min(features.shape[0], targets.shape[0])
                features_list.append(features[:min_len])
                targets_list.append(targets[:min_len])

        if not features_list:
            print("Warning: No data available for KAN training.")
            return

        # Add corrected data if provided
        if extra_data:
            for feature, target in extra_data:
                features_list.append(feature.reshape(1, -1))
                targets_list.append(target.reshape(1, -1))

        X = np.vstack(features_list)
        y = np.vstack(targets_list)

        # Fit scalers
        self.feature_scaler.fit(X)
        self.target_scaler.fit(y)

        # Scale data
        X_scaled = self.feature_scaler.transform(X)
        y_scaled = self.target_scaler.transform(y)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)

        # Training loop
        self.kan_predictor.train()
        for epoch in range(self.kan_train_epochs):
            self.optimizer.zero_grad()
            
            mean, std = self.kan_predictor(X_tensor)
            
            # Gaussian Negative Log-Likelihood Loss
            loss = self.nll_loss(mean, y_tensor, std)
            
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{self.kan_train_epochs}], Loss: {loss.item():.6f}')
        
        self.scheduler.step()


    def predict_linear_trajectory(self, track_history, num_steps=50):
        """Predicts future trajectory using simple linear extrapolation."""
        if len(track_history) < 2:
            return [], []
        
        # Use last segment for prediction
        last_segment = track_history[-2:]
        
        # Calculate deltas
        deltas = np.array([
            [last_segment[1][0] - last_segment[0][0], last_segment[1][1] - last_segment[0][1]]
        ], dtype=np.float32)
        
        # Repeat last position for prediction
        last_position = np.array(last_segment[-1], dtype=np.float32)
        predicted_points = [last_position]
        uncertainties = [[0.5, 0.5]]  # Initial uncertainty
        
        for step in range(1, num_steps):
            # Simple linear prediction
            next_position = last_position + deltas[0]
            predicted_points.append(next_position)
            
            # Update last position
            last_position = next_position
        
        return predicted_points, uncertainties
    
    def predict_future_trajectory(self, track):
        """
        Use robust KAN to predict future trajectory points with uncertainty estimation
        """
        # If KAN not trained yet, use simple linear extrapolation
        if not self.kan_trained:
            return self.predict_linear_trajectory(track)
            
        # Use KAN prediction if trained
            
        features = self.prepare_trajectory_features(track['path'])
        if features is None:
            return [], []
        
        # Scale features before prediction
        features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
        
        # Convert to GPU tensor
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
        
        predicted_points = []
        uncertainties = []
        current_features = features_tensor.clone()
        
        # Set model to evaluation mode for inference
        self.kan_predictor.eval()
        
        # Predict multiple steps into the future
        with torch.no_grad():
            for step in range(self.prediction_horizon):
                # Predict next position with uncertainty
                pred_pos_scaled, uncertainty = self.kan_predictor(current_features, return_uncertainty=True)
                
                # Inverse transform predicted position to original scale
                pred_pos = self.target_scaler.inverse_transform(pred_pos_scaled.cpu().numpy())
                
                # Apply uncertainty-based adaptive weighting
                uncertainty_weight = 1.0 - uncertainty.mean().cpu().numpy()
                if uncertainty_weight < self.min_confidence:
                    # High uncertainty, reduce prediction confidence
                    pred_pos = pred_pos * uncertainty_weight + self.target_scaler.inverse_transform(current_features[:, :2].cpu().numpy()) * (1 - uncertainty_weight)
                
                predicted_points.append(pred_pos[0])
                uncertainties.append(uncertainty.cpu().numpy()[0])
                
                # Robust feature update with momentum
                current_pos_scaled = current_features[:, :2].cpu().numpy()
                current_pos = self.target_scaler.inverse_transform(current_pos_scaled)[0]
                
                new_x, new_y = pred_pos[0, 0], pred_pos[0, 1]
                
                # Calculate new motion features in original scale
                new_vel_x, new_vel_y = new_x - current_pos[0], new_y - current_pos[1]
                new_speed = np.sqrt(new_vel_x**2 + new_vel_y**2)
                new_direction = np.arctan2(new_vel_y, new_vel_x)
                
                # Create new feature vector and scale it
                new_features = np.array([new_x, new_y, new_vel_x, new_vel_y, 
                                         current_features[0, 4].item(), current_features[0, 5].item(), 
                                         new_speed, new_direction], dtype=np.float32)
                new_features_scaled = self.feature_scaler.transform(new_features.reshape(1, -1))
                
                # Adaptive feature update based on uncertainty
                alpha = self.adaptation_rate * uncertainty_weight
                current_features = torch.tensor(alpha * new_features_scaled + (1 - alpha) * current_features.cpu().numpy(), dtype=torch.float32).to(self.device)
        
        return predicted_points, uncertainties
    
    def train_kan_predictor(self, all_tracks):
        """
        Train robust KAN predictor with uncertainty estimation
        """
        training_data = []
        targets = []
        
        # Collect training data from all tracks
        for track_id, track in all_tracks.items():
            if len(track['path']) >= 5:  # Need at least 5 points for robust features
                path = track['path']
                
                # Create training samples from path history
                for i in range(4, len(path)):
                    # Features from positions i-4 to i-1
                    features = self.prepare_trajectory_features(path[:i])
                    if features is not None:
                        target = path[i]  # Actual next position
                        training_data.append(features)
                        targets.append(target)
        
        if len(training_data) < 10:  # Reduced minimum training data requirement
            print(f"Insufficient training data: {len(training_data)} samples (need at least 10)")
            return
        
        self.training_data_count = len(training_data)
        print(f"Training KAN with {len(training_data)} samples")
        
        # Fit scalers and transform data
        X_scaled = self.feature_scaler.fit_transform(np.array(training_data))
        y_scaled = self.target_scaler.fit_transform(np.array(targets))
        
        # Convert to tensors with explicit dtype to prevent type errors
        X = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        
        # Create data loader with larger batch size
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Robust training loop with uncertainty loss
        self.kan_predictor.train()
        for epoch in range(20):  # More epochs for better convergence
            total_loss = 0
            total_uncertainty_loss = 0
            
            for batch_X, batch_y in dataloader:
                self.kan_optimizer.zero_grad()
                
                # Forward pass with uncertainty
                predictions, uncertainty = self.kan_predictor(batch_X, return_uncertainty=True)
                
                # Position prediction loss
                position_loss = nn.MSELoss()(predictions, batch_y)
                
                # Uncertainty regularization loss
                prediction_error = torch.mean((predictions - batch_y)**2, dim=1, keepdim=True)
                uncertainty_loss = nn.MSELoss()(uncertainty, prediction_error.detach())
                
                # Combined loss with adaptive weighting
                total_loss_batch = position_loss + 0.1 * uncertainty_loss
                
                # Backward pass
                total_loss_batch.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.kan_predictor.parameters(), max_norm=1.0)
                
                self.kan_optimizer.step()
                
                total_loss += position_loss.item()
                total_uncertainty_loss += uncertainty_loss.item()
            
            # Update learning rate
            self.kan_scheduler.step()
            
            if epoch % 5 == 0:
                avg_loss = total_loss / len(dataloader)
                avg_uncertainty = total_uncertainty_loss / len(dataloader)
                lr = self.kan_optimizer.param_groups[0]['lr']
                print(f"KAN Epoch {epoch}: Loss={avg_loss:.4f}, Uncertainty={avg_uncertainty:.4f}, LR={lr:.6f}")
        
        # Mark as trained
        self.kan_trained = True
        print("KAN training completed successfully")
    
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
        Merge clusters that have significant overlap (>30% by default).
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
    
    def draw_dense_clusters_and_tracks(self, rgb_frame, event_mask, dense_clusters, frame_idx):
        """
        Visualize:
        1. All event pixels as light green overlay
        2. Dense clusters as green circles (cyan for merged clusters)
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
        
        # 3. Draw tracked objects with unique colors and paths
        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
                 (0, 255, 255), (128, 0, 255), (255, 128, 0), (128, 255, 0)]
        
        active_tracks = [tid for tid, track in self.tracked_clusters.items() if track['active']]
        
        for i, track_id in enumerate(active_tracks):
            track = self.tracked_clusters[track_id]
            color = colors[i % len(colors)]
            
            # Draw track path
            if len(track['path']) > 1:
                # Draw original trajectory (thin line)
                for j in range(1, len(track['path'])):
                    pt1 = tuple(map(int, track['path'][j-1]))
                    pt2 = tuple(map(int, track['path'][j]))
                    cv2.line(output_frame, pt1, pt2, color, 1)
                
                # Draw B-spline smoothed trajectory if enough points
                if len(track['path']) >= 4:  # Need at least 4 points for cubic B-spline
                    spline_points = self.calculate_cubic_bspline_path(track['path'])
                    if len(spline_points) > 1:
                        # Draw smoothed B-spline path (thicker line)
                        for j in range(1, len(spline_points)):
                            pt1 = tuple(map(int, spline_points[j-1]))
                            pt2 = tuple(map(int, spline_points[j]))
                            cv2.line(output_frame, pt1, pt2, color, 3)
                
                # Draw KAN-predicted future trajectory with uncertainty
                if len(track['path']) >= 3:  # Reduced requirement to 3 points
                    try:
                        future_points, uncertainties = self.predict_future_trajectory(track)
                        if frame_idx % 50 == 0 and track_id == list(active_tracks)[0]:  # Debug info for first track
                            print(f"Track {track_id}: Predicted {len(future_points)} points, KAN trained: {self.kan_trained}")
                    except Exception as e:
                        print(f"Warning: KAN prediction failed for track {track_id}: {e}")
                        future_points, uncertainties = [], []
                    
                    if len(future_points) > 0:
                        # Connect last actual position to first predicted position with very thick line
                        last_actual = tuple(map(int, track['path'][-1]))
                        first_predicted = tuple(map(int, future_points[0]))
                        cv2.line(output_frame, last_actual, first_predicted, color, 6)
                        
                        # Draw predicted trajectory with varying thickness based on uncertainty
                        for j in range(1, len(future_points)):
                            pt1 = tuple(map(int, future_points[j-1]))
                            pt2 = tuple(map(int, future_points[j]))
                            
                            # Calculate line thickness based on confidence (inverse of uncertainty)
                            if j-1 < len(uncertainties):
                                uncertainty = np.mean(uncertainties[j-1])
                                confidence = 1.0 - uncertainty
                                thickness = max(4, int(8 * confidence))  # Thicker lines
                            else:
                                thickness = 5  # Default thick line
                            
                            # Draw prediction line with confidence-based thickness
                            cv2.line(output_frame, pt1, pt2, color, thickness)
                            
                            # Add uncertainty indicators every 10 points
                            if j % 10 == 0 and j-1 < len(uncertainties):
                                uncertainty = np.mean(uncertainties[j-1])
                                # Draw uncertainty circle (larger = more uncertain)
                                uncertainty_radius = int(3 + uncertainty * 5)
                                cv2.circle(output_frame, pt2, uncertainty_radius, color, 1)
                        
                        # Different markers for different prediction stages
                        prediction_length = len(future_points)
                        
                        # Short-term prediction marker (triangle)
                        if prediction_length > 10:
                            short_term = tuple(map(int, future_points[9]))
                            triangle_pts = np.array([
                                [short_term[0], short_term[1] - 8],
                                [short_term[0] - 7, short_term[1] + 4],
                                [short_term[0] + 7, short_term[1] + 4]
                            ], np.int32)
                            cv2.fillPoly(output_frame, [triangle_pts], color)
                        
                        # Medium-term prediction marker (square)
                        if prediction_length > 25:
                            medium_term = tuple(map(int, future_points[24]))
                            cv2.rectangle(output_frame, 
                                        (medium_term[0] - 6, medium_term[1] - 6),
                                        (medium_term[0] + 6, medium_term[1] + 6), 
                                        color, -1)
                        
                        # Long-term prediction marker (diamond)
                        if prediction_length > 40:
                            long_term = tuple(map(int, future_points[39]))
                            diamond_pts = np.array([
                                [long_term[0], long_term[1] - 8],
                                [long_term[0] - 8, long_term[1]],
                                [long_term[0], long_term[1] + 8],
                                [long_term[0] + 8, long_term[1]]
                            ], np.int32)
                            cv2.fillPoly(output_frame, [diamond_pts], color)
                        
                        # Final prediction marker (star-like)
                        if len(future_points) > 0:
                            end_pred = tuple(map(int, future_points[-1]))
                            # Draw star-like marker
                            star_size = 8
                            for angle in range(0, 360, 45):
                                rad = np.radians(angle)
                                end_x = int(end_pred[0] + star_size * np.cos(rad))
                                end_y = int(end_pred[1] + star_size * np.sin(rad))
                                cv2.line(output_frame, end_pred, (end_x, end_y), color, 3)
            
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
        merged_count = len([c for c in dense_clusters if c.get('is_merged', False)])
        info_text = f"Frame {frame_idx} | Events: {total_event_pixels} | Clusters: {len(dense_clusters)} | Merged: {merged_count} | Tracks: {active_track_count}"
        cv2.putText(output_frame, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return output_frame

    def create_prediction_snapshot(self, rgb_frame, frame_idx, output_dir):
        """
        Creates a snapshot image with all current predicted trajectories drawn on it.
        """
        snapshot_frame = rgb_frame.copy()

        for track_id, track in self.tracked_clusters.items():
            if 'predicted_path' in track and len(track['predicted_path']) > 0:
                future_path = track['predicted_path']
                uncertainty = track.get('prediction_uncertainty', [0] * len(future_path))

                # Draw the predicted path
                for i in range(len(future_path) - 1):
                    cv2.line(snapshot_frame, tuple(future_path[i].astype(int)), tuple(future_path[i+1].astype(int)), (255, 0, 255), 2)

                # Draw uncertainty markers
                for i, point in enumerate(future_path):
                    radius = int(uncertainty[i] * 5) # Scale uncertainty for visibility
                    cv2.circle(snapshot_frame, tuple(point.astype(int)), radius, (0, 255, 255), 1)

        # Save the snapshot
        output_filename = os.path.join(output_dir, f"prediction_snapshot_frame_{frame_idx}.png")
        cv2.imwrite(output_filename, snapshot_frame)
        print(f"Saved prediction snapshot: {output_filename}")
        return snapshot_frame


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
    Create video showing OPTICS-based clustering with overlap merging:
    - Light green overlay for all event pixels
    - Green circles for single clusters, cyan for merged clusters
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
    total_merged_clusters = 0
    total_tracks_created = 0

    print("Processing frames for OPTICS event clustering with overlap merging...")
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

        # Count new tracks created this frame
        new_tracks_this_frame = len([a for a in track_assignments if tracker.tracked_clusters[a[0]]['frames_tracked'] == 1])
        total_tracks_created += new_tracks_this_frame

        # Train KAN predictor periodically and take a snapshot
        if frame_idx > 0 and frame_idx % 50 == 0:
            try:
                print("Training KAN predictor...")
                tracker.train_kan_predictor(dense_clusters)
                
                # After training, create a snapshot of the predictions
                if tracker.kan_trained:
                    print(f"Creating prediction snapshot at frame {frame_idx}...")
                    # Pass the output directory to the snapshot function
                    tracker.create_prediction_snapshot(rgb_frame, frame_idx, os.path.dirname(output_path))

            except Exception as e:
                print(f"Warning: KAN training or snapshot failed: {e}")
                traceback.print_exc()
                # Continue without training or snapshot

        # Create visualization for the video frame
        output_frame = tracker.draw_dense_clusters_and_tracks(rgb_frame, event_mask, dense_clusters, frame_idx)

        # Write frame
        out.write(output_frame)
        frame_idx += 1

        if frame_idx % 25 == 0:
            active_tracks = len([t for t in tracker.tracked_clusters.values() if t['active']])
            merged_count = len([c for c in dense_clusters if c.get('is_merged', False)])
            print(f"Frame {frame_idx}/{total_frames}: {len(dense_clusters)} clusters ({merged_count} merged), {active_tracks} active tracks")
        
        # Train KAN predictor more frequently with accumulated trajectory data
        if frame_idx % 10 == 0 and frame_idx > 0:  # Train every 10 frames instead of 100
            try:
                print("Training KAN predictor...")
                tracker.train_kan_predictor(tracker.tracked_clusters)
            except Exception as e:
                print(f"Warning: KAN training failed: {e}")
                # Continue without training

    # Cleanup
    cap.release()
    out.release()

    # Final statistics
    active_tracks = len([t for t in tracker.tracked_clusters.values() if t['active']])
    total_tracks = len(tracker.tracked_clusters)
    
    print(f"\nProcessing complete!")
    print(f"Frames processed: {frame_idx}")
    print(f"Total dense clusters found: {total_dense_clusters}")
    print(f"Total merged clusters: {total_merged_clusters}")
    print(f"Total tracks created: {total_tracks}")
    print(f"Final active tracks: {active_tracks}")
    print(f"Average clusters per frame: {total_dense_clusters/frame_idx:.1f}")
    print(f"Merge rate: {total_merged_clusters/total_dense_clusters*100:.1f}%")
    print(f"Output saved: {output_path}")

if __name__ == "__main__":
    # File paths
    input_video = "../sample_videos_to_test/clip_002.mp4"  # RGB input video (for background)
    h5_path = "../v2e-output/traffic_scenev.h5"  # Event data
    output_path = "../v2e-output/not_an_Aspline_baby_its_B_uwxu.mp4"
    
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
    print("\nCreating Event Cluster Tracking video with OPTICS and overlap merging...")
    create_event_cluster_tracking_video(input_video, event_masks, output_path)