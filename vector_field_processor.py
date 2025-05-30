import cupy as cp
import logging
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.ndimage
import scipy.spatial
from scipy.spatial import Delaunay
import time
from functools import wraps
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

def log_execution_time(func):
    """Decorator to log method execution time"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.logger.isEnabledFor(logging.INFO):
            return func(self, *args, **kwargs)
            
        start_time = time.time()
        self.logger.info(f"Starting {func.__name__}")
        result = func(self, *args, **kwargs)
        end_time = time.time()
        self.logger.info(f"Finished {func.__name__} in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class ForwardTrajectoryAttractorAnalyzer:
    """
    An analyzer that extracts attractors from a vector field by analyzing forward trajectories.
    
    This class computes trajectories in the forward direction from grid points,
    extracts a short terminal segment from each trajectory to measure curvature,
    and then clusters the endpoints (augmented with curvature information) to
    identify distinct attractors. The basin of attraction is then the union of all
    trajectories that lead to each attractor.
    """
    def __init__(self, df, interpolation_factor=1, base_flow_threshold=1e-6, divergence_threshold=0.7,
                 terminal_segment_length=5, sensitivity=1.0, enable_logging=False):
        self.df = df
        self.interpolation_factor = interpolation_factor
        self.base_flow_threshold = base_flow_threshold
        self.divergence_threshold = divergence_threshold
        self.terminal_segment_length = terminal_segment_length
        self.sensitivity = sensitivity

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO if enable_logging else logging.WARNING)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.trajectories = []          # List of forward trajectories (each a list of (i,j) indices)
        self.trajectory_endpoints = []  # List of endpoint coordinates (x,y)
        self.endpoint_curvatures = []   # Curvature measures from the terminal segments
        self.attractors = None          # Representative attractor positions (centroid or medoid)
        self.basins = None              # Basin assignments for each grid cell

        self._prepare_data()
        x_range = self.x_fine.max() - self.x_fine.min()
        y_range = self.y_fine.max() - self.y_fine.min()
        self.base_step_size = min(
            x_range / (len(self.x_fine) - 1),
            y_range / (len(self.y_fine) - 1)
        ) / 2

    def _prepare_data(self):
        """Interpolate the vector field on a fine grid and compute a validity mask."""
        x = self.df['from_x'].values
        y = self.df['from_y'].values
        u = self.df['mu_dx'].values
        v = self.df['mu_dy'].values

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        # Center grid cells
        step_size_field = (x_max - x_min) / (self.interpolation_factor * (x_max - x_min) - 1)
        self.x_fine = np.linspace(x_min, x_max, int((x_max - x_min) * self.interpolation_factor), endpoint=True) + (step_size_field / 2)
        self.y_fine = np.linspace(y_min, y_max, int((y_max - y_min) * self.interpolation_factor), endpoint=True) + (step_size_field / 2)
        self.X_fine, self.Y_fine = np.meshgrid(self.x_fine, self.y_fine)

        # Interpolation using Delaunay triangulation
        points = np.column_stack((x, y))
        tri = Delaunay(points)
        self.U_interp = LinearNDInterpolator(points, u, fill_value=np.nan)
        self.V_interp = LinearNDInterpolator(points, v, fill_value=np.nan)
        self.U_fine = self.U_interp(self.X_fine, self.Y_fine)
        self.V_fine = self.V_interp(self.X_fine, self.Y_fine)

        fine_points = np.column_stack((self.X_fine.ravel(), self.Y_fine.ravel()))
        in_simplex = tri.find_simplex(fine_points) >= 0
        has_valid_values = ~np.isnan(self.U_fine.ravel()) & ~np.isnan(self.V_fine.ravel())
        self.valid_mask = (in_simplex & has_valid_values).reshape(self.X_fine.shape)
        self.U_fine[~self.valid_mask] = np.nan
        self.V_fine[~self.valid_mask] = np.nan
        self.grid_shape = self.X_fine.shape

    def discretize_point(self, point):
        """Convert a continuous point to grid indices."""
        j = np.clip(np.searchsorted(self.x_fine, point[0]), 0, len(self.x_fine)-1)
        i = np.clip(np.searchsorted(self.y_fine, point[1]), 0, len(self.y_fine)-1)
        return int(i), int(j)

    def rk4_step(self, x, y, step_size, U, V):
        """Perform a single RK4 integration step."""
        k1x = U(x, y)
        k1y = V(x, y)
        k2x = U(x + 0.5 * step_size * k1x, y + 0.5 * step_size * k1y)
        k2y = V(x + 0.5 * step_size * k1x, y + 0.5 * step_size * k1y)
        k3x = U(x + 0.5 * step_size * k2x, y + 0.5 * step_size * k2y)
        k3y = V(x + 0.5 * step_size * k2x, y + 0.5 * step_size * k2y)
        k4x = U(x + step_size * k3x, y + step_size * k3y)
        k4y = V(x + step_size * k3x, y + step_size * k3y)
        x_next = x + (step_size / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        y_next = y + (step_size / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)
        return x_next, y_next

    def generate_trajectory(self, start_index, max_steps=1000):
        """
        Generate a forward trajectory starting from a grid index.
        Integration stops if:
          - The point leaves the valid grid.
          - The local flow magnitude drops below a threshold.
          - The trajectory loops.
        """
        trajectory = [start_index]
        i, j = start_index
        x = self.X_fine[i, j]
        y = self.Y_fine[i, j]
        
        visited = set([start_index])
        step_size = self.base_step_size / self.sensitivity

        def U_func(x, y):
            ii, jj = self.discretize_point((x, y))
            if np.isnan(self.U_fine[ii, jj]):
                return 0.0
            return self.U_fine[ii, jj]
        def V_func(x, y):
            ii, jj = self.discretize_point((x, y))
            if np.isnan(self.V_fine[ii, jj]):
                return 0.0
            return self.V_fine[ii, jj]
            
        for _ in range(max_steps):
            ii, jj = self.discretize_point((x, y))
            if not (0 <= ii < self.grid_shape[0] and 0 <= jj < self.grid_shape[1]):
                break
            if not self.valid_mask[ii, jj]:
                break

            # Stop if flow magnitude is too low (i.e. converged to an attractor)
            flow_mag = np.hypot(U_func(x, y), V_func(x, y))
            if flow_mag < self.base_flow_threshold:
                self.logger.info(f"Trajectory stopped at grid ({ii},{jj}) with flow {flow_mag:.3e}")
                break

            x_next, y_next = self.rk4_step(x, y, step_size, U_func, V_func)
            ii_next, jj_next = self.discretize_point((x_next, y_next))
            if not (0 <= ii_next < self.grid_shape[0] and 0 <= jj_next < self.grid_shape[1]):
                break
            if not self.valid_mask[ii_next, jj_next]:
                break
            if (ii_next, jj_next) in visited:
                break
            visited.add((ii_next, jj_next))
            trajectory.append((ii_next, jj_next))
            x, y = x_next, y_next

        return trajectory

    def compute_terminal_curvature(self, trajectory):
        """
        Compute an average curvature for the terminal segment of the trajectory.
        Only the last terminal_segment_length points are used.
        """
        if len(trajectory) < self.terminal_segment_length + 1:
            return 0.0
        # Convert grid indices to continuous coordinates
        points = []
        for idx in trajectory[-(self.terminal_segment_length+1):]:
            i, j = idx
            points.append((self.X_fine[i, j], self.Y_fine[i, j]))
        points = np.array(points)
        angles = []
        for k in range(1, len(points)):
            dx = points[k][0] - points[k-1][0]
            dy = points[k][1] - points[k-1][1]
            angles.append(np.arctan2(dy, dx))
        angles = np.array(angles)
        angle_diffs = np.diff(angles)
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
        curvature = np.mean(np.abs(angle_diffs))
        return curvature

    @log_execution_time
    def analyze_forward_attractors(self):
        """
        Complete pipeline:
         1. Compute forward trajectories from valid grid points.
         2. Extract endpoint coordinates and compute terminal curvature.
         3. Cluster endpoints augmented with curvature information.
         4. Define attractors (centroids or medoids) and basins as all trajectories belonging to each attractor.
        
        Returns:
            attractors: (N x 2) array of attractor positions.
            basins: 2D array mapping grid cells to attractor labels.
        """
        valid_indices = list(zip(*np.where(self.valid_mask)))
        self.logger.info(f"Computing trajectories from {len(valid_indices)} valid grid points.")

        endpoints = []
        curvatures = []
        traj_collection = []
        for idx in valid_indices:
            traj = self.generate_trajectory(idx)
            if len(traj) == 0:
                continue
            traj_collection.append(traj)
            # Use the last grid point as the endpoint
            end_i, end_j = traj[-1]
            endpoints.append((self.X_fine[end_i, end_j], self.Y_fine[end_i, end_j]))
            curvature = self.compute_terminal_curvature(traj)
            curvatures.append(curvature)

        self.trajectories = traj_collection
        self.trajectory_endpoints = np.array(endpoints)
        self.endpoint_curvatures = np.array(curvatures)

        if len(endpoints) < 2:
            self.logger.warning("Not enough endpoints for clustering.")
            self.basins = np.full(self.grid_shape, -1, dtype=int)
            return np.array([]), self.basins

        # Form feature vector: [x, y, curvature_scaled]
        spatial_features = self.trajectory_endpoints
        curvature_feature = self.endpoint_curvatures.reshape(-1, 1)
        curvature_scale = self.sensitivity  # Adjust to weight curvature vs. spatial coordinates
        combined_features = np.hstack([spatial_features, curvature_scale * curvature_feature])

        # Cluster using DBSCAN
        avg_spacing = np.mean([self.x_fine[1] - self.x_fine[0], self.y_fine[1] - self.y_fine[0]])
        eps = avg_spacing * 2 / self.sensitivity
        clustering = DBSCAN(eps=eps, min_samples=1).fit(combined_features)
        labels = clustering.labels_
        unique_labels = np.unique(labels)
        self.logger.info(f"Clustering produced {len(unique_labels)} clusters.")

        # Compute attractors as centroids of clusters
        attractors = []
        for label in unique_labels:
            mask = (labels == label)
            cluster_points = spatial_features[mask]
            centroid = np.mean(cluster_points, axis=0)
            attractors.append(centroid)
        attractors = np.array(attractors)
        self.attractors = attractors

        # Assign each valid grid cell (that is part of any trajectory) to the nearest attractor.
        basin_map = np.full(self.grid_shape, -1, dtype=int)
        if attractors.size > 0:
            tree = cKDTree(attractors)
            # For each trajectory endpoint, assign all grid points along the trajectory to that attractor.
            for traj, label in zip(traj_collection, labels):
                # Find the representative attractor label for this endpoint.
                _, attractor_label = tree.query(self.trajectory_endpoints[0])
                # Alternatively, you can simply use the cluster label from DBSCAN.
                for (i, j) in traj:
                    basin_map[i, j] = label
        self.basins = basin_map
        return attractors, basin_map

    def plot_trajectory_endpoints(self, figsize=(12, 10)):
        """Plot the endpoints colored by cluster."""
        plt.figure(figsize=figsize)
        plt.scatter(self.trajectory_endpoints[:, 0], self.trajectory_endpoints[:, 1],
                    c='blue', s=50, alpha=0.7, label='Endpoints')
        if self.attractors is not None:
            plt.scatter(self.attractors[:, 0], self.attractors[:, 1],
                        c='red', s=100, label='Attractors')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Trajectory Endpoints and Attractors")
        plt.legend()
        plt.show()

    def plot_basins(self, figsize=(12, 10)):
        """Visualize the basins of attraction over the vector field."""
        plt.figure(figsize=figsize)
        plt.imshow(self.basins, extent=[self.x_fine.min(), self.x_fine.max(),
                                        self.y_fine.min(), self.y_fine.max()],
                   origin='lower', alpha=0.6, cmap='viridis')
        plt.streamplot(self.X_fine, self.Y_fine, self.U_fine, self.V_fine,
                       density=2, color='gray', arrowsize=1)
        if self.attractors is not None:
            plt.scatter(self.attractors[:, 0], self.attractors[:, 1],
                        c='red', s=100, label='Attractors')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Basins of Attraction")
        plt.legend()
        plt.show()

class DiscreteVectorFieldAnalyzer:
    def __init__(self, df, interpolation_factor=1, base_flow_threshold=1e-6, divergence_threshold = .7, enable_logging=False):
        self.df = df
        self.interpolation_factor = interpolation_factor
        self.base_flow_threshold = base_flow_threshold
        self.divergence_threshold = divergence_threshold

        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO if enable_logging else logging.WARNING)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Store trajectory analysis results
        self.trajectory_endpoints = None  # Will store (x, y) coordinates of endpoints
        self.endpoint_flows = None  # Will store flow magnitudes at endpoints
        self.endpoint_clusters = None  # Will store cluster labels
        
        self._prepare_data()

        # Calculate base step size for the rk4 step
        x_range = self.x_fine.max() - self.x_fine.min()
        y_range = self.y_fine.max() - self.y_fine.min()
        self.base_step_size = min(
            x_range / (len(self.x_fine) - 1),
            y_range / (len(self.y_fine) - 1)
        ) / 2

    def _prepare_data(self):
        # Extract original data
        x = self.df['from_x'].values
        y = self.df['from_y'].values
        u = self.df['mu_dx'].values
        v = self.df['mu_dy'].values

        # Create fine grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        # self.x_fine = np.linspace(x_min, x_max, int((x_max - x_min) * self.interpolation_factor))
        # self.y_fine = np.linspace(y_min, y_max, int((y_max - y_min) * self.interpolation_factor))

        # Center "pixels" to avoid systematic side bias
        step_size_field = (x_max - x_min) / (self.interpolation_factor * (x_max - x_min) - 1)
        self.x_fine = np.linspace(x_min, x_max, int((x_max - x_min) * self.interpolation_factor),endpoint=True) + (step_size_field / 2)
        self.y_fine = np.linspace(y_min, y_max, int((y_max - y_min) * self.interpolation_factor),endpoint=True) + (step_size_field / 2)


        self.X_fine, self.Y_fine = np.meshgrid(self.x_fine, self.y_fine)

        # Prepare points for interpolation
        points = np.column_stack((x, y))

        # Create Delaunay triangulation for the original points
        tri = Delaunay(points)

        # Interpolate U and V using LinearNDInterpolator
        self.U_interp = LinearNDInterpolator(points, u, fill_value=np.nan)
        self.V_interp = LinearNDInterpolator(points, v, fill_value=np.nan)

        # Apply interpolation
        self.U_fine = self.U_interp(self.X_fine, self.Y_fine)
        self.V_fine = self.V_interp(self.X_fine, self.Y_fine)

        # Create mask for valid data points
        # fine_points = np.column_stack((self.X_fine.ravel(), self.Y_fine.ravel()))
        # self.valid_mask = tri.find_simplex(fine_points) >= 0
        # self.valid_mask = self.valid_mask.reshape(self.X_fine.shape)

        # Create a more restrictive mask for valid data points
        fine_points = np.column_stack((self.X_fine.ravel(), self.Y_fine.ravel()))
        in_simplex = tri.find_simplex(fine_points) >= 0
        has_valid_values = ~np.isnan(self.U_fine.ravel()) & ~np.isnan(self.V_fine.ravel())
        self.valid_mask = (in_simplex & has_valid_values).reshape(self.X_fine.shape)

        # Apply mask to U_fine and V_fine
        self.U_fine[~self.valid_mask] = np.nan
        self.V_fine[~self.valid_mask] = np.nan

        self.grid_shape = self.X_fine.shape


    def generate_trajectory_rk4(self, start_index, sensitivity=1.0):
        """
        Generates a trajectory with divergence detection.
        """
        step_size = self.base_step_size / sensitivity
        max_steps = int(3 * np.sqrt(self.grid_shape[0]**2 + self.grid_shape[1]**2))
        
        trajectory = [start_index]
        current_x = self.X_fine[start_index[0], start_index[1]]
        current_y = self.Y_fine[start_index[0], start_index[1]]
        
        visited = set([start_index])
        
        # Keep track of flow directions for divergence detection
        flow_history = []
        
        def U_func(x, y):
            i, j = self.discretize_point((x, y))
            if np.isnan(self.U_fine[i, j]):
                return 0.0
            return self.U_fine[i, j]
            
        def V_func(x, y):
            i, j = self.discretize_point((x, y))
            if np.isnan(self.V_fine[i, j]):
                return 0.0
            return self.V_fine[i, j]
        
        flow_threshold = self.base_flow_threshold * sensitivity
        
        for step in range(max_steps):
            i, j = self.discretize_point((current_x, current_y))
            if not (0 <= i < self.grid_shape[0] and 0 <= j < self.grid_shape[1]):
                break
            if not self.valid_mask[i, j]:
                break
                
            # Get current flow vector
            u = U_func(current_x, current_y)
            v = V_func(current_x, current_y)
            delta_mag = np.hypot(u, v)
            
            if delta_mag < flow_threshold:
                self.logger.info(f"Stopping at ({i},{j}): flow magnitude {delta_mag:.3e} below threshold {flow_threshold:.3e}")
                break
                
            # Calculate flow direction and add to history
            flow_direction = np.arctan2(v, u)
            flow_history.append(flow_direction)
            
            # Check for sudden direction changes (potential separatrix crossing)
            if len(flow_history) > 5:  # Need some history to detect changes
                # Calculate rate of direction change
                direction_changes = np.diff(flow_history[-5:])
                # Wrap angles to [-pi, pi]
                direction_changes = np.mod(direction_changes + np.pi, 2*np.pi) - np.pi
                max_change = np.abs(direction_changes).max()
                
                # If we detect significant direction change and aren't near zero flow
                if max_change > 0:
                    #self.logger.info(f"Direction change at ({i},{j}): max_change={max_change:.3f}rad, flow_mag={delta_mag:.3e}")
                    # Check neighborhood for divergent flows
                    neighborhood = self.get_neighborhood_flows(i, j, radius=2)
                    if self.is_divergent_region(neighborhood,self.divergence_threshold):
                        self.logger.info(f"Stopping at ({i},{j}): divergent region detected")
                        break  # Stop at potential separatrix
            
            next_x, next_y = self.rk4_step(current_x, current_y, step_size, U_func, V_func)
            
            next_i, next_j = self.discretize_point((next_x, next_y))
            if not (0 <= next_i < self.grid_shape[0] and 0 <= next_j < self.grid_shape[1]):
                break
            if not self.valid_mask[next_i, next_j]:
                break
                
            if (next_i, next_j) in visited:
                break
            visited.add((next_i, next_j))
            
            trajectory.append((next_i, next_j))
            current_x, current_y = next_x, next_y
            
        return trajectory

    def get_neighborhood_flows(self, i, j, radius=2):
        """Get flow vectors in neighborhood of point (i,j)."""
        neighborhood = []
        flow_mags = []  # Store magnitudes for logging
        for di in range(-radius, radius+1):
            for dj in range(-radius, radius+1):
                ni, nj = i + di, j + dj
                if (0 <= ni < self.grid_shape[0] and 
                    0 <= nj < self.grid_shape[1] and 
                    self.valid_mask[ni, nj]):
                    u = self.U_fine[ni, nj]
                    v = self.V_fine[ni, nj]
                    if not (np.isnan(u) or np.isnan(v)):
                        mag = np.hypot(u, v)
                        flow_mags.append(mag)
                        neighborhood.append((u, v))
        
        if len(neighborhood) >= 4:
            avg_mag = np.mean(flow_mags)
            #self.logger.info(f"Neighborhood at ({i},{j}): {len(neighborhood)} flows, avg_magnitude={avg_mag:.3e}")
            
        return neighborhood

    def is_divergent_region(self, flows, threshold=0.7):
        """
        Check if flows in a region are divergent.
        Uses cosine similarity to check if flows are pointing in different directions.
        """
        if len(flows) < 4:  # Need minimum number of valid flows
            return False
            
        flows = np.array(flows)
        # Normalize flows
        magnitudes = np.sqrt(np.sum(flows**2, axis=1))
        normalized_flows = flows / magnitudes[:, np.newaxis]
        
        # Calculate pairwise cosine similarities
        similarities = normalized_flows @ normalized_flows.T
        
        # Get minimum similarity for logging
        min_similarity = similarities.min()
        is_divergent = min_similarity < threshold
        
        if is_divergent:
            self.logger.info(f"Divergent region detected: min_similarity={min_similarity:.3f}")
            
        return is_divergent

    def rk4_step(self,x, y, step_size, U, V):
        """
        One RK4 step at (x,y) with vector field (U,V).
        U and V are functions: U(x, y) -> dx/dt, V(x, y) -> dy/dt.
        step_size is the integration step.
        Returns (x_next, y_next).
        """
        # k1
        k1x = U(x, y)
        k1y = V(x, y)

        # k2
        k2x = U(x + 0.5 * step_size * k1x, y + 0.5 * step_size * k1y)
        k2y = V(x + 0.5 * step_size * k1x, y + 0.5 * step_size * k1y)

        # k3
        k3x = U(x + 0.5 * step_size * k2x, y + 0.5 * step_size * k2y)
        k3y = V(x + 0.5 * step_size * k2x, y + 0.5 * step_size * k2y)

        # k4
        k4x = U(x + step_size * k3x, y + step_size * k3y)
        k4y = V(x + step_size * k3x, y + step_size * k3y)

        # Combine
        x_next = x + (step_size / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        y_next = y + (step_size / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)

        return x_next, y_next


    @log_execution_time
    def analyze_vector_field(self, sensitivity=1.0):
        """
        Analyze the vector field with a single control parameter.
        
        Args:
            sensitivity: Float between 0.1 and 2.0 that scales the derived parameters.
                Lower values are more conservative (fewer, larger attractors),
                higher values allow for more, smaller attractors.
        """
        valid_indices = list(zip(*np.where(self.valid_mask)))
        args = [(index, sensitivity) for index in valid_indices]
        
        with mp.Pool() as pool:
            trajectories = pool.map(self.process_point_rk4, args)
            
        return trajectories


    def process_point_rk4(self, args):
        index, sensitivity = args
        return self.generate_trajectory_rk4(index, sensitivity=sensitivity)

    def discretize_point(self, point):
        """Convert a continuous point to the nearest grid index."""
        j = np.clip(np.searchsorted(self.x_fine, point[0]), 0, self.grid_shape[1] - 1)
        i = np.clip(np.searchsorted(self.y_fine, point[1]), 0, self.grid_shape[0] - 1)
        return (int(i), int(j))


    def vector_field(self, i, j):
        """Return the vector field at the given grid indices."""
        return np.array([self.U_fine[i, j], self.V_fine[i, j]])


    def find_attractors_and_basins(self, sensitivity=1.0, min_samples=5, similarity_threshold=0.7):
        """
        Find attractors and their basins using trajectory-based clustering.
        
        Args:
            sensitivity (float): Controls precision of trajectory generation
            min_samples (int): Minimum number of trajectories to form an attractor cluster
            similarity_threshold (float): Threshold for trajectory similarity (0-1)
            
        Returns:
            tuple: (attractors, basins)
                - attractors: dict mapping attractor IDs to their properties
                - basins: 2D array of attractor IDs for each point (-1 for no basin)
        """
        # Initialize basin array
        basins = np.full(self.grid_shape, -1, dtype=int)
        
        # Generate and analyze trajectories
        self.analyze_attractors(
            sensitivity=sensitivity,
            min_samples=min_samples,
            similarity_threshold=similarity_threshold
        )
        
        if self.endpoint_clusters is None or len(self.endpoint_clusters) == 0:
            self.logger.warning("No clusters found in trajectory analysis")
            return {}, basins
            
        # Create attractor dictionary
        attractors = {}
        for label in self.cluster_info.keys():
            info = self.cluster_info[label]
            
            # Only consider clusters with sufficiently low speed as attractors
            if info['min_speed'] > self.base_flow_threshold * 10:  # Adjustable threshold
                continue
                
            attractors[label] = {
                'center': info['center'],
                'size': info['size'],
                'avg_speed': info['avg_speed'],
                'min_speed': info['min_speed']
            }
            
        # If no valid attractors found, return empty results
        if not attractors:
            self.logger.warning("No valid attractors found")
            return attractors, basins
            
        # Generate trajectories from all valid points to identify basins
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if not self.valid_mask[i,j]:
                    continue
                    
                # Generate trajectory
                traj = self.generate_trajectory_rk4((i,j), sensitivity)
                if len(traj) < 3:
                    continue
                    
                # Compute trajectory features
                features = self.compute_trajectory_features(traj)
                if features is None:
                    continue
                    
                # Find the most similar attractor
                best_similarity = -1
                best_label = -1
                
                for label in attractors.keys():
                    # Create a synthetic feature set for the attractor
                    attractor_features = {
                        'endpoint': attractors[label]['center'],
                        'end_speed': attractors[label]['min_speed'],
                        'center': attractors[label]['center'],
                        'avg_radius': 0,  # Attractor is a point
                        'radius_std': 0,
                        'total_length': 0,
                        'avg_speed': attractors[label]['avg_speed'],
                        'min_speed': attractors[label]['min_speed']
                    }
                    
                    sim = self.compute_trajectory_similarity(features, attractor_features)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_label = label
                
                # Assign point to basin if similarity exceeds threshold
                if best_similarity > similarity_threshold * 0.8:  # Slightly lower threshold for basin assignment
                    basins[i,j] = best_label
        
        # Store results
        self.attractors = attractors
        self.basins = basins
        
        return attractors, basins

    def analyze_attractors(self, sensitivity=1.0, min_samples=5, similarity_threshold=0.7):
        """
        Analyze attractors using trajectory-based clustering.
        
        Args:
            sensitivity (float): Controls precision of trajectory generation
            min_samples (int): Minimum number of trajectories to form an attractor cluster
            similarity_threshold (float): Threshold for trajectory similarity (0-1)
        """
        # Generate trajectories from each valid point
        trajectories = []
        start_points = []
        
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if self.valid_mask[i,j]:
                    traj = self.generate_trajectory_rk4((i,j), sensitivity)
                    if len(traj) > 2:  # Minimum trajectory length
                        trajectories.append(traj)
                        start_points.append((i,j))
        
        # Cluster trajectories
        labels, features = self.cluster_trajectories(
            trajectories,
            min_samples=min_samples,
            similarity_threshold=similarity_threshold
        )
        
        # Store results
        self.trajectory_endpoints = np.array([
            features[i]['endpoint'] for i in range(len(features))
            if features[i] is not None
        ])
        
        self.endpoint_flows = np.array([
            features[i]['end_speed'] for i in range(len(features))
            if features[i] is not None
        ])
        
        self.endpoint_clusters = labels
        
        # Store additional information about clusters
        self.cluster_info = {}
        for label in set(labels):
            if label == -1:  # Noise points
                continue
                
            cluster_mask = (labels == label)
            cluster_features = [f for f, m in zip(features, cluster_mask) if m and f is not None]
            
            self.cluster_info[label] = {
                'size': np.sum(cluster_mask),
                'avg_speed': np.mean([f['avg_speed'] for f in cluster_features]),
                'min_speed': np.min([f['min_speed'] for f in cluster_features]),
                'center': np.mean([f['center'] for f in cluster_features], axis=0)
            }
        
        return self.endpoint_clusters

    def analyze_field(self, sensitivity=1.0, min_samples=5, similarity_threshold=0.7):
        """
        Main method to analyze the vector field and identify attractors and basins.
        
        Args:
            sensitivity (float): Controls precision of trajectory generation
            min_samples (int): Minimum number of trajectories to form an attractor cluster
            similarity_threshold (float): Threshold for trajectory similarity (0-1)
        """
        self.attractors, self.basins = self.find_attractors_and_basins(
            sensitivity=sensitivity,
            min_samples=min_samples,
            similarity_threshold=similarity_threshold
        )
        
        # Log analysis results
        n_attractors = len(self.attractors)
        n_assigned = np.sum(self.basins >= 0)
        total_valid = np.sum(self.valid_mask)
        
        self.logger.info(f"Found {n_attractors} attractors")
        self.logger.info(f"Assigned {n_assigned}/{total_valid} points to basins")
        
        for label, attractor in self.attractors.items():
            self.logger.info(f"Attractor {label}:")
            self.logger.info(f"  Center: ({attractor['center'][0]:.2f}, {attractor['center'][1]:.2f})")
            self.logger.info(f"  Size: {attractor['size']}")
            self.logger.info(f"  Min Speed: {attractor['min_speed']:.2e}")

    def plot_trajectory_endpoints(self, with_clusters=True, with_flows=True, with_qualities=True, figsize=(12, 10)):
        """
        Visualize trajectory endpoints and their clustering.
        
        Args:
            with_clusters (bool): Color points by cluster assignment
            with_flows (bool): Scale point sizes by flow magnitude
            with_qualities (bool): Show cluster qualities in legend
            figsize (tuple): Figure size
        """
        if self.trajectory_endpoints is None:
            self.logger.warning("No trajectory endpoints available. Run analysis first.")
            return
            
        plt.figure(figsize=figsize)
        
        # Plot vector field in background
        X, Y = np.meshgrid(self.x_fine, self.y_fine)
        U = self.U_fine
        V = self.V_fine
        
        # Normalize vectors for visualization
        magnitudes = np.sqrt(U**2 + V**2)
        U_norm = U / (magnitudes + 1e-10)
        V_norm = V / (magnitudes + 1e-10)
        
        # Plot vectors with reduced density
        stride = max(1, self.grid_shape[0] // 20)
        plt.quiver(X[::stride, ::stride], Y[::stride, ::stride],
                  U_norm[::stride, ::stride], V_norm[::stride, ::stride],
                  color='gray', alpha=0.2)
        
        # Prepare point sizes based on flow magnitudes
        if with_flows:
            sizes = 50 * (1 + 4 * self.endpoint_flows / np.max(self.endpoint_flows))
        else:
            sizes = 50
            
        # Plot endpoints
        if with_clusters and self.endpoint_clusters is not None:
            # Get unique clusters excluding noise (-1)
            unique_clusters = np.unique(self.endpoint_clusters)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
            
            for cluster_id, color in zip(unique_clusters, colors):
                mask = self.endpoint_clusters == cluster_id
                if cluster_id >= 0:
                    quality = self.cluster_qualities.get(cluster_id, 'N/A')
                    label = f'Cluster {cluster_id} (quality: {quality})' if with_qualities else f'Cluster {cluster_id}'
                else:
                    label = 'Noise'
                    
                plt.scatter(self.trajectory_endpoints[mask, 0],
                          self.trajectory_endpoints[mask, 1],
                          s=sizes[mask] if with_flows else sizes,
                          c=[color], label=label, alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(self.trajectory_endpoints[:, 0],
                       self.trajectory_endpoints[:, 1],
                       s=sizes, alpha=0.6)
            
        plt.title('Trajectory Endpoints\n(Pruned clusters shown as noise)')
        if with_flows:
            plt.colorbar(label='Normalized Flow Magnitude')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()


    def compute_trajectory_features(self, trajectory):
        """
        Compute features for a trajectory that are invariant to direction.
        
        Args:
            trajectory: List of (i, j) points representing the trajectory
            
        Returns:
            dict: Features of the trajectory
        """
        if len(trajectory) < 3:
            return None
            
        # Convert to continuous coordinates
        points = np.array([(self.X_fine[i,j], self.Y_fine[i,j]) for i,j in trajectory])
        
        # Compute endpoint
        endpoint = points[-1]
        
        # Compute flow speed at endpoint
        end_i, end_j = trajectory[-1]
        end_speed = np.hypot(self.U_fine[end_i, end_j], self.V_fine[end_i, end_j])
        
        # Compute center of mass of trajectory
        center = np.mean(points, axis=0)
        
        # Compute average distance from center (helps identify spiral vs straight paths)
        distances_from_center = np.linalg.norm(points - center, axis=1)
        avg_radius = np.mean(distances_from_center)
        radius_std = np.std(distances_from_center)
        
        # Compute path properties
        segments = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        total_length = np.sum(segment_lengths)
        
        # Compute average speed along trajectory
        speeds = np.array([
            np.hypot(self.U_fine[i,j], self.V_fine[i,j])
            for i,j in trajectory
        ])
        avg_speed = np.mean(speeds)
        min_speed = np.min(speeds)
        
        return {
            'endpoint': endpoint,
            'end_speed': end_speed,
            'center': center,
            'avg_radius': avg_radius,
            'radius_std': radius_std,
            'total_length': total_length,
            'avg_speed': avg_speed,
            'min_speed': min_speed
        }

    def compute_trajectory_similarity(self, features1, features2):
        """
        Compute similarity between two trajectories based on their features.
        Returns a similarity score between 0 and 1.
        """
        if features1 is None or features2 is None:
            return 0.0
            
        # Weight parameters - these can be tuned
        w_endpoint = 1.0
        w_center = 0.5
        w_speed = 2.0  # Higher weight on speed characteristics
        w_shape = 0.5
        
        # Normalize distances by the characteristic length of the field
        field_size = np.sqrt(
            (self.x_fine.max() - self.x_fine.min())**2 +
            (self.y_fine.max() - self.y_fine.min())**2
        )
        
        # Endpoint proximity (normalized by field size)
        endpoint_dist = np.linalg.norm(features1['endpoint'] - features2['endpoint']) / field_size
        endpoint_sim = np.exp(-endpoint_dist)
        
        # Center proximity (normalized)
        center_dist = np.linalg.norm(features1['center'] - features2['center']) / field_size
        center_sim = np.exp(-center_dist)
        
        # Speed similarity (both endpoint and average)
        speed_sim = np.exp(-np.abs(features1['end_speed'] - features2['end_speed']))
        avg_speed_sim = np.exp(-np.abs(features1['avg_speed'] - features2['avg_speed']))
        min_speed_sim = np.exp(-np.abs(features1['min_speed'] - features2['min_speed']))
        
        # Shape similarity (using radius statistics)
        radius_sim = np.exp(-np.abs(features1['avg_radius'] - features2['avg_radius']) / field_size)
        radius_std_sim = np.exp(-np.abs(features1['radius_std'] - features2['radius_std']) / field_size)
        
        # Combine similarities with weights
        similarity = (
            w_endpoint * endpoint_sim +
            w_center * center_sim +
            w_speed * (speed_sim + avg_speed_sim + min_speed_sim) / 3 +
            w_shape * (radius_sim + radius_std_sim) / 2
        ) / (w_endpoint + w_center + w_speed + w_shape)
        
        return similarity

    def cluster_trajectories(self, trajectories, min_samples=5, similarity_threshold=0.7):
        """
        Cluster trajectories using custom similarity metric.
        """
        # Compute features for all trajectories
        trajectory_features = [self.compute_trajectory_features(traj) for traj in trajectories]
        
        # Create similarity matrix
        n_trajectories = len(trajectories)
        similarity_matrix = np.zeros((n_trajectories, n_trajectories))
        
        for i in range(n_trajectories):
            for j in range(i+1, n_trajectories):
                sim = self.compute_trajectory_similarity(
                    trajectory_features[i],
                    trajectory_features[j]
                )
                similarity_matrix[i,j] = sim
                similarity_matrix[j,i] = sim
            similarity_matrix[i,i] = 1.0
            
        # Convert similarity to distance for DBSCAN
        distance_matrix = 1 - similarity_matrix
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(
            metric='precomputed',
            eps=1 - similarity_threshold,
            min_samples=min_samples
        ).fit(distance_matrix)
        
        return clustering.labels_, trajectory_features

    def plot_results(self, attractors, basins):
        plt.figure(figsize=(12, 10))
        masked_basins = np.ma.masked_where(~self.valid_mask, basins)
        plt.imshow(masked_basins, extent=[self.x_fine.min(), self.x_fine.max(), 
                                            self.y_fine.min(), self.y_fine.max()],
                    origin='lower', alpha=0.6, cmap='viridis')
        plt.streamplot(self.X_fine, self.Y_fine, self.U_fine, self.V_fine, 
                        density=3, color='gray', arrowsize=0.5)
        plt.scatter(attractors[:, 0], attractors[:, 1], c='red', s=100, label='Attractors')
        plt.colorbar(label='Basin of Attraction')
        plt.title('Vector Field with Basins of Attraction')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.show()

    def plot_flow_divergence(self, radius=2, threshold=0.7, figsize=(12, 10)):
        """
        Create a heatmap visualization of flow divergence across the vector field.
        
        Args:
            radius (int): Radius for neighborhood analysis
            threshold (float): Threshold for cosine similarity (same as in is_divergent_region)
            figsize (tuple): Figure size for the plot
            
        Returns:
            numpy.ndarray: The computed divergence scores (0 to 1 scale, where scores > 0
                          indicate regions that would be classified as divergent)
        """
        # Initialize divergence score matrix
        divergence_scores = np.zeros(self.grid_shape)
        
        # Compute divergence score for each valid point
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if self.valid_mask[i, j]:
                    # Get neighborhood flows
                    flows = self.get_neighborhood_flows(i, j, radius)
                    if len(flows) >= 4:  # Only compute if we have enough neighbors
                        flows = np.array(flows)
                        # Normalize flows
                        magnitudes = np.sqrt(np.sum(flows**2, axis=1))
                        normalized_flows = flows / magnitudes[:, np.newaxis]
                        # Calculate pairwise cosine similarities
                        similarities = normalized_flows @ normalized_flows.T
                        # Compute divergence score as how far below threshold the minimum similarity is
                        min_similarity = similarities.min()
                        # Scale to [0, 1] where values > 0 indicate divergent regions
                        divergence_scores[i, j] = max(0, (threshold - min_similarity) / threshold)
        
        # Create visualization
        plt.figure(figsize=figsize)
        # Use a custom colormap that emphasizes the threshold
        colors = [(0.95, 0.95, 0.95), (0.7, 0.7, 1), (1, 0, 0)]  # white -> light blue -> red
        positions = [0, 0.001, 1]
        cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('custom_divergence', 
                                                                      list(zip(positions, colors)))
        
        plt.imshow(divergence_scores, origin='lower', cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(label='Divergence Score\n(>0 indicates divergent region)')
        
        # Add vector field overlay
        X, Y = np.meshgrid(np.arange(self.grid_shape[1]), np.arange(self.grid_shape[0]))
        U = self.U_fine
        V = self.V_fine
        
        # Normalize vectors for visualization
        magnitudes = np.sqrt(U**2 + V**2)
        U_norm = U / (magnitudes + 1e-10)
        V_norm = V / (magnitudes + 1e-10)
        
        # Plot vectors with reduced density
        stride = max(1, self.grid_shape[0] // 20)  # Adjust density of arrows
        plt.quiver(X[::stride, ::stride], Y[::stride, ::stride],
                  U_norm[::stride, ::stride], V_norm[::stride, ::stride],
                  color='black', alpha=0.3)
        
        plt.title('Flow Divergence Heatmap\n(Red regions indicate divergent flows)')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.tight_layout()
        
        return divergence_scores

    def plot_trajectory_endpoints(self, with_clusters=True, with_flows=True, with_qualities=True, figsize=(12, 10)):
        """
        Visualize trajectory endpoints and their clustering.
        
        Args:
            with_clusters (bool): Color points by cluster assignment
            with_flows (bool): Scale point sizes by flow magnitude
            with_qualities (bool): Show cluster qualities in legend
            figsize (tuple): Figure size
        """
        if self.trajectory_endpoints is None:
            self.logger.warning("No trajectory endpoints available. Run analysis first.")
            return
            
        plt.figure(figsize=figsize)
        
        # Plot vector field in background
        X, Y = np.meshgrid(self.x_fine, self.y_fine)
        U = self.U_fine
        V = self.V_fine
        
        # Normalize vectors for visualization
        magnitudes = np.sqrt(U**2 + V**2)
        U_norm = U / (magnitudes + 1e-10)
        V_norm = V / (magnitudes + 1e-10)
        
        # Plot vectors with reduced density
        stride = max(1, self.grid_shape[0] // 20)
        plt.quiver(X[::stride, ::stride], Y[::stride, ::stride],
                  U_norm[::stride, ::stride], V_norm[::stride, ::stride],
                  color='gray', alpha=0.2)
        
        # Prepare point sizes based on flow magnitudes
        if with_flows:
            sizes = 50 * (1 + 4 * self.endpoint_flows / np.max(self.endpoint_flows))
        else:
            sizes = 50
            
        # Plot endpoints
        if with_clusters and self.endpoint_clusters is not None:
            # Get unique clusters excluding noise (-1)
            unique_clusters = np.unique(self.endpoint_clusters)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
            
            for cluster_id, color in zip(unique_clusters, colors):
                mask = self.endpoint_clusters == cluster_id
                if cluster_id >= 0:
                    quality = self.cluster_qualities.get(cluster_id, 'N/A')
                    label = f'Cluster {cluster_id} (quality: {quality})' if with_qualities else f'Cluster {cluster_id}'
                else:
                    label = 'Noise'
                    
                plt.scatter(self.trajectory_endpoints[mask, 0],
                          self.trajectory_endpoints[mask, 1],
                          s=sizes[mask] if with_flows else sizes,
                          c=[color], label=label, alpha=0.6)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(self.trajectory_endpoints[:, 0],
                       self.trajectory_endpoints[:, 1],
                       s=sizes, alpha=0.6)
            
        plt.title('Trajectory Endpoints\n(Pruned clusters shown as noise)')
        if with_flows:
            plt.colorbar(label='Normalized Flow Magnitude')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()


class VectorFieldProcessor:
    """
    A class that implements the complete pipeline for vector field analysis:
    1. Vector field creation
    2. Bayesian smoothing
    3. Attractor and basin identification
    4. Morphological cleanup
    """
    def __init__(self, grid_size=100, interpolation_factor=1, smoothing_radius=1, enable_logging=False):
        """
        Initialize the processor with configuration parameters.
        
        Args:
            grid_size (int): Size of the grid for vector field discretization
            interpolation_factor (float): Factor for interpolation refinement
            smoothing_radius (int): Radius for Bayesian smoothing
            enable_logging (bool): Whether to enable logging
        """
        self.grid_size = grid_size
        self.interpolation_factor = interpolation_factor
        self.smoothing_radius = smoothing_radius
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO if enable_logging else logging.WARNING)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Storage for processed data
        self.vector_field = None
        self.smoothed_field = None
        self.interpolated_field = None
        self.attractors = None
        self.basins = None
        self.streamlines = None
        
    def enable_logging(self, enable=True):
        """Enable or disable logging"""
        self.logger.setLevel(logging.INFO if enable else logging.WARNING)
        
    @log_execution_time
    def create_vector_field(self, df, x_col='x', y_col='y', time_col='dt', id_col='u_id'):
        """
        Create initial vector field from trajectory data.
        """
        # Store original dataframe
        self.df = df.copy()
        
        # Calculate transitions
        self.df.sort_values([id_col, time_col], inplace=True)
        
        # Normalize coordinates to grid
        self.df['x_grid'] = ((self.df[x_col] - self.df[x_col].min()) / 
                            (self.df[x_col].max() - self.df[x_col].min()) * 
                            (self.grid_size - 1)).astype(int)
        self.df['y_grid'] = ((self.df[y_col] - self.df[y_col].min()) / 
                            (self.df[y_col].max() - self.df[y_col].min()) * 
                            (self.grid_size - 1)).astype(int)
        
        # Calculate transitions in grid space (matching notebook exactly)
        self.df['from_x'] = self.df['x_grid'].shift(1)
        self.df['from_y'] = self.df['y_grid'].shift(1)
        self.df['to_x'] = self.df['x_grid']
        self.df['to_y'] = self.df['y_grid']
        
        # Filter transitions to same user
        transitions = self.df[self.df[id_col] == self.df[id_col].shift(1)].copy()
        transitions['dx'] = transitions['to_x'] - transitions['from_x']
        transitions['dy'] = transitions['to_y'] - transitions['from_y']
        
        # Create grids for mean and variance
        mu_dx = np.zeros((self.grid_size, self.grid_size))
        mu_dy = np.zeros((self.grid_size, self.grid_size))
        var_dx = np.zeros((self.grid_size, self.grid_size))
        var_dy = np.zeros((self.grid_size, self.grid_size))
        counts = np.zeros((self.grid_size, self.grid_size))
        
        # Aggregate statistics - compute actual variance, not normalized
        for _, row in transitions.iterrows():
            x, y = int(row['from_x']), int(row['from_y'])
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                dx, dy = row['dx'], row['dy']
                counts[x, y] += 1
                delta_x = dx - mu_dx[x, y]
                mu_dx[x, y] += delta_x / counts[x, y]
                var_dx[x, y] += delta_x * (dx - mu_dx[x, y])
                
                delta_y = dy - mu_dy[x, y]
                mu_dy[x, y] += delta_y / counts[x, y]
                var_dy[x, y] += delta_y * (dy - mu_dy[x, y])
        
        # Finalize variance calculation
        mask = counts > 1
        var_dx[mask] = var_dx[mask] / (counts[mask] - 1)  # Unbiased variance estimator
        var_dy[mask] = var_dy[mask] / (counts[mask] - 1)
        
        # Set variance to high value for single-count cells
        single_count = counts == 1
        var_dx[single_count] = np.max(var_dx[mask]) if np.any(mask) else 1.0
        var_dy[single_count] = np.max(var_dy[mask]) if np.any(mask) else 1.0
        
        self.vector_field = {
            'mu_dx': mu_dx,
            'mu_dy': mu_dy,
            'var_dx': var_dx,
            'var_dy': var_dy,
            'counts': counts,
            'mask': counts > 0,
            'x_min': self.df[x_col].min(),
            'x_max': self.df[x_col].max(),
            'y_min': self.df[y_col].min(),
            'y_max': self.df[y_col].max()
        }
        
        return transitions, self.vector_field

    def bayesian_update(self, mu_prior, var_prior, mean_sample, var_sample, count, epsilon=1e-10):
        """
        Perform Bayesian update of mean and variance.
        """
        if count == 0:
            return mu_prior, var_prior
        if count == 1:
            var_sample = np.max([var_prior, var_sample])  # Use maximum variance for single samples
        if var_sample == 0:
            var_sample = epsilon
            
        precision_prior = 1 / var_prior
        precision_sample = count / var_sample  # Scale precision by count
        combined_precision = precision_prior + precision_sample
        mu_posterior = (precision_prior * mu_prior + precision_sample * mean_sample) / combined_precision
        var_posterior = 1 / combined_precision
        return mu_posterior, var_posterior

    def spatial_weighted_average(self, mu_grid, var_grid, i, j, radius):
        """
        Compute spatially weighted average for a point.
        """
        weighted_mu, weighted_var, total_weight = 0, 0, 0
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni, nj = i + di, j + dj
                if (0 <= ni < mu_grid.shape[0] and 
                    0 <= nj < mu_grid.shape[1] and 
                    not np.isnan(mu_grid[ni, nj])):
                    # Include center point and use inverse distance weighting
                    dist = np.sqrt(di**2 + dj**2) + 1e-10
                    weight = 1 / dist if dist > 0 else 1.0
                    weighted_mu += weight * mu_grid[ni, nj]
                    weighted_var += weight * var_grid[ni, nj]
                    total_weight += weight
        
        if total_weight > 0:
            return weighted_mu / total_weight, weighted_var / total_weight
        return mu_grid[i, j], var_grid[i, j]

    @log_execution_time
    def smooth_vector_field(self):
        """
        Apply Bayesian updating and spatial smoothing to the vector field.
        """
        if self.vector_field is None:
            raise ValueError("Vector field must be created first")
            
        # First, convert the counts and sums to means and variances per cell
        mu_dx = np.zeros_like(self.vector_field['mu_dx'])
        mu_dy = np.zeros_like(self.vector_field['mu_dy'])
        var_dx = np.zeros_like(self.vector_field['var_dx'])
        var_dy = np.zeros_like(self.vector_field['var_dy'])
        
        # Apply Bayesian updating with prior mu=0, var=1
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.vector_field['mask'][i, j]:
                    # Get the raw values
                    dx_mean = self.vector_field['mu_dx'][i, j]
                    dy_mean = self.vector_field['mu_dy'][i, j]
                    dx_var = self.vector_field['var_dx'][i, j]
                    dy_var = self.vector_field['var_dy'][i, j]
                    count = self.vector_field['counts'][i, j]
                    
                    # Perform Bayesian updates
                    mu_dx[i, j], var_dx[i, j] = self.bayesian_update(0, 1, dx_mean, dx_var, count)
                    mu_dy[i, j], var_dy[i, j] = self.bayesian_update(0, 1, dy_mean, dy_var, count)
                else:
                    mu_dx[i, j] = np.nan
                    mu_dy[i, j] = np.nan
                    var_dx[i, j] = np.nan
                    var_dy[i, j] = np.nan
        
        # Apply spatial smoothing
        smoothed_mu_dx = np.copy(mu_dx)
        smoothed_mu_dy = np.copy(mu_dy)
        smoothed_var_dx = np.copy(var_dx)
        smoothed_var_dy = np.copy(var_dy)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if not np.isnan(mu_dx[i, j]):
                    smoothed_mu_dx[i, j], smoothed_var_dx[i, j] = self.spatial_weighted_average(
                        mu_dx, var_dx, i, j, self.smoothing_radius)
                    smoothed_mu_dy[i, j], smoothed_var_dy[i, j] = self.spatial_weighted_average(
                        mu_dy, var_dy, i, j, self.smoothing_radius)
        
        self.smoothed_field = {
            'mu_dx': smoothed_mu_dx,
            'mu_dy': smoothed_mu_dy,
            'var_dx': smoothed_var_dx,
            'var_dy': smoothed_var_dy,
            'mask': ~np.isnan(smoothed_mu_dx),
            'x_min': self.vector_field['x_min'],
            'x_max': self.vector_field['x_max'],
            'y_min': self.vector_field['y_min'],
            'y_max': self.vector_field['y_max']
        }
        
        return self.smoothed_field

    @log_execution_time
    def interpolate_field(self):
        """
        Interpolate the smoothed vector field to create a finer grid.
        """
        if self.smoothed_field is None:
            raise ValueError("Smoothed field must be created first")
        
        # Create fine grid
        x_min = self.smoothed_field['x_min']
        x_max = self.smoothed_field['x_max']
        y_min = self.smoothed_field['y_min']
        y_max = self.smoothed_field['y_max']
        
        # Center pixels to avoid systematic bias
        step_size = (x_max - x_min) / (self.interpolation_factor * (x_max - x_min) - 1)
        x_fine = np.linspace(x_min, x_max, int((x_max - x_min) * self.interpolation_factor), endpoint=True) + (step_size / 2)
        y_fine = np.linspace(y_min, y_max, int((y_max - y_min) * self.interpolation_factor), endpoint=True) + (step_size / 2)
        
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        
        # Create interpolation points
        x_grid = np.linspace(0, self.grid_size-1, self.grid_size)
        y_grid = np.linspace(0, self.grid_size-1, self.grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
        
        # Interpolate vector components
        U_interp = LinearNDInterpolator(points, self.smoothed_field['mu_dx'].ravel())
        V_interp = LinearNDInterpolator(points, self.smoothed_field['mu_dy'].ravel())
        
        # Normalize coordinates for interpolation
        X_norm = ((X_fine - x_min) / (x_max - x_min) * (self.grid_size - 1))
        Y_norm = ((Y_fine - y_min) / (y_max - y_min) * (self.grid_size - 1))
        
        # Apply interpolation
        U_fine = U_interp(X_norm, Y_norm)
        V_fine = V_interp(X_norm, Y_norm)
        
        # Create mask for valid data points
        valid_mask = ~np.isnan(U_fine) & ~np.isnan(V_fine)
        
        self.interpolated_field = {
            'X': X_fine,
            'Y': Y_fine,
            'U': U_fine,
            'V': V_fine,
            'x_fine': x_fine,
            'y_fine': y_fine,
            'mask': valid_mask
        }
        
        return self.interpolated_field

    @log_execution_time
    def identify_attractors(self, step_size=0.1, max_steps=1000, eps=0.5, min_samples=5, min_quality=10):
        """
        Identify attractors and their basins using RK4 integration.
        
        Args:
            step_size (float): Step size for RK4 integration
            max_steps (int): Maximum number of steps for trajectory computation
            eps (float): DBSCAN epsilon parameter for attractor clustering
            min_samples (int): DBSCAN min_samples parameter
            min_quality (int): Minimum trajectory length sum for valid attractors
        """
        if self.interpolated_field is None:
            raise ValueError("Interpolated field must be created first")
        
        def rk4_step(x, y):
            """RK4 integration step in real coordinates."""
            def U_func(x, y):
                i, j = self.discretize_point(x, y)
                if not (0 <= i < self.interpolated_field['U'].shape[0] and 
                       0 <= j < self.interpolated_field['U'].shape[1]):
                    return 0.0
                return self.interpolated_field['U'][i, j]
            
            def V_func(x, y):
                i, j = self.discretize_point(x, y)
                if not (0 <= i < self.interpolated_field['V'].shape[0] and 
                       0 <= j < self.interpolated_field['V'].shape[1]):
                    return 0.0
                return self.interpolated_field['V'][i, j]
            
            # k1
            k1x = U_func(x, y)
            k1y = V_func(x, y)
            
            # k2
            k2x = U_func(x + 0.5 * step_size * k1x, y + 0.5 * step_size * k1y)
            k2y = V_func(x + 0.5 * step_size * k1x, y + 0.5 * step_size * k1y)
            
            # k3
            k3x = U_func(x + 0.5 * step_size * k2x, y + 0.5 * step_size * k2y)
            k3y = V_func(x + 0.5 * step_size * k2x, y + 0.5 * step_size * k2y)
            
            # k4
            k4x = U_func(x + step_size * k3x, y + step_size * k3y)
            k4y = V_func(x + step_size * k3x, y + step_size * k3y)
            
            dx = (k1x + 2*k2x + 2*k3x + k4x) / 6
            dy = (k1y + 2*k2y + 2*k3y + k4y) / 6
            
            return x + dx * step_size, y + dy * step_size
        
        def discretize_point(x, y):
            """Convert real coordinates to nearest grid indices."""
            j = np.clip(np.searchsorted(self.interpolated_field['x_fine'], x), 
                       0, len(self.interpolated_field['x_fine']) - 1)
            i = np.clip(np.searchsorted(self.interpolated_field['y_fine'], y), 
                       0, len(self.interpolated_field['y_fine']) - 1)
            return i, j
        
        self.discretize_point = discretize_point
        
        # Generate trajectories from original grid points
        trajectories = []
        endpoints = []
        
        # Use original grid points from vector_field
        x_grid = np.linspace(self.vector_field['x_min'], self.vector_field['x_max'], self.grid_size)
        y_grid = np.linspace(self.vector_field['y_min'], self.vector_field['y_max'], self.grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.vector_field['mask'][i, j]:
                    x = X_grid[i, j]
                    y = Y_grid[i, j]
                    trajectory = [(x, y)]
                    
                    for _ in range(max_steps):
                        x_new, y_new = rk4_step(x, y)
                        i_new, j_new = discretize_point(x_new, y_new)
                        
                        if not (0 <= i_new < self.interpolated_field['U'].shape[0] and 
                               0 <= j_new < self.interpolated_field['U'].shape[1]):
                            break
                        if not self.interpolated_field['mask'][i_new, j_new]:
                            break
                        if np.sqrt((x_new - x)**2 + (y_new - y)**2) < 1e-6:
                            break
                        
                        trajectory.append((x_new, y_new))
                        x, y = x_new, y_new
                    
                    trajectories.append(trajectory)
                    endpoints.append(trajectory[-1])
        
        # Cluster endpoints
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(endpoints)
        unique_labels = np.unique(clustering.labels_)
        
        # Process clusters and identify attractors
        attractors = []
        valid_labels = {}  # Dictionary to map DBSCAN labels to new attractor indices
        new_label = 0
        
        for label in unique_labels:
            if label != -1:  # Skip noise points
                cluster_indices = np.where(clustering.labels_ == label)[0]
                quality = sum(len(trajectories[idx]) for idx in cluster_indices)
                
                if quality > min_quality:
                    cluster_points = np.array([endpoints[i] for i in cluster_indices])
                    attractors.append(np.mean(cluster_points, axis=0))
                    valid_labels[label] = new_label
                    new_label += 1
        
        self.attractors = np.array(attractors)
        self.streamlines = trajectories
        
        # Assign basin labels to original grid
        self.basins = np.full((self.grid_size, self.grid_size), -1)
        valid_point_count = 0
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.vector_field['mask'][i, j]:
                    if clustering.labels_[valid_point_count] in valid_labels:
                        self.basins[i, j] = valid_labels[clustering.labels_[valid_point_count]]
                    valid_point_count += 1
        
        # Fill pruned basins with nearest valid basin
        valid_points = []
        valid_labels_list = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.vector_field['mask'][i, j] and self.basins[i, j] != -1:
                    valid_points.append((X_grid[i, j], Y_grid[i, j]))
                    valid_labels_list.append(self.basins[i, j])
        
        if valid_points:
            tree = cKDTree(valid_points)
            
            pruned_points = []
            pruned_indices = []
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.vector_field['mask'][i, j] and self.basins[i, j] == -1:
                        pruned_points.append((X_grid[i, j], Y_grid[i, j]))
                        pruned_indices.append((i, j))
            
            if pruned_points:
                distances, indices = tree.query(pruned_points, k=1)
                for (i, j), idx in zip(pruned_indices, indices):
                    self.basins[i, j] = valid_labels_list[idx]
        
        return self.attractors, self.basins, self.streamlines
    
    @log_execution_time
    def clean_basins_gpu(self, kernel_size=9, threshold=0.8, max_iter=10, use_gpu=True):
        """
        Apply morphological cleanup to basin boundaries using convolution-based mode computation.
        
        Args:
            kernel_size (int): Size of the sliding window (odd integer)
            threshold (float): Fraction threshold for label changes
            max_iter (int): Maximum number of iterations
            use_gpu (bool): Whether to use GPU acceleration
        """
        if self.basins is None:
            raise ValueError("Basins must be identified first")
            
        def compute_local_mode_convolution(image, valid_label_set, device_xp):
            """
            Computes local mode and fraction using convolution on indicator arrays.
            
            Args:
                image: 2D array of labels (-1 for no data)
                valid_label_set: array of valid labels
                device_xp: numpy or cupy module depending on device
            """
            # Create the kernel
            kernel = device_xp.ones((kernel_size, kernel_size), dtype=int)
            
            # Use ndimage module for convolution with proper boundary handling
            if device_xp == np:
                from scipy import ndimage
                convolve_func = lambda x, k: ndimage.convolve(x, k, mode='constant', cval=0)
            else:
                import cupyx.scipy.ndimage as cupy_ndimage
                convolve_func = lambda x, k: cupy_ndimage.convolve(x, k, mode='constant', cval=0)
            
            # Convolve valid mask to get count of valid pixels
            valid_mask = (image != -1).astype(int)
            valid_count = convolve_func(valid_mask, kernel)
            
            # Count occurrences of each label
            count_stack = device_xp.zeros((len(valid_label_set),) + image.shape, dtype=int)
            for idx, label in enumerate(valid_label_set):
                label_mask = (image == label).astype(int)
                count_stack[idx] = convolve_func(label_mask, kernel)
            
            # Find label with maximum count
            max_counts = device_xp.max(count_stack, axis=0)
            mode_indices = device_xp.argmax(count_stack, axis=0)
            mode_arr = device_xp.array(valid_label_set)[mode_indices]
            
            # Compute fraction
            fraction_arr = device_xp.zeros_like(image, dtype=float)
            nonzero = valid_count > 0
            fraction_arr[nonzero] = max_counts[nonzero] / valid_count[nonzero]
            
            # Set mode to -1 where no valid neighbors
            mode_arr[~nonzero] = -1
            
            return mode_arr, fraction_arr
        
        try:
            if use_gpu:
                device_xp = cp
                consolidated = cp.array(self.basins)
            else:
                device_xp = np
                consolidated = self.basins.copy()
                
            # Get unique valid labels
            modes = device_xp.unique(consolidated)
            modes = modes[modes != -1]
            
            for iteration in range(max_iter):
                # Compute modes and fractions
                mode_arr, fraction_arr = compute_local_mode_convolution(
                    consolidated, modes, device_xp
                )
                
                # Create mask for pixels that should change
                valid_mask = consolidated != -1
                change_mask = (
                    (fraction_arr >= threshold) & 
                    (consolidated != mode_arr) & 
                    valid_mask
                )
                
                changes = int(device_xp.sum(change_mask))
                print(f"Iteration {iteration+1}: {changes} pixels updated")
                
                if changes == 0:
                    break
                    
                # Update pixels
                consolidated[change_mask] = mode_arr[change_mask]
            
            # Transfer back to CPU if needed
            self.basins = cp.asnumpy(consolidated) if use_gpu else consolidated
            
        except ImportError:
            warnings.warn("GPU acceleration not available, using CPU", RuntimeWarning)
            # Fall back to CPU implementation
            self.basins = compute_local_mode_convolution(
                self.basins.copy(),
                np.unique(self.basins[self.basins != -1]),
                np
            )[0]
        
        return self.basins
    
    @log_execution_time
    def get_features(self):
        """
        Extract features from the processed vector field for predictive modeling.
        
        Returns:
            dict: Dictionary containing various features
        """
        if any(x is None for x in [self.vector_field, self.smoothed_field, self.interpolated_field, self.attractors, self.basins]):
            raise ValueError("All processing steps must be completed first")
            
        features = {
            'n_attractors': len(self.attractors),
            'attractor_locations': self.attractors,
            'basin_sizes': np.array([np.sum(self.basins == i) for i in range(len(self.attractors))]),
            'vector_field_stats': {
                'mean_magnitude': np.sqrt(
                    self.interpolated_field['U']**2 + 
                    self.interpolated_field['V']**2
                )[self.interpolated_field['mask']].mean(),
                'variance': np.var(
                    self.interpolated_field['U'][self.interpolated_field['mask']]
                ) + np.var(
                    self.interpolated_field['V'][self.interpolated_field['mask']]
                )
            }
        }
        
        return features
    
    def vector_field_to_dataframe(self, field_dict=None):
        """
        Convert vector field data to a pandas DataFrame for visualization.
        If field_dict is None, uses self.vector_field.
        """
        if field_dict is None:
            if self.vector_field is None:
                raise ValueError("Vector field must be created first")
            field_dict = self.vector_field
            
        # Create coordinate meshgrid
        x_coords = np.linspace(field_dict['x_min'], field_dict['x_max'], self.grid_size)
        y_coords = np.linspace(field_dict['y_min'], field_dict['y_max'], self.grid_size)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Create DataFrame with all necessary columns
        df = pd.DataFrame({
            'from_x': X.flatten(),
            'from_y': Y.flatten(),
            'mu_dx': field_dict['mu_dx'].T.flatten(),  # Transpose to match original orientation
            'mu_dy': field_dict['mu_dy'].T.flatten(),
            'var_dx': field_dict['var_dx'].T.flatten() if 'var_dx' in field_dict else np.zeros_like(X.flatten()),
            'var_dy': field_dict['var_dy'].T.flatten() if 'var_dy' in field_dict else np.zeros_like(X.flatten()),
            'mask': field_dict['mask'].T.flatten()
        })
        
        # Filter out invalid points
        df = df[df['mask']]
        
        return df
    
    def plot_vector_field_with_var(self, field_dict=None, scale=1, figsize=(10, 8)):
        """
        Plot vector field with variance indication, matching the original notebook style.
        
        Args:
            field_dict: Dictionary containing field data. If None, uses self.vector_field
            scale: Scale factor for vector sizes
            figsize: Figure size tuple
        """
        df = self.vector_field_to_dataframe(field_dict)
        
        # Calculate average variance
        df['avg_var'] = (df['var_dx'] + df['var_dy']) / 2
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        Q = ax.quiver(
            df['from_x'], df['from_y'],
            df['mu_dx'] * scale, df['mu_dy'] * scale,
            df['avg_var'],
            angles='xy', scale_units='xy', scale=1,
            cmap='viridis', pivot='mid'
        )
        
        # Add colorbar
        cbar = fig.colorbar(Q, ax=ax)
        cbar.set_label('Average Variance of Vector Components')
        
        # Set labels and title
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('Vector Field with Variance Indication')
        plt.grid(True)
        
        return fig, ax
    
    def plot_vector_field_simple(self, field_dict=None, scale=1, figsize=(10, 8)):
        """
        Plot vector field without variance indication.
        
        Args:
            field_dict: Dictionary containing field data. If None, uses self.vector_field
            scale: Scale factor for vector sizes
            figsize: Figure size tuple
        """
        df = self.vector_field_to_dataframe(field_dict)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.quiver(
            df['from_x'], df['from_y'],
            df['mu_dx'] * scale, df['mu_dy'] * scale,
            angles='xy', scale_units='xy', scale=1,
            pivot='mid'
        )
        
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('Vector Field')
        plt.grid(True)
        
        return fig, ax
    
    def plot_basins(self, with_attractors=True, figsize=(10, 8)):
        """
        Plot basins of attraction and optionally overlay attractors.
        
        Args:
            with_attractors: Whether to plot attractors as scatter points
            figsize: Figure size tuple
        """
        if self.basins is None:
            raise ValueError("Basins must be identified first")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create coordinate meshgrid for the original grid
        x_grid = np.linspace(self.vector_field['x_min'], self.vector_field['x_max'], self.grid_size)
        y_grid = np.linspace(self.vector_field['y_min'], self.vector_field['y_max'], self.grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Plot basins with proper masking
        masked_basins = np.ma.masked_where(~self.vector_field['mask'], self.basins)
        im = ax.pcolormesh(X, Y, masked_basins, cmap='viridis', alpha=0.6)
        plt.colorbar(im, ax=ax, label='Basin Label')
        
        if with_attractors and self.attractors is not None:
            ax.scatter(self.attractors[:, 0], self.attractors[:, 1], 
                      c='red', s=100, label='Attractors')
            ax.legend()
        
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('Basins of Attraction')
        
        return fig, ax
