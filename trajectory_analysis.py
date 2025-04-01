import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from tqdm.notebook import tqdm
import warnings

class TrajectoryAnalysis:
    """
    A class for analyzing vector field trajectories from a grid-based vector field.
    Specifically designed for belief dynamics analysis in social media data.
    """
    
    def __init__(self, df, x_col='from_x', y_col='from_y', dx_col='mu_dx', dy_col='mu_dy',
                 var_dx_col='var_dx', var_dy_col='var_dy', grid_size=None):
        """
        Initialize the trajectory analysis with a dataframe containing vector field data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe containing grid coordinates and vector components
        x_col, y_col : str
            Column names for x and y coordinates
        dx_col, dy_col : str
            Column names for vector components in x and y directions
        var_dx_col, var_dy_col : str
            Column names for variance of vector components (for uncertainty analysis)
        grid_size : tuple of int, optional
            Size of the grid (nx, ny). If None, will be inferred from the data.
        """
        self.df = df.copy()
        self.x_col = x_col
        self.y_col = y_col
        self.dx_col = dx_col
        self.dy_col = dy_col
        self.var_dx_col = var_dx_col
        self.var_dy_col = var_dy_col
        
        # Extract unique grid points
        self.x_unique = sorted(df[x_col].dropna().unique())
        self.y_unique = sorted(df[y_col].dropna().unique())
        
        # Infer or set grid size
        if grid_size is None:
            self.nx = len(self.x_unique)
            self.ny = len(self.y_unique)
        else:
            self.nx, self.ny = grid_size
            
        # Set up grid
        self.x_grid = np.array(self.x_unique)
        self.y_grid = np.array(self.y_unique)
        
        # Initialize trajectory storage
        self.trajectories = {}
        self.basins = None
        self.attractors = None
        
        # Debug info
        print(f"Grid shape: {self.nx} x {self.ny}")
        print(f"X range: {self.x_grid.min()} to {self.x_grid.max()}")
        print(f"Y range: {self.y_grid.min()} to {self.y_grid.max()}")
        
        # Prepare vector field arrays
        self._prepare_vector_field()
        
    def _prepare_vector_field(self):
        """
        Prepare the vector field data for interpolation.
        """
        # *** CRITICAL CHANGE: Using consistent [x, y] indexing throughout ***
        # Initialize arrays - shape is [x, y] -> U[i,j] is the vector at x_grid[i], y_grid[j]
        self.U = np.full((self.nx, self.ny), np.nan)
        self.V = np.full((self.nx, self.ny), np.nan)
        self.var_U = np.full((self.nx, self.ny), np.nan)
        self.var_V = np.full((self.nx, self.ny), np.nan)
        
        # Create a mapping from coordinates to indices
        x_idx = {x: i for i, x in enumerate(self.x_unique)}
        y_idx = {y: i for i, y in enumerate(self.y_unique)}
        
        # Fill the arrays
        for _, row in self.df.dropna(subset=[self.dx_col, self.dy_col]).iterrows():
            i = x_idx.get(row[self.x_col])
            j = y_idx.get(row[self.y_col])
            
            if i is not None and j is not None:
                self.U[i, j] = row[self.dx_col]
                self.V[i, j] = row[self.dy_col]
                
                if self.var_dx_col in row and self.var_dy_col in row:
                    self.var_U[i, j] = row[self.var_dx_col]
                    self.var_V[i, j] = row[self.var_dy_col]
        
        # Create a mask for valid data points
        self.valid_mask = ~np.isnan(self.U) & ~np.isnan(self.V)
        
        # Create a distance transform from valid data points
        from scipy.ndimage import distance_transform_edt
        self.distance_from_valid = distance_transform_edt(~self.valid_mask)
        
        # Prepare interpolation-friendly arrays by filling NaNs with zeros
        U_filled = np.nan_to_num(self.U, nan=0.0)
        V_filled = np.nan_to_num(self.V, nan=0.0)
        
        # *** CRITICAL CHANGE: No need to transpose for interpolation if we're consistent ***
        # Create interpolators - x coordinates are 1st dim, y are 2nd dim
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if np.any(self.valid_mask):
                self.u_interp = RegularGridInterpolator(
                    (self.x_grid, self.y_grid), 
                    U_filled,  # No transpose
                    bounds_error=False,
                    fill_value=0.0,
                    method='linear'
                )
                
                self.v_interp = RegularGridInterpolator(
                    (self.x_grid, self.y_grid), 
                    V_filled,  # No transpose
                    bounds_error=False,
                    fill_value=0.0,
                    method='linear'
                )
            else:
                self.u_interp = lambda p: np.zeros((len(p), 1))
                self.v_interp = lambda p: np.zeros((len(p), 1))
                
        # Also create an interpolator for the distance transform
        self.dist_interp = RegularGridInterpolator(
            (self.x_grid, self.y_grid),
            self.distance_from_valid,  # No transpose
            bounds_error=False,
            fill_value=np.max(self.distance_from_valid),
            method='linear'
        )
        
    def vector_field_func(self, t, state):
        """
        Vector field function for trajectory integration.
        
        Parameters:
        -----------
        t : float
            Time parameter (not used, but required by solve_ivp)
        state : array-like
            Current state [x, y]
            
        Returns:
        --------
        array-like
            Vector [dx, dy] at the given position
        """
        x, y = state
        
        # Early rejection for points far outside the grid
        if (x < np.min(self.x_grid) - 0.1 or x > np.max(self.x_grid) + 0.1 or 
            y < np.min(self.y_grid) - 0.1 or y > np.max(self.y_grid) + 0.1):
            return np.array([0.0, 0.0])
        
        # Add a small regularization to prevent exact boundary issues
        eps = 1e-10
        x = min(max(x, np.min(self.x_grid) + eps), np.max(self.x_grid) - eps)
        y = min(max(y, np.min(self.y_grid) + eps), np.max(self.y_grid) - eps)
        
        try:
            # Check distance from valid data points
            distance = float(self.dist_interp(np.array([[x, y]])).flatten()[0])
            max_allowed_distance = 1.0
            
            if distance > max_allowed_distance:
                # Return a small vector pointing back toward valid data
                eps = 1e-5
                dx_dist = float((self.dist_interp(np.array([[x + eps, y]])) - 
                                self.dist_interp(np.array([[x - eps, y]]))).flatten()[0]) / (2 * eps)
                dy_dist = float((self.dist_interp(np.array([[x, y + eps]])) - 
                                self.dist_interp(np.array([[x, y - eps]]))).flatten()[0]) / (2 * eps)
                
                # Normalize and invert
                grad_norm = np.sqrt(dx_dist**2 + dy_dist**2)
                if grad_norm > 1e-10:
                    return np.array([-dx_dist / grad_norm, -dy_dist / grad_norm]) * 0.01
                else:
                    return np.array([1e-10, 1e-10])
            
            # For valid data regions, proceed normally
            dx = float(self.u_interp(np.array([[x, y]])).flatten()[0])
            dy = float(self.v_interp(np.array([[x, y]])).flatten()[0])
            
            # Check for NaN or Inf values
            if np.isnan(dx) or np.isnan(dy) or np.isinf(dx) or np.isinf(dy):
                return np.array([1e-10, 1e-10])
            
            # Prevent excessive magnitudes
            mag = np.sqrt(dx**2 + dy**2)
            if mag > 10.0:
                scale_factor = 10.0 / mag
                dx = dx * scale_factor
                dy = dy * scale_factor
                
            # Add tiny noise to prevent exact stagnation points
            if abs(dx) < 1e-8 and abs(dy) < 1e-8:
                dx += np.random.normal(0, 1e-8)
                dy += np.random.normal(0, 1e-8)
                
            return np.array([dx, dy])
        except Exception as e:
            print(f"Error in vector_field_func at ({x}, {y}): {str(e)}")
            return np.array([1e-10, 1e-10])
            
    def slow_speed_event(self, threshold_speed):
        """
        Create an event function to detect when speed falls below threshold.
        
        Parameters:
        -----------
        threshold_speed : float
            Speed threshold for termination
            
        Returns:
        --------
        function
            Event function for solve_ivp
        """
        def event(t, state):
            x, y = state
            # Check boundary
            if (x < np.min(self.x_grid) or x > np.max(self.x_grid) or 
                y < np.min(self.y_grid) or y > np.max(self.y_grid)):
                return 0.0
                
            # Calculate speed
            try:
                dx = float(self.u_interp(np.array([[x, y]])).flatten()[0])
                dy = float(self.v_interp(np.array([[x, y]])).flatten()[0])
                speed = np.sqrt(dx**2 + dy**2)
                return speed - threshold_speed
            except Exception as e:
                print(f"Error in event function at ({x}, {y}): {str(e)}")
                return 0.0
        
        event.terminal = True
        return event
    
    def interior_slow_speed_event(self, threshold_speed, interior_buffer=0.05):
        """
        Create an event function that detects slow speed but only in the interior of the domain.
        
        Parameters:
        -----------
        threshold_speed : float
            Speed threshold for termination
        interior_buffer : float
            Buffer from the boundary to consider a point "interior" (as fraction of domain size)
            
        Returns:
        --------
        function
            Event function for solve_ivp
        """
        x_min, x_max = np.min(self.x_grid), np.max(self.x_grid)
        y_min, y_max = np.min(self.y_grid), np.max(self.y_grid)
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        x_buffer = interior_buffer * x_range
        y_buffer = interior_buffer * y_range
        
        def event(t, state):
            x, y = state
            
            # Check if we're near the boundary - if so, don't trigger the event
            if (x < x_min + x_buffer or x > x_max - x_buffer or 
                y < y_min + y_buffer or y > y_max - y_buffer):
                return 1.0  # Positive value means event not triggered
                
            # In the interior, check for slow speed
            try:
                dx = float(self.u_interp(np.array([[x, y]])).flatten()[0])
                dy = float(self.v_interp(np.array([[x, y]])).flatten()[0])
                speed = np.sqrt(dx**2 + dy**2)
                return speed - threshold_speed  # Event triggers when this goes to zero
            except Exception as e:
                print(f"Error in interior_event function at ({x}, {y}): {str(e)}")
                return 1.0  # Don't trigger
        
        event.terminal = True
        return event
            
    def compute_trajectories(self, threshold_speed=0.01, max_time=100.0, 
                             sample_points=100, sample_factor=1.0, solver='RK45',
                             rtol=None, atol=None, max_step=None,
                             enforce_interior_endpoints=True, interior_buffer=0.05,
                             debug_mode=False):
        """
        Compute trajectories from each grid cell.
        
        Parameters:
        -----------
        threshold_speed : float
            Speed threshold for terminating trajectories
        max_time : float
            Maximum integration time
        sample_points : int
            Number of points to sample along each trajectory
        sample_factor : float
            Fraction of grid cells to sample (for faster testing)
        solver : str
            ODE solver method ('RK45', 'Radau', 'LSODA', 'BDF', or 'DOP853')
        rtol : float, optional
            Relative tolerance for the solver (overrides default)
        atol : float, optional
            Absolute tolerance for the solver (overrides default)
        max_step : float, optional
            Maximum step size for the solver (overrides default)
        enforce_interior_endpoints : bool
            If True, enforce that trajectories don't end on boundaries
        interior_buffer : float
            Buffer from the boundary to consider a point "interior" (as fraction of domain size)
        debug_mode : bool
            If True, print additional diagnostic information
            
        Returns:
        --------
        self
        """
        # Clear previous results
        self.trajectories = {}
        
        # Create a slow speed event function that also checks for boundary
        if enforce_interior_endpoints:
            interior_speed_event = self.interior_slow_speed_event(threshold_speed, interior_buffer)
            event_func = interior_speed_event
        else:
            slow_speed = self.slow_speed_event(threshold_speed)
            event_func = slow_speed
        
        # Sample grid cells
        if sample_factor < 1.0:
            total_cells = self.nx * self.ny
            sample_size = int(total_cells * sample_factor)
            flat_indices = np.random.choice(total_cells, sample_size, replace=False)
            grid_indices = [(idx // self.ny, idx % self.ny) for idx in flat_indices]
        else:
            grid_indices = [(i, j) for i in range(self.nx) for j in range(self.ny)]
        
        # Set solver parameters based on the solver type
        solver_params = {
            'RK45': {
                'rtol': 1e-3,
                'atol': 1e-3,
                'max_step': 0.5,
                'vectorized': False
            },
            'Radau': {
                'rtol': 1e-2,
                'atol': 1e-2,
                'max_step': 1.0,
                'vectorized': False
            },
            'LSODA': {
                'rtol': 1e-1,
                'atol': 1e-1,
                'max_step': 2.0,
                'vectorized': False,
                'min_step': 1e-5
            },
            'BDF': {
                'rtol': 1e-1,
                'atol': 1e-1,
                'max_step': 0.5,
                'vectorized': False,
                'first_step': 1e-3
            },
            'DOP853': {
                'rtol': 1e-3,
                'atol': 1e-3,
                'max_step': 0.5,
                'vectorized': False
            }
        }
        
        # Get solver params or use defaults for RK45
        params = solver_params.get(solver, solver_params['RK45'])
        
        # Override with user-specified parameters if provided
        if rtol is not None:
            params['rtol'] = rtol
        if atol is not None:
            params['atol'] = atol
        if max_step is not None:
            params['max_step'] = max_step
        
        # Compute trajectories
        success_count = 0
        failure_count = 0
        boundary_endpoint_count = 0
        slow_endpoint_count = 0
        
        # Track speed statistics
        all_speeds = []
        
        for i, j in tqdm(grid_indices, desc=f"Computing trajectories with {solver}"):
            # Get starting coordinates
            x_start = self.x_grid[i]
            y_start = self.y_grid[j]
            
            # Skip cells with NaN vector
            if not self.valid_mask[i, j] or np.isnan(self.U[i, j]) or np.isnan(self.V[i, j]):
                continue
            
            # Skip cells with zero or very small vectors
            if abs(self.U[i, j]) < 1e-6 and abs(self.V[i, j]) < 1e-6:
                if debug_mode:
                    print(f"Skipping cell ({i},{j}) - near-zero vector")
                continue
            
            # Integrate the trajectory
            try:
                # Add tiny noise to starting points to avoid exact singularities
                x_start_noisy = x_start + np.random.normal(0, 1e-10)
                y_start_noisy = y_start + np.random.normal(0, 1e-10)
                
                # For LSODA, use a fallback mechanism with increasingly relaxed tolerances
                if solver == 'LSODA':
                    # Try first with standard parameters
                    try:
                        result = solve_ivp(
                            self.vector_field_func,
                            [0, max_time],
                            [x_start_noisy, y_start_noisy],
                            events=event_func,
                            dense_output=True,
                            method=solver,
                            **params
                        )
                    except Exception as e:
                        if "too much accuracy requested" in str(e):
                            # If that fails with precision error, try with extremely relaxed tolerances
                            fallback_params = params.copy()
                            fallback_params['rtol'] = 1e-1
                            fallback_params['atol'] = 1e-1
                            result = solve_ivp(
                                self.vector_field_func,
                                [0, max_time],
                                [x_start_noisy, y_start_noisy],
                                events=event_func,
                                dense_output=True,
                                method='RK45',  # Fall back to RK45 instead
                                **fallback_params
                            )
                        else:
                            raise
                # For BDF solver, use special handling
                elif solver == 'BDF':
                    try:
                        # First check if we're starting in a valid region
                        if self.valid_mask[i, j]:
                            result = solve_ivp(
                                self.vector_field_func,
                                [0, max_time],
                                [x_start_noisy, y_start_noisy],
                                events=event_func,
                                dense_output=True,
                                method=solver,
                                **params
                            )
                        else:
                            # Skip invalid starting points for BDF
                            raise ValueError("Starting point in invalid region")
                    except Exception as e:
                        if "infs or NaNs" in str(e) or "invalid region" in str(e):
                            # Fall back to RK45 for problematic points
                            fallback_params = solver_params['RK45'].copy()
                            fallback_params['rtol'] = params['rtol']
                            fallback_params['atol'] = params['atol']
                            result = solve_ivp(
                                self.vector_field_func,
                                [0, max_time],
                                [x_start_noisy, y_start_noisy],
                                events=event_func,
                                dense_output=True,
                                method='RK45',  # Fall back to RK45
                                **fallback_params
                            )
                        else:
                            raise
                else:
                    # For other solvers, use the standard approach
                    result = solve_ivp(
                        self.vector_field_func,
                        [0, max_time],
                        [x_start_noisy, y_start_noisy],
                        events=event_func,
                        dense_output=True,
                        method=solver,
                        **params
                    )
                
                # Track integration status
                if result.status == -1:
                    if debug_mode:
                        print(f"Integration failed for cell ({i},{j})")
                    failure_count += 1
                elif result.status == 0:
                    if debug_mode:
                        print(f"Completed full time span for ({i},{j})")
                    success_count += 1
                elif result.status == 1:
                    if debug_mode:
                        print(f"Terminated by event at t={result.t[-1]:.2f} for ({i},{j})")
                    success_count += 1
                
                # Check if we have a valid trajectory
                if len(result.t) > 1:
                    # Generate points along the trajectory
                    t_fine = np.linspace(0, result.t[-1], sample_points)
                    x_traj, y_traj = result.sol(t_fine)
                    
                    # Calculate final speed
                    final_speed = np.linalg.norm(self.vector_field_func(0, [x_traj[-1], y_traj[-1]]))
                    all_speeds.append(final_speed)
                    
                    # Check if endpoint is near boundary
                    x_range = np.max(self.x_grid) - np.min(self.x_grid)
                    y_range = np.max(self.y_grid) - np.min(self.y_grid)
                    x_buffer = interior_buffer * x_range
                    y_buffer = interior_buffer * y_range
                    
                    is_near_boundary = (
                        x_traj[-1] < np.min(self.x_grid) + x_buffer or 
                        x_traj[-1] > np.max(self.x_grid) - x_buffer or
                        y_traj[-1] < np.min(self.y_grid) + y_buffer or 
                        y_traj[-1] > np.max(self.y_grid) - y_buffer
                    )
                    
                    if is_near_boundary:
                        boundary_endpoint_count += 1
                        if debug_mode:
                            print(f"Endpoint near boundary for ({i},{j})")
                    
                    if final_speed < threshold_speed:
                        slow_endpoint_count += 1
                    
                    # Store the trajectory
                    self.trajectories[(i, j)] = {
                        'x': x_traj,
                        'y': y_traj,
                        't': t_fine,
                        'endpoint': (x_traj[-1], y_traj[-1]),
                        'speed_at_end': final_speed,
                        'is_near_boundary': is_near_boundary,
                        'termination_reason': 'slow_speed' if final_speed < threshold_speed else 'time_limit'
                    }
            except Exception as e:
                if debug_mode:
                    print(f"Error at ({i},{j}): {str(e)}")
                failure_count += 1
                continue
        
        # Compute speed statistics
        if all_speeds:
            speed_stats = {
                'min': np.min(all_speeds),
                'max': np.max(all_speeds),
                'mean': np.mean(all_speeds),
                'median': np.median(all_speeds),
                'p25': np.percentile(all_speeds, 25),
                'p75': np.percentile(all_speeds, 75)
            }
            
            print(f"\nSpeed statistics:")
            print(f"  Min: {speed_stats['min']:.6f}")
            print(f"  25th percentile: {speed_stats['p25']:.6f}")
            print(f"  Median: {speed_stats['median']:.6f}")
            print(f"  Mean: {speed_stats['mean']:.6f}")
            print(f"  75th percentile: {speed_stats['p75']:.6f}")
            print(f"  Max: {speed_stats['max']:.6f}")
            
            self.speed_stats = speed_stats
        
        print(f"\nTrajectories computed: {success_count} successful, {failure_count} failed")
        print(f"Endpoints: {boundary_endpoint_count} near boundary, {slow_endpoint_count} below speed threshold")
        
        return self
    
     # --- New Helper Methods for Attractor Identification ---
  
    def get_endpoint_weight(self, point, tol=None):
        """
        Determine the weight (i.e. number of trajectories) near a given endpoint.
        The tolerance is based on grid spacing.
        """
        if tol is None:
            tol = self.x_grid[1] - self.x_grid[0] if len(self.x_grid) > 1 else 1e-6
            tol = tol / 10.0
        count = 0
        pt = np.array(point)
        for traj in self.trajectories.values():
            endpoint = np.array(traj['endpoint'])
            if np.linalg.norm(endpoint - pt) < tol:
                count += 1
        return count

    def compute_weighted_medoid(self, points, weights):
        """
        Compute the weighted medoid of a set of points.
        The medoid minimizes the weighted sum of distances to all other points.
        """
        best_idx = 0
        best_weighted_sum = float('inf')
        for i in range(len(points)):
            dists = np.linalg.norm(points[i] - points, axis=1)
            weighted_sum = np.sum(weights * dists)
            if weighted_sum < best_weighted_sum:
                best_weighted_sum = weighted_sum
                best_idx = i
        return points[best_idx]

    def trajectory_length(self, traj_x, traj_y):
        diffs = np.sqrt(np.diff(traj_x)**2 + np.diff(traj_y)**2)
        return np.sum(diffs)

    def identify_attractors_hdbscan(self, min_cluster_size=5, min_samples=None, 
                               length_weight=0.4, speed_weight=0.6,confidence_factor=1.0):
        """
        Identify attractors using HDBSCAN with a confidence-adjusted distance metric.
        
        Parameters:
        -----------
        min_cluster_size : int
            Minimum size of clusters (HDBSCAN parameter)
        min_samples : int or None
            Minimum samples for core points
        length_weight, speed_weight : float
            Weights for confidence calculation
            
        Returns:
        --------
        self
        """
        import hdbscan
        from sklearn.metrics import pairwise_distances
        import numpy as np
        
        # Extract endpoints and calculate confidences
        endpoints = []
        traj_lengths = []
        speeds = []
        traj_indices = []
        
        for traj_idx, traj in self.trajectories.items():
            endpoints.append(traj['endpoint'])
            traj_lengths.append(self.trajectory_length(traj['x'], traj['y']))
            speeds.append(traj['speed_at_end'])
            traj_indices.append(traj_idx)
        
        endpoints = np.array(endpoints)
        traj_lengths = np.array(traj_lengths)
        speeds = np.array(speeds)
        
        if len(endpoints) == 0:
            print("No endpoints to cluster.")
            return self
            
        # Normalize trajectory lengths
        
        distance_matrix = pairwise_distances(endpoints)
        distance_matrix /= distance_matrix.max()

        # Compute confidence scores
        normalized_lengths = traj_lengths / max(traj_lengths.max(), 1)
        valid_speeds = speeds[~np.isnan(speeds)]
        log_speeds = np.log1p(valid_speeds) if len(valid_speeds) > 0 else np.zeros_like(speeds)

        # Normalize speed values
        max_speed = max(log_speeds.max(), 1)
        normalized_speeds = log_speeds / max_speed

        # Compute confidence weights
        confidence_weights = length_weight * normalized_lengths + speed_weight * normalized_speeds
     
        print(f"Confidence scores: min = {np.min(confidence_weights):.3f}, max = {np.max(confidence_weights):.3f}, "
            f"mean = {np.mean(confidence_weights):.3f}")
        # Adjust the distance matrix using confidence
        confidence_adjustment = np.outer(confidence_weights, confidence_weights)
        adjusted_distances = distance_matrix * (1 - confidence_factor) + confidence_adjustment * confidence_factor
        
        # Apply HDBSCAN with precomputed distances
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='precomputed',
            cluster_selection_epsilon=0.0,
            alpha=1.0
        )
        
        labels = clusterer.fit_predict(adjusted_distances)
        
        # Process clusters as before
        self.attractors = {}
        unique_labels = set(labels)
        
        # Count noise points
        noise_count = np.sum(labels == -1)
        print(f"HDBSCAN found {len(unique_labels) - (1 if -1 in unique_labels else 0)} clusters "
            f"and {noise_count} noise points ({noise_count/len(labels)*100:.1f}%)")
        
        # Process clusters
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
                
            mask = labels == label
            cluster_points = endpoints[mask]
            cluster_confidences = confidence_weights[mask]

            weights = np.array([self.get_endpoint_weight(pt) for pt in cluster_points])
            medoid = self.compute_weighted_medoid(cluster_points, weights)
            self.attractors[label] = {
                'position': medoid,
                'size': np.sum(mask),
                'points': cluster_points,
                'avg_confidence': np.mean(cluster_confidences)
            }
          
            
            print(f"Attractor {label}: {np.sum(mask)} points, avg confidence: "
                f"{np.mean(cluster_confidences):.3f}")
            
        # Assign trajectories to basins
        self.basins = {}
        for i, traj_idx in enumerate(traj_indices):
            if labels[i] != -1:  # Skip noise points
                self.basins[traj_idx] = labels[i]
        
        return self, confidence_weights


  

    def _update_basins_from_grid(self):
        """
        Update the basins dictionary based on the current basin_grid.
        This ensures consistency between the basin_grid and the basins dict.
        """
        # Reset the basins dictionary
        self.basins = {}
        
        # For each trajectory, assign it to the basin of its starting cell
        for (i, j), traj in self.trajectories.items():
            if 0 <= i < self.nx and 0 <= j < self.ny:
                basin_id = self.basin_grid[i, j]
                if basin_id != -1:
                    self.basins[(i, j)] = basin_id
    


   
    
    def assign_basins_to_grid(self):
        """
        Assign basins to the original grid cells based on identified attractors.
        Uses nearest neighbor interpolation to assign basins only to cells with valid data
        (where the vector components are not NaN).
        
        Returns:
        --------
        numpy.ndarray
            2D array of basin IDs (-1 if not in any basin, no valid data, or no attractors found)
        """
        import numpy as np
        from scipy.interpolate import NearestNDInterpolator
        
        # Initialize grid with -1 (no basin)
        basin_grid = np.full((self.nx, self.ny), -1)
        
        # If we have basins, assign them to the grid
        if self.basins is not None and len(self.basins) > 0:
            # First assign the cells we know directly from trajectories
            known_points = []
            known_basins = []
            
            for (i, j), basin_id in self.basins.items():
                # Ensure indices are within bounds
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    basin_grid[i, j] = basin_id
                    known_points.append((i, j))
                    known_basins.append(basin_id)
            
            # If we have any known points, interpolate for valid cells without a basin
            if known_points:
                # Create a mask for cells to interpolate (valid data but no basin yet)
                to_interpolate = (basin_grid == -1) & self.valid_mask
                
                # If we have cells to interpolate
                if np.any(to_interpolate):
                    # Convert known points to coordinate space
                    coord_points = np.array([(self.x_grid[i], self.y_grid[j]) for i, j in known_points])
                    
                    # Create interpolator from known basin assignments
                    interpolator = NearestNDInterpolator(coord_points, known_basins)
                    
                    # Get the indices of cells to interpolate
                    i_indices, j_indices = np.where(to_interpolate)
                    
                    # Generate coordinates for those cells
                    coords_to_interpolate = np.array([(self.x_grid[i], self.y_grid[j]) 
                                                    for i, j in zip(i_indices, j_indices)])
                    
                    # Interpolate basins for these coordinates
                    interpolated_basins = interpolator(coords_to_interpolate)
                    
                    # Assign the interpolated basins back to the grid
                    for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
                        basin_grid[i, j] = interpolated_basins[idx]
        
        # Store for later use
        self.basin_grid = basin_grid.astype(int)
        
        # Print diagnostic information
        unique_basins, counts = np.unique(self.basin_grid, return_counts=True)
        print(f"Basin assignment complete. Grid shape: {self.basin_grid.shape}")
        print(f"Valid data points: {np.sum(self.valid_mask)}")
        print(f"Assigned data points: {np.sum(basin_grid != -1)}")
        for basin_id, count in zip(unique_basins, counts):
            if basin_id == -1:
                print(f"  Unassigned: {count} cells")
            else:
                print(f"  Basin {basin_id}: {count} cells ({count/(self.nx*self.ny)*100:.1f}%)")
        
        return self.basin_grid

  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
   

## All of the following need to be fixed to use the trajectory_analysis member

class TrajectoryAnalysisInspector:

    def __init__(self, trajectory_analysis):
        self.trajectory_analysis = trajectory_analysis

    def plot_basin_heatmap(self, ax=None, cmap='tab10', alpha=0.7, show_attractors=True,
                    show_vector_field=False, vector_density=5, vector_color='black',
                    vector_alpha=0.3, slow_region_threshold=None):
        """
        Plot basins as a colored heatmap over the original grid.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        cmap : str or matplotlib.colors.Colormap
            Colormap for basin colors
        alpha : float
            Transparency of the basin colors
        show_attractors : bool
            If True, mark attractors on the map
        show_vector_field : bool
            If True, overlay the vector field
        vector_density : int
            Density of vector field if shown
        vector_color : str
            Color of vector field arrows if shown
        vector_alpha : float
            Transparency of vector field if shown
        slow_region_threshold : float, optional
            If provided, highlight regions slower than this threshold
            
        Returns:
        --------
        matplotlib.axes.Axes
        """

        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Make sure we have basins assigned to the grid
        if not hasattr(self, 'basin_grid'):
            self.assign_basins_to_grid()
        
        # Create a mask for areas with no basin
        no_basin_mask = (self.basin_grid == -1)
        
        # Create a copy of the basin grid for visualization
        vis_grid = self.basin_grid.copy()
        
        # Count unique basins (excluding -1)
        unique_basins = np.unique(vis_grid[vis_grid >= 0])
        n_basins = len(unique_basins)
        
        if n_basins > 0:
            # Create a custom colormap that reserves color 0 for no basin (white/transparent)
            base_cmap = plt.cm.get_cmap(cmap, n_basins)
            colors = [base_cmap(i) for i in range(n_basins)]
            basin_cmap = ListedColormap(colors)
            
            # Create a normalized colormap
            norm = mcolors.BoundaryNorm(np.arange(-0.5, n_basins + 0.5), basin_cmap.N)
            
            # Replace -1 with masked value for visualization
            vis_grid = np.ma.masked_where(no_basin_mask, vis_grid)
            
            # Create meshgrid with consistent indexing
            X, Y = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
            
            # Plot the basin heatmap
            basin_plot = ax.pcolormesh(X, Y, vis_grid, cmap=basin_cmap, norm=norm, 
                                    alpha=alpha, shading='auto')
            
            # Add a colorbar
            cbar = plt.colorbar(basin_plot, ax=ax)
            cbar.set_label('Basin ID')
            
            # If slow_region_threshold is provided, highlight slow regions
            if slow_region_threshold is not None:
                # Calculate vector magnitude
                magnitude = np.sqrt(self.U**2 + self.V**2)
                
                # Create a mask for slow regions
                slow_mask = magnitude < slow_region_threshold
                
                # Create a meshgrid for the heatmap
                Xs, Ys = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
                
                # Highlight slow regions with a hatched pattern
                slow_regions = ax.contourf(Xs, Ys, slow_mask.astype(float), 
                                        levels=[0.5, 1.5], colors='none', 
                                        hatches=['//'], alpha=0.3)
                
                # Add a label for slow regions
                ax.text(0.02, 0.98, f"Slow regions (speed < {slow_region_threshold:.6f})",
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            print("No basins found to visualize.")
        
        # Show attractors if requested
        if show_attractors and self.attractors is not None:
            for label, attractor in self.attractors.items():
                pos = attractor['position']
                size = attractor['size']
                
                # Scale marker size by the basin size
                marker_size = 5 + 20 * (size / max(1, max(att['size'] for att in self.attractors.values())))
                
                ax.plot(pos[0], pos[1], 'o', markersize=marker_size, 
                    color='black', markeredgecolor='white', alpha=1.0)
                
                # Add label
                ax.text(pos[0], pos[1], f'A{label}', fontsize=10, 
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Overlay vector field if requested
        if show_vector_field:
            # Sample the grid
            Xv, Yv = np.meshgrid(self.x_grid[::vector_density], self.y_grid[::vector_density], 
                            indexing='ij')
            Uv = self.U[::vector_density, ::vector_density]  # No transpose needed
            Vv = self.V[::vector_density, ::vector_density]
            
            # Plot the vector field
            ax.quiver(Xv, Yv, Uv, Vv, color=vector_color, alpha=vector_alpha, 
                    angles='xy', scale_units='xy', scale=1.0/0.25,
                    width=0.001, headwidth=3)
        
        # Set limits and labels
        ax.set_xlim(np.min(self.x_grid), np.max(self.x_grid))
        ax.set_ylim(np.min(self.y_grid), np.max(self.y_grid))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Basin Map')
        
        return ax

    def plot_vector_field(self, ax=None, scale=1.0, density=1, **kwargs):
        """
        Plot the vector field.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        scale : float
            Scaling factor for vectors
        density : int
            Density of the quiver plot (1 = every point, 2 = every other point, etc.)
        **kwargs : dict
            Additional arguments to pass to plt.quiver
            
        Returns:
        --------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            
        # *** CRITICAL CHANGE: Use indexing='ij' for consistent coordinate orientation ***
        # Sample the grid
        X, Y = np.meshgrid(self.x_grid[::density], self.y_grid[::density], indexing='ij')
        U = self.U[::density, ::density]  # No transpose 
        V = self.V[::density, ::density]
        
        # Plot vector field
        quiver_args = {
            'angles': 'xy',
            'scale_units': 'xy',
            'scale': 1.0 / scale,
            'width': 0.002,
            'headwidth': 3,
            'headlength': 4,
            'alpha': 0.6
        }
        quiver_args.update(kwargs)
        
        q = ax.quiver(X, Y, U, V, **quiver_args)
        
        # Add a quiver key for scale
        if 'color' in quiver_args:
            key_color = quiver_args['color']
        else:
            key_color = 'k'
            
        avg_mag = np.nanmean(np.sqrt(self.U**2 + self.V**2))
        ax.quiverkey(q, 0.9, 0.08, avg_mag, f'{avg_mag:.2f} units', 
                    labelpos='E', coordinates='figure', color=key_color)
        
        ax.set_xlim(np.min(self.x_grid), np.max(self.x_grid))
        ax.set_ylim(np.min(self.y_grid), np.max(self.y_grid))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        return ax
        
    def plot_trajectories(self, ax=None, sample_factor=0.1, line_width=0.5, 
                         color_by_basin=False, show_endpoints=True, show_attractors=True,
                         cmap='viridis', **kwargs):
        """
        Plot computed trajectories.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        sample_factor : float
            Fraction of trajectories to plot (for clearer visualization)
        line_width : float
            Width of trajectory lines
        color_by_basin : bool
            If True, color trajectories by their basin of attraction
        show_endpoints : bool
            If True, mark trajectory endpoints
        show_attractors : bool
            If True, mark attractor positions
        cmap : str or matplotlib.colors.Colormap
            Colormap for trajectories
        **kwargs : dict
            Additional arguments to pass to plotting functions
            
        Returns:
        --------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
            
        # Sample trajectories for visualization
        if sample_factor < 1.0:
            indices = list(self.trajectories.keys())
            sample_size = max(1, int(len(indices) * sample_factor))
            sampled_indices = np.random.choice(len(indices), sample_size, replace=False)
            traj_to_plot = [indices[i] for i in sampled_indices]
        else:
            traj_to_plot = list(self.trajectories.keys())
            
        # Create colormap for basins
        if color_by_basin and self.basins is not None:
            basin_colors = {}
            basin_cmap = cm.get_cmap(cmap)
            
            attractor_labels = sorted(list(self.attractors.keys()))
            for i, label in enumerate(attractor_labels):
                basin_colors[label] = basin_cmap(i / max(1, len(attractor_labels) - 1))
                
        # Get endpoints for diagnostic visualization
        all_endpoints_x = []
        all_endpoints_y = []
        all_speeds = []
        
        # Plot trajectories
        for idx in traj_to_plot:
            traj = self.trajectories[idx]
            x_traj = traj['x']
            y_traj = traj['y']
            
            # Determine color
            if color_by_basin and self.basins is not None and idx in self.basins:
                color = basin_colors[self.basins[idx]]
            else:
                color = kwargs.get('color', 'blue')
                
            # Plot the trajectory
            ax.plot(x_traj, y_traj, linewidth=line_width, color=color, alpha=0.7)
            
            # Collect endpoints for later analysis
            all_endpoints_x.append(x_traj[-1])
            all_endpoints_y.append(y_traj[-1])
            all_speeds.append(traj['speed_at_end'])
            
            # Mark endpoint
            if show_endpoints:
                ax.plot(x_traj[-1], y_traj[-1], 'o', markersize=3, 
                       color='red', alpha=0.5)
                       
        # Plot attractors
        if show_attractors and self.attractors is not None:
            for label, attractor in self.attractors.items():
                pos = attractor['position']
                size = attractor['size']
                
                # Scale marker size by the basin size
                marker_size = 5 + 20 * (size / max(1, max(att['size'] for att in self.attractors.values())))
                
                if color_by_basin:
                    color = basin_colors[label]
                else:
                    color = 'black'
                    
                ax.plot(pos[0], pos[1], 'o', markersize=marker_size, 
                       color=color, alpha=0.8)
                
                # Add label
                ax.text(pos[0], pos[1], f'A{label}', fontsize=10, 
                       ha='center', va='bottom', color='black',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Set limits and labels
        ax.set_xlim(np.min(self.x_grid), np.max(self.x_grid))
        ax.set_ylim(np.min(self.y_grid), np.max(self.y_grid))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        return ax
    
    def plot_combined(self, figsize=(15, 12), sample_factor=0.1, 
                     vector_density=3, **kwargs):
        """
        Create a combined plot with vector field and trajectories.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        sample_factor : float
            Fraction of trajectories to plot
        vector_density : int
            Density of the vector field
        **kwargs : dict
            Additional arguments to pass to plotting functions
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot vector field
        self.plot_vector_field(ax=ax, density=vector_density, color='lightgray', **kwargs)
        
        # Plot trajectories
        self.plot_trajectories(ax=ax, sample_factor=sample_factor, **kwargs)
        
        # Add title and legend
        ax.set_title('Vector Field and Trajectories')
        
        return fig
    
    def plot_vector_field_magnitude(self, ax=None, cmap='viridis', vmin=None, vmax=None, log_scale=False):
        """
        Plot the magnitude of the vector field as a heatmap.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        cmap : str or matplotlib.colors.Colormap
            Colormap for the heatmap
        vmin, vmax : float, optional
            Minimum and maximum values for the colormap
        log_scale : bool
            If True, use a logarithmic scale for the magnitude
            
        Returns:
        --------
        matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            
        # Calculate vector magnitude
        magnitude = np.sqrt(self.U**2 + self.V**2)
        
        # Apply log scale if requested
        if log_scale:
            magnitude = np.log10(magnitude + 1e-10)  # Add small value to avoid log(0)
            
        # *** CRITICAL CHANGE: Use indexing='ij' for consistent coordinate orientation ***
        # Create a meshgrid for the heatmap
        X, Y = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
        
        # Plot the heatmap
        im = ax.pcolormesh(X, Y, magnitude, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        
        # Add a colorbar
        cb = plt.colorbar(im, ax=ax)
        if log_scale:
            cb.set_label('Log10(Speed)')
        else:
            cb.set_label('Speed')
            
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Vector Field Magnitude')
        
        return ax
    
    def export_to_dataframe(self):
        """
        Export the trajectory data to a pandas DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with trajectory data
        """
        records = []
        
        for (i, j), traj in self.trajectories.items():
            x_start = self.x_grid[i]
            y_start = self.y_grid[j]
            x_end = traj['endpoint'][0]
            y_end = traj['endpoint'][1]
            
            basin_id = self.basins.get((i, j), -1) if self.basins is not None else -1
            
            records.append({
                'start_idx_x': i,
                'start_idx_y': j,
                'start_x': x_start,
                'start_y': y_start,
                'end_x': x_end,
                'end_y': y_end,
                'traj_length': len(traj['x']),
                'end_speed': traj['speed_at_end'],
                'basin_id': basin_id,
                'is_near_boundary': traj.get('is_near_boundary', False),
                'termination_reason': traj.get('termination_reason', 'unknown')
            })
            
        return pd.DataFrame(records)
    
    def trace_from_point(self, x, y, threshold_speed=0.01, max_time=100.0, 
                        sample_points=100):
        """
        Trace a trajectory from a specific point (not necessarily on the grid).
        
        Parameters:
        -----------
        x, y : float
            Starting coordinates
        threshold_speed : float
            Speed threshold for termination
        max_time : float
            Maximum integration time
        sample_points : int
            Number of points to sample along the trajectory
            
        Returns:
        --------
        dict
            Trajectory data
        """
        # Create a slow speed event function
        slow_speed = self.slow_speed_event(threshold_speed)
        
        # Integrate the trajectory
        try:
            result = solve_ivp(
                self.vector_field_func,
                [0, max_time],
                [x, y],
                events=slow_speed,
                dense_output=True,
                method='LSODA',  # Use LSODA by default
                rtol=1e-1,       # Relaxed tolerances
                atol=1e-1
            )
            
            # Check if we have a valid trajectory
            if len(result.t) > 1:
                # Generate points along the trajectory
                t_fine = np.linspace(0, result.t[-1], sample_points)
                x_traj, y_traj = result.sol(t_fine)
                
                # Return the trajectory
                return {
                    'x': x_traj,
                    'y': y_traj,
                    't': t_fine,
                    'endpoint': (x_traj[-1], y_traj[-1]),
                    'speed_at_end': np.linalg.norm(self.vector_field_func(0, [x_traj[-1], y_traj[-1]]))
                }
            else:
                return None
        except Exception as e:
            print(f"Error tracing from point ({x}, {y}): {str(e)}")
            return None