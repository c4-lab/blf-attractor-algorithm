import numpy as np
from vector_field_processor import DiscreteVectorFieldAnalyzer
import logging
from scipy.spatial import cKDTree
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, FrozenSet
import numpy.typing as npt

@dataclass(frozen=True)
class TrajectoryInfo:
    """Store information about a trajectory and its endpoint."""
    points: FrozenSet[Tuple[int, int]]  # List of (i,j) points in the trajectory made hashable
    endpoint: Tuple[int, int]      # The final point (i,j)
    endpoint_speed: float          # Vector magnitude at endpoint
    approach_angles: Tuple[float, ...]   # List of angles in the final approach made hashable

    @classmethod
    def from_trajectory(cls, points: List[Tuple[int, int]], endpoint_speed: float, approach_angles: List[float]):
        """Factory method to create TrajectoryInfo from mutable sequences."""
        return cls(
            points=frozenset(points),
            endpoint=points[-1],
            endpoint_speed=endpoint_speed,
            approach_angles=tuple(approach_angles)
        )

class TrajectoryBasedAttractorAnalyzer(DiscreteVectorFieldAnalyzer):
    def __init__(self, df, interpolation_factor=1, base_flow_threshold=1e-6,
                 slow_point_threshold=0.1, curvature_threshold=0.5,
                 min_trajectory_length=5, enable_logging=False):
        """
        Initialize the analyzer with configuration parameters.
        
        Args:
            df: DataFrame with vector field data
            interpolation_factor: Factor for interpolation refinement
            base_flow_threshold: Threshold for stopping trajectory integration
            slow_point_threshold: Threshold for identifying slow points (as fraction of max speed)
            curvature_threshold: Threshold for separating trajectories based on curvature
            min_trajectory_length: Minimum length for a valid trajectory
            enable_logging: Whether to enable detailed logging
        """
        super().__init__(df, interpolation_factor, base_flow_threshold, enable_logging=enable_logging)
        
        self.slow_point_threshold = slow_point_threshold
        self.curvature_threshold = curvature_threshold
        self.min_trajectory_length = min_trajectory_length
        
        # Storage for analysis results
        self.trajectories: Set[TrajectoryInfo] = set()
        self.slow_points: Set[Tuple[int, int]] = set()
        self.attractor_groups: List[Set[Tuple[int, int]]] = []

    def get_vector_magnitude(self, i: int, j: int) -> float:
        """Calculate vector magnitude at grid point (i,j)."""
        return np.hypot(self.U_fine[i,j], self.V_fine[i,j])
    
    def identify_slow_points(self) -> Set[Tuple[int, int]]:
        """
        Identify grid points where vector magnitude is below threshold.
        Returns set of (i,j) coordinates of slow points.
        """
        magnitudes = np.hypot(self.U_fine, self.V_fine)
        threshold = self.slow_point_threshold * np.nanmax(magnitudes)
        slow_mask = (magnitudes < threshold) & self.valid_mask
        slow_points = set(zip(*np.where(slow_mask)))
        self.slow_points = slow_points
        return slow_points
    
    def compute_approach_angles(self, trajectory: List[Tuple[int, int]], 
                              window_size: int = 5) -> List[float]:
        """
        Compute approach angles for the final segment of a trajectory.
        Uses a window of points to compute average direction changes.
        """
        if len(trajectory) < window_size:
            return []
            
        # Get final window_size points
        final_points = trajectory[-window_size:]
        angles = []
        
        for i in range(len(final_points)-1):
            p1 = final_points[i]
            p2 = final_points[i+1]
            # Convert to real coordinates
            x1, y1 = self.X_fine[p1[0], p1[1]], self.Y_fine[p1[0], p1[1]]
            x2, y2 = self.X_fine[p2[0], p2[1]], self.Y_fine[p2[0], p2[1]]
            angle = np.arctan2(y2-y1, x2-x1)
            angles.append(angle)
            
        return angles
    
    def are_trajectories_compatible(self, angles1: List[float], 
                                  angles2: List[float]) -> bool:
        """
        Determine if two trajectories should be grouped together based on their
        approach angles. Returns True if trajectories should be grouped.
        """
        if not angles1 or not angles2:
            return False
            
        # Compare the average approach directions
        mean_angle1 = np.mean(angles1)
        mean_angle2 = np.mean(angles2)
        
        # Compute angle difference (handling circular nature of angles)
        angle_diff = np.abs(np.mod(mean_angle1 - mean_angle2 + np.pi, 2*np.pi) - np.pi)
        
        # Check if trajectories are approaching from similar directions
        # or if they form a spiral pattern (approximately opposite directions)
        return angle_diff < self.curvature_threshold or \
               abs(angle_diff - np.pi) < self.curvature_threshold
    
    def group_trajectories(self) -> List[Set[Tuple[int, int]]]:
        """
        Group trajectories based on their endpoints and approach patterns.
        Returns list of sets, where each set contains endpoint coordinates
        belonging to the same attractor.
        """
        # First, group by connected slow points
        slow_point_groups = self._group_connected_slow_points()
        
        # For each group of connected slow points, analyze trajectories
        final_groups = []
        
        for slow_group in slow_point_groups:
            # Get all trajectories ending in this group
            group_trajectories = [t for t in self.trajectories if t.endpoint in slow_group]
            
            if not group_trajectories:
                continue
                
            # Start with each trajectory in its own subgroup
            subgroups: List[Set[TrajectoryInfo]] = [{t} for t in group_trajectories]
            
            # Iteratively merge compatible subgroups
            while True:
                merged = False
                for i in range(len(subgroups)):
                    if merged:
                        break
                    for j in range(i+1, len(subgroups)):
                        # Check if any trajectories between groups are compatible
                        compatible = any(
                            any(self.are_trajectories_compatible(t1.approach_angles, t2.approach_angles)
                                for t2 in subgroups[j])
                            for t1 in subgroups[i]
                        )
                        
                        if compatible:
                            # Merge groups
                            subgroups[i].update(subgroups[j])
                            subgroups.pop(j)
                            merged = True
                            break
                            
                if not merged:
                    break
            
            # Convert trajectory groups to endpoint groups
            for subgroup in subgroups:
                endpoints = {t.endpoint for t in subgroup}
                if len(endpoints) > 0:
                    final_groups.append(endpoints)
        
        self.attractor_groups = final_groups
        return final_groups
    
    def _group_connected_slow_points(self) -> List[Set[Tuple[int, int]]]:
        """
        Group slow points that are connected in the grid.
        Returns list of sets, where each set contains connected slow points.
        """
        def get_neighbors(point):
            i, j = point
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if (ni, nj) in self.slow_points:
                        neighbors.append((ni, nj))
            return neighbors
        
        # Use depth-first search to find connected components
        visited = set()
        groups = []
        
        for point in self.slow_points:
            if point in visited:
                continue
                
            # Start new group
            group = set()
            stack = [point]
            
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    group.add(current)
                    stack.extend(n for n in get_neighbors(current)
                               if n not in visited)
            
            groups.append(group)
            
        return groups
    
    def find_attractors_and_basins(self, trajectories, sensitivity=1.0):
        """
        Identify attractors and their basins using trajectory-based analysis.
        Overrides the parent class method to use trajectory curvature analysis
        instead of DBSCAN clustering.
        
        Args:
            trajectories: List of trajectories from analyze_vector_field
            sensitivity: Controls analysis parameters (used to scale thresholds)
            
        Returns:
            tuple: (attractors, basins) where:
                - attractors is an array of (x,y) coordinates for each attractor
                - basins is a grid-shaped array with labels for each basin
        """
        # First identify slow points
        magnitudes = np.hypot(self.U_fine, self.V_fine)
        threshold = self.slow_point_threshold * sensitivity * np.nanmax(magnitudes)
        slow_mask = (magnitudes < threshold) & self.valid_mask
        self.slow_points = set(zip(*np.where(slow_mask)))
        
        # Process trajectories and store endpoint information
        self.trajectories = set()
        endpoints_dict = {}  # Maps endpoint to list of trajectories ending there
        
        for traj in trajectories:
            if len(traj) < self.min_trajectory_length:
                continue
                
            endpoint = traj[-1]
            if endpoint not in self.slow_points:
                continue
                
            # Compute approach angles for the trajectory
            approach_angles = self.compute_approach_angles(traj)
            
            traj_info = TrajectoryInfo.from_trajectory(
                points=traj,
                endpoint_speed=self.get_vector_magnitude(*endpoint),
                approach_angles=approach_angles
            )
            
            self.trajectories.add(traj_info)
            endpoints_dict.setdefault(endpoint, []).append(traj_info)
        
        # Group connected slow points and analyze their trajectories
        self.attractor_groups = self._group_trajectories()
        
        # Convert attractor groups to the expected format
        attractors = []
        basins = np.full(self.grid_shape, -1, dtype=int)
        
        for idx, group in enumerate(self.attractor_groups):
            # Compute attractor location as mean of group points
            group_x = np.mean([self.X_fine[i,j] for i,j in group])
            group_y = np.mean([self.Y_fine[i,j] for i,j in group])
            attractors.append([group_x, group_y])
            
            # Assign basin labels for all trajectories leading to this group
            for point in group:
                for traj_info in endpoints_dict.get(point, []):
                    for i, j in traj_info.points:
                        basins[i,j] = idx
        
        # Store results for visualization
        self.trajectory_endpoints = np.array([[self.X_fine[t.endpoint[0], t.endpoint[1]],
                                            self.Y_fine[t.endpoint[0], t.endpoint[1]]]
                                           for t in self.trajectories])
        self.endpoint_flows = np.array([t.endpoint_speed for t in self.trajectories])
        self.endpoint_clusters = np.array([next(idx for idx, group in enumerate(self.attractor_groups)
                                              if t.endpoint in group) 
                                         for t in self.trajectories])
        
        return np.array(attractors), basins
    
    def analyze_vector_field(self, sensitivity=1.0) -> List[List[Tuple[int, int]]]:
        """
        Main analysis method that implements the complete pipeline.
        Returns trajectories in the format expected by find_attractors_and_basins.
        
        Args:
            sensitivity: Float controlling the sensitivity of trajectory generation
            
        Returns:
            List[List[Tuple[int, int]]]: List of trajectories, where each trajectory
            is a list of (i,j) grid coordinates
        """
        # Generate trajectories from grid points
        trajectories = []
        
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                if not self.valid_mask[i,j]:
                    continue
                    
                # Generate trajectory
                traj = self.generate_trajectory_rk4((i,j), sensitivity=sensitivity)
                if len(traj) >= self.min_trajectory_length:
                    trajectories.append(traj)
        
        return trajectories

    def plot_attractors(self, figsize=(12,10)):
        """Plot the vector field with identified attractors."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot vector field
        ax.quiver(self.X_fine, self.Y_fine, 
                 self.U_fine, self.V_fine,
                 alpha=0.3)
        
        # Plot slow points
        slow_x = [self.X_fine[i,j] for i,j in self.slow_points]
        slow_y = [self.Y_fine[i,j] for i,j in self.slow_points]
        ax.scatter(slow_x, slow_y, c='gray', alpha=0.5, label='Slow Points')
        
        # Plot attractor groups with different colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        for idx, group in enumerate(self.attractor_groups):
            color = colors[idx % len(colors)]
            x = [self.X_fine[i,j] for i,j in group]
            y = [self.Y_fine[i,j] for i,j in group]
            ax.scatter(x, y, c=[color], label=f'Attractor {idx+1}')
        
        ax.legend()
        ax.set_title('Vector Field with Identified Attractors')
        plt.show()

    def _group_trajectories(self) -> List[Set[Tuple[int, int]]]:
        """
        Group trajectories based on their endpoints and approach patterns.
        Returns list of sets, where each set contains endpoint coordinates
        belonging to the same attractor.
        """
        # First, group by connected slow points
        slow_point_groups = self._group_connected_slow_points()
        
        # For each group of connected slow points, analyze trajectories
        final_groups = []
        
        for slow_group in slow_point_groups:
            # Get all trajectories ending in this group
            group_trajectories = [t for t in self.trajectories if t.endpoint in slow_group]
            
            if not group_trajectories:
                continue
                
            # Start with each trajectory in its own subgroup
            subgroups: List[Set[TrajectoryInfo]] = [{t} for t in group_trajectories]
            
            # Iteratively merge compatible subgroups
            while True:
                merged = False
                for i in range(len(subgroups)):
                    if merged:
                        break
                    for j in range(i+1, len(subgroups)):
                        # Check if any trajectories between groups are compatible
                        compatible = any(
                            any(self.are_trajectories_compatible(t1.approach_angles, t2.approach_angles)
                                for t2 in subgroups[j])
                            for t1 in subgroups[i]
                        )
                        
                        if compatible:
                            # Merge groups
                            subgroups[i].update(subgroups[j])
                            subgroups.pop(j)
                            merged = True
                            break
                            
                if not merged:
                    break
            
            # Convert trajectory groups to endpoint groups
            for subgroup in subgroups:
                endpoints = {t.endpoint for t in subgroup}
                if len(endpoints) > 0:
                    final_groups.append(endpoints)
        
        return final_groups
