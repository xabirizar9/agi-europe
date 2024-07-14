import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astar import AStar


def load_grid(path: Path):
    try:
        grid = np.load(path)
    except FileNotFoundError:
        print("The file doesn't exist, loading sample")
        grid = generate_sample_grid(25, 25)
    return grid


def generate_sample_grid(rows, cols):
    grid = np.random.random((rows, cols))
    grid[0, 0] = 0  # Ensure start position is not an obstacle
    grid[rows - 1, cols - 1] = 0  # Ensure goal position is not an obstacle
    return grid


class DronePlanning(AStar):
    def __init__(self, start, goal, grid):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.cost_estimates = np.zeros_like(grid, dtype=float)

    def neighbors(self, node):
        x, y = node
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [
            n
            for n in neighbors
            if 0 <= n[0] < self.grid.shape[0] and 0 <= n[1] < self.grid.shape[1]
        ]

    def distance_between(self, n1, n2):
        return 1

    def heuristic_cost_estimate(self, current, goal):
        x, y = current
        estimate = self.grid[x, y]
        self.cost_estimates[x, y] = (
            estimate  # Store the cost estimate for visualization
        )
        return estimate

    def is_goal_reached(self, current, goal):
        return current == goal

    def path_planning(self):
        path = self.astar(self.start, self.goal)
        if path is None:
            raise ValueError("No path found")
        return list(path)


def increase_risk_around_path(grid, path, increment, radius=5):
    updated_grid = np.copy(grid)
    for node in path:
        x, y = node
        for i in range(max(0, x - radius), min(grid.shape[0], x + radius + 1)):
            for j in range(max(0, y - radius), min(grid.shape[1], y + radius + 1)):
                if np.sqrt((x - i) ** 2 + (y - j) ** 2) <= radius:
                    updated_grid[i, j] += increment
    return updated_grid


def path_planning(start, goal, grid: np.array, previous_paths=None, increment=0):
    if previous_paths:
        updated_grid = np.copy(grid)
        for path in previous_paths:
            updated_grid = increase_risk_around_path(updated_grid, path, increment)
    else:
        updated_grid = grid

    planner = DronePlanning(start, goal, updated_grid)
    path = planner.path_planning()
    return path, planner.cost_estimates


def ensure_path(start, goal, grid, previous_paths, increment, max_attempts=10):
    for attempt in range(max_attempts):
        try:
            path, cost_estimates = path_planning(
                start, goal, grid, previous_paths=previous_paths, increment=increment
            )
            return path, cost_estimates
        except ValueError:
            increment *= 0.9  # Decrease the increment
            print(
                f"Attempt {attempt + 1}: No path found. Decreasing increment to {increment:.2f}"
            )

    # If still no path, try decreasing grid values
    for attempt in range(max_attempts):
        grid *= 0.9  # Make the grid more traversable
        try:
            path, cost_estimates = path_planning(
                start, goal, grid, previous_paths=previous_paths, increment=increment
            )
            return path, cost_estimates
        except ValueError:
            print(
                f"Attempt {attempt + max_attempts + 1}: No path found. Reducing grid values."
            )

    raise ValueError("No path found after maximum attempts")


def visualize_grid_with_paths(
    initial_grid, paths, start, goal, title="Grid with Paths"
):
    fig, ax = plt.subplots()
    cax = ax.imshow(initial_grid, cmap="jet", interpolation="nearest")
    fig.colorbar(cax)
    colors = [
        "black",
        "black",
        "black",
        "purple",
    ]  # Different colors for different paths
    for idx, path in enumerate(paths):
        path = np.array(path)
        ax.plot(
            path[:, 1],
            path[:, 0],
            color=colors[idx % len(colors)],
            label=f"Path {idx + 1}",
        )
    ax.scatter([start[1]], [start[0]], color="cyan", label="Start")
    ax.scatter([goal[1]], [goal[0]], color="magenta", label="Goal")
    ax.set_title(title)
    ax.legend()
    plt.show(block=True)


if __name__ == "__main__":
    start = (450, 50)
    goal = (50, 470)

    cost_increment = 70
    num_robots = 3  # Number of robots
    grid_path = Path("risk_value.npy")
    grid = load_grid(grid_path)
    initial_grid = np.copy(grid)  # Save the initial grid for visualization

    previous_paths = []
    for i in range(num_robots):
        try:
            path, cost_estimates = ensure_path(
                start, goal, grid, previous_paths, cost_increment
            )
            previous_paths.append(path)
            # Calculate and print total loss
            total_loss = sum(cost_estimates[node[0], node[1]] for node in path)
            print(f"Total loss for Path {i+1}: {total_loss:.2f}")
        except ValueError as e:
            print(f"No path found for robot {i+1}")

    # Visualize all paths on the same grid
    visualize_grid_with_paths(initial_grid, previous_paths, start, goal)
