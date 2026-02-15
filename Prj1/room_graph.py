import random
import numpy as np
from collections import deque
from typing import List, Tuple, Iterator, Optional

FLOOR = 0
OBSTACLE = 1

UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
DIRECTIONS = [
    (-1, 0),  # UP
    (1, 0),   # DOWN
    (0, 1),   # RIGHT
    (0, -1),  # LEFT
]


def _rasterize_line(r0: int, c0: int, r1: int, c1: int) -> Iterator[Tuple[int, int]]:
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    r, c = r0, c0
    if dr >= dc:
        err = 2 * dc - dr
        for _ in range(dr + 1):
            yield (r, c)
            if err > 0:
                c += sc
                err -= 2 * dr
            err += 2 * dc
            r += sr
    else:
        err = 2 * dr - dc
        for _ in range(dc + 1):
            yield (r, c)
            if err > 0:
                r += sr
                err -= 2 * dc
            err += 2 * dr
            c += sc


def _fill_interior_of_obstacles(grid: np.ndarray) -> None:
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)

    def on_border(r: int, c: int) -> bool:
        return r == 0 or r == rows - 1 or c == 0 or c == cols - 1

    q: deque[Tuple[int, int]] = deque()
    for r in range(rows):
        for c in range(cols):
            if on_border(r, c) and grid[r, c] == FLOOR and not visited[r, c]:
                q.append((r, c))
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in DIRECTIONS:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == FLOOR and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == FLOOR and not visited[r, c]:
                grid[r, c] = OBSTACLE


class RoomGraph:

    def __init__(self, rows: int, cols: int):
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        self.rows = rows
        self.cols = cols
        self._grid = np.full((rows, cols), FLOOR, dtype=np.int32)

    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def get_state(self, r: int, c: int) -> int:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise IndexError(f"(r,c)=({r},{c}) out of grid")
        return int(self._grid[r, c])

    def is_floor(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols and self._grid[r, c] == FLOOR

    def set_obstacle(self, r: int, c: int) -> None:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self._grid[r, c] = OBSTACLE

    def set_floor(self, r: int, c: int) -> None:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self._grid[r, c] = FLOOR

    def add_obstacle_line(self, r0: int, c0: int, r1: int, c1: int) -> None:
        for r, c in _rasterize_line(r0, c0, r1, c1):
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self._grid[r, c] = OBSTACLE

    def add_obstacle_contour(self, points: List[Tuple[int, int]], fill_interior: bool = True) -> None:
        for i in range(len(points)):
            r0, c0 = points[i]
            r1, c1 = points[(i + 1) % len(points)]
            self.add_obstacle_line(r0, c0, r1, c1)
        if fill_interior:
            _fill_interior_of_obstacles(self._grid)

    def add_obstacle_circle(self, center_r: int, center_c: int, radius: int) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                if (r - center_r) ** 2 + (c - center_c) ** 2 <= radius**2:
                    self._grid[r, c] = OBSTACLE

    def add_obstacle_gaussian(self, center_r: float, center_c: float, sigma: float, threshold: float = 0.2) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                d2 = (r - center_r) ** 2 + (c - center_c) ** 2
                if np.exp(-d2 / (2 * sigma * sigma)) >= threshold:
                    self._grid[r, c] = OBSTACLE

    def get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if self.is_floor(nr, nc):
                out.append((nr, nc))
        return out

    def get_actions_for(self, r: int, c: int) -> List[int]:
        actions: List[int] = []
        for a, (dr, dc) in enumerate(DIRECTIONS):
            if self.is_floor(r + dr, c + dc):
                actions.append(a)
        return actions

    def step(self, r: int, c: int, action: int) -> Tuple[int, int]:
        if not self.is_floor(r, c):
            return (r, c)
        dr, dc = DIRECTIONS[action]
        nr, nc = r + dr, c + dc
        if self.is_floor(nr, nc):
            return (nr, nc)
        return (r, c)

    def floor_cells(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if self._grid[r, c] == FLOOR]

    def obstacle_count(self) -> int:
        return int(np.sum(self._grid == OBSTACLE))

    def is_floor_connected(self) -> bool:
        floors = self.floor_cells()
        if not floors:
            return True
        start = floors[0]
        visited: set[Tuple[int, int]] = set()
        q: deque[Tuple[int, int]] = deque([start])
        visited.add(start)
        while q:
            r, c = q.popleft()
            for nr, nc in self.get_neighbors(r, c):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return len(visited) == len(floors)

    def copy(self) -> RoomGraph:
        other = RoomGraph(self.rows, self.cols)
        other._grid = self._grid.copy()
        return other


def _wall_cell_count(rows: int, cols: int) -> int:
    if rows <= 0 or cols <= 0:
        return 0
    return 2 * rows + 2 * cols - 4


def _build_random_room_attempt(
    rows: int,
    cols: int,
    add_walls: bool,
    target_obstacle_ratio: float,
    margin: int,
    rng: random.Random,
) -> RoomGraph:
    total = rows * cols
    wall_count = _wall_cell_count(rows, cols) if add_walls else 0
    interior_cells = total - wall_count
    interior_budget = max(0, int(interior_cells * target_obstacle_ratio))
    g = RoomGraph(rows, cols)
    if add_walls:
        g.add_obstacle_contour(
            [(0, 0), (0, cols - 1), (rows - 1, cols - 1), (rows - 1, 0)],
            fill_interior=False,
        )
    if interior_budget == 0:
        return g
    L = min(rows, cols)
    r_max = max(1, L // 6)
    r_min = max(1, L // 8)
    sigma_max = max(1.0, L / 5)
    sigma_min = max(0.8, L / 12)
    lo, hi = margin, max(margin, rows - 1 - margin)
    clo, chi = margin, max(margin, cols - 1 - margin)
    if hi < lo or chi < clo:
        return g
    max_obstacles_total = wall_count + interior_budget
    max_attempts = 80
    while max_attempts > 0:
        if g.obstacle_count() >= max_obstacles_total:
            break
        choice = rng.choice(["circle", "gaussian", "line"])
        if choice == "circle":
            radius = rng.randint(r_min, r_max)
            center_r = rng.randint(lo, hi)
            center_c = rng.randint(clo, chi)
            if g.is_floor(center_r, center_c):
                g.add_obstacle_circle(center_r, center_c, radius)
                max_attempts -= 1
        elif choice == "gaussian":
            sigma = rng.uniform(sigma_min, sigma_max)
            center_r = rng.uniform(lo, hi)
            center_c = rng.uniform(clo, chi)
            cr, cc = int(round(center_r)), int(round(center_c))
            if 0 <= cr < rows and 0 <= cc < cols and g.is_floor(cr, cc):
                g.add_obstacle_gaussian(center_r, center_c, sigma)
                max_attempts -= 1
        else:
            r0, c0 = rng.randint(lo, hi), rng.randint(clo, chi)
            r1, c1 = rng.randint(lo, hi), rng.randint(clo, chi)
            if (r0, c0) != (r1, c1):
                g.add_obstacle_line(r0, c0, r1, c1)
                max_attempts -= 1
    return g


def build_random_room(
    rows: int = 12,
    cols: int = 16,
    add_walls: bool = True,
    margin: int = 2,
    max_obstacle_ratio: float = 0.3,
    rng: Optional[random.Random] = None,
) -> RoomGraph:
    if rng is None:
        rng = random.Random()
    wall_count = _wall_cell_count(rows, cols) if add_walls else 0
    interior_cells = rows * cols - wall_count
    ratio = max_obstacle_ratio
    min_ratio = 0.005
    ratio_decay = 0.85
    while ratio >= min_ratio:
        g = _build_random_room_attempt(
            rows=rows,
            cols=cols,
            add_walls=add_walls,
            target_obstacle_ratio=ratio,
            margin=margin,
            rng=rng,
        )
        interior_obstacles = g.obstacle_count() - wall_count
        # if interior_obstacles > interior_cells * max_obstacle_ratio:
        #     ratio *= ratio_decay
        #     continue
        if not g.is_floor_connected():
            ratio *= ratio_decay
            continue
        return g
    g = RoomGraph(rows, cols)
    if add_walls:
        g.add_obstacle_contour(
            [(0, 0), (0, cols - 1), (rows - 1, cols - 1), (rows - 1, 0)],
            fill_interior=False,
        )
    return g
