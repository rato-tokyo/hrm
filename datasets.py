"""
Dataset classes for HRM training
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import random
from collections import deque


class CopyDataset(Dataset):
    """Copy task: model must learn to copy input to output."""

    def __init__(self, num_samples: int = 1000, seq_len: int = 16, vocab_size: int = 10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = self._generate_data()

    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        data = []
        for _ in range(self.num_samples):
            x = torch.randint(0, self.vocab_size, (self.seq_len,))
            y = x.clone()
            data.append((x, y))
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


class ReverseDataset(Dataset):
    """Reverse task: model must learn to reverse the input sequence."""

    def __init__(self, num_samples: int = 1000, seq_len: int = 16, vocab_size: int = 10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = self._generate_data()

    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        data = []
        for _ in range(self.num_samples):
            x = torch.randint(0, self.vocab_size, (self.seq_len,))
            y = x.flip(0)
            data.append((x, y))
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


class SortDataset(Dataset):
    """Sort task: model must learn to sort the input sequence."""

    def __init__(self, num_samples: int = 1000, seq_len: int = 16, vocab_size: int = 10):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = self._generate_data()

    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        data = []
        for _ in range(self.num_samples):
            x = torch.randint(0, self.vocab_size, (self.seq_len,))
            y = torch.sort(x).values
            data.append((x, y))
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


class AdditionDataset(Dataset):
    """
    Addition task: given two numbers encoded as sequences, output their sum.
    Example: [1, 2, 3, +, 4, 5, 6] -> [5, 7, 9, PAD, ...]
    """

    def __init__(self, num_samples: int = 1000, max_digits: int = 3):
        self.num_samples = num_samples
        self.max_digits = max_digits
        self.vocab_size = 13  # 0-9, +, PAD, EOS
        self.pad_token = 11
        self.plus_token = 10
        self.seq_len = max_digits * 2 + 1
        self.data = self._generate_data()

    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        data = []
        max_val = 10 ** self.max_digits - 1

        for _ in range(self.num_samples):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            c = a + b

            a_str = str(a).zfill(self.max_digits)
            b_str = str(b).zfill(self.max_digits)
            x = [int(d) for d in a_str] + [self.plus_token] + [int(d) for d in b_str]

            c_str = str(c).zfill(self.max_digits + 1)
            y = [int(d) for d in c_str]
            while len(y) < self.seq_len:
                y.append(self.pad_token)

            data.append((torch.tensor(x), torch.tensor(y[:self.seq_len])))

        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


class MazeDataset(Dataset):
    """
    Maze task: find the shortest path from start to goal.

    Encoding:
        Input: 0=passage, 1=wall, 2=start, 3=goal
        Output: 0=not on path, 1=on path

    The maze is flattened to a 1D sequence for transformer input.
    """

    # Token IDs
    PASSAGE = 0
    WALL = 1
    START = 2
    GOAL = 3

    def __init__(
        self,
        num_samples: int = 1000,
        grid_size: int = 10,
        min_path_length: int = 10,
        wall_density: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Args:
            num_samples: Number of maze samples to generate
            grid_size: Size of the maze grid (grid_size x grid_size)
            min_path_length: Minimum shortest path length (for difficulty)
            wall_density: Probability of a cell being a wall
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.min_path_length = min_path_length
        self.wall_density = wall_density
        self.seq_len = grid_size * grid_size
        self.vocab_size = 4  # passage, wall, start, goal

        if seed is not None:
            random.seed(seed)

        self.data = self._generate_data()

    def _generate_maze(self) -> Tuple[List[List[int]], Tuple[int, int], Tuple[int, int]]:
        """Generate a random maze with guaranteed path from start to goal."""
        while True:
            # Initialize maze with passages
            maze = [[self.PASSAGE for _ in range(self.grid_size)] for _ in range(self.grid_size)]

            # Add random walls
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if random.random() < self.wall_density:
                        maze[i][j] = self.WALL

            # Set start and goal positions
            start = (0, 0)
            goal = (self.grid_size - 1, self.grid_size - 1)

            # Ensure start and goal are passages
            maze[start[0]][start[1]] = self.START
            maze[goal[0]][goal[1]] = self.GOAL

            # Check if path exists and meets minimum length
            path = self._bfs_shortest_path(maze, start, goal)
            if path is not None and len(path) >= self.min_path_length:
                return maze, start, goal

    def _bfs_shortest_path(
        self,
        maze: List[List[int]],
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find shortest path using BFS. Returns path as list of (row, col) tuples."""
        rows, cols = self.grid_size, self.grid_size
        queue = deque([(start, [start])])
        visited = {start}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

        while queue:
            (row, col), path = queue.popleft()

            if (row, col) == goal:
                return path

            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc

                if (0 <= new_row < rows and
                    0 <= new_col < cols and
                    (new_row, new_col) not in visited and
                    maze[new_row][new_col] != self.WALL):

                    visited.add((new_row, new_col))
                    queue.append(((new_row, new_col), path + [(new_row, new_col)]))

        return None  # No path found

    def _generate_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate maze samples with shortest paths."""
        data = []

        for _ in range(self.num_samples):
            maze, start, goal = self._generate_maze()
            path = self._bfs_shortest_path(maze, start, goal)

            # Flatten maze to 1D input
            x_list: List[int] = []
            for maze_row in maze:
                x_list.extend(maze_row)

            # Create path mask as output (1 = on path, 0 = not on path)
            y_list = [0] * self.seq_len
            if path is not None:
                for (r, c) in path:
                    flat_idx = r * self.grid_size + c
                    y_list[flat_idx] = 1

            data.append((torch.tensor(x_list), torch.tensor(y_list)))

        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]

    def visualize(self, sample_idx: int) -> str:
        """Visualize a maze sample as a string."""
        x_tensor, y_tensor = self.data[sample_idx]
        x_list = x_tensor.tolist()
        y_list = y_tensor.tolist()

        symbols = {self.PASSAGE: '.', self.WALL: '#', self.START: 'S', self.GOAL: 'G'}

        lines: List[str] = []
        lines.append("Input Maze:")
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                flat_idx = i * self.grid_size + j
                row_str += symbols[x_list[flat_idx]] + " "
            lines.append(row_str)

        lines.append("\nShortest Path:")
        for i in range(self.grid_size):
            row_str = ""
            for j in range(self.grid_size):
                flat_idx = i * self.grid_size + j
                if y_list[flat_idx] == 1:
                    if x_list[flat_idx] == self.START:
                        row_str += "S "
                    elif x_list[flat_idx] == self.GOAL:
                        row_str += "G "
                    else:
                        row_str += "* "
                else:
                    row_str += symbols[x_list[flat_idx]] + " "
            lines.append(row_str)

        return "\n".join(lines)
