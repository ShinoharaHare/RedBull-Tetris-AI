import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing_extensions import Self

from .tetrimino import Tetrimino

Position = list[int, int]


@dataclass
class TetrisState:
    # Game state
    board: torch.ByteTensor
    current_position: Position | None = None
    current_tetrimino: Tetrimino | None = None
    next_tetrimino: Tetrimino | None = None
    held_tetrimino: Tetrimino | None = None
    already_held: bool = False
    combo: int = 0
    score: int = 0
    score_multiplier: int = 1
    score_multiplier_meter: float = 0.0
    special_item_meter: float = 0.0
    gameover: bool = False
    survived: bool | None = None
    
    special_item_index: int = 0
    
    # Statistics
    num_lines_cleared: int = 0
    num_tetriminoes_placed: int = 0
    max_combo: int = 0
    max_tetriminoes: int | None = None

    # Last action info
    last_hard_drop_distance: int = 0
    last_lines_cleared: int = 0
    last_golden_blocks_cleared: int = 0
    last_special_items_cleared: int = 0
    last_score_gained: int = 0
    last_reward_gained: float = 0.0

    @property
    def num_special_items(self) -> int:
        return torch.sum(self.board[1] > 0).item()
    
    @property
    def height(self) -> torch.Tensor:
        board = self.board[0] > 0
        return torch.where(
            torch.any(board > 0, dim=0),
            board.size(0) - board.byte().argmax(dim=0),
            0
        )
    
    @property
    def max_height(self) -> int:
        return self.height.max().item()
    
    @property
    def num_holes(self) -> int:
        board = self.board[0] > 0
        covered_mask = torch.cumsum(board, dim=0) > 0
        holes = torch.sum(covered_mask & ~board)        
        return holes.item()
    
    @property
    def bumpiness(self) -> int:
        return torch.abs(self.height[1:] - self.height[:-1]).sum().item()
    
    @property
    def can_tetris(self) -> bool:
        board = self.board[0] > 0
        row_gap = board.size(1) - board.sum(dim=1)
        candidate_rows = (row_gap == 1).float()
        empty_mask = (board == 0).float()
        valid_positions = empty_mask * candidate_rows.unsqueeze(1)
        x = valid_positions.T.unsqueeze(1)
        kernel = torch.ones(1, 1, 4)
        conv_out = F.conv1d(x, kernel)
        candidates = (conv_out == 4).squeeze(1)

        if not candidates.any():
            return False

        col_idx, row_start = candidates.nonzero(as_tuple=True)
        if len(col_idx) == 0:
            return False

        blocked = board[:row_start.max(), col_idx].cumsum(0)
        padded = torch.zeros(
            board.size(0),
            col_idx.numel(),
            dtype=board.dtype,
            device=board.device
        )
        padded[:blocked.shape[0]] = blocked

        clear_path = (padded[row_start-1, torch.arange(col_idx.numel())] == 0) if (row_start > 0).any() else True
        return clear_path.any().item()

    def copy(self) -> Self:
        return copy.deepcopy(self)
    