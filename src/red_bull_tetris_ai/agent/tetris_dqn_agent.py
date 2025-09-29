from dataclasses import asdict, dataclass

import safetensors.torch
import torch
from torch import nn
from typing_extensions import Self

from red_bull_tetris_ai.env.tetris import TetrisAction, TetrisState


@dataclass
class TetrisDQNState:
    state: torch.Tensor
    current_tetrimino: torch.Tensor
    next_tetrimino: torch.Tensor
    held_tetrimino: torch.Tensor

    def __getitem__(self, index: int | slice) -> Self:
        if isinstance(index, int):
            index = slice(index, index + 1)

        return self.__class__(
            state=self.state[index],
            current_tetrimino=self.current_tetrimino[index],
            next_tetrimino=self.next_tetrimino[index],
            held_tetrimino=self.held_tetrimino[index]
        )

    def to_dict(self) -> dict[str, torch.Tensor]:
        return asdict(self)
    
    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            state=self.state.to(*args, **kwargs),
            current_tetrimino=self.current_tetrimino.to(*args, **kwargs),
            next_tetrimino=self.next_tetrimino.to(*args, **kwargs),
            held_tetrimino=self.held_tetrimino.to(*args, **kwargs)
        )

    @classmethod
    def from_state(cls, state: TetrisState) -> Self:
        height = state.height
        bumpiness = torch.abs(height[1:] - height[:-1])
        time_ratio = (
            0.0
            if state.max_tetriminoes is None
            else state.num_tetriminoes_placed / state.max_tetriminoes
        )
        return cls(
            state=torch.tensor(
                [
                    [
                        state.last_lines_cleared,
                        state.num_holes,
                        state.height.sum().item(),
                        state.height.max().item(),
                        state.height.min().item(),
                        bumpiness.sum().item(),
                        bumpiness.max().item(),
                        time_ratio,
                        state.score_multiplier,
                        state.score_multiplier_meter,
                        state.last_golden_blocks_cleared,
                        state.last_special_items_cleared
                    ]
                ]
            ),
            current_tetrimino=torch.tensor([state.current_tetrimino.type]),
            next_tetrimino=torch.tensor([state.next_tetrimino.type]),
            held_tetrimino=torch.tensor([0 if state.held_tetrimino is None else state.held_tetrimino.type])
        )

    @classmethod
    def from_batch(cls, batch: list[TetrisState] | list[Self]) -> Self:
        if not isinstance(batch[0], cls):
            batch = [cls.from_state(f) for f in batch]

        return cls(
            state=torch.cat([x.state for x in batch]),
            current_tetrimino=torch.cat([x.current_tetrimino for x in batch]),
            next_tetrimino=torch.cat([x.next_tetrimino for x in batch]),
            held_tetrimino=torch.cat([x.held_tetrimino for x in batch])
        )


class TetrisDQNAgent(nn.Module):
    """
    Input state features:
         0: Number of Lines Cleared
         1: Number of Holes
         2: Total Height
         3: Max Height
         4: Min Height
         5: Total Bumpiness
         6: Max Bumpiness
         7: Time Ratio
         8: Score Multiplier
         9: Score Multiplier Meter
        10: Number of Golden Blocks Cleared
        11: Number of Special Items Cleared
        12: Current Tetrimino
        13: Next Tetrimino
        14: Held Tetrimino
    Output:
        Score
    """
    def __init__(self) -> None:
        super().__init__()

        embedding_dim = 32
        self.tetrimino_embeds = nn.Embedding(9, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(12 + embedding_dim * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        state: torch.Tensor,
        current_tetrimino: torch.Tensor,
        next_tetrimino: torch.Tensor,
        held_tetrimino: torch.Tensor
    ) -> torch.Tensor:
        """
        state: (N, 12)
        current_tetrimino: (N,) dtype long, 0-8
        next_tetrimino: (N,) dtype long, 0-8
        held_tetrimino: (N,) dtype long, 0-8
        return: (N, 1)
        """
        return self.model(
            torch.cat(
                [
                    state,
                    self.tetrimino_embeds(current_tetrimino),
                    self.tetrimino_embeds(next_tetrimino),
                    self.tetrimino_embeds(held_tetrimino)
                ],
                dim=1
            )
        )

    @torch.no_grad()
    def predict(self, next_states: dict[TetrisAction, TetrisState]) -> TetrisAction:
        actions, next_states = zip(*next_states.items())
        next_states: TetrisDQNState = TetrisDQNState.from_batch(next_states)
        next_states = next_states.to(self.device, non_blocking=True)
        next_states.state = next_states.state.to(self.dtype)
        values = self(**next_states.to_dict())[:, 0]
        index = torch.argmax(values).item()
        return actions[index]

    @classmethod
    def from_pretrained(cls, path: str, *args, **kwargs) -> Self:
        model = cls(*args, **kwargs)
        safetensors.torch.load_model(model, path)
        return model
