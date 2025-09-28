from dataclasses import dataclass, replace
from functools import cache

from typing_extensions import Self

from .tetrimino import TetriminoType


@dataclass(frozen=True)
class TetrisAction:
    hold: bool
    rotation: int
    translation: int

    @cache
    def __new__(cls, *arg, **kwargs):
        return super().__new__(cls)

    def __post_init__(self) -> None:
        if self.rotation < 0 or self.rotation > 3:
            raise ValueError('Rotation must be between 0 and 3.')

        if self.translation < -5 or self.translation > 5:
            raise ValueError('Translation must be between -5 and 5.')
        
    def set_hold(self, hold: bool) -> Self:
        return replace(self, hold=hold)


POSIBLE_ACTIONS = {
    TetriminoType.GOLDEN: {
        (0, 0), (0, -1), (0, -2), (0, -3), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, -1), (1, -2), (1, -3), (1, -4), (1, -5), (1, 1), (1, 2), (1, 3), (1, 4)
    },
    TetriminoType.I: {
        (0, 0), (0, -1), (0, -2), (0, -3), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, -1), (1, -2), (1, -3), (1, -4), (1, -5), (1, 1), (1, 2), (1, 3), (1, 4)
    },
    TetriminoType.J: {
        (0, 0), (0, -1), (0, -2), (0, -3), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, -1), (1, -2), (1, -3), (1, -4), (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 0), (2, -1), (2, -2), (2, -3), (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 0), (3, -1), (3, -2), (3, -3), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)
    },
    TetriminoType.L: {
        (0, 0), (0, -1), (0, -2), (0, -3), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, -1), (1, -2), (1, -3), (1, -4), (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 0), (2, -1), (2, -2), (2, -3), (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 0), (3, -1), (3, -2), (3, -3), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)
    },
    TetriminoType.O: {
        (0, 0), (0, -1), (0, -2), (0, -3), (0, -4), (0, 1), (0, 2), (0, 3), (0, 4)
    },
    TetriminoType.S: {
        (0, 0), (0, -1), (0, -2), (0, -3), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, -1), (1, -2), (1, -3), (1, -4), (1, 1), (1, 2), (1, 3), (1, 4)
    },
    TetriminoType.T: {
        (0, 0), (0, -1), (0, -2), (0, -3), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, -1), (1, -2), (1, -3), (1, -4), (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 0), (2, -1), (2, -2), (2, -3), (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 0), (3, -1), (3, -2), (3, -3), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)
    },
    TetriminoType.Z: {
        (0, 0), (0, -1), (0, -2), (0, -3), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (1, -1), (1, -2), (1, -3), (1, -4), (1, 1), (1, 2), (1, 3), (1, 4)
    }
}
