import copy
from dataclasses import dataclass
from enum import IntEnum

from typing_extensions import Self

Vector4 = tuple[int, int, int, int]
Matrix4x4 = tuple[Vector4, Vector4, Vector4, Vector4]


class TetriminoType(IntEnum):
    I = 1
    J = 2
    L = 3
    O = 4
    S = 5
    T = 6
    Z = 7
    GOLDEN = 8


class TetriminoShape:
    def __init__(
        self,
        *shapes: Matrix4x4,
    ) -> None:
        assert len(shapes) == 4, "A Tetrimino must have exactly 4 rotation states."

        self.shapes = shapes

    def __getitem__(self, rotation: int) -> Matrix4x4:
        if rotation not in (0, 1, 2, 3):
            raise IndexError("Rotation must be in the range [0, 3].")

        return self.shapes[rotation]


class TetriminoMeta(type):
    registry: dict[TetriminoType, TetriminoShape] = {}
    I: "Tetrimino"
    J: "Tetrimino"
    L: "Tetrimino"
    O: "Tetrimino"
    S: "Tetrimino"
    T: "Tetrimino"
    Z: "Tetrimino"
    GOLDEN: "Tetrimino"

    def register(
        cls,
        type: TetriminoType,
        shape: TetriminoShape
    ) -> None:
        if type in cls.registry:
            raise ValueError(f'Tetrimino with type {type} is already registered.')

        cls.registry[type] = shape

    def __getattr__(cls, name: str) -> "Tetrimino":
        if name in TetriminoType.__members__:
            return Tetrimino(TetriminoType[name])
        raise AttributeError(f'Tetrimino with type {type} is not registered.')

    def __getitem__(cls, type: TetriminoType) -> "Tetrimino":
        return Tetrimino(type)


@dataclass
class Tetrimino(metaclass=TetriminoMeta):
    type: TetriminoType
    rotation: int = 0
    special_item: int = 0

    @property
    def shape(self) -> Matrix4x4:
        return self.__class__.registry[self.type][self.rotation]
    
    def rotate(self, rotation: int | None = None) -> None:
        if rotation is not None:
            if rotation not in (0, 1, 2, 3):
                raise ValueError("Rotation must be in the range [0, 3].")
            self.rotation = rotation
        else:
            self.rotation = (self.rotation + 1) % 4

    def copy(self) -> Self:
        return copy.deepcopy(self)


Tetrimino.register(
    type=TetriminoType.I,
    shape=(
        (
            (0, 0, 0, 0),
            (1, 2, 1, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 0, 1, 0),
            (0, 0, 1, 0),
            (0, 0, 2, 0),
            (0, 0, 1, 0)
        ),
        (
            (0, 0, 0, 0),
            (1, 2, 1, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 2, 0, 0),
            (0, 1, 0, 0)
        )
    )
)


Tetrimino.register(
    type=TetriminoType.J,
    shape=(
        (
            (1, 0, 0, 0),
            (1, 2, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 1, 1, 0),
            (0, 2, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 0, 0, 0),
            (2, 1, 1, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 1, 0, 0),
            (0, 1, 0, 0),
            (1, 2, 0, 0),
            (0, 0, 0, 0)
        )
    )
)


Tetrimino.register(
    type=TetriminoType.L,
    shape=(
        (
            (0, 0, 1, 0),
            (1, 2, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 1, 2, 0),
            (0, 0, 0, 0)
        ),
        (
            (2, 1, 1, 0),
            (1, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (1, 1, 0, 0),
            (0, 2, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0)
        )
    )
)


Tetrimino.register(
    type=TetriminoType.O,
    shape=(
        (
            (0, 1, 1, 0),
            (0, 2, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 1, 1, 0),
            (0, 2, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 1, 1, 0),
            (0, 2, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 1, 1, 0),
            (0, 2, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        )
    )
)


Tetrimino.register(
    type=TetriminoType.S,
    shape=(
        (
            (0, 1, 1, 0),
            (1, 2, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 1, 0, 0),
            (0, 2, 1, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (1, 2, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (1, 0, 0, 0),
            (2, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0)
        )
    )
)


Tetrimino.register(
    type=TetriminoType.T,
    shape=(
        (
            (0, 1, 0, 0),
            (1, 2, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 1, 0, 0),
            (0, 2, 1, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 0, 0, 0),
            (2, 1, 1, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 1, 0, 0),
            (2, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 0, 0)
        )
    )
)


Tetrimino.register(
    type=TetriminoType.Z,
    shape=(
        (
            (1, 1, 0, 0),
            (0, 2, 1, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 0, 1, 0),
            (0, 1, 1, 0),
            (0, 2, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 0, 0, 0),
            (1, 1, 0, 0),
            (0, 2, 1, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 1, 0, 0),
            (1, 1, 0, 0),
            (2, 0, 0, 0),
            (0, 0, 0, 0)
        )
    )
)


Tetrimino.register(
    type=TetriminoType.GOLDEN,
    shape=(
        (
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 0, 1, 0),
            (0, 0, 1, 0),
            (0, 0, 1, 0),
            (0, 0, 1, 0)
        ),
        (
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        ),
        (
            (0, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 1, 0, 0),
            (0, 1, 0, 0)
        )
    )
)
