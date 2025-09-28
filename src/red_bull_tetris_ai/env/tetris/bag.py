import random

from .tetrimino import Tetrimino, TetriminoType


class TetriminoBag:
    def __init__(
        self,
        tetrimino_types: list[TetriminoType] | None = None,
        golden_tetrimino_after_n: int | None = None
    ) -> None:
        if tetrimino_types is None:
            tetrimino_types = [
                TetriminoType.I,
                TetriminoType.J,
                TetriminoType.L,
                TetriminoType.O,
                TetriminoType.S,
                TetriminoType.T,
                TetriminoType.Z
            ]
        
        self._tetrimino_types = tetrimino_types.copy()
        self._golden_tetrimino_after_n = golden_tetrimino_after_n
        self.reset()

    @property
    def _safe_bag(self) -> list[TetriminoType]:
        if not self._bag:
            self._bag = self._tetrimino_types.copy()
            random.shuffle(self._bag)
        return self._bag
    
    def set_special_item(self, special_item: int) -> None:
        self._special_item = special_item
    
    def reset(self) -> None:
        self._bag = []
        self._next_tetrimino = None
        self._num_generated = 0
        self._special_item = None

    def __next__(self) -> tuple[Tetrimino, Tetrimino]:
        if (
            self._golden_tetrimino_after_n is not None and
            self._num_generated == self._golden_tetrimino_after_n
        ):
            self._safe_bag.append(TetriminoType.GOLDEN)

        if self._next_tetrimino is None:
            next_tetrimino_type = self._safe_bag.pop()
            self._next_tetrimino = Tetrimino(next_tetrimino_type)
        
        current_tetrimino = self._next_tetrimino
        if (
            self._special_item is not None and
            current_tetrimino.type != TetriminoType.GOLDEN
        ):
            current_tetrimino.special_item = self._special_item
            self._special_item = None

        next_tetrimino_type = self._safe_bag.pop()
        self._next_tetrimino = Tetrimino(next_tetrimino_type)
        self._num_generated += 1
        return current_tetrimino, self._next_tetrimino
