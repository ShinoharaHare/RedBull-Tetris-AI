from functools import cache
from pathlib import Path

from PIL import Image


class ImageGroupMeta(type):
    _base_dir: Path
    
    @cache
    def __getattr__(cls, name: str) -> Image.Image:
        return Image.open(cls._base_dir / f'{name.lower()}.png')


class NextTetriminoImage(metaclass=ImageGroupMeta):
    _base_dir = Path(__file__).parent / 'next_tetrimino'
    I: Image.Image
    J: Image.Image
    L: Image.Image
    O: Image.Image
    S: Image.Image
    T: Image.Image
    Z: Image.Image
    GOLDEN: Image.Image


class HeldTetriminoImage(metaclass=ImageGroupMeta):
    _base_dir = Path(__file__).parent / 'held_tetrimino'
    I: Image.Image
    J: Image.Image
    L: Image.Image
    O: Image.Image
    S: Image.Image
    T: Image.Image
    Z: Image.Image
    GOLDEN: Image.Image
    NONE: Image.Image


class RedbullTerisImage(metaclass=ImageGroupMeta):
    _base_dir = Path(__file__).parent

    TITLE: Image.Image
    EMPTY_BOARD: Image.Image
    NEXT_TETRIMINO = NextTetriminoImage
    HELD_TETRIMINO = HeldTetriminoImage
