from functools import cache
from pathlib import Path

import pygame

base_dir = Path(__file__).parent


class ImageAssetGroupMeta(type):
    _base_dir: Path
    
    @cache
    def __getattr__(cls, name: str) -> pygame.Surface:
        return pygame.image.load(cls._base_dir / f'{name.lower()}.png')
    
    @cache
    def get_image(cls, name: str, size: tuple[int, int]) -> pygame.Surface:
        image = cls.__getattr__(name)
        if image.get_size() != size:
            image = pygame.transform.scale(image, size)
        return image


class BlockImage(metaclass=ImageAssetGroupMeta):
    _base_dir = base_dir / 'block'

    I: pygame.Surface
    J: pygame.Surface
    L: pygame.Surface
    O: pygame.Surface
    S: pygame.Surface
    T: pygame.Surface
    Z: pygame.Surface
    GOLDEN: pygame.Surface
    SPECIAL_ITEM: pygame.Surface


class TetrisAssets:
    BLOCK = BlockImage
