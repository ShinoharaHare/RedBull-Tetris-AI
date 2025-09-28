import pygame

from .assets import TetrisAssets
from .state import TetrisState
from .tetrimino import Tetrimino


class UpperBoard(pygame.sprite.Sprite):
    def __init__(
        self,
        size: tuple[int, int],
        block_size: tuple[int, int] = (20, 20),
        position: tuple[int, int] = (0, 0)
    ) -> None:
        super().__init__()
        
        self.block_size = block_size
        self.image = pygame.Surface(size)
        self.rect = self.image.get_rect(topleft=position)

    def draw_tetrimino(self, tetrimino: Tetrimino) -> pygame.Surface:
        block_image = TetrisAssets.BLOCK.get_image(tetrimino.type.name, self.block_size)
        image = pygame.Surface(
            (self.block_size[0] * 4, self.block_size[1] * 4),
            pygame.SRCALPHA
        )
        for y in range(4):
            for x in range(4):
                if tetrimino.shape[y][x] == 0:
                    continue
                
                image.blit(block_image, (x * self.block_size[0], y * self.block_size[1]))
        return image

    def update(self, state: TetrisState) -> None:
        self.image.fill((0, 0, 0))

        if state.held_tetrimino is not None:
            self.image.blit(
                self.draw_tetrimino(state.held_tetrimino),
                (64, 24)
            )

        self.image.blit(
            self.draw_tetrimino(state.next_tetrimino),
            (self.image.get_width() - self.block_size[0] * 4, 24)
        )
