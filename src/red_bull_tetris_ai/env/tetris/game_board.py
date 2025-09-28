import pygame

from .assets import TetrisAssets
from .state import TetrisState
from .tetrimino import TetriminoType


class TetrisGameBoard(pygame.sprite.Sprite):
    def __init__(
        self,
        board_size: tuple[int, int],
        block_size: tuple[int, int],
        position: tuple[int, int] = (0, 0)
    ) -> None:
        super().__init__()

        self.board_size = board_size
        self.block_size = block_size
        self.position = position

        self.image = pygame.Surface(
            (self.image_width, self.image_height),
            pygame.SRCALPHA
        )
        self.rect = self.image.get_rect(topleft=position)

    @property
    def image_width(self) -> int:
        return self.board_size[0] * self.block_size[0] + (self.board_size[0] - 1)
    
    @property
    def image_height(self) -> int:
        return self.board_size[1] * self.block_size[1] + (self.board_size[1] - 1)

    def render_background(self) -> None:
        board_width = self.board_size[0]
        board_height = self.board_size[1]
        block_width = self.block_size[0]
        block_height = self.block_size[1]

        self.image.fill((194, 208, 222))
        for x in range(board_width):
            x = block_width * x + x
            pygame.draw.line(
                self.image,
                color=(0, 0, 0),
                start_pos=(x, 0),
                end_pos=(x, self.image_height),
                width=1
            )
        
        for y in range(board_height):
            y = block_height * y + y
            pygame.draw.line(
                self.image,
                color=(0, 0, 0),
                start_pos=(0, y),
                end_pos=(self.image_width, y),
                width=1
            )

    def update(self, state: TetrisState) -> None:
        self.render_background()

        for y in range(self.board_size[1]):
            for x in range(self.board_size[0]):
                x0 = x * self.block_size[0] + (x + 1) * (x > 0)
                y0 = y * self.block_size[1] + (y + 1) * (y > 0)

                t = state.board[0, y, x].item()

                if t == 0:
                    continue

                t = TetriminoType(t)
                block_image = TetrisAssets.BLOCK.get_image(t.name, self.block_size)
                self.image.blit(block_image, (x0, y0))

                special_item = state.board[1, y, x].item() > 0
                if special_item:
                    self.image.blit(
                        TetrisAssets.BLOCK.get_image('SPECIAL_ITEM', self.block_size),
                        (x0, y0)
                    )
