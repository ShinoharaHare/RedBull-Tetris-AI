from functools import cached_property
import pygame

from .state import TetrisState


class TetrisScoreBoard(pygame.sprite.Sprite):
    def __init__(
        self,
        size: tuple[int, int],
        position: tuple[int, int] = (0, 0)
    ) -> None:
        super().__init__()
        
        self.image = pygame.Surface(size)
        self.rect = self.image.get_rect(topleft=position)

    @cached_property
    def font(self) -> pygame.font.Font:
        return pygame.font.SysFont('Arial', 24)

    def update(self, state: TetrisState) -> None:
        self.image.fill((0, 0, 0))
        
        x = 32
        y = 0
        self.image.blit(
            self.font.render(f'Score: {state.score}', True, (255, 255, 255)),
            (x, y)
        )
        y += 32

        self.image.blit(
            self.font.render(f'Combo: {state.combo}', True, (255, 255, 255)),
            (x, y)
        )
        y += 32

        self.image.blit(
            self.font.render(f'Placed: {state.num_tetriminoes_placed}', True, (255, 255, 255)),
            (x, y)
        )
        y += 32

        self.image.blit(
            self.font.render(f'Lines: {state.num_lines_cleared}', True, (255, 255, 255)),
            (x, y)
        )
        y += 32
