from functools import cached_property
import pygame

from .state import TetrisState


class ScoreMultiplierMeter(pygame.sprite.Sprite):
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
        
        outline_rect = self.image.get_rect().inflate(-8, -8)
        
        text = self.font.render(f'{state.score_multiplier}x', True, (255, 255, 255))
        self.image.blit(text, (outline_rect.centerx - text.get_width() // 2, 4))
        
        outline_rect.top += text.get_height() + 16
        outline_rect.height -= text.get_height() + 16
        pygame.draw.rect(self.image, (255, 255, 255), outline_rect, 2)

        bar_rect = pygame.Rect(
            outline_rect.left,
            outline_rect.top + outline_rect.height * (1 - state.score_multiplier_meter),
            outline_rect.width,
            int(outline_rect.height * state.score_multiplier_meter)
        )
        self.image.fill((255, 255, 255), bar_rect)
