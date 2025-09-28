import pygame
import torch

from .action import POSIBLE_ACTIONS, TetrisAction
from .bag import TetriminoBag
from .game_board import TetrisGameBoard
from .score_board import TetrisScoreBoard
from .score_multiplier_meter import ScoreMultiplierMeter
from .scoring import TetrisScoring
from .special_item_meter import SpecialItemMeter
from .state import Position, TetrisState
from .tetrimino import Tetrimino, TetriminoType
from .upper_board import UpperBoard


class Tetris:
    def __init__(
        self,
        board_size: tuple[int, int] = (10, 20),
        block_size: tuple[int, int] = (40, 40),
        max_tetriminoes: int | None = None
    ) -> None:
        self.board_size = board_size
        self.block_size = block_size
        self.max_tetriminoes = max_tetriminoes

        self.bag = TetriminoBag(golden_tetrimino_after_n=None if max_tetriminoes is None else max_tetriminoes // 2)
        self.scoring = TetrisScoring()

        self.score_multiplier_meter_width = 64
        self.special_item_meter_width = 64
        self.score_board_width = 200
        self.upper_board_height = 100

        self.game_board = TetrisGameBoard(
            board_size,
            block_size,
            position=(self.score_multiplier_meter_width, self.upper_board_height)
        )
        self.score_multiplier_meter = ScoreMultiplierMeter(
            (self.score_multiplier_meter_width, self.game_board.image_height),
            position=(0, self.upper_board_height)
        )
        self.special_item_meter = SpecialItemMeter(
            (self.special_item_meter_width, self.game_board.image_height),
            position=(self.score_multiplier_meter_width + self.game_board.image_width, self.upper_board_height)
        )
        self.score_board = TetrisScoreBoard(
            (self.score_board_width, self.game_board.image_height),
            position=(self.score_multiplier_meter_width + self.special_item_meter_width + self.game_board.image_width, self.upper_board_height)
        )
        self.upper_board = UpperBoard(
            (self.game_board.image_width, self.upper_board_height),
            position=(0, 0)
        )

        self.sprites = pygame.sprite.Group(
            self.game_board,
            self.score_multiplier_meter,
            self.special_item_meter,
            self.score_board,
            self.upper_board
        )

        self.screen_width = self.game_board.image_width + self.score_multiplier_meter_width + self.special_item_meter_width + self.score_board_width
        self.screen_height = self.game_board.image_height + self.upper_board_height
        self.screen = None

        self.reset()

    @property
    def board_width(self) -> int:
        return self.board_size[0]
    
    @property
    def board_height(self) -> int:
        return self.board_size[1]
    
    @property
    def score(self) -> int:
        return self.state.score
    
    @property
    def combo(self) -> int:
        return self.state.combo
    
    @property
    def max_combo(self) -> int:
        return self.state.max_combo
    
    @property
    def num_lines_cleared(self) -> int:
        return self.state.num_lines_cleared
    
    @property
    def num_tetriminoes_placed(self) -> int:
        return self.state.num_tetriminoes_placed
    
    @property
    def score_multiplier(self) -> int:
        return self.state.score_multiplier
    
    @property
    def survived(self) -> bool:
        return self.state.survived
    
    @property
    def gameover(self) -> bool:
        return self.state.gameover

    def reset(self) -> None:
        self.state = TetrisState(
            board=torch.zeros(
                (2, self.board_height, self.board_width),
                dtype=torch.uint8
            ),
            max_tetriminoes=self.max_tetriminoes
        )
        self.bag.reset()
        self.new_tetrimino(self.state)

    def new_tetrimino(self, game_state: TetrisState) -> None:
        game_state.current_tetrimino, game_state.next_tetrimino = next(self.bag)
        game_state.current_position = [self.board_width // 2 - 2, 0]
        game_state.already_held = False

    def check_collision(
        self,
        board: torch.Tensor,
        tetrimino: Tetrimino,
        position: Position
    ) -> bool:
        x, y = position
        shape = torch.tensor(tetrimino.shape, dtype=torch.uint8)
        coords = shape.nonzero() + torch.tensor([y, x])

        if torch.any(
            (coords[:, 0] >= self.board_height) |
            (coords[:, 0] < 0) |
            (coords[:, 1] >= self.board_width) |
            (coords[:, 1] < 0)
        ):
            return True
        
        board_values = board[0, coords[:, 0], coords[:, 1]]
        
        if torch.any(board_values):
            return True
        
        return False
    
    def update_board(self, state: TetrisState) -> None:
        board = state.board.clone()
        tetrimino = state.current_tetrimino
        x, y = state.current_position

        shape = torch.tensor(tetrimino.shape, dtype=torch.uint8)
        offset = torch.tensor([y, x])
        coords = shape.nonzero() + offset
        coords[:, 0] = coords[:, 0].clamp(0, self.board_height - 1)
        coords[:, 1] = coords[:, 1].clamp(0, self.board_width - 1)
        board[0, coords[:, 0], coords[:, 1]] = tetrimino.type.value

        special_item_coords = torch.argwhere(shape == 2) + offset
        board[1, special_item_coords[:, 0], special_item_coords[:, 1]] = tetrimino.special_item
        
        state.board = board

    def hold(self, state: TetrisState) -> None:
        if state.already_held:
            return
        
        (
            state.held_tetrimino,
            state.current_tetrimino
        ) = state.current_tetrimino, state.held_tetrimino

        state.held_tetrimino.rotate(0)

        if state.current_tetrimino is None:
            current_tetrimino = state.next_tetrimino
            self.new_tetrimino(state)
            state.current_tetrimino = current_tetrimino
        else:
            state.current_position = [self.board_width // 2 - 4 // 2, 0]

        state.already_held = True

    def rotate(self, state: TetrisState) -> None:
        tetrimino = state.current_tetrimino.copy()
        tetrimino.rotate()
        if not self.check_collision(
            state.board,
            tetrimino,
            state.current_position
        ):
            state.current_tetrimino = tetrimino

    def move_left(self, state: TetrisState) -> None:
        position = state.current_position.copy()
        position[0] -= 1
        if not self.check_collision(
            state.board,
            state.current_tetrimino,
            position
        ):
            state.current_position = position

    def move_right(self, state: TetrisState) -> None:
        position = state.current_position.copy()
        position[0] += 1
        if not self.check_collision(
            state.board,
            state.current_tetrimino,
            position
        ):
            state.current_position = position

    def hard_drop(self, state: TetrisState) -> None:
        position = state.current_position.copy()
        while not self.check_collision(
            state.board,
            state.current_tetrimino,
            position
        ):
            position[1] += 1
        position[1] -= 1
        state.last_hard_drop_distance = position[1] - state.current_position[1]
        state.current_position = position

    def clear_completed_lines(self, state: TetrisState) -> None:
        board = state.board
        is_completed = torch.all(board[0] > 0, dim=1)
        num_lines_cleared = is_completed.sum().item()
        new_board = torch.zeros_like(board)
        new_board[:, num_lines_cleared:] = board[:, ~is_completed]

        special_items_cleared = board[1, is_completed]
        special_items_cleared = special_items_cleared[special_items_cleared > 0]
        special_items_cleared = special_items_cleared.sort(descending=False)[0].tolist()
        for special_item in special_items_cleared:
            item_type = (special_item - 1) % 3 + 1
            variant = (special_item - 1) // 3 % 3

            # if item_type == 1:
            #     new_board = Shifter(variant).apply(new_board)
            # elif item_type == 2:
            #     new_board = Pusher(variant).apply(new_board)
            # elif item_type == 3:
            #     new_board = Filler().apply(new_board)

        state.board = new_board
        state.last_golden_blocks_cleared = (board[0, is_completed] == TetriminoType.GOLDEN).sum().item()
        state.last_special_items_cleared = len(special_items_cleared)
        state.last_lines_cleared = num_lines_cleared
        state.num_lines_cleared += num_lines_cleared

    def update(
        self,
        state: TetrisState,
        new_tetrimino: bool = True
    ) -> None:
        self.update_board(state)
        self.clear_completed_lines(state)

        score_multiplier = state.score_multiplier
        score_multiplier_meter = state.score_multiplier_meter

        score = 0
        score += self.scoring.get_hard_drop_score(state.last_hard_drop_distance)
        score += self.scoring.get_line_clear_score(state.last_lines_cleared) * score_multiplier
        score += self.scoring.get_golden_block_score(state.last_golden_blocks_cleared) * score_multiplier
        score += self.scoring.get_special_item_score(state.last_special_items_cleared) * score_multiplier
        
        state.last_score_gained = score
        state.score += score
        state.num_tetriminoes_placed += 1

        decay = [0.00085, 0.0012, 0.002, 0.00355, 0.0075, 0.015, 0.0375, 0.05, 0.1, 0.0][score_multiplier - 1]
        # decay = [0.00085, 0.0012, 0.002, 0.00355, 0.0075, 0.015, 0.0375, 0.05, 0.1, 0.0][score_multiplier - 1]
        # decay = [0.0012, 0.0018, 0.003, 0.005, 0.01, 0.02, 0.05, 0.09, 0.15, 0.0][score_multiplier - 1]
        state.score_multiplier_meter -= decay
        state.score_multiplier_meter = max(0.0, state.score_multiplier_meter)
        state.score_multiplier_meter += 0.2 * state.last_golden_blocks_cleared

        if state.last_lines_cleared > 0:
            state.combo += 1
            state.special_item_meter += 0.25
            state.score_multiplier_meter += 0.2
        
        else:
            state.combo = 0
            state.special_item_meter += 0.05

        state.max_combo = max(state.max_combo, state.combo)

        if state.score_multiplier_meter >= 1.0:
            state.score_multiplier += 1
            state.score_multiplier = min(state.score_multiplier, 10)
            if state.score_multiplier == 10:
                state.score_multiplier_meter = 1.0
            else:
                state.score_multiplier_meter = 0.0
        
        if state.special_item_meter >= 1.0:
            state.special_item_meter = 0.0
            state.special_item_index += 1
            self.bag.set_special_item(state.special_item_index)

        reward = score * 0.01
        # reward -= max(0, state.max_height - 10) ** 2
        state.last_reward_gained = reward

        if not new_tetrimino:
            return

        if (
            self.max_tetriminoes is not None and
            state.num_tetriminoes_placed >= self.max_tetriminoes
        ):
            state.gameover = True
            state.survived = True

        else:
            self.new_tetrimino(state)
            state.gameover = self.check_collision(
                state.board,
                state.current_tetrimino,
                state.current_position
            )
            state.survived = False

    def init(self, title: str = 'Tetris') -> None:
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

    def render(self, state: TetrisState) -> None:
        if not pygame.get_init():
            self.init()

        if not state.gameover:
            state = state.copy()
            self.update_board(state)

        self.screen.fill((0, 0, 0))
        self.sprites.update(state)
        self.sprites.draw(self.screen)
        pygame.display.flip()

    def step(
        self,
        action: TetrisAction,
        render: bool = True
    ) -> tuple[float, bool]:
        state = self.state

        if action.hold:
            self.hold(state)

        for _ in range(action.rotation):
            self.rotate(state)

        for _ in range(abs(action.translation)):
            if action.translation < 0:
                self.move_left(state)
            elif action.translation > 0:
                self.move_right(state)

        self.hard_drop(state)        
        self.update(state)

        if render:
            self.render(state)
            pygame.event.pump()

        return state.last_reward_gained, state.gameover

    def get_state(self) -> TetrisState:
        return self.state.copy()

    def get_next_states(
        self,
        hold: bool | tuple[bool, ...] = (False, True)
    ) -> dict[TetrisAction, TetrisState]:
        if isinstance(hold, bool):
            hold = (hold,)

        next_states = {}
        for h in set(hold):
            hold_state = self.state.copy()
            
            if h:
                self.hold(hold_state)
            
            for rotation, translation in POSIBLE_ACTIONS[hold_state.current_tetrimino.type]:
                next_state = hold_state.copy()

                for _ in range(rotation):
                    self.rotate(next_state)

                for _ in range(abs(translation)):
                    if translation < 0:
                        self.move_left(next_state)
                    elif translation > 0:
                        self.move_right(next_state)
                
                self.hard_drop(next_state)
                self.update(next_state, new_tetrimino=False)
                next_states[TetrisAction(h, rotation, translation)] = next_state
        return next_states

    def start(self, control: bool = True) -> None:
        if not pygame.get_init():
            self.init()

        running = True
        while running and not self.state.gameover:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN and control:
                    key = event.key
                    if key == pygame.K_UP:
                        self.rotate(self.state)
                    elif key == pygame.K_c:
                        self.hold(self.state)
                    elif key == pygame.K_LEFT:
                        self.move_left(self.state)
                    elif key == pygame.K_RIGHT:
                        self.move_right(self.state)
                    elif key == pygame.K_SPACE:
                        self.hard_drop(self.state)
                        self.update(self.state)
                
            self.render(self.state)
            pygame.time.Clock().tick(30)

        pygame.quit()
