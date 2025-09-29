import torch

from red_bull_tetris_ai.env import *

from .tetris_dqn_agent import TetrisDQNAgent


class RedBullTetrisDQNAgent(TetrisDQNAgent):
    def __init__(
        self,
        env: RedBullTetris | None = None,
        allow_action_filtering: bool = False
    ) -> None:
        super().__init__()

        self.env = env
        self.allow_action_filtering = allow_action_filtering
    
    def filter_actions(self, next_states: dict[TetrisAction, TetrisState]) -> dict[TetrisAction, TetrisState]:
        state = self.env.simulation.state
        score_multiplier = state.score_multiplier
        current_golden = (
            state.current_tetrimino is not None and
            state.current_tetrimino.type == TetriminoType.GOLDEN
        )
        holding_golden = (
            state.held_tetrimino is not None and
            state.held_tetrimino.type == TetriminoType.GOLDEN
        )

        if score_multiplier < 10 and state.max_height <= 10:
            # 消行：1 > 2 > 3 > 4 > 0
            # 特殊方塊：1 > 2 > 3 > ... > 0
            next_states = sorted(
                next_states.items(),
                key=lambda kv: (
                    kv[1].last_lines_cleared == 0,
                    kv[1].last_lines_cleared,
                    kv[1].last_special_items_cleared == 0,
                    kv[1].last_special_items_cleared
                )
            )
            next_states = {
                k: v for k, v in next_states
                if (
                    v.last_lines_cleared == next_states[0][1].last_lines_cleared and
                    v.last_special_items_cleared == next_states[0][1].last_special_items_cleared
                )
            }

        elif score_multiplier >= 9 and (current_golden or holding_golden):
            # 倍率 >= 9 才消金色方塊，而且必須一次消 4 個
            next_states = {
                k: v for k, v in next_states.items()
                if (
                    v.last_golden_blocks_cleared == 4 or k.rotation == 0 or (holding_golden and not k.hold)
                )
            }

        elif score_multiplier == 10:
            if state.max_height <= 10 and state.can_tetris:
                tetris_states = {
                    k: v for k, v in next_states.items()
                    if (
                        v.last_lines_cleared == 4 or
                        v.last_special_items_cleared > 0 or
                        v.can_tetris
                    )
                }

                if tetris_states:
                    next_states = tetris_states

            # 消行：4 > 3 > 2 > 1 > 0
            # 特殊方塊：1 > 2 > 3 > ... > 0
            next_states = sorted(
                next_states.items(),
                key=lambda kv: (
                    -kv[1].num_lines_cleared,
                    kv[1].last_special_items_cleared == 0,
                    kv[1].last_special_items_cleared
                )
            )
            next_states = {
                k: v for k, v in next_states
                if (
                    v.last_lines_cleared == next_states[0][1].last_lines_cleared and
                    v.last_special_items_cleared == next_states[0][1].last_special_items_cleared
                )
            }
        return next_states

    @torch.no_grad()
    def predict(self, next_states: dict[TetrisAction, TetrisState]) -> TetrisAction:
        if self.allow_action_filtering:
            next_states = self.filter_actions(next_states)

        if len(next_states) == 1:
            return next_states.popitem()[0]

        return super().predict(next_states)
