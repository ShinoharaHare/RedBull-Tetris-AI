from dataclasses import dataclass


@dataclass
class TetrisScoring:
    single: int = 100
    double: int = 300
    triple: int = 500
    tetris: int = 800

    soft_drop: int = 1
    hard_drop: int = 2

    one_golden_block: int = 400
    two_golden_blocks: int = 1200
    three_golden_blocks: int = 2000
    four_golden_blocks: int = 2800

    t_spin: int = 200
    t_spin_single: int = 800
    t_spin_double: int = 1200
    t_spin_triple: int = 1600

    special_item: int = 1000

    def get_line_clear_score(self, num_cleared_lines: int) -> int:
        if num_cleared_lines <= 0:
            return 0
        elif num_cleared_lines == 1:
            return self.single
        elif num_cleared_lines == 2:
            return self.double
        elif num_cleared_lines == 3:
            return self.triple
        elif num_cleared_lines >= 4:
            return self.tetris

    def get_golden_block_score(self, num_cleared_golden_blocks: int) -> int:
        if num_cleared_golden_blocks <= 0:
            return 0
        elif num_cleared_golden_blocks == 1:
            return self.one_golden_block
        elif num_cleared_golden_blocks == 2:
            return self.two_golden_blocks
        elif num_cleared_golden_blocks == 3:
            return self.three_golden_blocks
        elif num_cleared_golden_blocks >= 4:
            return self.four_golden_blocks

    def get_special_item_score(self, num_cleared_special_items: int) -> int:
        return num_cleared_special_items * self.special_item

    def get_soft_drop_score(self, distance: int) -> int:
        return distance * self.soft_drop

    def get_hard_drop_score(self, distance: int) -> int:
        return distance * self.hard_drop
