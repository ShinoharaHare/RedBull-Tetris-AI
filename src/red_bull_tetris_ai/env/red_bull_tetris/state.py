import time

from .frame import RedBullTetrisFrame


class RedbullTetrisState:
    def __init__(self) -> None:
        self.reset()

    @property
    def score(self) -> int:
        return self._score
    
    @score.setter
    def score(self, value: int) -> None:
        self._score = value
    
    @property
    def score_multiplier(self) -> int:
        return self._score_multiplier
    
    @property
    def score_multiplier_meter(self) -> float:
        return self._score_multiplier_meter
    
    @property
    def special_item_meter(self) -> float:
        return self._special_item_meter

    @property
    def time_left(self) -> int:
        return 180 - self._current_time + self._start_time

    @property
    def gameover(self) -> bool:
        return self._gameover

    @gameover.setter
    def gameover(self, value: bool) -> None:
        self._gameover = value

    def reset(self) -> None:
        self._gameover = False
        self._score = 0
        self._score_multiplier = 1
        self._score_multiplier_meter = 0.0
        self._special_item_meter = 0.0
        self._start_time = int(time.time())
        self._current_time = self._start_time
        self._line_clear_counter = 0

    def update(
        self,
        frame: RedBullTetrisFrame,
        line_clear: bool = False
    ) -> None:
        self._frame = frame
        self._current_time = int(time.time())

        # self._special_item_meter = frame.special_item_meter
        
        # if self._score_multiplier < 10:
        #     self._score_multiplier_meter = frame.score_multiplier_meter
        # else:
        #     self._score_multiplier_meter = 1.0

        if line_clear:
            self._line_clear_counter += 1
            if (
                self._score_multiplier < 10 and
                self._line_clear_counter % 6 == 0 and
                frame.score_multiplier > self._score_multiplier
            ):
                self._score_multiplier = frame.score_multiplier

            # if frame.score > -1 and frame.score > self.score:
            #     self._score = frame.score
