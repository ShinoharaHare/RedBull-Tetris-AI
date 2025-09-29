import base64
import logging
import re
import time
from enum import Enum
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from selenium.common.exceptions import (NoSuchElementException,
                                        StaleElementReferenceException,
                                        TimeoutException)
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from red_bull_tetris_ai.env.tetris import *
from red_bull_tetris_ai.utils import tesseract_available

from .block_recognizer import BlockRecognizer
from .frame import RedBullTetrisFrame
from .state import RedbullTetrisState


class Region(Enum):
    JA = 'JA'
    TW = 'TW'
    US = 'US'


REGION_TO_URL = {
    Region.JA: 'https://www.redbull.com/jp-ja/events/red-bull-tetris-japan',
    Region.TW: 'https://www.redbull.com/tw-zh/events/red-bull-tetris-taiwan',
    Region.US: 'https://www.redbull.com/us-en/events/red-bull-tetris-united-states',
}


class RedBullTetris:
    def __init__(
        self,
        region: Region,
        block_recognizer: BlockRecognizer | None = None,
        chrome_user_data_dir: str | None = None,
        max_tetriminoes: int = 350
    ) -> None:
        self.region = region
        self.block_recognizer = block_recognizer

        options = ChromeOptions()
        options.add_argument('--window-size=400,1000')

        if chrome_user_data_dir is not None:
            options.add_argument(f'--user-data-dir={chrome_user_data_dir}')
        # options.add_argument('--auto-open-devtools-for-tabs')；
        options.set_capability('pageLoadStrategy', 'none')
        self.driver = Chrome(options)
        self.driver.execute_cdp_cmd(
            'Page.addScriptToEvaluateOnNewDocument',
            {'source': Path(__file__).parent.joinpath('script.js').read_text()}
        )

        self.frame = None
        self.state = RedbullTetrisState()
        self.simulation = Tetris(
            board_size=(10, 20),
            block_size=(40, 40),
            max_tetriminoes=max_tetriminoes
        )

    @property
    def score(self) -> int:
        return self.state.score
    
    @property
    def score_multiplier(self) -> int:
        return self.state.score_multiplier
    
    @property
    def score_multiplier_meter(self) -> float:
        return self.state.score_multiplier_meter
    
    @property
    def special_item_meter(self) -> float:
        return self.state.special_item_meter

    @property
    def time_left(self) -> int:
        return self.state.time_left

    @property
    def current_tetrimino(self) -> Tetrimino:
        return self.frame.current_tetrimino

    @property
    def next_tetrimino(self) -> Tetrimino:
        return self.frame.next_tetrimino

    @property
    def held_tetrimino(self) -> Tetrimino | None:
        return self.frame.held_tetrimino
    
    @property
    def board(self) -> torch.Tensor:
        return self.frame.board

    @property
    def gameover(self) -> bool:
        return self.state.gameover

    def _start(self) -> bool:
        self.driver.get(REGION_TO_URL[self.region])

        wait = WebDriverWait(self.driver, timeout=10)
        try:
            if self.driver.get_cookie('OptanonAlertBoxClosed') is None:
                wait.until(
                    EC.presence_of_element_located((By.ID, 'onetrust-accept-btn-handler'))
                ).click()

            button = wait.until(
                EC.element_to_be_clickable(
                    (
                        By.CSS_SELECTOR,
                        '.unified-event-hero__ctas cosmos-button.hydrated'
                    )
                )
            )
        except (TimeoutException, StaleElementReferenceException):
            return False
        
        ActionChains(self.driver).move_to_element(button).perform()
        button.click()

        time.sleep(0.5)
        
        try:
            self.driver.find_element(By.XPATH, '//cosmos-button[text()="Skip" and contains(@class, "hydrated")]').click()
        except NoSuchElementException: ...

        try:
            overlay = self.driver.find_element(By.CSS_SELECTOR, 'div.fixed.top-0.left-0.w-full.h-full')
            wait.until(lambda driver: overlay.value_of_css_property('opacity') == '0')
        except TimeoutException:
            return False

        frame = self.get_frame()
        if not frame.valid or not frame.is_title_screen:
            return False
        
        self.touch(260, 520)
        return True

    def start(self) -> None:
        while not self._start(): ...
        
        t = time.time()
        time.sleep(1.0)
        self.update_frame()
        time.sleep(max(0, 3.0 - (time.time() - t)))

    def reset(self) -> None:
        self.frame = None
        self.start()
        self.simulation.reset()
        self.state.reset()
        self.simulation.state.current_tetrimino = self.next_tetrimino
        self.update_frame()
        self.simulation.state.next_tetrimino = self.next_tetrimino

    def touch(self, x: int, y: int) -> None:
        self.driver.execute_script('controller.touch(arguments[0], arguments[1]);', x, y)

    def move_left(self) -> None:
        self.driver.execute_script('controller.moveLeft();')

    def move_right(self) -> None:
        self.driver.execute_script('controller.moveRight();')

    def rotate(self) -> None:
        self.driver.execute_script('controller.rotate();')

    def hard_drop(self) -> None:
        self.driver.execute_script('controller.hardDrop();')

    def hold(self) -> None:
        self.driver.execute_script('controller.hold();')

    def check_gameover(self) -> bool:
        try:
            self.driver.find_element(By.XPATH, '//cosmos-button[text()="Play again"]')
            return True
        except NoSuchElementException:
            return False
        
    def get_final_score(self) -> int:
        element = self.driver.find_element(By.XPATH, '//span[contains(text(),"Your current score is")]')
        score = re.search(r'Your current score is (\d+).', element.text)[1]
        score = int(score)
        return score

    def update_state(self, line_clear: bool = False) -> None:
        self.state.update(self.frame, line_clear)
        self.state.gameover = self.check_gameover()
        if self.state.gameover:
            self.state.score = self.get_final_score()

    def get_screenshot(self) -> Image.Image:
        data = self.driver.execute_script('return screenshot();')
        image_data = data.split('base64,', 1)[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        return image

    def get_frame(self) -> RedBullTetrisFrame:
        image = self.get_screenshot()
        frame = RedBullTetrisFrame(image, self.block_recognizer)
        return frame

    def update_frame(self) -> None:
        self.frame = self.get_frame()

    def get_state(self) -> TetrisState:
        return self.simulation.get_state()

    def get_next_states(self, hold: tuple[bool] = (False, True)) -> dict[TetrisAction, TetrisState]:
        return self.simulation.get_next_states(hold)
    
    def wait_for_pending_commands(self) -> None:
        while self.driver.execute_script('return controller.hasPendingCommands();'):
            time.sleep(0.001)

    def wait_for_next_tetrimino(self) -> None:
        while True:
            self.update_frame()
            if self.frame.next_tetrimino_spawned:
                break

    def step(self, action: TetrisAction) -> tuple[float, bool]:
        simulation_state = self.simulation.state
        special_item_index = simulation_state.special_item_index
        self.simulation.step(action, render=False)

        if action.hold:
            self.hold()
            time.sleep(0.125)

            if self.held_tetrimino is None:
                self.update_frame()
        
        for _ in range(action.rotation):
            self.rotate()

        for _ in range(abs(action.translation)):
            if action.translation < 0:
                self.move_left()
            else:
                self.move_right()

        self.hard_drop()
        self.wait_for_pending_commands()
        time.sleep(0.01)

        current_tetrimino = self.next_tetrimino
        line_clear = simulation_state.last_lines_cleared > 0
        if simulation_state.special_item_index != special_item_index:
            current_tetrimino.special_item = simulation_state.special_item_index

        if line_clear:
            self.wait_for_next_tetrimino()
            simulation_state.board.copy_(self.frame.board)
        else:
            self.update_frame()

        self.update_state(line_clear)
        
        simulation_state.current_tetrimino = current_tetrimino
        simulation_state.next_tetrimino = self.next_tetrimino

        if tesseract_available():
            simulation_state.score_multiplier = self.score_multiplier

        # simulation_state.score_multiplier_meter = self.score_multiplier_meter
        # simulation_state.special_item_meter = self.special_item_meter
        max_tetriminoes = self.simulation.max_tetriminoes
        simulation_state.num_tetriminoes_placed = min(int((1.0 - self.time_left / 180) * max_tetriminoes), max_tetriminoes - 5)
        if action.hold:
            simulation_state.held_tetrimino = self.held_tetrimino
        
        return 0.0, self.gameover
