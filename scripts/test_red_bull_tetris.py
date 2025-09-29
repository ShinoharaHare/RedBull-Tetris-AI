from threading import Thread

import fire
import pytesseract
import torch

from red_bull_tetris_ai.agent import *
from red_bull_tetris_ai.env.red_bull_tetris import *


def main(
    dqn_agent_path: str,
    block_recognizer_path: str,
    chrome_user_data_dir: str | None = None,
    region: Region | str = Region.US,
    allow_action_filtering: bool = False,
    render_simulation: bool = False,
    tesseract_cmd: str = 'tesseract',
    device: torch.device | None = None,
    dtype: torch.dtype = torch.half
):

    if isinstance(region, str):
        region = Region[region.upper()]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    block_recognizer = BlockRecognizer.from_pretrained(block_recognizer_path)
    block_recognizer.eval().to(device, dtype)

    env = RedBullTetris(
        region=region,
        block_recognizer=block_recognizer,
        chrome_user_data_dir=chrome_user_data_dir,
    )

    model = RedBullTetrisDQNAgent.from_pretrained(dqn_agent_path, env, allow_action_filtering)
    model.eval().to(device, dtype)

    if render_simulation:
        render_thread = Thread(
            target=env.simulation.start,
            kwargs={'control': False},
            daemon=True
        )
        render_thread.start()
    
    env.reset()
    while not env.gameover:
        next_states = env.get_next_states(hold=False)
        action = model.predict(next_states)
        env.step(action)

    if render_simulation:
        env.simulation.state.gameover = True
        render_thread.join()

    input('Press Enter to exit...')


if __name__ == '__main__':
    fire.Fire(main)
