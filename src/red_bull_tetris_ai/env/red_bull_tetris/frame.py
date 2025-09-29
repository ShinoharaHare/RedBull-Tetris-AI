import logging
from functools import cache, cached_property

import pytesseract
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from red_bull_tetris_ai.env.tetris import Tetrimino, TetriminoType
from red_bull_tetris_ai.utils import tesseract_available

from .block_recognizer import BlockRecognizer
from .images import RedbullTerisImage

logger = logging.getLogger(__name__)

EMPTY_BOARD_TENSOR = TF.to_tensor(RedbullTerisImage.EMPTY_BOARD)[:, 140:195, 190:310].half()
PLAY_BUTTON_TENSOR = TF.to_tensor(RedbullTerisImage.TITLE)[:, 495:545, 130:380].half()
NEXT_TETRIMMINO_TENSORS = torch.stack(
    [
        TF.to_tensor(RedbullTerisImage.NEXT_TETRIMINO.I),
        TF.to_tensor(RedbullTerisImage.NEXT_TETRIMINO.J),
        TF.to_tensor(RedbullTerisImage.NEXT_TETRIMINO.L),
        TF.to_tensor(RedbullTerisImage.NEXT_TETRIMINO.O),
        TF.to_tensor(RedbullTerisImage.NEXT_TETRIMINO.S),
        TF.to_tensor(RedbullTerisImage.NEXT_TETRIMINO.T),
        TF.to_tensor(RedbullTerisImage.NEXT_TETRIMINO.Z),
        TF.to_tensor(RedbullTerisImage.NEXT_TETRIMINO.GOLDEN)
    ]
).half()
HELD_TETRIMINO_TENSORS = torch.stack(
    [
        TF.to_tensor(RedbullTerisImage.HELD_TETRIMINO.I),
        TF.to_tensor(RedbullTerisImage.HELD_TETRIMINO.J),
        TF.to_tensor(RedbullTerisImage.HELD_TETRIMINO.L),
        TF.to_tensor(RedbullTerisImage.HELD_TETRIMINO.O),
        TF.to_tensor(RedbullTerisImage.HELD_TETRIMINO.S),
        TF.to_tensor(RedbullTerisImage.HELD_TETRIMINO.T),
        TF.to_tensor(RedbullTerisImage.HELD_TETRIMINO.Z),
        TF.to_tensor(RedbullTerisImage.HELD_TETRIMINO.GOLDEN),
        TF.to_tensor(RedbullTerisImage.HELD_TETRIMINO.NONE)
    ]
).half()


class RedBullTetrisFrame:
    def __init__(
        self,
        image: Image.Image,
        detector: BlockRecognizer | None = None
    ) -> None:
        self.warn_once_if_tesseract_not_available()

        self._block_recognizer = detector
        self._image = image.convert('RGB')

    @classmethod
    @cache
    def warn_once_if_tesseract_not_available(cls) -> None:
        if not tesseract_available():
            logger.warning(
                'Tesseract is not available. '
                'The following game states will not be recognized: '
                'score, score multiplier, time left.'
            )

    @property
    def block_recognizer(self) -> BlockRecognizer:
        if self._block_recognizer is None:
            raise ValueError('To perform block recognition, a `BlockRecognizer` instance must be provided.')
        return self._block_recognizer

    @property
    def image(self) -> Image.Image:
        return self._image
    
    @cached_property
    def valid(self) -> bool:
        return self.image.size == (501, 855)

    @cached_property
    def grayscale_image(self) -> Image.Image:
        return self.image.convert('L')
    
    @cached_property
    def image_tensor(self) -> torch.Tensor:
        return TF.to_tensor(self.image).half()
    
    @cached_property
    def grayscale_image_tensor(self) -> torch.Tensor:
        return TF.to_tensor(self.grayscale_image).half()

    @cached_property
    def is_title_screen(self) -> bool:
        play_button_region = self.image_tensor[:, 495:545, 130:380]
        distance = torch.abs(play_button_region - PLAY_BUTTON_TENSOR).mean().item()
        return distance < 0.01

    @cached_property
    def block_tensor(self) -> torch.Tensor:
        """
        Return shape: (3, 22, 22, 20, 10)
        """
        board = self.image_tensor[None, :, 140:753, 98:404]
        blocks = F.unfold(board, kernel_size=22, stride=31)
        blocks = blocks.view(3, 22, 22, 20, 10)
        return blocks

    @cached_property
    def block_recognization_result(self):
        block_type, ghost, special_item = self.block_recognizer.predict(
            self.block_tensor.permute(3, 4, 0, 1, 2).reshape(-1, 3, 22, 22),
            torch.arange(200, dtype=torch.long)
        )
        block_type = block_type.view(20, 10)
        ghost = ghost.view(20, 10) > 0.5
        special_item = special_item.view(20, 10) > 0.5
        block_type = block_type.cpu()
        ghost = ghost.cpu()
        special_item = special_item.cpu()
        return block_type, ghost, special_item

    @cached_property
    def full_board(self) -> torch.Tensor:
        block_type, ghost, special_item = self.block_recognization_result
        board = torch.zeros((2, 20, 10), dtype=torch.uint8)
        board[0] = block_type * ~ghost
        board[1] = special_item.byte() * ~ghost
        return board

    @cached_property
    def current_board(self) -> torch.Tensor:
        board = torch.zeros_like(self.full_board)
        if self.current_tetrimino is not None:
            coords = torch.tensor(self.current_tetrimino.shape).nonzero()
            coords += torch.tensor([0, board.size(2) // 2 - 2])
            coords[:, 0] = coords[:, 0].clamp(0, board.size(1) - 1)
            coords[:, 1] = coords[:, 1].clamp(0, board.size(2) - 1)
            board[0, coords[:, 0], coords[:, 1]] = self.current_tetrimino.type.value
        return board

    @cached_property
    def board(self) -> torch.Tensor:
        return self.full_board - self.current_board

    @cached_property
    def next_tetrimino_spawned(self) -> bool:
        distance = torch.abs(self.image_tensor[:, 140:195, 190:310] - EMPTY_BOARD_TENSOR)
        distance = distance.mean().item()
        return distance > 0.05

    @cached_property
    def current_tetrimino(self) -> Tetrimino | None:
        index_y = [0, 0, 0, 1, 1, 1, 1]
        index_x = [3, 4, 5, 3, 4, 5, 6]

        if 'full_board' in self.__dict__:
            block_type = self.full_board[0, index_y, index_x]
        else:
            block_tensor = self.block_tensor[:, :, :, index_y, index_x]
            block_type = self.block_recognizer.predict(
                image=block_tensor.permute(3, 0, 1, 2),
                position=torch.tensor([3, 4, 5, 13, 14, 15, 16])
            )[0]
            block_type = block_type.cpu()

        # (0, 3) -> 0
        # (0, 4) -> 1
        # (0, 5) -> 2
        # (1, 3) -> 3
        # (1, 4) -> 4
        # (1, 5) -> 5
        # (1, 6) -> 6
        indices = {
            TetriminoType.I: [3, 4, 5, 6],
            TetriminoType.J: [0, 3, 4, 5],
            TetriminoType.L: [2, 3, 4, 5],
            TetriminoType.O: [1, 2, 4, 5],
            TetriminoType.S: [1, 2, 3, 4],
            TetriminoType.T: [1, 3, 4, 5],
            TetriminoType.Z: [0, 1, 4, 5],
            TetriminoType.GOLDEN: [3, 4, 5, 6]
        }
        for tetrimino_type, index in indices.items():
            if torch.all(block_type[index] == tetrimino_type.value):
                return Tetrimino(tetrimino_type)
        return None

    @cached_property
    def next_tetrimino(self) -> Tetrimino:
        next_tetrimino_region = self.image_tensor[None, :, 65:110, 370:430]
        distance = torch.abs(NEXT_TETRIMMINO_TENSORS - next_tetrimino_region).sum(dim=(1, 2, 3))
        index = distance.argmin().item()
        tetrimino_type = TetriminoType(index + 1)
        tetrimino = Tetrimino(tetrimino_type)
        return tetrimino

    @cached_property
    def held_tetrimino(self) -> Tetrimino | None:
        held_tetrimino_region = self.image_tensor[None, :, 65:110, 70:130]
        distance = torch.abs(HELD_TETRIMINO_TENSORS - held_tetrimino_region).sum(dim=(1, 2, 3))
        index = distance.argmin().item()
        if index == 8:
            return None
        tetrimino_type = TetriminoType(index + 1)
        tetrimino = Tetrimino(tetrimino_type)
        return tetrimino

    @cached_property
    def score(self) -> int:
        if not tesseract_available():
            return -1

        score_region = self.image.crop((320, 780, 470, 830))
        score = pytesseract.image_to_string(
            score_region,
            lang='eng',
            config='--psm 11 -c tessedit_char_whitelist=,0123456789'
        )
        score = score.replace('\n', '')
        score = score.replace(',', '')
        if not score.isdigit():
            return -1
        return int(score)

    @cached_property
    def score_multiplier(self) -> int:
        if not tesseract_available():
            return -1

        multiplier_region = self.image.crop((25, 135, 75, 185))
        multiplier = pytesseract.image_to_string(
            multiplier_region,
            lang='eng',
            config='--psm 11 -c tessedit_char_whitelist=x0123456789'
        )
        multiplier = multiplier.replace('\n', '')
        multiplier = multiplier.replace('\n', '')
        multiplier = multiplier.removesuffix('x')

        if not multiplier.isdigit():
            return -1
        
        multiplier = int(multiplier)
        multiplier = min(max(1, multiplier), 10)
        return multiplier

    @cached_property
    def score_multiplier_meter(self) -> float:
        meter_tensor = self.grayscale_image_tensor[:, 195:745, 60:70]
        meter_tensor = torch.where(meter_tensor > 0.4, 1.0, 0.0)
        filled_pixels = torch.sum(meter_tensor == 1.0)
        meter = filled_pixels / meter_tensor.numel()
        meter = meter.item()
        meter = max(0.0, meter - 0.01)
        return meter

    @cached_property
    def special_item_meter(self) -> float:
        meter_tensor = self.grayscale_image_tensor[:, 195:745, 430:440]
        meter_tensor = torch.where(meter_tensor > 0.5, 1.0, 0.0)
        filled_pixels = torch.sum(meter_tensor == 1.0)
        meter = filled_pixels / meter_tensor.numel()
        meter = meter.item()
        meter = max(0.0, meter - 0.01)
        return meter

    @cached_property
    def time_left(self) -> int:
        if not tesseract_available():
            return -1

        time_region = self.image.crop((205, 775, 295, 825))
        time_left = pytesseract.image_to_string(
            time_region,
            lang='eng',
            config='--psm 7 -c tessedit_char_whitelist=:0123456789'
        )
        time_left = time_left.replace('\n', '')

        if time_left.count(':') != 1:
            return -1
        
        minutes, seconds = time_left.split(':')
        
        if (
            not minutes.isdigit()
            or not seconds.isdigit()
            or len(minutes) != 1
            or len(seconds) != 2
        ):
            return -1
        
        return int(minutes) * 60 + int(seconds)

    def debug(self) -> Image.Image:
        from PIL import ImageDraw, ImageFont

        font = ImageFont.load_default(size=8)

        image = self.image.copy()
        block_type, ghost, special_item = self.block_recognization_result
        draw = ImageDraw.Draw(image)
        for i in range(20):
            for j in range(10):
                x0 = 98 + j * 31
                y0 = 140 + i * 31
                
                t = block_type[i, j].item()
                if t == 0:
                    name = ' '
                else:
                    name = TetriminoType(t).name[0]
                draw.text((x0 + 11, y0 - 4), name, fill=(0, 0, 0), font=font)

                if ghost[i, j] > 0.5:
                    draw.text((x0 + 2, y0 + 2), 'Ghost', fill=(0, 0, 0), font=font)

                if special_item[i, j] > 0.5:
                    draw.text((x0 + 2, y0 + 8), 'Special', fill=(0, 0, 0), font=font)
        return image
