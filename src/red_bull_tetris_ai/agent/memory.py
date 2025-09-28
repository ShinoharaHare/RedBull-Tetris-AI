import random
from collections import deque
from typing import Generic, Iterable, TypeVar

T = TypeVar('T')

class Memory(Generic[T]):
    def __init__(self, capacity: int | None = None) -> None:
        self.buffer = deque(maxlen=capacity)

    def add(self, data: T) -> None:
        self.buffer.append(data)

    def sample(self, batch_size: int) -> list[T]:
        return random.sample(self.buffer, batch_size)
    
    def __iter__(self) -> Iterable[T]:
        return iter(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()

    def state_dict(self) -> dict:
        return {
            'capacity': self.buffer.maxlen,
            'buffer': list(self.buffer)
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.buffer = deque(state_dict['buffer'], maxlen=state_dict['capacity'])
