import json
from pathlib import Path

import fire
import safetensors.torch
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from red_bull_tetris_ai.env.red_bull_tetris import BlockRecognizer


class BlockDataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)

        with self.data_dir.joinpath('data.json').open() as f:
            data = json.load(f)
        
        self.data = []
        for path in self.data_dir.glob('images/*.png'):
            annotation = data[path.name]

            image = Image.open(path).convert('RGB')
            image = TF.to_tensor(image)
            image = image[None, :, 140:753, 98:404]
            image = F.unfold(image, kernel_size=22, stride=31)
            image = image.view(3, 22, 22, 20, 10)
            position = torch.arange(200).view(20, 10)
            for i in range(20):
                for j in range(10):
                    block_type = annotation['board'][i][j]
                    ghost = annotation['ghost'][i][j]
                    special_item = annotation['special_item'][i][j]
                    self.data.append(
                        {
                            'image': image[:, :, :, i, j],
                            'position': position[i, j],
                            'block_type': torch.tensor(block_type, dtype=torch.long),
                            'ghost': torch.tensor(ghost, dtype=torch.float),
                            'special_item': torch.tensor(special_item, dtype=torch.float)
                        }
                    )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[index]

    def collate_fn(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            'image': torch.stack([x['image'] for x in batch]),
            'position': torch.stack([x['position'] for x in batch]),
            'block_type': torch.stack([x['block_type'] for x in batch]),
            'ghost': torch.stack([x['ghost'] for x in batch]),
            'special_item': torch.stack([x['special_item'] for x in batch])
        }


def main(
    data_dir: str,
    save_dir: str,
    num_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 3e-4,
    num_workers: int = 0,
    device: torch.device | str | None = None
):
    data_dir: Path = Path(data_dir)
    save_dir: Path = Path(save_dir)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    dataset = BlockDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
    )

    model = BlockRecognizer()
    model.train().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    with tqdm(total=len(dataloader)) as progress_bar:
        for epoch in range(num_epochs):
            progress_bar.reset()
            progress_bar.set_description(f'Epoch {epoch} / {num_epochs}')

            for batch in dataloader:
                loss = model.compute_loss(
                    image=batch['image'].to(device),
                    position=batch['position'].to(device),
                    block_type=batch['block_type'].to(device),
                    ghost=batch['ghost'].to(device),
                    special_item=batch['special_item'].to(device)
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                progress_bar.update()

    save_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_model(model, save_dir / f'epoch-{epoch:04d}.safetensors')


if __name__ == '__main__':
    fire.Fire(main)
