import safetensors.torch
import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import Self


class BlockRecognizer(nn.Module):
    """
    Input:
        Image: (N, 3, 22, 22), float, [0.0, 1.0]
        Position: (N,), long, 0-199
    Output:
        Block Type: (N,), long, 0-8
            0: None
            1-7: I, J, L, O, S, T, Z
            8: Golden
    Ghost: (N,), float, [0.0, 1.0]
    Special Item: (N,), float, [0.0, 1.0]
    """
    def __init__(self) -> None:
        super().__init__()

        self.position_embeds = nn.Embedding(200, 64)
        self.conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(64),
            nn.SiLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.SiLU(inplace=True),            

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.SiLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 11)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        image: torch.Tensor,
        position: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv_in(image)
        x = x + self.position_embeds(position).view(-1, 64, 1, 1)
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def compute_loss(
        self,
        image: torch.Tensor,
        position: torch.Tensor,
        block_type: torch.Tensor,
        ghost: torch.Tensor,
        special_item: torch.Tensor
    ) -> torch.Tensor:
        """
        image: (N, 3, 22, 22)
        position: (N,) dtype long, 0-199
        block_type: (N,) dtype long, 0-8
        ghost: (N,) dtype float, [0.0, 1.0]
        special_item: (N,) dtype float, [0.0, 1.0]
        """

        logits = self(image, position)
        loss = F.cross_entropy(logits[:, :9].float(), block_type)
        loss += F.binary_cross_entropy_with_logits(logits[:, 9].float(), ghost)
        loss += F.binary_cross_entropy_with_logits(logits[:, 10].float(), special_item)
        return loss

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        image: (N, 3, 22, 22)
        returns:
            block_type: (N,) dtype long, 0-8
            ghost: (N,) dtype float, [0.0, 1.0]
            special_item: (N,) dtype float, [0.0, 1.0]
        """
        image = image.to(self.device, self.dtype, non_blocking=True)
        position = position.to(self.device, non_blocking=True)
        logits = self(image, position)
        block_type = torch.argmax(logits[:, :9], dim=1)
        ghost = torch.sigmoid(logits[:, 9])
        special_item = torch.sigmoid(logits[:, 10])
        return block_type, ghost, special_item

    @classmethod
    def from_pretrained(cls, path: str) -> Self:
        model = cls()
        safetensors.torch.load_model(model, path)
        return model
