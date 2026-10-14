"""CPU 可跑的 PyTorch 二元分類：Dataset、訓練、驗證、早停與最佳權重。"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def make_loaders(seed: int = 42) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(800, 2, generator=generator)
    y = (x[:, 0] + 0.7 * x[:, 1] > 0).long()
    dataset = TensorDataset(x.float(), y)
    train, valid = random_split(dataset, [640, 160], generator=generator)
    return (
        DataLoader(train, batch_size=64, shuffle=True, num_workers=0),
        DataLoader(valid, batch_size=160, shuffle=False, num_workers=0),
    )


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += len(y)
    return total_loss / count, correct / count


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = make_loaders()
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    best_loss = float("inf")
    wait = 0

    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "best-model.pt"
        for epoch in range(1, 31):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            val_loss, val_accuracy = evaluate(model, valid_loader, criterion, device)
            scheduler.step(val_loss)
            print(f"epoch={epoch:02d} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.3f}")
            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                wait = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                wait += 1
                if wait >= 5:
                    print("early stopping")
                    break

        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        best_val_loss, best_val_accuracy = evaluate(model, valid_loader, criterion, device)
        print(f"best val_loss={best_val_loss:.4f} best val_accuracy={best_val_accuracy:.3f}")
        assert best_val_accuracy > 0.9


if __name__ == "__main__":
    main()
