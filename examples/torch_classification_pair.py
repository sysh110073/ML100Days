"""PyTorch：與 tf_classification.py 共用資料和模型層數的明確訓練迴圈。"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from paired_data import make_binary_data


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    loss_sum = correct = count = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for xb, yb in loader:
            logits = model(xb)
            loss_sum += criterion(logits, yb).item()
            correct += (logits.argmax(dim=1) == yb).sum().item()
            count += len(yb)
    return loss_sum / count, correct / count


def main() -> None:
    torch.manual_seed(42)
    x_train, y_train, x_valid, y_valid = make_binary_data()
    train = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).long()),
                       batch_size=64, shuffle=True)
    valid = DataLoader(TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid).long()),
                       batch_size=64)
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "best.pt"
        best_loss, wait = float("inf"), 0
        for _ in range(30):
            model.train()
            for xb, yb in train:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
            val_loss, _ = evaluate(model, valid)
            if val_loss < best_loss:
                best_loss, wait = val_loss, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                wait += 1
                if wait >= 5:
                    break

        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        loss, accuracy = evaluate(model, valid)
        print(f"PyTorch best val_loss={loss:.4f} val_accuracy={accuracy:.3f}")
        assert accuracy > 0.9


if __name__ == "__main__":
    main()
