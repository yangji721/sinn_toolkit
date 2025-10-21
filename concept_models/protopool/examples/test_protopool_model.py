import torch
from concept_models.protopool.adapter import ProtoPoolAdapter
from concept_models.protopool.settings import img_size, prototype_shape, num_classes


def test_protopool_model():
    model = ProtoPoolAdapter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train(False)

    x = torch.randn(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        out = model.forward(x)
        print('Output shape:', out.shape)
        interp = model.interpret(x)
        print('Interpret keys:', list(interp.keys()))

if __name__ == '__main__':
    test_protopool_model()

    # ------------------- Tiny training demo -------------------
    def train_demo(model, device, epochs=2, batch_size=8, lr=1e-3):
        model.train()
        mm = model.module if hasattr(model, 'module') else model
        in_ch = 3
        import torch.nn as nn
        for layer in mm.modules():
            if isinstance(layer, nn.Conv2d):
                in_ch = layer.in_channels
                break

        # infer classes
        with torch.no_grad():
            sample = torch.randn(1, in_ch, img_size, img_size).to(device)
            out = model(sample)
            if isinstance(out, (tuple, list)):
                out = out[0]
            n_classes = out.shape[1]

        from torch.utils.data import TensorDataset, DataLoader
        N = 64
        X = torch.randn(N, in_ch, img_size, img_size)
        y = torch.randint(0, n_classes, (N,))
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        optim = torch.optim.Adam(model.parameters(), lr=lr)
        crit = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            lsum = 0.0
            correct = 0
            total = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optim.zero_grad()
                out = model(xb)
                if isinstance(out, (tuple, list)):
                    logits = out[0]
                else:
                    logits = out
                loss = crit(logits, yb)
                loss.backward()
                optim.step()
                lsum += loss.item() * xb.size(0)
                _, preds = logits.max(1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
            print(f"Epoch {epoch+1}/{epochs} - loss: {lsum/total:.4f}, acc: {correct/total:.4f}")

    # run demo
    if __name__ == '__main__':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_for_train = ProtoPoolAdapter().to(device)
        try:
            train_demo(model_for_train, device, epochs=2, batch_size=8)
        except Exception as e:
            print('ProtoPool training demo failed:', e)