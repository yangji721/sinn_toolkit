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