import torch
import torch.nn as nn

from src.models import ImageClassifier, ScratchCNN


def test_scratch_cnn_output_shape():
    model = ScratchCNN(feature_dim=128)
    logits = model(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, 128)


def test_classifier_output_shape():
    model = ImageClassifier(ScratchCNN(feature_dim=128), num_classes=200)
    logits = model(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, 200)


def test_forward_features_shape():
    model = ScratchCNN(feature_dim=128)
    features = model.forward_features(torch.randn(2, 3, 128, 128))
    assert features.shape == (2, 512, 4, 4)


def test_cam_target_is_conv():
    model = ScratchCNN(feature_dim=128)
    assert isinstance(model.cam_target_layer(), nn.Conv2d)


def test_classifier_backward():
    model = ImageClassifier(ScratchCNN(feature_dim=128), num_classes=10)
    x = torch.randn(2, 3, 128, 128)
    y = torch.tensor([1, 3])

    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()

    assert model.encoder.conv1[0].weight.grad is not None
