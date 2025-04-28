

def test_focal_loss_function():
    """Test the focal loss function"""
    from utils.loss_functions import FocalLoss
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    import numpy as np

    # Create a dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.fc1 = nn.Linear(10, 10)
            self.fc2 = nn.Linear(10, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    model = DummyModel()
    # Create a dummy loss function
    loss_function = FocalLoss()
    # Create a dummy input and target
    input_tensor = torch.rand(10, 10)
    target_tensor = torch.randint(0, 10, (10,))
    # Compute the loss
    loss = loss_function(input_tensor, target_tensor)
    assert loss.shape == torch.Size([10])



