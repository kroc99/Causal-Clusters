import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    CNN for image classification (lung-cancer images), same as before.
    """
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.activations = {}

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        flattened_size = 128 * 28 * 28  # for input 224x224

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Register forward hooks
        self.conv1.register_forward_hook(self.save_activation('conv1'))
        self.conv2.register_forward_hook(self.save_activation('conv2'))
        self.conv3.register_forward_hook(self.save_activation('conv3'))
        self.fc1.register_forward_hook(self.save_activation('fc1'))

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def forward(self, x, return_embedding=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        embedding = F.relu(self.fc2(x))
        logits = self.fc3(embedding)

        if return_embedding:
            return embedding
        else:
            return logits


class SimpleMLP(nn.Module):
    """
    MLP for tabular data (Insurance, etc.), with forward hooks for latent-space analysis.
    - input_dim: e.g., 6 for [age, sex, bmi, children, smoker, region]
    - output_dim: number of classes (for classification) or 1 (for regression).
    """
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=3):
        super(SimpleMLP, self).__init__()
        self.activations = {}

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Register forward hooks for each layer
        self.fc1.register_forward_hook(self.save_activation('fc1'))
        self.fc2.register_forward_hook(self.save_activation('fc2'))
        # Optionally hook fc3 if you want to track final layer as well
        self.fc3.register_forward_hook(self.save_activation('fc3'))

    def save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def forward(self, x, return_embedding=False):
        # x shape: [batch_size, input_dim]
        x = F.relu(self.fc1(x))
        embedding = F.relu(self.fc2(x))  # treat this as "embedding"
        logits = self.fc3(embedding)

        if return_embedding:
            return embedding
        else:
            return logits
