import torch
import cv2
import numpy as np


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):

        self.gradients = None
        self.activations = None

        output = self.model(input_tensor)

        class_idx = torch.argmax(output)

        score = output[:, class_idx]

        self.model.zero_grad()

        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return None

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2,3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam / (cam.max() + 1e-8)

        cam = cv2.resize(cam,(224,224))

        return cam