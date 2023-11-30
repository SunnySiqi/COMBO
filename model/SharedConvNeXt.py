from copy import deepcopy

import numpy as np
from timm import create_model
import torch
from torch import nn
import torch.nn.functional as F


# model here: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L410
# lr=4.0e-3 (mentioned in  A ConvNet for the 2020s paper)


class SharedConvNeXt(nn.Module):
    def __init__(self):
        # pretrained_model_name "convnext_tiny.fb_in22k"
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420

        super().__init__()
        model = create_model("convnext_tiny.fb_in22k", pretrained=True)

        self.first_layer = self._get_first_layer(model, 5)

        ## Store reference to sel.first_layer for later access
        self.adaptive_interface = nn.ModuleList([self.first_layer])

        ## shared feature_extractor
        self.feature_extractor = nn.Sequential(
            model.stem[1],
            model.stages[0],
            model.stages[1],
            model.stages[2].downsample,
            *[model.stages[2].blocks[i] for i in range(9)],
            model.stages[3].downsample,
            *[model.stages[3].blocks[i] for i in range(3)],
        )

        ## Loss
        num_proxies = 4
        self.dim = 768
        self.proxies = torch.nn.Parameter((torch.randn(num_proxies, self.dim) / 8))
        init_temperature = 0.11111  # scale = sqrt(1/T)
        self.scale = np.sqrt(1.0 / init_temperature)

    def _get_first_layer(self, model, new_in_dim):
        out_dim, original_in_dim, kh, kw = model.stem[0].weight.shape
        new_shape = (out_dim, new_in_dim, kh, kw)
        layer_1 = nn.Parameter(torch.zeros(new_shape))
        nn.init.kaiming_normal_(layer_1, mode="fan_out", nonlinearity="relu")
        conv1 = deepcopy(model.stem[0])
        conv1.weight = layer_1
        return conv1

    def _reset_params(self, model):
        for m in model.children():
            if len(list(m.children())) > 0:
                self._reset_params(m)

            elif isinstance(m, nn.Conv2d):
                print("resetting", m)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                print("resetting", m)

            elif isinstance(m, nn.Linear):
                print("resetting", m)

                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            else:
                print("skipped", m)

    def _init_bias(self, model):
        ## Init bias of the first layer
        if model.stem[0].bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.stem[0].weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(model.stem[0].bias, -bound, bound)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.feature_extractor(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        y = x.view(x.size(0), -1)
        out = F.linear(y, self.proxies)
        return y, out
