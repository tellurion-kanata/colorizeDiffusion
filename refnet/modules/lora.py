import torch.nn as nn
import torch.optim as optim

from sgm.util import exists


def torch_dfs(model: nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class LoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, dropout_p=0.):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )

        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = 1.0
        self.use_lora = True

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        if self.use_lora:
            return (
                self.linear(input) + self.lora_up(self.dropout(self.lora_down(input))) * self.scale
            )
        else:
            return self.linear(input)

    def init_linear_weight(self, weight, bias=None):
        self.linear.weight = weight
        if exists(self.linear.bias):
            assert exists(bias), "Injected linear layer includes bias"
            self.linear.bias = bias

    def get_trainable_params(self):
        return [self.lora_up.weight, self.lora_down.weight]


class RefLoraModules(nn.Module):
    def inject_lora_modules(self):
        if self.is_xl:
            from sgm.modules.attention import BasicTransformerBlock
        else:
            from ldm.modules.attention import BasicTransformerBlock

        attn_modules = []
        modules = torch_dfs(self.model.diffusion_model) if not self.controlled else \
            torch_dfs(self.model.diffusion_model) + torch_dfs(self.control_model)

        for module in modules:
            if isinstance(module, BasicTransformerBlock):
                attn_modules.append(module.attn2)

        for attn_module in attn_modules:
            for name, layer in attn_module.named_modules():
                if isinstance(layer, nn.Linear):
                    _tmp_module = LoraInjectedLinear(
                        in_features=layer.weight.shape[1],
                        out_features=layer.weight.shape[0],
                        r=self.r
                    )
                    _tmp_module.init_linear_weight(layer.weight)
                    attn_module._modules[name] = _tmp_module

    def get_lora_weights(self):
        modules = torch_dfs(self.model.diffusion_model) if not self.controlled else \
            torch_dfs(self.model.diffusion_model) + torch_dfs(self.control_model)

        require_grad_params = []
        for module in modules:
            if isinstance(module, LoraInjectedLinear):
                require_grad_params += module.get_trainable_params()
        return require_grad_params

    def configure_optimizers(self):
        lora_weights = self.get_lora_weights()
        opt = optim.AdamW(lr=self.lr, params=lora_weights)
        return opt

    def switch_lora_modules(self, v):
        modules = torch_dfs(self.model.diffusion_model) if not self.controlled else \
            torch_dfs(self.model.diffusion_model) + torch_dfs(self.control_model)

        for module in modules:
            if isinstance(module, LoraInjectedLinear):
                module.use_lora = v

    def activate_lora_weights(self):
         self.switch_lora_modules(True)

    def deactivate_lora_weights(self):
        self.switch_lora_modules(False)