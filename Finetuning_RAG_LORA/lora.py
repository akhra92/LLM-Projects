import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        # Freeze the original pre-trained weights
        self.original = nn.Linear(in_features, out_features, bias=False)
        self.original.weight.requires_grad = False
        print(f"Original Parameters: {self.original.weight.numel():,}")

        # Low-rank adaptation matrices
        self.A = nn.Parameter(torch.rand(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.rand(rank, out_features))
        print(f"LoRA Parameters: {(self.A.numel() + self.B.numel()):,}")

    def forward(self, x):
        """Combine frozen base model output with LoRA adaptation"""
        original = self.original(x)
        lora = x @ self.A @ self.B

        return original + lora
    


class QLoRA(LoRA):
    def __init__(self, in_features, out_features, rank=4, bits=4):
        super().__init__(in_features, out_features, rank)
        self.bits = bits

        # Quantize and store weights in int8
        quantized_weight, scale, zero_point = self.quantize(self.original.weight.data)

        # Delete the original float32 weight and store quantized version
        del self.original.weight
        self.original.register_buffer('quantized_weight', quantized_weight)
        self.original.register_buffer('scale', scale)
        self.original.register_buffer('zero_point', zero_point)

    def quantize(self, weight):
        """Quantize weights to int8 and store scale/zero_point for dequantization"""
        max_val = weight.max()
        min_val = weight.min()

        # Calculate scale and zero point for int8 quantization
        scale = (max_val - min_val) / 255
        zero_point = -torch.round(min_val / scale).to(torch.int8)

        # Quantize to int8
        quantized = torch.round(weight / scale + zero_point.float()).clamp(-128, 127).to(torch.int8)

        return quantized, scale, zero_point

    def forward(self, x):
        # Dequantize weights on-the-fly
        dequantized_weight = (self.original.quantized_weight.float() - self.original.zero_point.float()) * self.original.scale

        # Manual linear operation since we replaced the weight parameter
        original = x @ dequantized_weight.t()
        lora = x @ self.A @ self.B

        return original + lora


def get_model_memory(model):
    """Calculate total memory consumption of model parameters in MB"""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    return total_bytes / (1024 ** 2)


if __name__ == "__main__":
    bs, n, m = 3, 1000, 1000
    x = torch.rand(bs, n)

    print("=== LoRA ===")
    lora_model = LoRA(in_features=n, out_features=m, rank=4)
    lora_out = lora_model(x)
    lora_memory = get_model_memory(lora_model)
    print(f"Output shape: {lora_out.shape}")
    print(f"Memory consumption: {lora_memory:.2f} MB\n")

    print("=== QLoRA ===")
    qlora_model = QLoRA(in_features=n, out_features=m, rank=4, bits=4)
    qlora_out = qlora_model(x)
    qlora_memory = get_model_memory(qlora_model)
    print(f"Output shape: {qlora_out.shape}")
    print(f"Memory consumption: {qlora_memory:.2f} MB\n")

    print(f"Memory reduction: {((lora_memory - qlora_memory) / lora_memory * 100):.2f}%")
