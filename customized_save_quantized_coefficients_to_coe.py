import torch 
import os 

def save_all_params_to_coe(model, save_dir="Weight_Bias_QAT/hybrid_model/"):
    """
    保存量化模型和非量化模型层的参数
    :param model: 模型对象（量化后的或未量化的模型）
    :param save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    for name, module in model.named_children():
        # 对可量化层（如 Linear）进行量化保存
        if isinstance(module, nn.Linear):
            # 获取量化后的权重和偏置
            for param_name, param in module.named_parameters():
                quantized, scale, zero_point = quantize_tensor_non_symmetric(param.data.cpu())

                # 生成文件名
                file_name = f"{name}_{param_name}.coe"
                save_coefficients_to_coe(file_name, quantized, scale, zero_point, save_dir)
                print(f"Saved quantized parameter '{name}.{param_name}' to {file_name}.")
        
        # 对非量化层（如 MultiheadAttention, LayerNorm）进行浮点权重保存
        elif isinstance(module, nn.MultiheadAttention) or isinstance(module, nn.LayerNorm):
            for param_name, param in module.named_parameters():
                # 保存浮点权重
                file_name = f"{name}_{param_name}.coe"
                save_float_coefficients_to_coe(file_name, param.data.cpu(), save_dir)
                print(f"Saved float parameter '{name}.{param_name}' to {file_name}.")
        
        # 递归检查子模块
        save_all_params_to_coe(module, save_dir)

def save_float_coefficients_to_coe(filename, tensor, save_dir):
    """
    保存浮点张量到 .coe 文件（处理负数）
    :param filename: 文件名
    :param tensor: 浮点张量
    :param save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    
    with open(file_path, 'w') as f:
        f.write(f"; float tensor\n")
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        
        # 对负数进行处理：将浮点值乘以1000，转换为整数，并保存为16进制
        hex_values = []
        for value in tensor.flatten():
            int_value = int(value.item() * 1000)  # 放大浮点数，保留三位小数
            
            # 如果值为负数，保持负号
            hex_value = format(int_value & 0xFFFFFFFF, '08X')  # 保证以8个字符16进制显示
            hex_values.append(hex_value)
        
        f.write(", ".join(hex_values) + ";\n")
        
    print(f"Saved float parameter to {file_path}.")


def quantize_tensor_non_symmetric(tensor):
    """
    量化张量（非对称量化），返回量化后的数据、scale 和 zero_point。
    """
    min_val = tensor.min()
    max_val = tensor.max()

    if min_val == max_val:
        scale = 1.0
        zero_point = 0
        quantized = tensor.to(torch.int8)
        return quantized, scale, zero_point
    
    scale = (max_val - min_val) / 255.0  # 量化范围 [0, 255]
    zero_point = int(-min_val / scale)   # 计算零点
    
    quantized = torch.round((tensor - min_val) / scale).clamp(0, 255).to(torch.int8)
    return quantized, scale, zero_point

def save_coefficients_to_coe(filename, quantized_tensor, scale, zero_point, save_dir):
    """
    保存量化的 tensor 到 .coe 文件（16进制）
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    
    with open(file_path, 'w') as f:
        f.write(f"; scale factor: {scale}\n")
        f.write(f"; zero point: {zero_point}\n")
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        
        if quantized_tensor.ndimension() == 1:  # 一维张量 (bias)
            hex_values = [format((value + 128) & 0xFF, '02X') for value in quantized_tensor]
            f.write(", ".join(hex_values) + ";\n")
        elif quantized_tensor.ndimension() == 2:  # 二维张量 (weight)
            for row in quantized_tensor:
                hex_row = [format((value + 128) & 0xFF, '02X') for value in row]
                f.write(", ".join(hex_row) + ";\n")

    print(f"Saved quantized parameter to {file_path}.")


# 训练完成后转换量化模型并保存权重
model.cpu()  # 确保模型在 CPU 上
torch.quantization.convert(model, inplace=True)  # 转换为量化模型
save_all_params_to_coe(model)  # 保存每层参数
