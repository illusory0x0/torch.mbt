import torch
from torchvision import datasets, transforms
import os

# 创建保存目录
os.makedirs('samples', exist_ok=True)

# 定义数据转换（与训练时一致）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载原始测试集（无转换）
raw_dataset = datasets.MNIST(
    root='../data',
    train=False,
    download=True,
    transform=None  # 禁用转换以获取原始图像
)

# 加载转换后的测试集
transformed_dataset = datasets.MNIST(
    root='../data',
    train=False,
    download=True,
    transform=transform
)

# 处理前 5 个样本
for idx in range(5):
    # 获取原始图像和标签
    raw_img, label = raw_dataset[idx]
    
    # 保存 PNG 文件
    png_path = f'samples/mnist_{idx+1}.png'
    raw_img.save(png_path)
    
    # 获取转换后的张量
    transformed_tensor, _ = transformed_dataset[idx]
    
    # 保存 PT 文件
    pt_path = f'samples/mnist_{idx+1}.pt'
    torch.save(transformed_tensor, pt_path)
    
    print(f"Saved: {png_path} ({raw_img.size}) | {pt_path} {transformed_tensor.shape}")
