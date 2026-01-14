# 测试显卡是否到导入的脚本
print("1. 正在开始运行脚本...")

try:
    print("2. 正在尝试导入 torch (如果卡在这里说明 PyTorch 安装有问题)...")
    import torch
    print(f"3. torch 导入成功！版本号: {torch.__version__}")
except ImportError:
    print("❌ 错误: 你没有安装 PyTorch！请运行 pip install torch")
    exit()
except Exception as e:
    print(f"❌ 发生未知错误: {e}")
    exit()

def check_gpu_torch():
    print("4. 正在检查显卡状态...")
    if torch.cuda.is_available():
        print(f"✅ 检测到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ 未检测到 GPU (程序使用 CPU 运行)")

if __name__ == '__main__':
    # 这一行开始执行函数
    check_gpu_torch()
    print("5. 脚本运行结束。")