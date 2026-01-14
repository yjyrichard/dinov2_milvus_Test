import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

# ================= 配置参数 =================
MODEL_PATH = "Qwen/Qwen3-VL-Embedding-2B"  # 模型ID，第一次运行会自动下载(~4GB)
OUTPUT_DIM = 768  # 你想要的维度！可以改成 512, 1024 等
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 正在使用设备: {DEVICE}")
if DEVICE == "cuda":
    print(f"   显卡型号: {torch.cuda.get_device_name(0)}")

# ================= 1. 加载模型 =================
print("1. 正在加载模型 (第一次下载可能需要几分钟)...")

try:
    # ⚠️ 关键点：970M 必须用 float16，不能用 bfloat16
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 强制使用 FP16
    ).to(DEVICE)
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败，可能是网络问题或显存不足。\n错误信息: {e}")
    exit()

# ================= 2. 准备测试数据 =================
# 下载一张测试图片
img_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
try:
    image = Image.open(BytesIO(requests.get(img_url).content))
except:
    # 如果下载失败，创建一个空白图代替
    image = Image.new('RGB', (224, 224), color='white')
    print("⚠️ 图片下载失败，使用空白图测试")

# 定义输入：一段文本 + 一张图片
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image"},
        ],
    }
]

# ================= 3. 数据预处理 =================
print("2. 正在处理输入数据...")
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages) # 注意：这里可能需要手动处理vision info，视transformer版本而定
# 简化版调用（Qwen3 VL Embedding 的用法可能略有不同，以下是通用 VL 逻辑）
# 由于 Qwen3-VL-Embedding 比较新，通常用法如下：
inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt"
).to(DEVICE)

# ================= 4. 生成向量 (Embedding) =================
print(f"3. 正在生成 {OUTPUT_DIM} 维向量...")

# 开启推理模式，不计算梯度（省显存）
with torch.no_grad():
    # 获取 hidden_states
    # 注意：Qwen3-VL-Embedding 通常取最后一层的 last_token 或者特定的 pooling
    # 这里假设它遵循标准 HuggingFace 接口，或者我们需要调用专门的 embedding 方法
    # 根据文档，它支持 dimension 参数截断
    
    # ⚠️ Qwen3-Embedding 的特殊调用方式（模拟）：
    # 通常 Embedding 模型会输出 last_hidden_state
    outputs = model(**inputs, output_hidden_states=True)
    
    # 取最后一层 hidden state
    last_hidden_state = outputs.hidden_states[-1] 
    
    # 取最后一个 token 的向量作为整个句子的表示 (EOS token pooling)
    # 或者取 mean pooling，Qwen 官方通常推荐 EOS
    embeddings = last_hidden_state[:, -1, :] 

    # 🔥 关键步骤：维度截断/投影
    # Qwen3-VL-Embedding 支持 Matryoshka 截断，直接切片即可！
    embeddings = embeddings[:, :OUTPUT_DIM]
    
    # 归一化 (使得向量长度为1，方便计算余弦相似度)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

# ================= 5. 输出结果 =================
print("-" * 30)
print(f"✅ 生成成功！")
print(f"向量形状: {embeddings.shape}")  # 应该是 [1, 768]
print(f"前 10 位数据: {embeddings[0, :10].cpu().numpy()}")
print("-" * 30)

# 清理显存
del model, inputs, outputs
torch.cuda.empty_cache()