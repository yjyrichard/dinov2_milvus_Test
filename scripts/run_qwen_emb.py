import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

# ================= 配置参数 =================
MODEL_PATH = "Qwen/Qwen3-VL-Embedding-2B"
OUTPUT_DIM = 768 # 你想要的向量维度

# ⚠️ 必须设置 device_map="auto"，让它自动利用内存(RAM)
# 因为 3GB 显存放不下完整的 FP16 模型
print(f"🚀 正在初始化...")

# ================= 1. 加载模型 (按官网截图方式) =================
print("1. 正在加载模型 (自动分配显存/内存)...")

try:
    # 🌟 这里改成了官网推荐的 AutoModelForVision2Seq
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16, # 保持 FP16 精度
        device_map="auto",         # 核心：解决 3GB 显存不足的问题
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    # 如果这里报错 import error，可能需要升级 transformers: pip install --upgrade transformers
    exit()

# ================= 2. 准备数据 =================
img_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
try:
    image = Image.open(BytesIO(requests.get(img_url).content))
except:
    image = Image.new('RGB', (224, 224), color='white')

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image"}, # 对于Embedding模型，这里可以是具体的查询或描述
        ],
    }
]

# ================= 3. 推理生成向量 =================
print("2. 正在计算向量...")

# 预处理
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(
    text=[text],
    images=[image],
    padding=True,
    return_tensors="pt"
).to(model.device) # 自动跟随模型设备

with torch.no_grad():
    # 运行模型
    # AutoModelForVision2Seq 输出的是 BaseModelOutputWithPooling
    outputs = model(**inputs)
    
    # 获取最后一层的隐藏状态 (Batch_Size, Sequence_Length, Hidden_Size)
    last_hidden_state = outputs.last_hidden_state
    
    # 提取向量策略：通常取最后一个 Token (EOS) 的特征
    embeddings = last_hidden_state[:, -1, :] 

    # 🤏 维度截断 (Matryoshka Embedding)
    # Qwen3-VL-Embedding 支持弹性维度，如果你只需要 768 维，直接切片即可
    embeddings = embeddings[:, :OUTPUT_DIM]
    
    # 归一化 (这一步对余弦相似度搜索至关重要)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

# ================= 4. 打印结果 =================
print("-" * 30)
print(f"✅ 生成成功！")
print(f"向量维度: {embeddings.shape}") # 应该是 [1, 768]
print(f"前 10 位数值: {embeddings[0, :10].cpu().numpy()}")
print("-" * 30)

# 显存清理建议
# 如果后面还要跑其他东西，建议把 model del 掉
del model
torch.cuda.empty_cache()