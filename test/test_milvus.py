from pymilvus import connections, utility

# 这里填你的 K3s 机器 IP 和刚才查到的 31902 端口
milvus_host = '192.168.1.174'
milvus_port = '31902'

try:
    print(f"正在连接 Milvus ({milvus_host}:{milvus_port})...")
    # 建立连接
    connections.connect(alias="default", host=milvus_host, port=milvus_port)

    print("✅ 连接成功！")
    print(f"Milvus 版本: {utility.get_server_version()}")


except Exception as e:
    print(f"❌ 连接失败: {e}")