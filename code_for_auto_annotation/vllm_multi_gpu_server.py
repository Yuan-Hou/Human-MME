#!/usr/bin/env python3
"""
vLLM多GPU部署脚本 - Qwen2.5-VL-72B-Instruct-AWQ
支持手动指定GPU卡并提供OpenAI格式的API接口
"""

import argparse
import os
import subprocess
import sys
from typing import List, Optional

def validate_gpu_ids(gpu_ids: List[int], total_gpus: int = 8) -> bool:
    """验证GPU ID是否有效"""
    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= total_gpus:
            return False
    return True

def get_gpu_memory_info():
    """获取GPU显存信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.free', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, check=True)
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            gpu_info.append({
                'index': int(parts[0]),
                'total_memory': int(parts[1]),
                'free_memory': int(parts[2])
            })
        return gpu_info
    except subprocess.CalledProcessError:
        print("❌ 无法获取GPU信息，请确保nvidia-smi可用")
        return []

def estimate_gpu_requirements(model_name: str, num_gpus: int, use_tensor_parallel: bool = False) -> int:
    """估算模型所需显存 (GB)"""
    # Qwen2.5-VL-72B-Instruct-AWQ 大约需要 45-50GB 显存
    base_memory_gb = 50
    # 考虑KV cache和其他开销
    overhead_gb = 10
    
    if use_tensor_parallel:
        # 张量并行模式下，模型被分割到多个GPU上
        total_per_gpu = (base_memory_gb + overhead_gb) // num_gpus + 5
    else:
        # 数据并行模式下，每个GPU都需要加载完整模型
        total_per_gpu = base_memory_gb + overhead_gb
    
    return total_per_gpu

def build_vllm_command(
    model_name: str,
    gpu_ids: List[int],
    port: int = 2334,
    host: str = "0.0.0.0",
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 64000,
    max_num_seqs: int = 16,
    api_key: Optional[str] = None,
    additional_args: List[str] = None,
    use_tensor_parallel: bool = False
) -> List[str]:
    """构建vLLM启动命令"""
    
    # 设置CUDA_VISIBLE_DEVICES环境变量
    gpu_ids_str = ",".join(map(str, gpu_ids))
    
    cmd = [
        "vllm", "serve", model_name,
        "--port", str(port),
        "--host", host,
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--trust-remote-code",
        "--disable-log-stats",
        "--served-model-name", "qwen2.5-vl-72b"
    ]
    
    # 根据模式选择并行策略
    if use_tensor_parallel:
        # 张量并行模式：模型分割到多个GPU
        tensor_parallel_size = len(gpu_ids)
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    # 数据并行模式不需要额外参数，vLLM会自动处理
    
    # 添加API密钥（如果提供）
    if api_key:
        cmd.extend(["--api-key", api_key])
    
    # 添加额外参数
    if additional_args:
        cmd.extend(additional_args)
    
    return cmd, gpu_ids_str

class VLLMHealthMonitor:
    """vLLM服务健康监控器"""
    
    def __init__(self, host: str = "localhost", port: int = 2334, check_interval: int = 30):
        self.host = host
        self.port = port
        self.check_interval = check_interval
        self.base_url = f"http://{host}:{port}"
        self.is_healthy = False
        self.last_check_time = 0
        self.start_time = time.time()
        self.total_checks = 0
        self.failed_checks = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.monitor_thread = None
        self.stop_monitoring = False
        
    def check_health(self) -> bool:
        """检查服务健康状态"""
        try:
            # 检查健康端点
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                self.is_healthy = True
                self.consecutive_failures = 0
                return True
        except requests.exceptions.RequestException:
            pass
        
        # 如果健康检查失败，尝试检查模型端点
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                self.is_healthy = True
                self.consecutive_failures = 0
                return True
        except requests.exceptions.RequestException:
            pass
        
        # 所有检查都失败
        self.is_healthy = False
        self.failed_checks += 1
        self.consecutive_failures += 1
        return False
    
    def get_status(self) -> dict:
        """获取详细状态信息"""
        uptime = time.time() - self.start_time
        success_rate = ((self.total_checks - self.failed_checks) / self.total_checks * 100) if self.total_checks > 0 else 0
        
        status = {
            "is_healthy": self.is_healthy,
            "uptime_seconds": round(uptime, 1),
            "uptime_minutes": round(uptime / 60, 1),
            "total_checks": self.total_checks,
            "failed_checks": self.failed_checks,
            "consecutive_failures": self.consecutive_failures,
            "success_rate_percent": round(success_rate, 1),
            "last_check_time": self.last_check_time,
            "check_interval": self.check_interval,
            "base_url": self.base_url,
            "max_consecutive_failures": self.max_consecutive_failures,
            "needs_attention": self.consecutive_failures >= self.max_consecutive_failures
        }
        
        return status
    
    def start_monitoring(self):
        """启动监控线程"""
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"🔍 健康监控已启动 (检查间隔: {self.check_interval}秒)")
    
    def stop_monitoring_service(self):
        """停止监控服务"""
        self.stop_monitoring = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("🛑 健康监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        print(f"🔍 开始监控vLLM服务健康状态...")
        
        # 等待服务启动
        print("⏳ 等待vLLM服务启动...")
        startup_timeout = 300  # 5分钟启动超时
        startup_start = time.time()
        
        while time.time() - startup_start < startup_timeout:
            if self.stop_monitoring:
                return
            
            if self.check_health():
                print("✅ vLLM服务已启动并可用!")
                break
            
            print("⏳ 等待服务启动...")
            time.sleep(10)
        else:
            print("❌ vLLM服务启动超时")
            return
        
        # 正常监控循环
        while not self.stop_monitoring:
            self.total_checks += 1
            self.last_check_time = time.time()
            
            is_healthy = self.check_health()
            status = self.get_status()
            
            if is_healthy:
                if self.total_checks % 12 == 0:  # 每6分钟显示一次状态 (30s * 12)
                    print(f"💚 服务健康 (运行{status['uptime_minutes']:.1f}分钟, "
                          f"成功率{status['success_rate_percent']:.1f}%)")
            else:
                print(f"🔴 服务不健康 (连续失败{self.consecutive_failures}次)")
                
                if self.consecutive_failures >= self.max_consecutive_failures:
                    print(f"❌ 服务连续失败{self.consecutive_failures}次，需要关注!")
            
            time.sleep(self.check_interval)

def create_health_check_file(port: int):
    """创建健康检查脚本文件"""
    health_script_content = f'''#!/bin/bash
# vLLM健康检查脚本
# 用法: ./health_check.sh

HOST="localhost"
PORT="{port}"
BASE_URL="http://$HOST:$PORT"

echo "🔍 检查vLLM服务健康状态..."
echo "📍 服务地址: $BASE_URL"
echo "=" * 50

# 检查健康端点
echo "🩺 检查健康端点..."
if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    echo "✅ 健康端点响应正常"
    HEALTH_OK=1
else
    echo "❌ 健康端点无响应"
    HEALTH_OK=0
fi

# 检查模型端点
echo "📋 检查模型端点..."
if curl -s "$BASE_URL/v1/models" > /dev/null 2>&1; then
    echo "✅ 模型端点响应正常"
    MODEL_OK=1
else
    echo "❌ 模型端点无响应"
    MODEL_OK=0
fi

# 获取模型列表
echo "📝 获取模型列表..."
MODELS=$(curl -s "$BASE_URL/v1/models" | python3 -m json.tool 2>/dev/null || echo "无法解析响应")
echo "$MODELS"

echo "=" * 50

if [ $HEALTH_OK -eq 1 ] || [ $MODEL_OK -eq 1 ]; then
    echo "✅ vLLM服务运行正常"
    exit 0
else
    echo "❌ vLLM服务出现问题"
    exit 1
fi
'''
    
    health_script_path = "health_check.sh"
    with open(health_script_path, 'w') as f:
        f.write(health_script_content)
    
    # 给脚本添加执行权限
    os.chmod(health_script_path, 0o755)
    print(f"📋 已创建健康检查脚本: {health_script_path}")

def setup_signal_handlers(monitor: Optional[VLLMHealthMonitor] = None, process: Optional[subprocess.Popen] = None):
    """设置信号处理器"""
    def signal_handler(signum, frame):
        print(f"\n🛑 收到信号 {signum}，正在关闭服务...")
        
        if monitor:
            monitor.stop_monitoring_service()
        
        if process:
            try:
                print("🔄 正在终止vLLM进程...")
                process.terminate()
                process.wait(timeout=30)
                print("✅ vLLM进程已正常关闭")
            except subprocess.TimeoutExpired:
                print("⚠️  强制终止vLLM进程...")
                process.kill()
                process.wait()
                print("✅ vLLM进程已强制关闭")
        
        print("👋 服务已关闭，再见!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    parser = argparse.ArgumentParser(description="vLLM多GPU部署脚本")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct-AWQ", 
                       help="模型名称或本地路径 (默认: Qwen/Qwen2.5-VL-72B-Instruct-AWQ)")
    parser.add_argument("--model-path", type=str, 
                       help="模型本地路径 (如果指定，将覆盖--model参数)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", 
                       help="使用的GPU ID，用逗号分隔 (例如: 0,1,2,3)")
    parser.add_argument("--port", type=int, default=2334, 
                       help="API服务端口 (默认: 2334)")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="绑定主机地址 (默认: 0.0.0.0)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, 
                       help="GPU显存利用率 (默认: 0.8)")
    parser.add_argument("--max-model-len", type=int, default=64000, 
                       help="最大模型长度 (默认: 64000)")
    parser.add_argument("--max-num-seqs", type=int, default=512, 
                       help="最大并发序列数 (默认: 16)")
    parser.add_argument("--api-key", type=str, 
                       help="API密钥 (可选)")
    parser.add_argument("--use-tensor-parallel", action="store_true",
                       help="使用张量并行模式而不是数据并行模式")
    parser.add_argument("--check-memory", action="store_true", 
                       help="检查GPU显存信息")
    parser.add_argument("--dry-run", action="store_true", 
                       help="仅显示命令，不实际执行")
    parser.add_argument("--additional-args", nargs="*", 
                       help="传递给vLLM的额外参数")
    
    args = parser.parse_args()
    
    # 确定使用的模型路径
    model_path = args.model_path if args.model_path else args.model
    
    # 检查本地路径是否存在
    if args.model_path and not os.path.exists(args.model_path):
        print(f"❌ 指定的模型路径不存在: {args.model_path}")
        sys.exit(1)
    
    # 检查GPU显存信息
    if args.check_memory:
        print("🔍 GPU显存信息:")
        gpu_info = get_gpu_memory_info()
        for gpu in gpu_info:
            print(f"  GPU {gpu['index']}: {gpu['free_memory']}/{gpu['total_memory']} MB 可用")
        return
    
    # 解析GPU ID
    try:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    except ValueError:
        print("❌ GPU ID格式错误，请使用逗号分隔的数字")
        sys.exit(1)
    
    # 验证GPU ID
    if not validate_gpu_ids(gpu_ids):
        print("❌ 无效的GPU ID")
        sys.exit(1)
    
    # 检查显存需求
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        required_memory = estimate_gpu_requirements(model_path, len(gpu_ids), args.use_tensor_parallel)
        parallel_mode = "张量并行" if args.use_tensor_parallel else "数据并行"
        print(f"📊 预估每个GPU需要显存: ~{required_memory}GB ({parallel_mode}模式)")
        
        for gpu_id in gpu_ids:
            if gpu_id < len(gpu_info):
                free_gb = gpu_info[gpu_id]['free_memory'] / 1024
                if free_gb < required_memory:
                    print(f"⚠️  GPU {gpu_id} 可用显存({free_gb:.1f}GB)可能不足")
    
    # 构建命令
    cmd, gpu_ids_str = build_vllm_command(
        model_name=model_path,
        gpu_ids=gpu_ids,
        port=args.port,
        host=args.host,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        api_key=args.api_key,
        additional_args=args.additional_args,
        use_tensor_parallel=args.use_tensor_parallel
    )
    
    print(f"🚀 准备启动vLLM服务")
    print(f"📋 模型: {model_path}")
    print(f"🎯 使用GPU: {gpu_ids_str}")
    print(f"🌐 服务地址: http://{args.host}:{args.port}")
    
    if args.use_tensor_parallel:
        print(f"⚡ 张量并行度: {len(gpu_ids)} (模型分割到多个GPU)")
    else:
        print(f"⚡ 数据并行度: {len(gpu_ids)} (每个GPU运行完整模型)")
    
    # 如果是本地路径，显示额外信息
    if os.path.exists(model_path):
        print(f"📁 本地模型路径: {os.path.abspath(model_path)}")
        # 检查模型文件
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            print(f"✅ 找到配置文件: config.json")
        else:
            print(f"⚠️  未找到config.json，请确认模型文件完整")
    
    print()
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
    
    # 显示完整命令
    print("💻 执行命令:")
    print(f"CUDA_VISIBLE_DEVICES={gpu_ids_str} {' '.join(cmd)}")
    print()
    
    if args.dry_run:
        print("🔍 (dry-run模式，未实际执行)")
        return
    
    try:
        # 启动vLLM服务
        print("🔄 正在启动vLLM服务...")
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n⛔ 用户中断服务")
    except subprocess.CalledProcessError as e:
        print(f"❌ vLLM启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
