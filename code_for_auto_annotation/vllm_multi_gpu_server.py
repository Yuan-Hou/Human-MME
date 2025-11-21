#!/usr/bin/env python3
"""
vLLM.GPU.... - Qwen2.5-VL-72B-Instruct-AWQ
......GPU....OpenAI...API..
"""

import argparse
import os
import subprocess
import sys
from typing import List, Optional

def validate_gpu_ids(gpu_ids: List[int], total_gpus: int = 8) -> bool:
    """..GPU ID...."""
    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= total_gpus:
            return False
    return True

def get_gpu_memory_info():
    """..GPU...."""
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
        print("❌ ....GPU..，...nvidia-smi..")
        return []

def estimate_gpu_requirements(model_name: str, num_gpus: int, use_tensor_parallel: bool = False) -> int:
    """........ (GB)"""
    # Qwen2.5-VL-72B-Instruct-AWQ .... 45-50GB ..
    base_memory_gb = 50
    # ..KV cache.....
    overhead_gb = 10
    
    if use_tensor_parallel:
        # .......，........GPU.
        total_per_gpu = (base_memory_gb + overhead_gb) // num_gpus + 5
    else:
        # .......，..GPU.........
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
    """..vLLM...."""
    
    # ..CUDA_VISIBLE_DEVICES....
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
    
    # ..........
    if use_tensor_parallel:
        # ......：.......GPU
        tensor_parallel_size = len(gpu_ids)
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    # .............，vLLM.....
    
    # ..API..（....）
    if api_key:
        cmd.extend(["--api-key", api_key])
    
    # ......
    if additional_args:
        cmd.extend(additional_args)
    
    return cmd, gpu_ids_str

class VLLMHealthMonitor:
    """vLLM......."""
    
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
        """........"""
        try:
            # ......
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                self.is_healthy = True
                self.consecutive_failures = 0
                return True
        except requests.exceptions.RequestException:
            pass
        
        # ........，........
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                self.is_healthy = True
                self.consecutive_failures = 0
                return True
        except requests.exceptions.RequestException:
            pass
        
        # .......
        self.is_healthy = False
        self.failed_checks += 1
        self.consecutive_failures += 1
        return False
    
    def get_status(self) -> dict:
        """........"""
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
        """......"""
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"🔍 ....... (....: {self.check_interval}.)")
    
    def stop_monitoring_service(self):
        """......"""
        self.stop_monitoring = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("🛑 .......")
    
    def _monitor_loop(self):
        """...."""
        print(f"🔍 ....vLLM.........")
        
        # ......
        print("⏳ ..vLLM.......")
        startup_timeout = 300  # 5......
        startup_start = time.time()
        
        while time.time() - startup_start < startup_timeout:
            if self.stop_monitoring:
                return
            
            if self.check_health():
                print("✅ vLLM........!")
                break
            
            print("⏳ .........")
            time.sleep(10)
        else:
            print("❌ vLLM......")
            return
        
        # ......
        while not self.stop_monitoring:
            self.total_checks += 1
            self.last_check_time = time.time()
            
            is_healthy = self.check_health()
            status = self.get_status()
            
            if is_healthy:
                if self.total_checks % 12 == 0:  # .6........ (30s * 12)
                    print(f"💚 .... (..{status['uptime_minutes']:.1f}.., "
                          f"...{status['success_rate_percent']:.1f}%)")
            else:
                print(f"🔴 ..... (....{self.consecutive_failures}.)")
                
                if self.consecutive_failures >= self.max_consecutive_failures:
                    print(f"❌ ......{self.consecutive_failures}.，....!")
            
            time.sleep(self.check_interval)

def create_health_check_file(port: int):
    """.........."""
    health_script_content = f'''#!/bin/bash
# vLLM......
# ..: ./health_check.sh

HOST="localhost"
PORT="{port}"
BASE_URL="http://$HOST:$PORT"

echo "🔍 ..vLLM........."
echo "📍 ....: $BASE_URL"
echo "=" * 50

# ......
echo "🩺 ........."
if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    echo "✅ ........"
    HEALTH_OK=1
else
    echo "❌ ......."
    HEALTH_OK=0
fi

# ......
echo "📋 ........."
if curl -s "$BASE_URL/v1/models" > /dev/null 2>&1; then
    echo "✅ ........"
    MODEL_OK=1
else
    echo "❌ ......."
    MODEL_OK=0
fi

# ......
echo "📝 ........."
MODELS=$(curl -s "$BASE_URL/v1/models" | python3 -m json.tool 2>/dev/null || echo "......")
echo "$MODELS"

echo "=" * 50

if [ $HEALTH_OK -eq 1 ] || [ $MODEL_OK -eq 1 ]; then
    echo "✅ vLLM......"
    exit 0
else
    echo "❌ vLLM......"
    exit 1
fi
'''
    
    health_script_path = "health_check.sh"
    with open(health_script_path, 'w') as f:
        f.write(health_script_content)
    
    # .........
    os.chmod(health_script_path, 0o755)
    print(f"📋 .........: {health_script_path}")

def setup_signal_handlers(monitor: Optional[VLLMHealthMonitor] = None, process: Optional[subprocess.Popen] = None):
    """......."""
    def signal_handler(signum, frame):
        print(f"\n🛑 .... {signum}，.........")
        
        if monitor:
            monitor.stop_monitoring_service()
        
        if process:
            try:
                print("🔄 ....vLLM.....")
                process.terminate()
                process.wait(timeout=30)
                print("✅ vLLM.......")
            except subprocess.TimeoutExpired:
                print("⚠️  ....vLLM.....")
                process.kill()
                process.wait()
                print("✅ vLLM.......")
        
        print("👋 .....，..!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    parser = argparse.ArgumentParser(description="vLLM.GPU....")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct-AWQ", 
                       help="......... (..: Qwen/Qwen2.5-VL-72B-Instruct-AWQ)")
    parser.add_argument("--model-path", type=str, 
                       help="...... (....，...--model..)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", 
                       help="...GPU ID，..... (..: 0,1,2,3)")
    parser.add_argument("--port", type=int, default=2334, 
                       help="API.... (..: 2334)")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="...... (..: 0.0.0.0)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, 
                       help="GPU..... (..: 0.8)")
    parser.add_argument("--max-model-len", type=int, default=64000, 
                       help="...... (..: 64000)")
    parser.add_argument("--max-num-seqs", type=int, default=512, 
                       help="....... (..: 16)")
    parser.add_argument("--api-key", type=str, 
                       help="API.. (..)")
    parser.add_argument("--use-tensor-parallel", action="store_true",
                       help=".................")
    parser.add_argument("--check-memory", action="store_true", 
                       help="..GPU....")
    parser.add_argument("--dry-run", action="store_true", 
                       help=".....，.....")
    parser.add_argument("--additional-args", nargs="*", 
                       help="...vLLM.....")
    
    args = parser.parse_args()
    
    # .........
    model_path = args.model_path if args.model_path else args.model
    
    # ..........
    if args.model_path and not os.path.exists(args.model_path):
        print(f"❌ ..........: {args.model_path}")
        sys.exit(1)
    
    # ..GPU....
    if args.check_memory:
        print("🔍 GPU....:")
        gpu_info = get_gpu_memory_info()
        for gpu in gpu_info:
            print(f"  GPU {gpu['index']}: {gpu['free_memory']}/{gpu['total_memory']} MB ..")
        return
    
    # ..GPU ID
    try:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    except ValueError:
        print("❌ GPU ID....，..........")
        sys.exit(1)
    
    # ..GPU ID
    if not validate_gpu_ids(gpu_ids):
        print("❌ ...GPU ID")
        sys.exit(1)
    
    # ......
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        required_memory = estimate_gpu_requirements(model_path, len(gpu_ids), args.use_tensor_parallel)
        parallel_mode = "...." if args.use_tensor_parallel else "...."
        print(f"📊 ....GPU....: ~{required_memory}GB ({parallel_mode}..)")
        
        for gpu_id in gpu_ids:
            if gpu_id < len(gpu_info):
                free_gb = gpu_info[gpu_id]['free_memory'] / 1024
                if free_gb < required_memory:
                    print(f"⚠️  GPU {gpu_id} ....({free_gb:.1f}GB)....")
    
    # ....
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
    
    print(f"🚀 ....vLLM..")
    print(f"📋 ..: {model_path}")
    print(f"🎯 ..GPU: {gpu_ids_str}")
    print(f"🌐 ....: http://{args.host}:{args.port}")
    
    if args.use_tensor_parallel:
        print(f"⚡ .....: {len(gpu_ids)} (.......GPU)")
    else:
        print(f"⚡ .....: {len(gpu_ids)} (..GPU......)")
    
    # .......，......
    if os.path.exists(model_path):
        print(f"📁 ......: {os.path.abspath(model_path)}")
        # ......
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            print(f"✅ ......: config.json")
        else:
            print(f"⚠️  ...config.json，.........")
    
    print()
    
    # ......
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
    
    # ......
    print("💻 ....:")
    print(f"CUDA_VISIBLE_DEVICES={gpu_ids_str} {' '.join(cmd)}")
    print()
    
    if args.dry_run:
        print("🔍 (dry-run..，.....)")
        return
    
    try:
        # ..vLLM..
        print("🔄 ....vLLM.....")
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n⛔ ......")
    except subprocess.CalledProcessError as e:
        print(f"❌ vLLM....: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
