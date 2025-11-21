#!/usr/bin/env python3
"""
vLLM多实例路由器 - 真正的数据并行实现
为每个GPU启动单独的vLLM实例，每个实例运行完整模型
提供统一的API入口和负载均衡
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import List, Dict, Optional, Any
import aiohttp
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue


@dataclass
class QueuedRequest:
    """排队的请求"""
    request_id: str
    future: asyncio.Future
    timestamp: float
    timeout: float = 300.0  # 5分钟超时


class VLLMInstance:
    """单个vLLM实例管理"""
    
    def __init__(self, gpu_id: int, port: int, model: str, model_path: Optional[str] = None,
                 host: str = "0.0.0.0", gpu_memory_util: float = 0.8,
                 max_model_len: int = 32000, max_num_seqs: int = 16):
        self.gpu_id = gpu_id
        self.port = port
        self.model = model
        self.model_path = model_path
        self.host = host
        self.gpu_memory_util = gpu_memory_util
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.process: Optional[subprocess.Popen] = None
        self.is_healthy = False
        self.health_info = "未启动"  # 健康状态详细信息
        self.last_health_check = 0  # 最后健康检查时间
        self.startup_time = 0  # 启动时间
        self.url = f"http://{host}:{port}"
        self.load_score = 0  # 负载评分，用于负载均衡
        self.active_requests = 0  # 当前活跃请求数
        self.total_requests = 0  # 总请求数
        self.max_concurrent_requests = 64  # 最大并发请求数（默认值，会在初始化时覆盖）
        self.unhealthy_since = 0  # 开始不健康的时间戳
        self.restart_count = 0  # 重启次数
        
    def get_command(self) -> List[str]:
        """构建启动命令"""
        model_arg = self.model_path if self.model_path else self.model
        
        cmd = [
            "python3", "vllm_multi_gpu_server.py",
            "--model", model_arg,
            "--gpus", str(self.gpu_id),
            "--port", str(self.port),
            "--host", self.host,
            "--gpu-memory-utilization", str(self.gpu_memory_util),
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", str(self.max_num_seqs)
        ]
        
        if self.model_path:
            cmd.extend(["--model-path", self.model_path])
            
        return cmd
    
    def start(self):
        """启动vLLM实例"""
        cmd = self.get_command()
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        print(f"🚀 启动GPU {self.gpu_id}实例 (端口{self.port})...")
        print(f"   命令: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # 创建新的进程组
            )
            
            self.startup_time = time.time()
            self.health_info = "正在启动..."
            print(f"✅ GPU {self.gpu_id}实例启动完成 (PID: {self.process.pid})")
            
        except Exception as e:
            self.health_info = f"启动失败: {str(e)}"
            print(f"❌ GPU {self.gpu_id}实例启动失败: {e}")
            raise
    
    async def check_health(self) -> bool:
        """检查实例健康状态"""
        if not self.process or self.process.poll() is not None:
            new_healthy = False
            if not self.process:
                self.health_info = "进程未启动"
            else:
                exit_code = self.process.poll()
                self.health_info = f"进程已退出 (退出码: {exit_code})"
        else:
            try:
                # 注意：这里需要从router获取health_session，所以我们需要传递router引用
                # 暂时保持原有逻辑，在MultiInstanceRouter中统一管理健康检查
                new_healthy = True  # 临时设为True，实际检查在router中进行
                self.health_info = "等待健康检查"
            except Exception as e:
                new_healthy = False
                self.health_info = f"健康检查异常: {str(e)[:100]}"
        
        # 跟踪不健康状态的时间
        current_time = time.time()
        if not new_healthy:
            # 如果之前是健康的，现在不健康了，记录开始时间
            if self.is_healthy:
                self.unhealthy_since = current_time
                print(f"⚠️  GPU {self.gpu_id} 开始不健康: {self.health_info}")
        else:
            # 如果现在是健康的，重置不健康时间
            if self.unhealthy_since > 0:
                unhealthy_duration = current_time - self.unhealthy_since
                print(f"✅ GPU {self.gpu_id} 恢复健康，不健康持续了 {unhealthy_duration:.1f}秒")
            self.unhealthy_since = 0
        
        self.is_healthy = new_healthy
        return self.is_healthy
    
    def can_accept_request(self) -> bool:
        """检查是否可以接受新请求"""
        return self.is_healthy and self.active_requests < self.max_concurrent_requests
    
    def start_request(self) -> bool:
        """开始一个新请求，返回是否成功"""
        if self.can_accept_request():
            self.active_requests += 1
            self.total_requests += 1
            self.load_score = self.active_requests / self.max_concurrent_requests
            return True
        return False
    
    def end_request(self):
        """结束一个请求"""
        if self.active_requests > 0:
            self.active_requests -= 1
            self.load_score = self.active_requests / self.max_concurrent_requests
    
    def get_load_info(self) -> Dict[str, Any]:
        """获取负载信息"""
        return {
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "max_concurrent_requests": self.max_concurrent_requests,
            "load_score": self.load_score,
            "utilization_percent": round(self.load_score * 100, 1),
            "can_accept_request": self.can_accept_request()
        }
    
    def get_unhealthy_duration(self) -> float:
        """获取不健康持续时间（秒）"""
        if self.unhealthy_since > 0:
            return time.time() - self.unhealthy_since
        return 0
    
    def needs_restart(self, max_unhealthy_time: float = 120.0) -> bool:
        """检查是否需要重启（不健康超过指定时间）"""
        return not self.is_healthy and self.get_unhealthy_duration() > max_unhealthy_time
    
    def restart(self):
        """重启实例"""
        print(f"🔄 重启GPU {self.gpu_id}实例...")
        
        # 停止现有进程
        self.stop()
        
        # 等待一段时间确保资源释放
        time.sleep(5)
        
        # 增加重启计数并重置连接状态
        self.restart_count += 1
        self.reset_connection_state()
        self.health_info = "正在重启..."
        
        # 重新启动
        self.start()
        
        print(f"✅ GPU {self.gpu_id}实例重启完成 (第{self.restart_count}次重启)")
    
    def reset_connection_state(self):
        """重置连接状态 - 用于重启或故障后的状态清理"""
        self.is_healthy = False
        self.health_info = "状态重置"
        self.active_requests = 0
        self.load_score = 0
        self.last_health_check = 0
        self.unhealthy_since = 0
        print(f"🔄 GPU {self.gpu_id} 连接状态已重置")
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """获取详细状态信息"""
        status = {
            "gpu_id": self.gpu_id,
            "port": self.port,
            "url": self.url,
            "is_healthy": self.is_healthy,
            "health_info": self.health_info,
            "last_health_check": self.last_health_check,
            "uptime": 0,
            "process_info": {},
            "load_info": self.get_load_info(),
            "restart_info": {
                "restart_count": self.restart_count,
                "unhealthy_duration": self.get_unhealthy_duration(),
                "needs_restart": self.needs_restart()
            }
        }
        
        if self.process:
            status["process_info"] = {
                "pid": self.process.pid,
                "is_running": self.process.poll() is None,
                "exit_code": self.process.poll()
            }
            
            if self.startup_time > 0:
                status["uptime"] = time.time() - self.startup_time
                
        return status
    
    def stop(self):
        """停止vLLM实例"""
        if self.process:
            try:
                # 发送SIGTERM信号给进程组
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                # 等待进程结束
                self.process.wait(timeout=30)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                # 强制杀死
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process = None
        
        # 重置所有连接状态
        self.is_healthy = False
        self.health_info = "已停止"
        self.active_requests = 0  # 清零活跃请求
        self.load_score = 0  # 重置负载评分
        self.last_health_check = 0  # 重置健康检查时间
        self.unhealthy_since = 0  # 重置不健康开始时间
        self.startup_time = 0  # 重置启动时间


class MultiInstanceRouter:
    """多实例路由器"""
    
    def __init__(self, model: str, gpus: List[int], base_port: int, 
                 model_path: Optional[str] = None, host: str = "0.0.0.0",
                 gpu_memory_util: float = 0.8, max_model_len: int = 32000,
                 max_num_seqs_per_instance: int = 16, max_queue_size: int = 1000,
                 max_wait_time: float = 300.0, max_request_size: int = 50*1024*1024,
                 max_concurrent_requests_per_instance: Optional[int] = None,
                 max_connections_per_host: int = 200):
        self.model = model
        self.model_path = model_path
        self.host = host
        self.base_port = base_port
        self.gpu_memory_util = gpu_memory_util
        self.max_model_len = max_model_len
        self.max_num_seqs_per_instance = max_num_seqs_per_instance
        
        # 如果没有指定负载均衡并发数，使用更高的默认值
        if max_concurrent_requests_per_instance is None:
            max_concurrent_requests_per_instance = max(max_num_seqs_per_instance * 4, 64)  # 至少64个并发
        
        # 为每个GPU创建实例，使用不同的端口
        self.instances = []
        for i, gpu_id in enumerate(gpus):
            port = base_port + i + 1  # 主端口留给路由器
            instance = VLLMInstance(
                gpu_id=gpu_id,
                port=port,
                model=model,
                model_path=model_path,
                host=host,
                gpu_memory_util=gpu_memory_util,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs_per_instance
            )
            # 设置负载均衡的最大并发请求数
            instance.max_concurrent_requests = max_concurrent_requests_per_instance
            self.instances.append(instance)
        
        # 存储配置参数
        self.max_queue_size = max_queue_size
        self.max_wait_time = max_wait_time
        self.max_request_size = max_request_size
        self.max_connections_per_host = max_connections_per_host
        
        # 增加请求体大小限制，支持4K图片等大文件
        self.app = web.Application(client_max_size=max_request_size)
        self.setup_routes()
        
        # 创建高并发的ClientSession用于转发请求，优化连接管理防止文件描述符泄露
        connector = TCPConnector(
            limit=1000,  # 设置总连接数限制，防止无限制连接
            limit_per_host=min(self.max_connections_per_host, 100),  # 限制每个主机的连接数
            enable_cleanup_closed=True,
            force_close=True,  # 强制关闭连接，确保资源释放
            ttl_dns_cache=300,  # DNS缓存5分钟
            use_dns_cache=True,
        )
        timeout = ClientTimeout(total=600)  # 10分钟超时
        self.client_session = ClientSession(
            connector=connector,
            timeout=timeout,
            connector_owner=True,  # 确保session拥有connector，在关闭时一起清理
            auto_decompress=False  # 禁用自动解压缩以减少内存使用
        )
        
        # 创建专用的健康检查session，避免每次创建新session
        health_connector = TCPConnector(
            limit=50,  # 健康检查用较少连接
            limit_per_host=10,
            enable_cleanup_closed=True,
            keepalive_timeout=10,  # 健康检查连接保活时间更短
        )
        health_timeout = ClientTimeout(total=10)  # 健康检查10秒超时
        self.health_session = ClientSession(
            connector=health_connector,
            timeout=health_timeout,
            connector_owner=True
        )
        
        # 健康检查任务
        self.health_check_task = None
        
        # 请求排队机制
        self.request_queue = asyncio.Queue()
        self.queue_processor_task = None
        
    def setup_routes(self):
        """设置路由"""
        # OpenAI API兼容路由
        self.app.router.add_post('/v1/chat/completions', self.chat_completions)
        self.app.router.add_post('/v1/completions', self.completions)
        self.app.router.add_get('/v1/models', self.list_models)
        
        # 健康检查和状态
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/status', self.status)
        self.app.router.add_get('/diagnosis', self.diagnosis)
        self.app.router.add_get('/metrics', self.metrics)
    
    def start_instances(self):
        """启动所有vLLM实例"""
        print(f"🔄 启动{len(self.instances)}个vLLM实例...")
        
        # 使用线程池并行启动实例
        with ThreadPoolExecutor(max_workers=len(self.instances)) as executor:
            futures = []
            for instance in self.instances:
                future = executor.submit(instance.start)
                futures.append(future)
            
            # 等待所有实例启动完成
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ 实例启动失败: {e}")
                    self.stop_all_instances()
                    raise
        
        print("✅ 所有实例启动完成")
    
    async def queue_processor(self):
        """处理排队的请求"""
        print("🚀 启动请求队列处理器...")
        
        while True:
            try:
                # 从队列获取请求
                queued_request = await self.request_queue.get()
                
                # 检查请求是否超时
                current_time = time.time()
                if current_time - queued_request.timestamp > queued_request.timeout:
                    print(f"⏰ [请求{queued_request.request_id}] 排队超时，丢弃请求")
                    if not queued_request.future.done():
                        queued_request.future.set_exception(
                            web.HTTPRequestTimeout(text="请求排队超时")
                        )
                    continue
                
                # 等待有可用的实例
                instance = None
                wait_start = time.time()
                while instance is None:
                    # 检查是否有可以接受请求的实例
                    available_instances = [inst for inst in self.instances if inst.can_accept_request()]
                    if available_instances:
                        # 按负载评分排序，选择负载最低的实例
                        available_instances.sort(key=lambda x: x.load_score)
                        instance = available_instances[0]
                        break
                    
                    # 检查等待时间是否超时
                    if time.time() - wait_start > queued_request.timeout:
                        print(f"⏰ [请求{queued_request.request_id}] 等待实例超时")
                        if not queued_request.future.done():
                            queued_request.future.set_exception(
                                web.HTTPRequestTimeout(text="等待可用实例超时")
                            )
                        break
                    
                    # 短暂等待后重试
                    await asyncio.sleep(0.5)  # 减少等待间隔以提高响应性
                
                if instance and not queued_request.future.done():
                    wait_time = time.time() - queued_request.timestamp
                    load_info = instance.get_load_info()
                    print(f"✅ [请求{queued_request.request_id}] 分配到GPU {instance.gpu_id} "
                          f"(等待{wait_time:.1f}s, 负载: {load_info['active_requests']}/{load_info['max_concurrent_requests']})")
                    queued_request.future.set_result(instance)
                
                self.request_queue.task_done()
                
            except asyncio.CancelledError:
                print("🛑 队列处理器被取消")
                break
            except Exception as e:
                print(f"❌ 队列处理器错误: {e}")
                await asyncio.sleep(1)
    
    async def check_instance_health(self, instance: VLLMInstance) -> bool:
        """使用共享的健康检查session检查实例健康状态"""
        if not instance.process or instance.process.poll() is not None:
            new_healthy = False
            if not instance.process:
                instance.health_info = "进程未启动"
            else:
                exit_code = instance.process.poll()
                instance.health_info = f"进程已退出 (退出码: {exit_code})"
        else:
            try:
                # 使用专用的health_session进行健康检查
                async with self.health_session.get(f"{instance.url}/health") as response:
                    new_healthy = response.status == 200
                    if new_healthy:
                        instance.health_info = "健康"
                    else:
                        instance.health_info = f"HTTP状态码异常: {response.status}"
            except asyncio.TimeoutError:
                new_healthy = False
                instance.health_info = "健康检查超时"
            except aiohttp.ClientConnectorError as e:
                new_healthy = False
                instance.health_info = f"连接失败: {str(e)[:100]}"
            except Exception as e:
                new_healthy = False
                instance.health_info = f"健康检查异常: {str(e)[:100]}"
        
        # 跟踪不健康状态的时间
        current_time = time.time()
        if not new_healthy:
            # 如果之前是健康的，现在不健康了，记录开始时间
            if instance.is_healthy:
                instance.unhealthy_since = current_time
                print(f"⚠️  GPU {instance.gpu_id} 开始不健康: {instance.health_info}")
        else:
            # 如果现在是健康的，重置不健康时间
            if instance.unhealthy_since > 0:
                unhealthy_duration = current_time - instance.unhealthy_since
                print(f"✅ GPU {instance.gpu_id} 恢复健康，不健康持续了 {unhealthy_duration:.1f}秒")
            instance.unhealthy_since = 0
        
        instance.is_healthy = new_healthy
        return instance.is_healthy

    async def wait_for_instances_ready(self, timeout: int = 600):
        """等待所有实例就绪"""
        print("⏳ 等待所有实例就绪...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            ready_count = 0
            for instance in self.instances:
                if await self.check_instance_health(instance):
                    ready_count += 1
            
            print(f"📊 就绪状态: {ready_count}/{len(self.instances)} 个实例")
            
            if ready_count == len(self.instances):
                print("✅ 所有实例已就绪！")
                return True
            
            await asyncio.sleep(10)
        
        print(f"❌ 等待超时 ({timeout}秒)，部分实例未就绪")
        return False
    
    async def get_best_instance(self) -> VLLMInstance:
        """选择最佳实例（基于负载均衡，支持排队等待）"""
        # 获取可以接受新请求的健康实例
        available_instances = [inst for inst in self.instances if inst.can_accept_request()]
        
        if available_instances:
            # 按负载评分排序，选择负载最低的实例
            available_instances.sort(key=lambda x: x.load_score)
            selected = available_instances[0]
            
            # 记录选择的实例
            load_info = selected.get_load_info()
            print(f"   🎯 负载均衡分配: GPU {selected.gpu_id} "
                  f"(活跃: {load_info['active_requests']}/{load_info['max_concurrent_requests']}, "
                  f"利用率: {load_info['utilization_percent']}%)")
            print(f"      💡 可用实例: {len(available_instances)}/{len(self.instances)}")
            
            return selected
        
        # 检查是否有健康但已满载的实例
        healthy_instances = [inst for inst in self.instances if inst.is_healthy]
        if healthy_instances:
            # 显示负载状态
            print(f"   🔄 所有健康实例已满载:")
            for inst in healthy_instances:
                load_info = inst.get_load_info()
                print(f"      GPU {inst.gpu_id}: {load_info['active_requests']}/{load_info['max_concurrent_requests']} "
                      f"({load_info['utilization_percent']}%)")
        
        # 没有可用实例，加入排队
        if self.request_queue.qsize() >= self.max_queue_size:
            print(f"❌ 队列已满 ({self.max_queue_size})，拒绝请求")
            raise web.HTTPServiceUnavailable(text=f"服务队列已满，请稍后再试")
        
        request_id = f"{int(time.time() * 1000) % 100000:05d}"
        queued_request = QueuedRequest(
            request_id=request_id,
            future=asyncio.Future(),
            timestamp=time.time(),
            timeout=self.max_wait_time
        )
        
        await self.request_queue.put(queued_request)
        queue_size = self.request_queue.qsize()
        print(f"📝 [请求{request_id}] 加入排队 (队列长度: {queue_size})")
        
        try:
            # 等待分配实例
            instance = await queued_request.future
            return instance
        except asyncio.TimeoutError:
            print(f"⏰ [请求{request_id}] 排队超时")
            raise web.HTTPRequestTimeout(text="请求排队超时，请稍后再试")
    
    async def forward_request(self, instance: VLLMInstance, path: str, method: str, 
                             data: Optional[bytes] = None, 
                             headers: Optional[Dict] = None) -> web.Response:
        """转发请求到vLLM实例"""
        url = f"{instance.url}{path}"
        
        # 开始请求，更新负载计数
        if not instance.start_request():
            # 如果实例无法接受请求，记录错误但继续尝试
            print(f"⚠️  GPU {instance.gpu_id} 负载已满，但仍然尝试转发请求")
        
        # 记录请求开始时间
        start_time = time.time()
        request_id = f"{int(start_time * 1000) % 100000:05d}"  # 简短的请求ID
        
        load_info = instance.get_load_info()
        print(f"🔄 [请求{request_id}] 转发到GPU {instance.gpu_id} ({method} {path})")
        print(f"   📊 [请求{request_id}] 当前负载: {load_info['active_requests']}/{load_info['max_concurrent_requests']} "
              f"({load_info['utilization_percent']}%)")
        
        # 如果是聊天请求，尝试解析并显示请求内容
        if data and path == '/v1/chat/completions':
            try:
                request_json = json.loads(data.decode('utf-8'))
                messages = request_json.get('messages', [])
                if messages:
                    last_message = messages[-1]
                    content_preview = str(last_message.get('content', ''))[:100]
                    if len(content_preview) == 100:
                        content_preview += "..."
                    print(f"   📝 [请求{request_id}] 用户消息: {content_preview}")
                    print(f"   ⚙️  [请求{request_id}] 请求参数: max_tokens={request_json.get('max_tokens', 'default')}, "
                          f"temperature={request_json.get('temperature', 'default')}")
            except Exception as e:
                print(f"   ⚠️  [请求{request_id}] 无法解析请求内容: {e}")
        
        try:
            # 使用共享的ClientSession而不是每次创建新的
            kwargs = {}
            if data:
                kwargs['data'] = data
                print(f"   📊 [请求{request_id}] 请求数据大小: {len(data)} bytes")
            if headers:
                # 过滤掉hop-by-hop headers
                filtered_headers = {k: v for k, v in headers.items() 
                                  if k.lower() not in ['connection', 'transfer-encoding']}
                kwargs['headers'] = filtered_headers
                print(f"   📋 [请求{request_id}] 请求头数量: {len(filtered_headers)}")
            
            print(f"   🚀 [请求{request_id}] 发送到: {url}")
            
            async with self.client_session.request(method, url, **kwargs) as response:
                content = await response.read()
                
                # 计算处理时间
                process_time = time.time() - start_time
                
                print(f"   ✅ [请求{request_id}] 响应状态: {response.status}, "
                      f"大小: {len(content)} bytes, 用时: {process_time:.2f}s")
                
                # 如果是聊天响应，尝试解析并显示响应内容
                if path == '/v1/chat/completions' and response.status == 200:
                    try:
                        response_json = json.loads(content.decode('utf-8'))
                        choices = response_json.get('choices', [])
                        if choices:
                            message = choices[0].get('message', {})
                            response_content = str(message.get('content', ''))
                            content_preview = response_content[:100]
                            if len(content_preview) == 100:
                                content_preview += "..."
                            print(f"   💬 [请求{request_id}] AI响应: {content_preview}")
                            
                            usage = response_json.get('usage', {})
                            if usage:
                                print(f"   📈 [请求{request_id}] Token使用: "
                                      f"输入={usage.get('prompt_tokens', 0)}, "
                                      f"输出={usage.get('completion_tokens', 0)}, "
                                      f"总计={usage.get('total_tokens', 0)}")
                    except Exception as e:
                        print(f"   ⚠️  [请求{request_id}] 无法解析响应内容: {e}")
                    
                    # 构建响应
                    resp = web.Response(
                        body=content,
                        status=response.status,
                        content_type=response.content_type
                    )
                    
                    # 复制响应头
                    for key, value in response.headers.items():
                        if key.lower() not in ['connection', 'transfer-encoding']:
                            resp.headers[key] = value
                    
                    print(f"   🎯 [请求{request_id}] 转发完成 (GPU {instance.gpu_id})")
                    return resp
                    
        except Exception as e:
            process_time = time.time() - start_time
            error_msg = f"转发失败 (GPU {instance.gpu_id}, 用时{process_time:.2f}s): {e}"
            print(f"❌ [请求{request_id}] {error_msg}")
            
            # 更新实例健康状态和详细信息
            instance.is_healthy = False
            instance.health_info = f"请求转发失败: {str(e)[:100]}"
            
            # 如果这是实例第一次变成不健康，记录开始时间
            if instance.unhealthy_since == 0:
                instance.unhealthy_since = time.time()
                print(f"⚠️  GPU {instance.gpu_id} 因请求失败标记为不健康")
            
            raise web.HTTPInternalServerError(text=f"转发请求失败: {e}")
        finally:
            # 结束请求，更新负载计数
            instance.end_request()
            final_load_info = instance.get_load_info()
            print(f"   📉 [请求{request_id}] 请求完成，GPU {instance.gpu_id} 负载: "
                  f"{final_load_info['active_requests']}/{final_load_info['max_concurrent_requests']} "
                  f"({final_load_info['utilization_percent']}%)")
    
    async def chat_completions(self, request: web.Request) -> web.Response:
        """处理聊天完成请求"""
        client_ip = request.remote
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        print(f"🎯 收到聊天请求 (来源: {client_ip}, UA: {user_agent[:50]})")
        
        instance = await self.get_best_instance()
        print(f"   🎲 使用实例: GPU {instance.gpu_id} (端口{instance.port})")
        
        data = await request.read()
        return await self.forward_request(
            instance, '/v1/chat/completions', 'POST', 
            data=data, headers=dict(request.headers)
        )
    
    async def completions(self, request: web.Request) -> web.Response:
        """处理文本完成请求"""
        client_ip = request.remote
        
        print(f"📝 收到文本完成请求 (来源: {client_ip})")
        
        instance = await self.get_best_instance()
        print(f"   🎲 使用实例: GPU {instance.gpu_id} (端口{instance.port})")
        
        data = await request.read()
        return await self.forward_request(
            instance, '/v1/completions', 'POST',
            data=data, headers=dict(request.headers)
        )
    
    async def list_models(self, request: web.Request) -> web.Response:
        """列出可用模型"""
        instance = await self.get_best_instance()
        return await self.forward_request(instance, '/v1/models', 'GET')
    
    async def health_check(self, request: web.Request) -> web.Response:
        """健康检查"""
        client_ip = request.remote
        print(f"🩺 健康检查请求 (来源: {client_ip})")
        
        healthy_count = sum(1 for inst in self.instances if inst.is_healthy)
        total_count = len(self.instances)
        
        status = {
            "status": "healthy" if healthy_count > 0 else "unhealthy",
            "instances": {
                "total": total_count,
                "healthy": healthy_count,
                "unhealthy": total_count - healthy_count
            }
        }
        
        print(f"   📊 返回状态: {status['status']} ({healthy_count}/{total_count} 健康)")
        
        return web.json_response(status)
    
    async def status(self, request: web.Request) -> web.Response:
        """详细状态信息"""
        client_ip = request.remote
        print(f"📊 状态查询请求 (来源: {client_ip})")
        
        instance_status = []
        for inst in self.instances:
            detailed_status = inst.get_detailed_status()
            instance_status.append(detailed_status)
            
            # 打印详细状态
            status_icon = '🟢' if inst.is_healthy else '🔴'
            load_info = inst.get_load_info()
            restart_info = f" (已重启{inst.restart_count}次)" if inst.restart_count > 0 else ""
            unhealthy_duration = inst.get_unhealthy_duration()
            duration_str = f" [不健康{unhealthy_duration:.1f}s]" if unhealthy_duration > 0 else ""
            restart_warning = " ⚠️即将重启" if inst.needs_restart() else ""
            
            print(f"   🖥️  GPU {inst.gpu_id}: {status_icon} {inst.health_info}{duration_str}{restart_info}{restart_warning}")
            print(f"      📍 端口{inst.port} PID={inst.process.pid if inst.process else 'None'}")
            print(f"      📊 负载: {load_info['active_requests']}/{load_info['max_concurrent_requests']} "
                  f"({load_info['utilization_percent']}%) | 总请求: {load_info['total_requests']}")
            
            if detailed_status['uptime'] > 0:
                uptime_str = f"{detailed_status['uptime']:.1f}秒"
                if detailed_status['uptime'] > 60:
                    uptime_str = f"{detailed_status['uptime']/60:.1f}分钟"
                print(f"      ⏱️  运行时间: {uptime_str}")
            
            if detailed_status['last_health_check'] > 0:
                last_check_ago = time.time() - detailed_status['last_health_check']
                print(f"      🔍 最后检查: {last_check_ago:.1f}秒前")
        
        queue_size = self.request_queue.qsize()
        print(f"   📝 当前队列长度: {queue_size}")
        
        # 计算负载统计
        total_active_requests = sum(inst.active_requests for inst in self.instances)
        total_capacity = sum(inst.max_concurrent_requests for inst in self.instances)
        total_requests_served = sum(inst.total_requests for inst in self.instances)
        available_instances = sum(1 for inst in self.instances if inst.can_accept_request())
        
        print(f"   📈 负载统计: 活跃请求 {total_active_requests}/{total_capacity}, "
              f"可用实例 {available_instances}/{len(self.instances)}, 总服务 {total_requests_served}")
        
        status = {
            "model": self.model,
            "instances": instance_status,
            "total_instances": len(self.instances),
            "healthy_instances": sum(1 for inst in self.instances if inst.is_healthy),
            "available_instances": available_instances,
            "load_statistics": {
                "total_active_requests": total_active_requests,
                "total_capacity": total_capacity,
                "total_requests_served": total_requests_served,
                "overall_utilization_percent": round((total_active_requests / total_capacity * 100) if total_capacity > 0 else 0, 1)
            },
            "queue": {
                "current_size": queue_size,
                "max_size": self.max_queue_size,
                "max_wait_time": self.max_wait_time
            },
            "limits": {
                "max_request_size": self.max_request_size,
                "max_request_size_mb": round(self.max_request_size / (1024*1024), 1)
            }
        }
        
        return web.json_response(status)
    
    async def diagnosis(self, request: web.Request) -> web.Response:
        """详细诊断信息"""
        client_ip = request.remote
        print(f"🔬 诊断请求 (来源: {client_ip})")
        
        diagnosis_info = {
            "timestamp": time.time(),
            "router_info": {
                "model": self.model,
                "model_path": self.model_path,
                "host": self.host,
                "base_port": self.base_port,
                "max_queue_size": self.max_queue_size,
                "max_wait_time": self.max_wait_time,
                "max_request_size": self.max_request_size
            },
            "instances": [],
            "queue_info": {
                "current_size": self.request_queue.qsize(),
                "max_size": self.max_queue_size
            }
        }
        
        for inst in self.instances:
            # 获取详细状态
            detailed_status = inst.get_detailed_status()
            
            # 获取进程输出信息（如果有的话）
            stdout_info = "无输出"
            stderr_info = "无错误"
            
            if inst.process and inst.process.stdout and inst.process.stderr:
                try:
                    # 非阻塞读取输出
                    import select
                    import fcntl
                    import os
                    
                    # 设置非阻塞模式
                    fd_stdout = inst.process.stdout.fileno()
                    fd_stderr = inst.process.stderr.fileno()
                    
                    fl = fcntl.fcntl(fd_stdout, fcntl.F_GETFL)
                    fcntl.fcntl(fd_stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                    
                    fl = fcntl.fcntl(fd_stderr, fcntl.F_GETFL)
                    fcntl.fcntl(fd_stderr, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                    
                    # 尝试读取最新输出
                    try:
                        stdout_data = inst.process.stdout.read(1024)
                        if stdout_data:
                            stdout_info = stdout_data.decode('utf-8', errors='ignore')[-500:]  # 最后500字符
                    except:
                        pass
                        
                    try:
                        stderr_data = inst.process.stderr.read(1024)
                        if stderr_data:
                            stderr_info = stderr_data.decode('utf-8', errors='ignore')[-500:]  # 最后500字符
                    except:
                        pass
                        
                except Exception as e:
                    stdout_info = f"读取输出失败: {e}"
                    stderr_info = f"读取错误失败: {e}"
            
            instance_diagnosis = {
                **detailed_status,
                "command": inst.get_command(),
                "recent_stdout": stdout_info,
                "recent_stderr": stderr_info
            }
            
            diagnosis_info["instances"].append(instance_diagnosis)
            
            print(f"   🔬 GPU {inst.gpu_id} 诊断:")
            print(f"      状态: {inst.health_info}")
            if not inst.is_healthy and stderr_info != "无错误":
                print(f"      错误信息: {stderr_info[:100]}...")
        
        return web.json_response(diagnosis_info)
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        stats = {
            "business_session": {
                "closed": self.client_session.closed if hasattr(self, 'client_session') else True,
                "connector_stats": {}
            },
            "health_session": {
                "closed": self.health_session.closed if hasattr(self, 'health_session') else True,
                "connector_stats": {}
            }
        }
        
        # 获取业务session连接统计
        if hasattr(self, 'client_session') and not self.client_session.closed:
            connector = self.client_session.connector
            if hasattr(connector, '_conns'):
                stats["business_session"]["connector_stats"] = {
                    "total_connections": len(connector._conns),
                    "limit": getattr(connector, '_limit', 'unknown'),
                    "limit_per_host": getattr(connector, '_limit_per_host', 'unknown')
                }
        
        # 获取健康检查session连接统计
        if hasattr(self, 'health_session') and not self.health_session.closed:
            connector = self.health_session.connector
            if hasattr(connector, '_conns'):
                stats["health_session"]["connector_stats"] = {
                    "total_connections": len(connector._conns),
                    "limit": getattr(connector, '_limit', 'unknown'),
                    "limit_per_host": getattr(connector, '_limit_per_host', 'unknown')
                }
        
        return stats

    async def metrics(self, request: web.Request) -> web.Response:
        """系统性能指标和统计信息"""
        client_ip = request.remote
        print(f"📊 指标查询请求 (来源: {client_ip})")
        
        current_time = time.time()
        
        # 收集所有实例的指标
        instance_metrics = []
        total_requests_served = 0
        total_active_requests = 0
        total_capacity = 0
        total_restarts = 0
        healthy_instances = 0
        unhealthy_instances = 0
        
        for inst in self.instances:
            load_info = inst.get_load_info()
            
            # 计算运行时间
            uptime = current_time - inst.startup_time if inst.startup_time > 0 else 0
            
            # 计算平均每分钟请求数
            requests_per_minute = 0
            if uptime > 60:  # 至少运行1分钟
                requests_per_minute = (inst.total_requests / uptime) * 60
            
            # 计算不健康持续时间
            unhealthy_duration = inst.get_unhealthy_duration()
            
            # 实例指标
            instance_metric = {
                "gpu_id": inst.gpu_id,
                "port": inst.port,
                "is_healthy": inst.is_healthy,
                "health_info": inst.health_info,
                "uptime_seconds": round(uptime, 1),
                "requests": {
                    "total": inst.total_requests,
                    "active": inst.active_requests,
                    "capacity": inst.max_concurrent_requests,
                    "utilization_percent": load_info['utilization_percent'],
                    "requests_per_minute": round(requests_per_minute, 2)
                },
                "performance": {
                    "restart_count": inst.restart_count,
                    "unhealthy_duration_seconds": round(unhealthy_duration, 1),
                    "load_score": round(inst.load_score, 3),
                    "can_accept_request": inst.can_accept_request()
                }
            }
            
            instance_metrics.append(instance_metric)
            
            # 累计统计
            total_requests_served += inst.total_requests
            total_active_requests += inst.active_requests
            total_capacity += inst.max_concurrent_requests
            total_restarts += inst.restart_count
            
            if inst.is_healthy:
                healthy_instances += 1
            else:
                unhealthy_instances += 1
        
        # 计算系统级指标
        available_instances = sum(1 for inst in self.instances if inst.can_accept_request())
        overall_utilization = round((total_active_requests / total_capacity * 100) if total_capacity > 0 else 0, 1)
        
        # 队列指标
        queue_size = self.request_queue.qsize()
        queue_utilization = round((queue_size / self.max_queue_size * 100) if self.max_queue_size > 0 else 0, 1)
        
        # 系统运行时间（取最早启动的实例时间）
        earliest_startup = min((inst.startup_time for inst in self.instances if inst.startup_time > 0), default=current_time)
        system_uptime = current_time - earliest_startup if earliest_startup < current_time else 0
        
        # 系统平均每分钟请求数
        system_requests_per_minute = 0
        if system_uptime > 60:
            system_requests_per_minute = (total_requests_served / system_uptime) * 60
        
        # 获取连接池统计
        connection_stats = await self.get_connection_stats()
        
        # 构建metrics响应
        metrics = {
            "timestamp": current_time,
            "system": {
                "uptime_seconds": round(system_uptime, 1),
                "total_instances": len(self.instances),
                "healthy_instances": healthy_instances,
                "unhealthy_instances": unhealthy_instances,
                "available_instances": available_instances,
                "availability_percent": round((healthy_instances / len(self.instances) * 100) if len(self.instances) > 0 else 0, 1)
            },
            "requests": {
                "total_served": total_requests_served,
                "active_requests": total_active_requests,
                "total_capacity": total_capacity,
                "utilization_percent": overall_utilization,
                "requests_per_minute": round(system_requests_per_minute, 2)
            },
            "queue": {
                "current_size": queue_size,
                "max_size": self.max_queue_size,
                "utilization_percent": queue_utilization,
                "max_wait_time_seconds": self.max_wait_time
            },
            "reliability": {
                "total_restarts": total_restarts,
                "average_restarts_per_instance": round(total_restarts / len(self.instances), 2) if len(self.instances) > 0 else 0,
                "unhealthy_instances_count": unhealthy_instances
            },
            "configuration": {
                "model": self.model,
                "max_request_size_mb": round(self.max_request_size / (1024*1024), 1),
                "gpu_memory_utilization": self.gpu_memory_util,
                "max_model_length": self.max_model_len,
                "max_seqs_per_instance": self.max_num_seqs_per_instance
            },
            "connection_pools": connection_stats,
            "instances": instance_metrics
        }
        
        # 打印简要指标概览
        print(f"   📈 系统指标概览:")
        print(f"      🔥 活跃请求: {total_active_requests}/{total_capacity} ({overall_utilization}%)")
        print(f"      📊 实例状态: {healthy_instances}健康/{unhealthy_instances}不健康/{available_instances}可用")
        print(f"      📝 队列状态: {queue_size}/{self.max_queue_size} ({queue_utilization}%)")
        print(f"      🚀 总服务请求: {total_requests_served}")
        print(f"      ⚡ 平均QPS: {system_requests_per_minute/60:.2f}")
        print(f"      🔄 总重启次数: {total_restarts}")
        
        # 打印连接池状态
        if connection_stats:
            business_conns = connection_stats.get("business_session", {}).get("connector_stats", {}).get("total_connections", 0)
            health_conns = connection_stats.get("health_session", {}).get("connector_stats", {}).get("total_connections", 0)
            print(f"      🔗 连接池: 业务={business_conns}, 健康检查={health_conns}")
        
        return web.json_response(metrics)
    
    async def connection_cleanup_task(self):
        """定期清理连接池，防止连接泄露"""
        print("🧹 启动连接清理任务...")
        cleanup_count = 0
        
        while True:
            try:
                cleanup_count += 1
                await asyncio.sleep(120)  # 每2分钟执行一次清理
                
                print(f"🧹 连接清理 #{cleanup_count}")
                
                # 获取清理前的连接统计
                before_stats = await self.get_connection_stats()
                before_business = before_stats.get("business_session", {}).get("connector_stats", {}).get("total_connections", 0)
                before_health = before_stats.get("health_session", {}).get("connector_stats", {}).get("total_connections", 0)
                
                print(f"   📊 清理前连接数: 业务={before_business}, 健康检查={before_health}")
                
                # 清理业务session连接池
                business_cleaned = await self._cleanup_session_connections(self.client_session, "业务")
                
                # 清理健康检查session连接池
                health_cleaned = await self._cleanup_session_connections(self.health_session, "健康检查")
                
                # 获取清理后的连接统计
                after_stats = await self.get_connection_stats()
                after_business = after_stats.get("business_session", {}).get("connector_stats", {}).get("total_connections", 0)
                after_health = after_stats.get("health_session", {}).get("connector_stats", {}).get("total_connections", 0)
                
                # 计算实际清理的连接数
                actual_cleaned_business = before_business - after_business
                actual_cleaned_health = before_health - after_health
                
                if actual_cleaned_business > 0 or actual_cleaned_health > 0:
                    print(f"   🧹 实际清理了 {actual_cleaned_business} 个业务连接, {actual_cleaned_health} 个健康检查连接")
                
                print(f"   📊 清理后连接数: 业务={after_business}, 健康检查={after_health}")
                    
            except asyncio.CancelledError:
                print("🛑 连接清理任务被取消")
                break
            except Exception as e:
                print(f"⚠️  连接清理失败: {e}")
                import traceback
                print(f"   🔍 详细错误: {traceback.format_exc()}")
    
    async def _cleanup_session_connections(self, session, session_name: str) -> int:
        """清理指定session的连接，返回清理的连接数"""
        cleaned_count = 0
        
        if not session or session.closed:
            print(f"   ⚠️  {session_name}session不可用或已关闭")
            return cleaned_count
        
        try:
            connector = session.connector
            if not connector:
                print(f"   ⚠️  {session_name}session没有connector")
                return cleaned_count
            
            # 记录清理前的连接数
            before_count = len(getattr(connector, '_conns', {}))
            
            # 尝试各种清理方法
            cleanup_methods = [
                ('_cleanup_closed', False),  # 清理已关闭的连接，通常是同步方法
                ('_cleanup', None),          # 通用清理方法，可能是异步的
                ('close_expired', False),    # 关闭过期连接，通常是同步方法
            ]
            
            for method_name, is_async in cleanup_methods:
                if hasattr(connector, method_name):
                    method = getattr(connector, method_name)
                    if callable(method):
                        try:
                            print(f"   🔧 {session_name}session尝试调用{method_name}...")
                            
                            if is_async is None:
                                # 自动检测是否为异步方法
                                result = method()
                                if hasattr(result, '__await__'):
                                    await result
                                # 如果不是异步，result就是返回值，不需要特殊处理
                            elif is_async:
                                await method()
                            else:
                                method()
                            
                            print(f"   ✅ {session_name}session {method_name} 调用成功")
                        except Exception as e:
                            print(f"   ⚠️  {session_name}session {method_name} 调用失败: {e}")
            
            # 记录清理后的连接数
            after_count = len(getattr(connector, '_conns', {}))
            cleaned_count = before_count - after_count
            
            if cleaned_count > 0:
                print(f"   🧹 {session_name}session清理了{cleaned_count}个连接")
            else:
                print(f"   ✅ {session_name}session连接池状态良好，无需清理")
                
        except Exception as e:
            print(f"   ❌ {session_name}session连接清理异常: {e}")
        
        return cleaned_count

    async def start_health_monitor(self):
        """启动健康监控"""
        monitor_count = 0
        while True:
            try:
                monitor_count += 1
                print(f"🔍 健康检查 #{monitor_count}")
                
                health_results = []
                for instance in self.instances:
                    old_status = instance.is_healthy
                    old_health_info = getattr(instance, 'health_info', '未知')
                    
                    instance.last_health_check = time.time()
                    new_status = await self.check_instance_health(instance)
                    new_health_info = instance.health_info
                    
                    health_results.append((instance.gpu_id, old_status, new_status, new_health_info))
                    
                    # 状态变化时打印详细日志
                    if old_status != new_status:
                        status_text = "🟢 健康" if new_status else "🔴 不健康"
                        print(f"   📊 GPU {instance.gpu_id} 状态变化: {status_text}")
                        print(f"      💡 详细信息: {new_health_info}")
                        
                        # 如果实例不健康，显示更多诊断信息
                        if not new_status:
                            detailed_status = instance.get_detailed_status()
                            if detailed_status['process_info']:
                                proc_info = detailed_status['process_info']
                                print(f"      🔧 进程状态: PID={proc_info.get('pid', 'N/A')}, "
                                      f"运行中={proc_info.get('is_running', False)}, "
                                      f"退出码={proc_info.get('exit_code', 'N/A')}")
                            
                            if detailed_status['uptime'] > 0:
                                uptime_str = f"{detailed_status['uptime']:.1f}秒"
                                if detailed_status['uptime'] > 60:
                                    uptime_str = f"{detailed_status['uptime']/60:.1f}分钟"
                                print(f"      ⏱️  运行时间: {uptime_str}")
                    elif not new_status and old_health_info != new_health_info:
                        # 健康信息变化但状态未变
                        print(f"   ⚠️  GPU {instance.gpu_id} 健康信息更新: {new_health_info}")
                
                # 打印总体状态
                healthy_count = sum(1 for _, _, status, _ in health_results if status)
                total_count = len(health_results)
                available_count = sum(1 for inst in self.instances if inst.can_accept_request())
                total_active = sum(inst.active_requests for inst in self.instances)
                total_capacity = sum(inst.max_concurrent_requests for inst in self.instances)
                
                print(f"   💚 健康实例: {healthy_count}/{total_count}")
                print(f"   🔄 可用实例: {available_count}/{total_count}")
                print(f"   📊 系统负载: {total_active}/{total_capacity} "
                      f"({round(total_active/total_capacity*100, 1) if total_capacity > 0 else 0}%)")
                
                # 显示每个实例的负载状态
                if monitor_count % 6 == 0:  # 每3分钟显示一次详细负载信息
                    print(f"   📈 详细负载状态:")
                    for inst in self.instances:
                        load_info = inst.get_load_info()
                        status_icon = '🟢' if inst.is_healthy else '🔴'
                        available_icon = '✅' if inst.can_accept_request() else '🔄'
                        restart_info = f" (重启{inst.restart_count}次)" if inst.restart_count > 0 else ""
                        print(f"      GPU {inst.gpu_id}: {status_icon}{available_icon} "
                              f"{load_info['active_requests']}/{load_info['max_concurrent_requests']} "
                              f"({load_info['utilization_percent']}%) | 总计: {load_info['total_requests']}{restart_info}")
                
                # 检查是否有实例需要重启
                instances_to_restart = [inst for inst in self.instances if inst.needs_restart()]
                if instances_to_restart:
                    print(f"   🔄 发现{len(instances_to_restart)}个实例需要重启:")
                    for inst in instances_to_restart:
                        unhealthy_duration = inst.get_unhealthy_duration()
                        print(f"      GPU {inst.gpu_id}: 不健康{unhealthy_duration:.1f}秒 - {inst.health_info}")
                        print(f"         📊 当前状态: 活跃请求={inst.active_requests}, 负载={inst.load_score:.2f}")
                        
                        # 在线程中执行重启以避免阻塞健康监控
                        def restart_instance(instance):
                            try:
                                print(f"🔧 开始重启GPU {instance.gpu_id}...")
                                instance.restart()
                                print(f"🎯 GPU {instance.gpu_id}重启完成，等待健康检查...")
                            except Exception as e:
                                print(f"❌ GPU {instance.gpu_id}重启失败: {e}")
                                # 重启失败时也要重置连接状态
                                instance.reset_connection_state()
                                instance.health_info = f"重启失败: {str(e)[:50]}"
                        
                        # 使用线程池执行重启
                        import threading
                        restart_thread = threading.Thread(target=restart_instance, args=(inst,))
                        restart_thread.daemon = True
                        restart_thread.start()
                
                # 显示不健康实例的详细信息
                unhealthy_instances = [(gpu_id, health_info) for gpu_id, _, status, health_info in health_results if not status]
                if unhealthy_instances:
                    print(f"   🔴 不健康实例详情:")
                    for gpu_id, health_info in unhealthy_instances:
                        # 找到对应的实例以获取不健康持续时间
                        instance = next((inst for inst in self.instances if inst.gpu_id == gpu_id), None)
                        if instance:
                            unhealthy_duration = instance.get_unhealthy_duration()
                            restart_info = f" (已重启{instance.restart_count}次)" if instance.restart_count > 0 else ""
                            duration_str = f" [不健康{unhealthy_duration:.1f}s]" if unhealthy_duration > 0 else ""
                            restart_warning = " ⚠️即将重启" if instance.needs_restart() else ""
                            print(f"      GPU {gpu_id}: {health_info}{duration_str}{restart_info}{restart_warning}")
                
                await asyncio.sleep(30)  # 每30秒检查一次
            except asyncio.CancelledError:
                print("🛑 健康监控被取消")
                break
            except Exception as e:
                print(f"⚠️  健康检查失败: {e}")
                await asyncio.sleep(30)
    
    def stop_all_instances(self):
        """停止所有实例"""
        print("🛑 停止所有vLLM实例...")
        for instance in self.instances:
            try:
                instance.stop()
                print(f"✅ GPU {instance.gpu_id}实例已停止")
            except Exception as e:
                print(f"⚠️  停止GPU {instance.gpu_id}实例时出错: {e}")
    
    async def cleanup(self):
        """清理资源"""
        print("🧹 清理路由器资源...")
        
        # 取消健康检查任务
        if hasattr(self, 'health_check_task') and self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            print("✅ 健康检查任务已取消")
        
        # 取消队列处理任务
        if hasattr(self, 'queue_processor_task') and self.queue_processor_task:
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass
            print("✅ 队列处理任务已取消")
        
        # 取消连接清理任务
        if hasattr(self, 'connection_cleanup_task_obj') and self.connection_cleanup_task_obj:
            self.connection_cleanup_task_obj.cancel()
            try:
                await self.connection_cleanup_task_obj
            except asyncio.CancelledError:
                pass
            print("✅ 连接清理任务已取消")
        
        # 关闭业务请求ClientSession
        if hasattr(self, 'client_session') and not self.client_session.closed:
            await self.client_session.close()
            print("✅ 业务ClientSession已关闭")
        
        # 关闭健康检查ClientSession
        if hasattr(self, 'health_session') and not self.health_session.closed:
            await self.health_session.close()
            print("✅ 健康检查ClientSession已关闭")
        
        # 等待一小段时间确保连接完全关闭
        await asyncio.sleep(1.0)  # 增加等待时间确保连接释放
        
        # 停止所有实例
        self.stop_all_instances()
        
        print("✅ 路由器资源清理完成")
    
    async def run(self, port: int):
        """运行路由器"""
        print(f"🚀 开始启动多实例路由器...")
        
        # 启动所有实例
        self.start_instances()
        
        # 等待实例就绪
        print(f"⏳ 等待所有{len(self.instances)}个实例就绪...")
        if not await self.wait_for_instances_ready():
            print("❌ 实例启动失败，停止所有服务")
            self.stop_all_instances()
            raise RuntimeError("实例启动失败")
        
        # 启动健康监控
        print("🔍 启动健康监控...")
        self.health_check_task = asyncio.create_task(self.start_health_monitor())
        
        # 启动队列处理器
        print("📝 启动请求队列处理器...")
        self.queue_processor_task = asyncio.create_task(self.queue_processor())
        
        # 启动连接清理任务
        print("🧹 启动连接清理任务...")
        self.connection_cleanup_task_obj = asyncio.create_task(self.connection_cleanup_task())
        
        # 启动web服务
        print(f"🌐 启动路由器web服务...")
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, port)
        
        print(f"🌐 路由器启动在 http://{self.host}:{port}")
        print(f"📊 管理{len(self.instances)}个vLLM实例")
        print("✅ 数据并行服务已就绪！")
        print("=" * 60)
        print("📝 API端点:")
        print(f"   🔗 聊天完成: POST http://{self.host}:{port}/v1/chat/completions")
        print(f"   🔗 文本完成: POST http://{self.host}:{port}/v1/completions")
        print(f"   🔗 模型列表: GET  http://{self.host}:{port}/v1/models")
        print(f"   🔗 健康检查: GET  http://{self.host}:{port}/health")
        print(f"   🔗 状态查询: GET  http://{self.host}:{port}/status")
        print(f"   🔗 详细诊断: GET  http://{self.host}:{port}/diagnosis")
        print(f"   🔗 性能指标: GET  http://{self.host}:{port}/metrics")
        print("=" * 60)
        
        await site.start()
        
        # 等待停止信号
        try:
            print("🎯 路由器运行中，等待请求...")
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("\n🛑 收到中断信号...")
        except Exception as e:
            print(f"❌ 路由器运行异常: {e}")
        finally:
            print("🧹 正在清理资源...")
            await self.cleanup()  # 使用新的cleanup方法
            await runner.cleanup()
            print("✅ 清理完成")


def signal_handler(signum, frame, router):
    """信号处理器"""
    print(f"\n🛑 收到信号 {signum}，正在关闭...")
    
    # 创建事件循环来执行清理操作
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环正在运行，创建任务来执行清理
            loop.create_task(router.cleanup())
        else:
            # 如果事件循环未运行，直接运行清理
            loop.run_until_complete(router.cleanup())
    except Exception as e:
        print(f"❌ 清理过程出错: {e}")
        # 即使清理失败也要停止实例
        router.stop_all_instances()
    
    sys.exit(0)


def parse_gpu_list(gpu_str: str) -> List[int]:
    """解析GPU列表"""
    return [int(gpu.strip()) for gpu in gpu_str.split(',')]


async def main():
    parser = argparse.ArgumentParser(description='vLLM多实例数据并行路由器')
    parser.add_argument('--model', required=True, help='模型名称')
    parser.add_argument('--gpus', required=True, help='GPU列表，逗号分隔，如: 0,1,2,3')
    parser.add_argument('--base-port', type=int, default=2334, help='基础端口号')
    parser.add_argument('--model-path', help='本地模型路径（可选）')
    parser.add_argument('--host', default='0.0.0.0', help='绑定地址')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8, help='GPU内存利用率')
    parser.add_argument('--max-model-len', type=int, default=32000, help='最大模型长度')
    parser.add_argument('--max-num-seqs-per-instance', type=int, default=16, help='每个实例最大并发数')
    parser.add_argument('--max-concurrent-requests-per-instance', type=int, default=64, 
                        help='每个实例负载均衡的最大并发请求数（默认64，会自动调整为max-num-seqs-per-instance的4倍或64的较大值）')
    parser.add_argument('--max-queue-size', type=int, default=1000, help='最大排队数量')
    parser.add_argument('--max-wait-time', type=float, default=600.0, help='最大等待时间（秒）')
    parser.add_argument('--max-request-size', type=int, default=50*1024*1024, help='最大请求体大小（字节），默认50MB')
    parser.add_argument('--max-connections-per-host', type=int, default=50, help='每个主机最大连接数（默认50，减少以防止文件描述符耗尽）')
    
    args = parser.parse_args()
    
    # 解析GPU列表
    try:
        gpus = parse_gpu_list(args.gpus)
    except ValueError as e:
        print(f"❌ GPU列表格式错误: {e}")
        sys.exit(1)
    
    print("🚀 vLLM多实例数据并行路由器")
    print("=" * 50)
    print(f"模型: {args.model}")
    if args.model_path:
        print(f"本地路径: {args.model_path}")
    print(f"GPU列表: {gpus}")
    print(f"基础端口: {args.base_port}")
    print(f"路由器端口: {args.base_port}")
    print(f"实例端口: {args.base_port + 1} - {args.base_port + len(gpus)}")
    print(f"GPU内存利用率: {args.gpu_memory_utilization}")
    print(f"每实例最大并发: {args.max_num_seqs_per_instance}")
    print(f"最大排队数量: {args.max_queue_size}")
    print(f"最大等待时间: {args.max_wait_time}秒")
    print(f"最大请求体大小: {args.max_request_size / (1024*1024):.1f}MB")
    print(f"负载均衡并发限制: {args.max_concurrent_requests_per_instance}")
    print("=" * 50)
    
    # 创建路由器
    router = MultiInstanceRouter(
        model=args.model,
        gpus=gpus,
        base_port=args.base_port,
        model_path=args.model_path,
        host=args.host,
        gpu_memory_util=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs_per_instance=args.max_num_seqs_per_instance,
        max_queue_size=args.max_queue_size,
        max_wait_time=args.max_wait_time,
        max_request_size=args.max_request_size,
        max_concurrent_requests_per_instance=args.max_concurrent_requests_per_instance,
        max_connections_per_host=args.max_connections_per_host
    )
    
    # 设置信号处理
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda s, f: signal_handler(s, f, router))
    
    try:
        await router.run(args.base_port)
    except Exception as e:
        print(f"❌ 路由器运行失败: {e}")
        await router.cleanup()  # 确保异常时也清理资源
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
