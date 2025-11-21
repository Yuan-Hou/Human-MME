#!/usr/bin/env python3
"""
vLLM...... - .........
...GPU.....vLLM..，..........
.....API.......
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
    """....."""
    request_id: str
    future: asyncio.Future
    timestamp: float
    timeout: float = 300.0  # 5....


class VLLMInstance:
    """..vLLM...."""
    
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
        self.health_info = "..."  # ........
        self.last_health_check = 0  # ........
        self.startup_time = 0  # ....
        self.url = f"http://{host}:{port}"
        self.load_score = 0  # ....，......
        self.active_requests = 0  # .......
        self.total_requests = 0  # ....
        self.max_concurrent_requests = 64  # .......（...，........）
        self.unhealthy_since = 0  # .........
        self.restart_count = 0  # ....
        
    def get_command(self) -> List[str]:
        """......"""
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
        """..vLLM.."""
        cmd = self.get_command()
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        print(f"🚀 ..GPU {self.gpu_id}.. (..{self.port})...")
        print(f"   ..: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # .......
            )
            
            self.startup_time = time.time()
            self.health_info = "......."
            print(f"✅ GPU {self.gpu_id}...... (PID: {self.process.pid})")
            
        except Exception as e:
            self.health_info = f"....: {str(e)}"
            print(f"❌ GPU {self.gpu_id}......: {e}")
            raise
    
    async def check_health(self) -> bool:
        """........"""
        if not self.process or self.process.poll() is not None:
            new_healthy = False
            if not self.process:
                self.health_info = "....."
            else:
                exit_code = self.process.poll()
                self.health_info = f"..... (...: {exit_code})"
        else:
            try:
                # ..：.....router..health_session，........router..
                # ........，.MultiInstanceRouter.........
                new_healthy = True  # ....True，.....router...
                self.health_info = "......"
            except Exception as e:
                new_healthy = False
                self.health_info = f"......: {str(e)[:100]}"
        
        # ..........
        current_time = time.time()
        if not new_healthy:
            # ........，......，......
            if self.is_healthy:
                self.unhealthy_since = current_time
                print(f"⚠️  GPU {self.gpu_id} .....: {self.health_info}")
        else:
            # ........，.......
            if self.unhealthy_since > 0:
                unhealthy_duration = current_time - self.unhealthy_since
                print(f"✅ GPU {self.gpu_id} ....，...... {unhealthy_duration:.1f}.")
            self.unhealthy_since = 0
        
        self.is_healthy = new_healthy
        return self.is_healthy
    
    def can_accept_request(self) -> bool:
        """..........."""
        return self.is_healthy and self.active_requests < self.max_concurrent_requests
    
    def start_request(self) -> bool:
        """.......，......"""
        if self.can_accept_request():
            self.active_requests += 1
            self.total_requests += 1
            self.load_score = self.active_requests / self.max_concurrent_requests
            return True
        return False
    
    def end_request(self):
        """......"""
        if self.active_requests > 0:
            self.active_requests -= 1
            self.load_score = self.active_requests / self.max_concurrent_requests
    
    def get_load_info(self) -> Dict[str, Any]:
        """......"""
        return {
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "max_concurrent_requests": self.max_concurrent_requests,
            "load_score": self.load_score,
            "utilization_percent": round(self.load_score * 100, 1),
            "can_accept_request": self.can_accept_request()
        }
    
    def get_unhealthy_duration(self) -> float:
        """.........（.）"""
        if self.unhealthy_since > 0:
            return time.time() - self.unhealthy_since
        return 0
    
    def needs_restart(self, max_unhealthy_time: float = 120.0) -> bool:
        """........（.........）"""
        return not self.is_healthy and self.get_unhealthy_duration() > max_unhealthy_time
    
    def restart(self):
        """...."""
        print(f"🔄 ..GPU {self.gpu_id}.....")
        
        # ......
        self.stop()
        
        # ............
        time.sleep(5)
        
        # .............
        self.restart_count += 1
        self.reset_connection_state()
        self.health_info = "......."
        
        # ....
        self.start()
        
        print(f"✅ GPU {self.gpu_id}...... (.{self.restart_count}...)")
    
    def reset_connection_state(self):
        """...... - ............."""
        self.is_healthy = False
        self.health_info = "...."
        self.active_requests = 0
        self.load_score = 0
        self.last_health_check = 0
        self.unhealthy_since = 0
        print(f"🔄 GPU {self.gpu_id} .......")
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """........"""
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
        """..vLLM.."""
        if self.process:
            try:
                # ..SIGTERM......
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                # ......
                self.process.wait(timeout=30)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                # ....
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process = None
        
        # ........
        self.is_healthy = False
        self.health_info = "..."
        self.active_requests = 0  # ......
        self.load_score = 0  # ......
        self.last_health_check = 0  # ........
        self.unhealthy_since = 0  # .........
        self.startup_time = 0  # ......


class MultiInstanceRouter:
    """......"""
    
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
        
        # .............，........
        if max_concurrent_requests_per_instance is None:
            max_concurrent_requests_per_instance = max(max_num_seqs_per_instance * 4, 64)  # ..64...
        
        # ...GPU....，.......
        self.instances = []
        for i, gpu_id in enumerate(gpus):
            port = base_port + i + 1  # ........
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
            # ..............
            instance.max_concurrent_requests = max_concurrent_requests_per_instance
            self.instances.append(instance)
        
        # ......
        self.max_queue_size = max_queue_size
        self.max_wait_time = max_wait_time
        self.max_request_size = max_request_size
        self.max_connections_per_host = max_connections_per_host
        
        # .........，..4K......
        self.app = web.Application(client_max_size=max_request_size)
        self.setup_routes()
        
        # ......ClientSession......，...............
        connector = TCPConnector(
            limit=1000,  # ........，.......
            limit_per_host=min(self.max_connections_per_host, 100),  # ..........
            enable_cleanup_closed=True,
            force_close=True,  # ......，......
            ttl_dns_cache=300,  # DNS..5..
            use_dns_cache=True,
        )
        timeout = ClientTimeout(total=600)  # 10....
        self.client_session = ClientSession(
            connector=connector,
            timeout=timeout,
            connector_owner=True,  # ..session..connector，........
            auto_decompress=False  # ..............
        )
        
        # .........session，.......session
        health_connector = TCPConnector(
            limit=50,  # .........
            limit_per_host=10,
            enable_cleanup_closed=True,
            keepalive_timeout=10,  # ............
        )
        health_timeout = ClientTimeout(total=10)  # ....10...
        self.health_session = ClientSession(
            connector=health_connector,
            timeout=health_timeout,
            connector_owner=True
        )
        
        # ......
        self.health_check_task = None
        
        # ......
        self.request_queue = asyncio.Queue()
        self.queue_processor_task = None
        
    def setup_routes(self):
        """...."""
        # OpenAI API....
        self.app.router.add_post('/v1/chat/completions', self.chat_completions)
        self.app.router.add_post('/v1/completions', self.completions)
        self.app.router.add_get('/v1/models', self.list_models)
        
        # .......
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/status', self.status)
        self.app.router.add_get('/diagnosis', self.diagnosis)
        self.app.router.add_get('/metrics', self.metrics)
    
    def start_instances(self):
        """....vLLM.."""
        print(f"🔄 ..{len(self.instances)}.vLLM.....")
        
        # ...........
        with ThreadPoolExecutor(max_workers=len(self.instances)) as executor:
            futures = []
            for instance in self.instances:
                future = executor.submit(instance.start)
                futures.append(future)
            
            # ..........
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ ......: {e}")
                    self.stop_all_instances()
                    raise
        
        print("✅ ........")
    
    async def queue_processor(self):
        """......."""
        print("🚀 ............")
        
        while True:
            try:
                # .......
                queued_request = await self.request_queue.get()
                
                # ........
                current_time = time.time()
                if current_time - queued_request.timestamp > queued_request.timeout:
                    print(f"⏰ [..{queued_request.request_id}] ....，....")
                    if not queued_request.future.done():
                        queued_request.future.set_exception(
                            web.HTTPRequestTimeout(text="......")
                        )
                    continue
                
                # ........
                instance = None
                wait_start = time.time()
                while instance is None:
                    # ..............
                    available_instances = [inst for inst in self.instances if inst.can_accept_request()]
                    if available_instances:
                        # .......，.........
                        available_instances.sort(key=lambda x: x.load_score)
                        instance = available_instances[0]
                        break
                    
                    # ..........
                    if time.time() - wait_start > queued_request.timeout:
                        print(f"⏰ [..{queued_request.request_id}] ......")
                        if not queued_request.future.done():
                            queued_request.future.set_exception(
                                web.HTTPRequestTimeout(text="........")
                            )
                        break
                    
                    # .......
                    await asyncio.sleep(0.5)  # ............
                
                if instance and not queued_request.future.done():
                    wait_time = time.time() - queued_request.timestamp
                    load_info = instance.get_load_info()
                    print(f"✅ [..{queued_request.request_id}] ...GPU {instance.gpu_id} "
                          f"(..{wait_time:.1f}s, ..: {load_info['active_requests']}/{load_info['max_concurrent_requests']})")
                    queued_request.future.set_result(instance)
                
                self.request_queue.task_done()
                
            except asyncio.CancelledError:
                print("🛑 ........")
                break
            except Exception as e:
                print(f"❌ .......: {e}")
                await asyncio.sleep(1)
    
    async def check_instance_health(self, instance: VLLMInstance) -> bool:
        """.........session........"""
        if not instance.process or instance.process.poll() is not None:
            new_healthy = False
            if not instance.process:
                instance.health_info = "....."
            else:
                exit_code = instance.process.poll()
                instance.health_info = f"..... (...: {exit_code})"
        else:
            try:
                # .....health_session......
                async with self.health_session.get(f"{instance.url}/health") as response:
                    new_healthy = response.status == 200
                    if new_healthy:
                        instance.health_info = ".."
                    else:
                        instance.health_info = f"HTTP.....: {response.status}"
            except asyncio.TimeoutError:
                new_healthy = False
                instance.health_info = "......"
            except aiohttp.ClientConnectorError as e:
                new_healthy = False
                instance.health_info = f"....: {str(e)[:100]}"
            except Exception as e:
                new_healthy = False
                instance.health_info = f"......: {str(e)[:100]}"
        
        # ..........
        current_time = time.time()
        if not new_healthy:
            # ........，......，......
            if instance.is_healthy:
                instance.unhealthy_since = current_time
                print(f"⚠️  GPU {instance.gpu_id} .....: {instance.health_info}")
        else:
            # ........，.......
            if instance.unhealthy_since > 0:
                unhealthy_duration = current_time - instance.unhealthy_since
                print(f"✅ GPU {instance.gpu_id} ....，...... {unhealthy_duration:.1f}.")
            instance.unhealthy_since = 0
        
        instance.is_healthy = new_healthy
        return instance.is_healthy

    async def wait_for_instances_ready(self, timeout: int = 600):
        """........"""
        print("⏳ ...........")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            ready_count = 0
            for instance in self.instances:
                if await self.check_instance_health(instance):
                    ready_count += 1
            
            print(f"📊 ....: {ready_count}/{len(self.instances)} ...")
            
            if ready_count == len(self.instances):
                print("✅ .......！")
                return True
            
            await asyncio.sleep(10)
        
        print(f"❌ .... ({timeout}.)，.......")
        return False
    
    async def get_best_instance(self) -> VLLMInstance:
        """......（......，......）"""
        # ..............
        available_instances = [inst for inst in self.instances if inst.can_accept_request()]
        
        if available_instances:
            # .......，.........
            available_instances.sort(key=lambda x: x.load_score)
            selected = available_instances[0]
            
            # .......
            load_info = selected.get_load_info()
            print(f"   🎯 ......: GPU {selected.gpu_id} "
                  f"(..: {load_info['active_requests']}/{load_info['max_concurrent_requests']}, "
                  f"...: {load_info['utilization_percent']}%)")
            print(f"      💡 ....: {len(available_instances)}/{len(self.instances)}")
            
            return selected
        
        # ..............
        healthy_instances = [inst for inst in self.instances if inst.is_healthy]
        if healthy_instances:
            # ......
            print(f"   🔄 .........:")
            for inst in healthy_instances:
                load_info = inst.get_load_info()
                print(f"      GPU {inst.gpu_id}: {load_info['active_requests']}/{load_info['max_concurrent_requests']} "
                      f"({load_info['utilization_percent']}%)")
        
        # ......，....
        if self.request_queue.qsize() >= self.max_queue_size:
            print(f"❌ .... ({self.max_queue_size})，....")
            raise web.HTTPServiceUnavailable(text=f"......，.....")
        
        request_id = f"{int(time.time() * 1000) % 100000:05d}"
        queued_request = QueuedRequest(
            request_id=request_id,
            future=asyncio.Future(),
            timestamp=time.time(),
            timeout=self.max_wait_time
        )
        
        await self.request_queue.put(queued_request)
        queue_size = self.request_queue.qsize()
        print(f"📝 [..{request_id}] .... (....: {queue_size})")
        
        try:
            # ......
            instance = await queued_request.future
            return instance
        except asyncio.TimeoutError:
            print(f"⏰ [..{request_id}] ....")
            raise web.HTTPRequestTimeout(text="......，.....")
    
    async def forward_request(self, instance: VLLMInstance, path: str, method: str, 
                             data: Optional[bytes] = None, 
                             headers: Optional[Dict] = None) -> web.Response:
        """.....vLLM.."""
        url = f"{instance.url}{path}"
        
        # ....，......
        if not instance.start_request():
            # ..........，.........
            print(f"⚠️  GPU {instance.gpu_id} ....，.........")
        
        # ........
        start_time = time.time()
        request_id = f"{int(start_time * 1000) % 100000:05d}"  # .....ID
        
        load_info = instance.get_load_info()
        print(f"🔄 [..{request_id}] ...GPU {instance.gpu_id} ({method} {path})")
        print(f"   📊 [..{request_id}] ....: {load_info['active_requests']}/{load_info['max_concurrent_requests']} "
              f"({load_info['utilization_percent']}%)")
        
        # .......，...........
        if data and path == '/v1/chat/completions':
            try:
                request_json = json.loads(data.decode('utf-8'))
                messages = request_json.get('messages', [])
                if messages:
                    last_message = messages[-1]
                    content_preview = str(last_message.get('content', ''))[:100]
                    if len(content_preview) == 100:
                        content_preview += "..."
                    print(f"   📝 [..{request_id}] ....: {content_preview}")
                    print(f"   ⚙️  [..{request_id}] ....: max_tokens={request_json.get('max_tokens', 'default')}, "
                          f"temperature={request_json.get('temperature', 'default')}")
            except Exception as e:
                print(f"   ⚠️  [..{request_id}] ........: {e}")
        
        try:
            # .....ClientSession.........
            kwargs = {}
            if data:
                kwargs['data'] = data
                print(f"   📊 [..{request_id}] ......: {len(data)} bytes")
            if headers:
                # ...hop-by-hop headers
                filtered_headers = {k: v for k, v in headers.items() 
                                  if k.lower() not in ['connection', 'transfer-encoding']}
                kwargs['headers'] = filtered_headers
                print(f"   📋 [..{request_id}] .....: {len(filtered_headers)}")
            
            print(f"   🚀 [..{request_id}] ...: {url}")
            
            async with self.client_session.request(method, url, **kwargs) as response:
                content = await response.read()
                
                # ......
                process_time = time.time() - start_time
                
                print(f"   ✅ [..{request_id}] ....: {response.status}, "
                      f"..: {len(content)} bytes, ..: {process_time:.2f}s")
                
                # .......，...........
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
                            print(f"   💬 [..{request_id}] AI..: {content_preview}")
                            
                            usage = response_json.get('usage', {})
                            if usage:
                                print(f"   📈 [..{request_id}] Token..: "
                                      f"..={usage.get('prompt_tokens', 0)}, "
                                      f"..={usage.get('completion_tokens', 0)}, "
                                      f"..={usage.get('total_tokens', 0)}")
                    except Exception as e:
                        print(f"   ⚠️  [..{request_id}] ........: {e}")
                    
                    # ....
                    resp = web.Response(
                        body=content,
                        status=response.status,
                        content_type=response.content_type
                    )
                    
                    # .....
                    for key, value in response.headers.items():
                        if key.lower() not in ['connection', 'transfer-encoding']:
                            resp.headers[key] = value
                    
                    print(f"   🎯 [..{request_id}] .... (GPU {instance.gpu_id})")
                    return resp
                    
        except Exception as e:
            process_time = time.time() - start_time
            error_msg = f".... (GPU {instance.gpu_id}, ..{process_time:.2f}s): {e}"
            print(f"❌ [..{request_id}] {error_msg}")
            
            # .............
            instance.is_healthy = False
            instance.health_info = f"......: {str(e)[:100]}"
            
            # ..............，......
            if instance.unhealthy_since == 0:
                instance.unhealthy_since = time.time()
                print(f"⚠️  GPU {instance.gpu_id} ...........")
            
            raise web.HTTPInternalServerError(text=f"......: {e}")
        finally:
            # ....，......
            instance.end_request()
            final_load_info = instance.get_load_info()
            print(f"   📉 [..{request_id}] ....，GPU {instance.gpu_id} ..: "
                  f"{final_load_info['active_requests']}/{final_load_info['max_concurrent_requests']} "
                  f"({final_load_info['utilization_percent']}%)")
    
    async def chat_completions(self, request: web.Request) -> web.Response:
        """........"""
        client_ip = request.remote
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        print(f"🎯 ...... (..: {client_ip}, UA: {user_agent[:50]})")
        
        instance = await self.get_best_instance()
        print(f"   🎲 ....: GPU {instance.gpu_id} (..{instance.port})")
        
        data = await request.read()
        return await self.forward_request(
            instance, '/v1/chat/completions', 'POST', 
            data=data, headers=dict(request.headers)
        )
    
    async def completions(self, request: web.Request) -> web.Response:
        """........"""
        client_ip = request.remote
        
        print(f"📝 ........ (..: {client_ip})")
        
        instance = await self.get_best_instance()
        print(f"   🎲 ....: GPU {instance.gpu_id} (..{instance.port})")
        
        data = await request.read()
        return await self.forward_request(
            instance, '/v1/completions', 'POST',
            data=data, headers=dict(request.headers)
        )
    
    async def list_models(self, request: web.Request) -> web.Response:
        """......"""
        instance = await self.get_best_instance()
        return await self.forward_request(instance, '/v1/models', 'GET')
    
    async def health_check(self, request: web.Request) -> web.Response:
        """...."""
        client_ip = request.remote
        print(f"🩺 ...... (..: {client_ip})")
        
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
        
        print(f"   📊 ....: {status['status']} ({healthy_count}/{total_count} ..)")
        
        return web.json_response(status)
    
    async def status(self, request: web.Request) -> web.Response:
        """......"""
        client_ip = request.remote
        print(f"📊 ...... (..: {client_ip})")
        
        instance_status = []
        for inst in self.instances:
            detailed_status = inst.get_detailed_status()
            instance_status.append(detailed_status)
            
            # ......
            status_icon = '🟢' if inst.is_healthy else '🔴'
            load_info = inst.get_load_info()
            restart_info = f" (...{inst.restart_count}.)" if inst.restart_count > 0 else ""
            unhealthy_duration = inst.get_unhealthy_duration()
            duration_str = f" [...{unhealthy_duration:.1f}s]" if unhealthy_duration > 0 else ""
            restart_warning = " ⚠️...." if inst.needs_restart() else ""
            
            print(f"   🖥️  GPU {inst.gpu_id}: {status_icon} {inst.health_info}{duration_str}{restart_info}{restart_warning}")
            print(f"      📍 ..{inst.port} PID={inst.process.pid if inst.process else 'None'}")
            print(f"      📊 ..: {load_info['active_requests']}/{load_info['max_concurrent_requests']} "
                  f"({load_info['utilization_percent']}%) | ...: {load_info['total_requests']}")
            
            if detailed_status['uptime'] > 0:
                uptime_str = f"{detailed_status['uptime']:.1f}."
                if detailed_status['uptime'] > 60:
                    uptime_str = f"{detailed_status['uptime']/60:.1f}.."
                print(f"      ⏱️  ....: {uptime_str}")
            
            if detailed_status['last_health_check'] > 0:
                last_check_ago = time.time() - detailed_status['last_health_check']
                print(f"      🔍 ....: {last_check_ago:.1f}..")
        
        queue_size = self.request_queue.qsize()
        print(f"   📝 ......: {queue_size}")
        
        # ......
        total_active_requests = sum(inst.active_requests for inst in self.instances)
        total_capacity = sum(inst.max_concurrent_requests for inst in self.instances)
        total_requests_served = sum(inst.total_requests for inst in self.instances)
        available_instances = sum(1 for inst in self.instances if inst.can_accept_request())
        
        print(f"   📈 ....: .... {total_active_requests}/{total_capacity}, "
              f".... {available_instances}/{len(self.instances)}, ... {total_requests_served}")
        
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
        """......"""
        client_ip = request.remote
        print(f"🔬 .... (..: {client_ip})")
        
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
            # ......
            detailed_status = inst.get_detailed_status()
            
            # ........（.....）
            stdout_info = "..."
            stderr_info = "..."
            
            if inst.process and inst.process.stdout and inst.process.stderr:
                try:
                    # .......
                    import select
                    import fcntl
                    import os
                    
                    # .......
                    fd_stdout = inst.process.stdout.fileno()
                    fd_stderr = inst.process.stderr.fileno()
                    
                    fl = fcntl.fcntl(fd_stdout, fcntl.F_GETFL)
                    fcntl.fcntl(fd_stdout, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                    
                    fl = fcntl.fcntl(fd_stderr, fcntl.F_GETFL)
                    fcntl.fcntl(fd_stderr, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                    
                    # ........
                    try:
                        stdout_data = inst.process.stdout.read(1024)
                        if stdout_data:
                            stdout_info = stdout_data.decode('utf-8', errors='ignore')[-500:]  # ..500..
                    except:
                        pass
                        
                    try:
                        stderr_data = inst.process.stderr.read(1024)
                        if stderr_data:
                            stderr_info = stderr_data.decode('utf-8', errors='ignore')[-500:]  # ..500..
                    except:
                        pass
                        
                except Exception as e:
                    stdout_info = f"......: {e}"
                    stderr_info = f"......: {e}"
            
            instance_diagnosis = {
                **detailed_status,
                "command": inst.get_command(),
                "recent_stdout": stdout_info,
                "recent_stderr": stderr_info
            }
            
            diagnosis_info["instances"].append(instance_diagnosis)
            
            print(f"   🔬 GPU {inst.gpu_id} ..:")
            print(f"      ..: {inst.health_info}")
            if not inst.is_healthy and stderr_info != "...":
                print(f"      ....: {stderr_info[:100]}...")
        
        return web.json_response(diagnosis_info)
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """........."""
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
        
        # ....session....
        if hasattr(self, 'client_session') and not self.client_session.closed:
            connector = self.client_session.connector
            if hasattr(connector, '_conns'):
                stats["business_session"]["connector_stats"] = {
                    "total_connections": len(connector._conns),
                    "limit": getattr(connector, '_limit', 'unknown'),
                    "limit_per_host": getattr(connector, '_limit_per_host', 'unknown')
                }
        
        # ......session....
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
        """..........."""
        client_ip = request.remote
        print(f"📊 ...... (..: {client_ip})")
        
        current_time = time.time()
        
        # .........
        instance_metrics = []
        total_requests_served = 0
        total_active_requests = 0
        total_capacity = 0
        total_restarts = 0
        healthy_instances = 0
        unhealthy_instances = 0
        
        for inst in self.instances:
            load_info = inst.get_load_info()
            
            # ......
            uptime = current_time - inst.startup_time if inst.startup_time > 0 else 0
            
            # ..........
            requests_per_minute = 0
            if uptime > 60:  # ....1..
                requests_per_minute = (inst.total_requests / uptime) * 60
            
            # .........
            unhealthy_duration = inst.get_unhealthy_duration()
            
            # ....
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
            
            # ....
            total_requests_served += inst.total_requests
            total_active_requests += inst.active_requests
            total_capacity += inst.max_concurrent_requests
            total_restarts += inst.restart_count
            
            if inst.is_healthy:
                healthy_instances += 1
            else:
                unhealthy_instances += 1
        
        # .......
        available_instances = sum(1 for inst in self.instances if inst.can_accept_request())
        overall_utilization = round((total_active_requests / total_capacity * 100) if total_capacity > 0 else 0, 1)
        
        # ....
        queue_size = self.request_queue.qsize()
        queue_utilization = round((queue_size / self.max_queue_size * 100) if self.max_queue_size > 0 else 0, 1)
        
        # ......（..........）
        earliest_startup = min((inst.startup_time for inst in self.instances if inst.startup_time > 0), default=current_time)
        system_uptime = current_time - earliest_startup if earliest_startup < current_time else 0
        
        # ..........
        system_requests_per_minute = 0
        if system_uptime > 60:
            system_requests_per_minute = (total_requests_served / system_uptime) * 60
        
        # .......
        connection_stats = await self.get_connection_stats()
        
        # ..metrics..
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
        
        # ........
        print(f"   📈 ......:")
        print(f"      🔥 ....: {total_active_requests}/{total_capacity} ({overall_utilization}%)")
        print(f"      📊 ....: {healthy_instances}../{unhealthy_instances}.../{available_instances}..")
        print(f"      📝 ....: {queue_size}/{self.max_queue_size} ({queue_utilization}%)")
        print(f"      🚀 .....: {total_requests_served}")
        print(f"      ⚡ ..QPS: {system_requests_per_minute/60:.2f}")
        print(f"      🔄 .....: {total_restarts}")
        
        # .......
        if connection_stats:
            business_conns = connection_stats.get("business_session", {}).get("connector_stats", {}).get("total_connections", 0)
            health_conns = connection_stats.get("health_session", {}).get("connector_stats", {}).get("total_connections", 0)
            print(f"      🔗 ...: ..={business_conns}, ....={health_conns}")
        
        return web.json_response(metrics)
    
    async def connection_cleanup_task(self):
        """.......，......"""
        print("🧹 ...........")
        cleanup_count = 0
        
        while True:
            try:
                cleanup_count += 1
                await asyncio.sleep(120)  # .2........
                
                print(f"🧹 .... #{cleanup_count}")
                
                # ..........
                before_stats = await self.get_connection_stats()
                before_business = before_stats.get("business_session", {}).get("connector_stats", {}).get("total_connections", 0)
                before_health = before_stats.get("health_session", {}).get("connector_stats", {}).get("total_connections", 0)
                
                print(f"   📊 ......: ..={before_business}, ....={before_health}")
                
                # ....session...
                business_cleaned = await self._cleanup_session_connections(self.client_session, "..")
                
                # ......session...
                health_cleaned = await self._cleanup_session_connections(self.health_session, "....")
                
                # ..........
                after_stats = await self.get_connection_stats()
                after_business = after_stats.get("business_session", {}).get("connector_stats", {}).get("total_connections", 0)
                after_health = after_stats.get("health_session", {}).get("connector_stats", {}).get("total_connections", 0)
                
                # ..........
                actual_cleaned_business = before_business - after_business
                actual_cleaned_health = before_health - after_health
                
                if actual_cleaned_business > 0 or actual_cleaned_health > 0:
                    print(f"   🧹 ..... {actual_cleaned_business} ....., {actual_cleaned_health} .......")
                
                print(f"   📊 ......: ..={after_business}, ....={after_health}")
                    
            except asyncio.CancelledError:
                print("🛑 .........")
                break
            except Exception as e:
                print(f"⚠️  ......: {e}")
                import traceback
                print(f"   🔍 ....: {traceback.format_exc()}")
    
    async def _cleanup_session_connections(self, session, session_name: str) -> int:
        """....session...，........"""
        cleaned_count = 0
        
        if not session or session.closed:
            print(f"   ⚠️  {session_name}session.......")
            return cleaned_count
        
        try:
            connector = session.connector
            if not connector:
                print(f"   ⚠️  {session_name}session..connector")
                return cleaned_count
            
            # .........
            before_count = len(getattr(connector, '_conns', {}))
            
            # ........
            cleanup_methods = [
                ('_cleanup_closed', False),  # ........，.......
                ('_cleanup', None),          # ......，......
                ('close_expired', False),    # ......，.......
            ]
            
            for method_name, is_async in cleanup_methods:
                if hasattr(connector, method_name):
                    method = getattr(connector, method_name)
                    if callable(method):
                        try:
                            print(f"   🔧 {session_name}session....{method_name}...")
                            
                            if is_async is None:
                                # ...........
                                result = method()
                                if hasattr(result, '__await__'):
                                    await result
                                # ......，result.....，.......
                            elif is_async:
                                await method()
                            else:
                                method()
                            
                            print(f"   ✅ {session_name}session {method_name} ....")
                        except Exception as e:
                            print(f"   ⚠️  {session_name}session {method_name} ....: {e}")
            
            # .........
            after_count = len(getattr(connector, '_conns', {}))
            cleaned_count = before_count - after_count
            
            if cleaned_count > 0:
                print(f"   🧹 {session_name}session...{cleaned_count}...")
            else:
                print(f"   ✅ {session_name}session.......，....")
                
        except Exception as e:
            print(f"   ❌ {session_name}session......: {e}")
        
        return cleaned_count

    async def start_health_monitor(self):
        """......"""
        monitor_count = 0
        while True:
            try:
                monitor_count += 1
                print(f"🔍 .... #{monitor_count}")
                
                health_results = []
                for instance in self.instances:
                    old_status = instance.is_healthy
                    old_health_info = getattr(instance, 'health_info', '..')
                    
                    instance.last_health_check = time.time()
                    new_status = await self.check_instance_health(instance)
                    new_health_info = instance.health_info
                    
                    health_results.append((instance.gpu_id, old_status, new_status, new_health_info))
                    
                    # ...........
                    if old_status != new_status:
                        status_text = "🟢 .." if new_status else "🔴 ..."
                        print(f"   📊 GPU {instance.gpu_id} ....: {status_text}")
                        print(f"      💡 ....: {new_health_info}")
                        
                        # .......，........
                        if not new_status:
                            detailed_status = instance.get_detailed_status()
                            if detailed_status['process_info']:
                                proc_info = detailed_status['process_info']
                                print(f"      🔧 ....: PID={proc_info.get('pid', 'N/A')}, "
                                      f"...={proc_info.get('is_running', False)}, "
                                      f"...={proc_info.get('exit_code', 'N/A')}")
                            
                            if detailed_status['uptime'] > 0:
                                uptime_str = f"{detailed_status['uptime']:.1f}."
                                if detailed_status['uptime'] > 60:
                                    uptime_str = f"{detailed_status['uptime']/60:.1f}.."
                                print(f"      ⏱️  ....: {uptime_str}")
                    elif not new_status and old_health_info != new_health_info:
                        # ...........
                        print(f"   ⚠️  GPU {instance.gpu_id} ......: {new_health_info}")
                
                # ......
                healthy_count = sum(1 for _, _, status, _ in health_results if status)
                total_count = len(health_results)
                available_count = sum(1 for inst in self.instances if inst.can_accept_request())
                total_active = sum(inst.active_requests for inst in self.instances)
                total_capacity = sum(inst.max_concurrent_requests for inst in self.instances)
                
                print(f"   💚 ....: {healthy_count}/{total_count}")
                print(f"   🔄 ....: {available_count}/{total_count}")
                print(f"   📊 ....: {total_active}/{total_capacity} "
                      f"({round(total_active/total_capacity*100, 1) if total_capacity > 0 else 0}%)")
                
                # ...........
                if monitor_count % 6 == 0:  # .3............
                    print(f"   📈 ......:")
                    for inst in self.instances:
                        load_info = inst.get_load_info()
                        status_icon = '🟢' if inst.is_healthy else '🔴'
                        available_icon = '✅' if inst.can_accept_request() else '🔄'
                        restart_info = f" (..{inst.restart_count}.)" if inst.restart_count > 0 else ""
                        print(f"      GPU {inst.gpu_id}: {status_icon}{available_icon} "
                              f"{load_info['active_requests']}/{load_info['max_concurrent_requests']} "
                              f"({load_info['utilization_percent']}%) | ..: {load_info['total_requests']}{restart_info}")
                
                # ...........
                instances_to_restart = [inst for inst in self.instances if inst.needs_restart()]
                if instances_to_restart:
                    print(f"   🔄 ..{len(instances_to_restart)}.......:")
                    for inst in instances_to_restart:
                        unhealthy_duration = inst.get_unhealthy_duration()
                        print(f"      GPU {inst.gpu_id}: ...{unhealthy_duration:.1f}. - {inst.health_info}")
                        print(f"         📊 ....: ....={inst.active_requests}, ..={inst.load_score:.2f}")
                        
                        # .................
                        def restart_instance(instance):
                            try:
                                print(f"🔧 ....GPU {instance.gpu_id}...")
                                instance.restart()
                                print(f"🎯 GPU {instance.gpu_id}....，.........")
                            except Exception as e:
                                print(f"❌ GPU {instance.gpu_id}....: {e}")
                                # .............
                                instance.reset_connection_state()
                                instance.health_info = f"....: {str(e)[:50]}"
                        
                        # .........
                        import threading
                        restart_thread = threading.Thread(target=restart_instance, args=(inst,))
                        restart_thread.daemon = True
                        restart_thread.start()
                
                # ............
                unhealthy_instances = [(gpu_id, health_info) for gpu_id, _, status, health_info in health_results if not status]
                if unhealthy_instances:
                    print(f"   🔴 .......:")
                    for gpu_id, health_info in unhealthy_instances:
                        # .................
                        instance = next((inst for inst in self.instances if inst.gpu_id == gpu_id), None)
                        if instance:
                            unhealthy_duration = instance.get_unhealthy_duration()
                            restart_info = f" (...{instance.restart_count}.)" if instance.restart_count > 0 else ""
                            duration_str = f" [...{unhealthy_duration:.1f}s]" if unhealthy_duration > 0 else ""
                            restart_warning = " ⚠️...." if instance.needs_restart() else ""
                            print(f"      GPU {gpu_id}: {health_info}{duration_str}{restart_info}{restart_warning}")
                
                await asyncio.sleep(30)  # .30.....
            except asyncio.CancelledError:
                print("🛑 .......")
                break
            except Exception as e:
                print(f"⚠️  ......: {e}")
                await asyncio.sleep(30)
    
    def stop_all_instances(self):
        """......"""
        print("🛑 ....vLLM.....")
        for instance in self.instances:
            try:
                instance.stop()
                print(f"✅ GPU {instance.gpu_id}.....")
            except Exception as e:
                print(f"⚠️  ..GPU {instance.gpu_id}.....: {e}")
    
    async def cleanup(self):
        """...."""
        print("🧹 ..........")
        
        # ........
        if hasattr(self, 'health_check_task') and self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            print("✅ .........")
        
        # ........
        if hasattr(self, 'queue_processor_task') and self.queue_processor_task:
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass
            print("✅ .........")
        
        # ........
        if hasattr(self, 'connection_cleanup_task_obj') and self.connection_cleanup_task_obj:
            self.connection_cleanup_task_obj.cancel()
            try:
                await self.connection_cleanup_task_obj
            except asyncio.CancelledError:
                pass
            print("✅ .........")
        
        # ......ClientSession
        if hasattr(self, 'client_session') and not self.client_session.closed:
            await self.client_session.close()
            print("✅ ..ClientSession...")
        
        # ......ClientSession
        if hasattr(self, 'health_session') and not self.health_session.closed:
            await self.health_session.close()
            print("✅ ....ClientSession...")
        
        # ...............
        await asyncio.sleep(1.0)  # ............
        
        # ......
        self.stop_all_instances()
        
        print("✅ .........")
    
    async def run(self, port: int):
        """....."""
        print(f"🚀 .............")
        
        # ......
        self.start_instances()
        
        # ......
        print(f"⏳ ....{len(self.instances)}........")
        if not await self.wait_for_instances_ready():
            print("❌ ......，......")
            self.stop_all_instances()
            raise RuntimeError("......")
        
        # ......
        print("🔍 .........")
        self.health_check_task = asyncio.create_task(self.start_health_monitor())
        
        # .......
        print("📝 ............")
        self.queue_processor_task = asyncio.create_task(self.queue_processor())
        
        # ........
        print("🧹 ...........")
        self.connection_cleanup_task_obj = asyncio.create_task(self.connection_cleanup_task())
        
        # ..web..
        print(f"🌐 .....web.....")
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, port)
        
        print(f"🌐 ...... http://{self.host}:{port}")
        print(f"📊 ..{len(self.instances)}.vLLM..")
        print("✅ .........！")
        print("=" * 60)
        print("📝 API..:")
        print(f"   🔗 ....: POST http://{self.host}:{port}/v1/chat/completions")
        print(f"   🔗 ....: POST http://{self.host}:{port}/v1/completions")
        print(f"   🔗 ....: GET  http://{self.host}:{port}/v1/models")
        print(f"   🔗 ....: GET  http://{self.host}:{port}/health")
        print(f"   🔗 ....: GET  http://{self.host}:{port}/status")
        print(f"   🔗 ....: GET  http://{self.host}:{port}/diagnosis")
        print(f"   🔗 ....: GET  http://{self.host}:{port}/metrics")
        print("=" * 60)
        
        await site.start()
        
        # ......
        try:
            print("🎯 ......，.......")
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("\n🛑 .........")
        except Exception as e:
            print(f"❌ .......: {e}")
        finally:
            print("🧹 .........")
            await self.cleanup()  # ....cleanup..
            await runner.cleanup()
            print("✅ ....")


def signal_handler(signum, frame, router):
    """....."""
    print(f"\n🛑 .... {signum}，.......")
    
    # .............
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # ..........，.........
            loop.create_task(router.cleanup())
        else:
            # .........，......
            loop.run_until_complete(router.cleanup())
    except Exception as e:
        print(f"❌ ......: {e}")
        # ............
        router.stop_all_instances()
    
    sys.exit(0)


def parse_gpu_list(gpu_str: str) -> List[int]:
    """..GPU.."""
    return [int(gpu.strip()) for gpu in gpu_str.split(',')]


async def main():
    parser = argparse.ArgumentParser(description='vLLM..........')
    parser.add_argument('--model', required=True, help='....')
    parser.add_argument('--gpus', required=True, help='GPU..，....，.: 0,1,2,3')
    parser.add_argument('--base-port', type=int, default=2334, help='.....')
    parser.add_argument('--model-path', help='......（..）')
    parser.add_argument('--host', default='0.0.0.0', help='....')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8, help='GPU.....')
    parser.add_argument('--max-model-len', type=int, default=32000, help='......')
    parser.add_argument('--max-num-seqs-per-instance', type=int, default=16, help='.........')
    parser.add_argument('--max-concurrent-requests-per-instance', type=int, default=64, 
                        help='................（..64，......max-num-seqs-per-instance.4..64....）')
    parser.add_argument('--max-queue-size', type=int, default=1000, help='......')
    parser.add_argument('--max-wait-time', type=float, default=600.0, help='......（.）')
    parser.add_argument('--max-request-size', type=int, default=50*1024*1024, help='.......（..），..50MB')
    parser.add_argument('--max-connections-per-host', type=int, default=50, help='.........（..50，............）')
    
    args = parser.parse_args()
    
    # ..GPU..
    try:
        gpus = parse_gpu_list(args.gpus)
    except ValueError as e:
        print(f"❌ GPU......: {e}")
        sys.exit(1)
    
    print("🚀 vLLM..........")
    print("=" * 50)
    print(f"..: {args.model}")
    if args.model_path:
        print(f"....: {args.model_path}")
    print(f"GPU..: {gpus}")
    print(f"....: {args.base_port}")
    print(f".....: {args.base_port}")
    print(f"....: {args.base_port + 1} - {args.base_port + len(gpus)}")
    print(f"GPU.....: {args.gpu_memory_utilization}")
    print(f".......: {args.max_num_seqs_per_instance}")
    print(f"......: {args.max_queue_size}")
    print(f"......: {args.max_wait_time}.")
    print(f".......: {args.max_request_size / (1024*1024):.1f}MB")
    print(f"........: {args.max_concurrent_requests_per_instance}")
    print("=" * 50)
    
    # .....
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
    
    # ......
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda s, f: signal_handler(s, f, router))
    
    try:
        await router.run(args.base_port)
    except Exception as e:
        print(f"❌ .......: {e}")
        await router.cleanup()  # ..........
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
