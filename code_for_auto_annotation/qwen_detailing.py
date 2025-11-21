from utils import ask_about_image, manual_retry, check_retry_status
from task_processor import get_task, list_available_tasks, BaseTask
import os
import json
import asyncio
import numpy as np
import cv2 as cv
import sys
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
import threading

# 任务配置
TASK_NAME = "qwen_detailing"  # 默认任务名
MAX_CONCURRENCY = 16  # 设置最大并发数

# 分片执行相关参数
SHARD_INDEX = 0  # 当前分片索引（从0开始）
SHARD_COUNT = 1  # 总分片数

# 创建一个全局信号量
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
console = Console()

# 全局任务实例
current_task: BaseTask = None

# 用于线程安全地管理完成文件列表的锁
completed_files_lock = asyncio.Lock()

def start_user_input_listener():
    """启动用户输入监听线程，支持手动重试"""
    def input_listener():
        console.print("\n[bold cyan]🎮 控制面板启动！")
        console.print("[dim]输入命令来控制程序:")
        console.print("[dim]  'r' 或 'retry' - 手动触发重试")
        console.print("[dim]  's' 或 'status' - 查看重试状态") 
        console.print("[dim]  'h' 或 'help' - 显示帮助")
        console.print("[dim]  'q' 或 'quit' - 停止监听[/dim]")
        
        while True:
            try:
                user_input = input("\n[控制台] 输入命令: ").strip().lower()
                
                if user_input in ['r', 'retry']:
                    console.print("[bold yellow]🔄 手动触发重试...")
                    manual_retry()
                    console.print("[green]✅ 重试信号已发送！")
                    
                elif user_input in ['s', 'status']:
                    console.print("[bold blue]📊 检查重试状态...")
                    check_retry_status()
                    
                elif user_input in ['h', 'help']:
                    console.print("\n[bold cyan]📖 帮助信息:")
                    console.print("  [bold]r/retry[/bold] - 当程序在等待重试时，立即触发重试")
                    console.print("  [bold]s/status[/bold] - 查看当前API调用状态")
                    console.print("  [bold]h/help[/bold] - 显示此帮助信息")
                    console.print("  [bold]q/quit[/bold] - 停止控制台监听")
                    console.print("\n[dim]💡 提示: 当看到'等待X秒后重试'时，输入'r'可立即重试")
                    
                elif user_input in ['q', 'quit']:
                    console.print("[yellow]🛑 用户输入监听已停止")
                    break
                    
                elif user_input == '':
                    continue
                    
                else:
                    console.print(f"[red]❌ 未知命令: '{user_input}'，输入 'h' 查看帮助")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]🛑 用户输入监听已停止")
                break
            except EOFError:
                break
                
    # 启动输入监听线程
    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()
    return input_thread

# 加载进度文件
def load_progress():
    """加载已完成的文件列表"""
    progress_file = current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('completed_files', []))
        except (json.JSONDecodeError, KeyError):
            console.print("[yellow]Warning: 无法读取进度文件，将重新开始[/yellow]")
            return set()
    return set()

# 保存进度
def save_progress(completed_files):
    """保存已完成的文件列表，使用临时文件确保原子写入"""
    progress_file = current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)
    progress_data = {
        'completed_files': list(completed_files),
        'total_files': len(completed_files),
        'shard_index': SHARD_INDEX,
        'shard_count': SHARD_COUNT,
        'task_name': current_task.task_name
    }
    
    temp_progress_file = progress_file + '.tmp'
    try:
        # 先写入临时文件
        with open(temp_progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
        
        # 原子性替换原文件
        os.replace(temp_progress_file, progress_file)
        
    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_progress_file):
            try:
                os.remove(temp_progress_file)
            except:
                pass
        console.print(f"[red]保存进度失败: {e}[/red]")

# 获取当前分片的进度文件名（保留兼容性）
def get_progress_filename():
    """根据分片信息生成进度文件名"""
    return current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)

# 检查文件是否属于当前分片
def is_file_in_current_shard(file_path):
    """根据文件名的哈希值判断是否属于当前分片"""
    filename = os.path.basename(file_path)
    # 使用文件名的哈希值进行分片
    file_hash = hash(filename)
    return file_hash % SHARD_COUNT == SHARD_INDEX

# 清理临时文件
def cleanup_temp_files():
    """清理可能遗留的临时文件"""
    cleaned_count = 0
    
    # 清理进度文件的临时文件
    progress_file = current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)
    temp_progress_file = progress_file + '.tmp'
    if os.path.exists(temp_progress_file):
        try:
            os.remove(temp_progress_file)
            cleaned_count += 1
        except Exception as e:
            console.print(f"[yellow]清理临时进度文件失败: {e}[/yellow]")
    
    # 清理JSON文件的临时文件
    if os.path.exists(current_task.data_dir):
        for filename in os.listdir(current_task.data_dir):
            if filename.endswith('.json.tmp'):
                temp_file_path = os.path.join(current_task.data_dir, filename)
                try:
                    os.remove(temp_file_path)
                    cleaned_count += 1
                except Exception as e:
                    console.print(f"[yellow]清理临时文件失败 {temp_file_path}: {e}[/yellow]")
    
    if cleaned_count > 0:
        console.print(f"[cyan]清理了 {cleaned_count} 个临时文件[/cyan]")

# 检查文件是否已经处理过
def is_file_processed(file_path, data):
    """检查文件是否已经处理过，使用任务特定的逻辑"""
    return current_task.is_file_processed(file_path, data)

# 异步读取并处理一个 JSON 文件（受信号量控制）
async def process_file(file_path, progress: Progress, task_id: TaskID, completed_files: set):
    async with semaphore:
        max_retries = 3  # 最大重试次数
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 读取 JSON 内容（同步）
                with open(file_path, mode='r', encoding='utf-8') as f:
                    contents = f.read()
                    data = json.loads(contents)
                
                # 检查是否已经处理过
                if is_file_processed(file_path, data):
                    progress.update(task_id, advance=1, description=f"[green]跳过已处理: {os.path.basename(file_path)}")
                    # 使用锁来安全地添加到完成列表
                    async with completed_files_lock:
                        completed_files.add(file_path)
                    console.print(f"[dim]⏭️  文件已处理，跳过: {os.path.basename(file_path)}[/dim]")
                    return data
                
                if retry_count == 0:
                    progress.update(task_id, description=f"[blue]处理中: {os.path.basename(file_path)}")
                    console.print(f"[dim]🚀 开始处理新文件: {os.path.basename(file_path)}[/dim]")
                else:
                    progress.update(task_id, description=f"[yellow]重试中({retry_count}/{max_retries}): {os.path.basename(file_path)}")
                    console.print(f"[dim]🔄 重试文件: {os.path.basename(file_path)} (第{retry_count}次重试)[/dim]")
                
                # 处理数据（自定义逻辑）
                console.print(f"[dim]⚙️  调用数据处理函数: {os.path.basename(file_path)}[/dim]")
                updated_data = await process_json_data(file_path, data)

                # 使用临时文件安全写入，保护原文件完整性（同步）
                temp_file_path = file_path + '.tmp'
                console.print(f"[dim]💾 开始写入文件: {os.path.basename(file_path)}[/dim]")
                try:
                    # 写入临时文件（同步）
                    with open(temp_file_path, mode='w', encoding='utf-8') as f:
                        json_content = json.dumps(updated_data, indent=2, ensure_ascii=False)
                        f.write(json_content)
                        console.print(f"[dim]📝 临时文件写入完成 (大小: {len(json_content)} 字符)[/dim]")
                    
                    # 原子性替换原文件（同步）
                    os.replace(temp_file_path, file_path)
                    console.print(f"[dim]🔄 文件替换完成: {os.path.basename(file_path)}[/dim]")
                    
                except Exception as write_error:
                    console.print(f"[red]❌ 文件写入失败: {os.path.basename(file_path)} - {write_error}[/red]")
                    # 清理临时文件
                    if os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            console.print(f"[dim]🧹 已清理临时文件: {os.path.basename(temp_file_path)}[/dim]")
                        except:
                            pass
                    raise write_error
                
                # 使用锁来安全地添加到完成列表和检查进度保存
                async with completed_files_lock:
                    completed_files.add(file_path)
                    current_completed_count = len(completed_files)
                    # 定期保存进度（每10个文件保存一次）
                    if current_completed_count % 10 == 0:
                        save_progress(completed_files)
                        console.print(f"[dim]💾 进度保存检查点: {current_completed_count} 个文件已完成[/dim]")
                
                progress.update(task_id, advance=1, description=f"[green]已完成: {os.path.basename(file_path)}")
                console.print(f"[dim]✅ 文件处理成功: {os.path.basename(file_path)}[/dim]")
                
                # 成功处理，跳出重试循环
                break
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                if retry_count < max_retries:
                    console.print(f'[yellow]⚠️  处理失败，准备重试 ({retry_count}/{max_retries}): {os.path.basename(file_path)}[/yellow]')
                    console.print(f'[yellow]错误详情: {error_msg}[/yellow]')
                    import traceback
                    traceback.print_exc()
                    # 等待一段时间再重试
                    wait_time = 1 * retry_count
                    console.print(f'[dim]⏳ 等待 {wait_time} 秒后重试...[/dim]')
                    await asyncio.sleep(wait_time)  # 递增等待时间
                else:
                    # 所有重试都失败了
                    progress.update(task_id, advance=1, description=f"[red]失败: {os.path.basename(file_path)}")
                    console.print(f'[red]❌ 处理失败，已达到最大重试次数 ({max_retries}): {os.path.basename(file_path)}[/red]')
                    console.print(f'[red]最终错误: {error_msg}[/red]')
                    # 不添加到completed_files，这样下次运行时会再次尝试处理

# 你自定义的处理逻辑
async def process_json_data(file_path, data): 
    """处理JSON数据，使用任务特定的逻辑"""
    return await current_task.process_json_data(file_path, data)

# 主函数：并发处理所有 JSON 文件
async def main():
    # 清理可能遗留的临时文件
    cleanup_temp_files()
    
    # 启动用户输入监听线程
    input_thread = start_user_input_listener()
    
    # 加载已完成的文件
    completed_files = load_progress()
    
    # 获取所有JSON文件
    all_files = []
    for filename in os.listdir(current_task.data_dir):
        if filename.endswith('.json'):
            full_path = os.path.join(current_task.data_dir, filename)
            all_files.append(full_path)
    
    # 过滤出属于当前分片的文件
    shard_files = [f for f in all_files if is_file_in_current_shard(f)]
    
    # 过滤出需要处理的文件（在当前分片内且未完成）
    files_to_process = [f for f in shard_files if f not in completed_files]
    
    total_files = len(all_files)
    shard_total = len(shard_files)
    already_completed = len(completed_files)
    to_process = len(files_to_process)
    
    console.print(f"[bold cyan]分片信息: 第 {SHARD_INDEX + 1}/{SHARD_COUNT} 片")
    console.print(f"[bold cyan]全部文件数: {total_files}")
    console.print(f"[bold blue]当前分片文件数: {shard_total}")
    console.print(f"[bold green]已完成: {already_completed}")
    console.print(f"[bold yellow]待处理: {to_process}")
    
    if to_process == 0:
        console.print("[bold green]当前分片的所有文件都已处理完成！")
        return
    
    console.print(f"\n[bold magenta]💡 使用进度文件: {current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)}")
    console.print("[bold magenta]💡 提示: 程序运行过程中，你可以:")
    console.print("[dim]  - 输入 'r' 手动触发重试（当API连接有问题时）")
    console.print("[dim]  - 输入 's' 查看当前重试状态")
    console.print("[dim]  - 当看到'等待X秒后重试'时，可以立即输入'r'跳过等待[/dim]")
    
    # 创建进度条
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        
        task_id = progress.add_task(
            description="[cyan]开始处理文件...", 
            total=to_process
        )
        
        # 创建任务列表
        tasks = []
        for file_path in files_to_process:
            tasks.append(process_file(file_path, progress, task_id, completed_files))
        
        # 执行所有任务
        await asyncio.gather(*tasks)
        
        # 最终保存进度
        save_progress(completed_files)
        
        progress.update(task_id, description="[bold green]所有文件处理完成！")
        
    console.print(f"[bold green]✅ 分片处理完成！共处理 {to_process} 个文件")
    console.print(f"[bold blue]💾 进度已保存到 {current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)}")

if __name__ == '__main__':
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='通用任务处理脚本 - 支持分片并行执行')
    parser.add_argument('--task', type=str, default=TASK_NAME, 
                       help=f'要执行的任务名称，可选: {", ".join(list_available_tasks())}')
    parser.add_argument('--reset', action='store_true', help='重置进度文件并重新开始')
    parser.add_argument('--shard-index', type=int, default=0, help='当前分片索引（从0开始）')
    parser.add_argument('--shard-count', type=int, default=1, help='总分片数')
    parser.add_argument('--list-tasks', action='store_true', help='列出所有可用任务')
    
    args = parser.parse_args()
    
    # 列出任务并退出
    if args.list_tasks:
        console.print("[bold cyan]📋 可用任务列表:")
        for task_name in list_available_tasks():
            task_instance = get_task(task_name)
            console.print(f"  [bold]{task_name}[/bold] - {task_instance.get_task_description()}")
        sys.exit(0)
    
    # 初始化任务
    try:
        current_task = get_task(args.task)
        TASK_NAME = args.task
    except ValueError as e:
        console.print(f"[red]错误: {e}[/red]")
        sys.exit(1)
    
    # 设置分片参数
    SHARD_INDEX = args.shard_index
    SHARD_COUNT = args.shard_count
    
    # 验证分片参数
    if SHARD_INDEX < 0 or SHARD_INDEX >= SHARD_COUNT:
        console.print(f"[red]错误: shard-index ({SHARD_INDEX}) 必须在 0 到 {SHARD_COUNT-1} 之间[/red]")
        sys.exit(1)
    
    if SHARD_COUNT <= 0:
        console.print(f"[red]错误: shard-count ({SHARD_COUNT}) 必须大于 0[/red]")
        sys.exit(1)
    
    # 处理重置参数
    if args.reset:
        progress_file = current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)
        if os.path.exists(progress_file):
            os.remove(progress_file)
            console.print(f"[yellow]已重置进度文件 {progress_file}[/yellow]")
        else:
            console.print(f"[yellow]进度文件 {progress_file} 不存在[/yellow]")
        sys.exit(0)
    
    console.print(f"[bold cyan]🚀 启动通用任务处理脚本")
    console.print(f"[bold cyan]📋 当前任务: {current_task.task_name}")
    console.print(f"[bold cyan]📁 数据目录: {current_task.data_dir}")
    if SHARD_COUNT > 1:
        console.print(f"[bold cyan]📊 分片模式: 第 {SHARD_INDEX + 1}/{SHARD_COUNT} 片")
        console.print(f"[dim]进度文件: {current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)}[/dim]")
    console.print("[dim]提示: 使用 --reset 参数可以重置进度并重新开始[/dim]")
    console.print("[dim]提示: 使用 --shard-index 和 --shard-count 参数可以并行执行多个分片[/dim]")
    console.print("[dim]提示: 使用 --task 参数可以选择不同的任务[/dim]")
    console.print("[dim]提示: 使用 --list-tasks 查看所有可用任务[/dim]")
    console.print("[bold yellow]🆕 新功能: 支持手动重试！在程序运行时可以随时输入命令控制重试[/bold yellow]")
    
    asyncio.run(main())
