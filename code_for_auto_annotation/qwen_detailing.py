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

# ....
TASK_NAME = "qwen_detailing"  # .....
MAX_CONCURRENCY = 16  # .......

# ........
SHARD_INDEX = 0  # ......（.0..）
SHARD_COUNT = 1  # ....

# .........
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
console = Console()

# ......
current_task: BaseTask = None

# .................
completed_files_lock = asyncio.Lock()

def start_user_input_listener():
    """..........，......"""
    def input_listener():
        console.print("\n[bold cyan]🎮 ......！")
        console.print("[dim].........:")
        console.print("[dim]  'r' . 'retry' - ......")
        console.print("[dim]  's' . 'status' - ......") 
        console.print("[dim]  'h' . 'help' - ....")
        console.print("[dim]  'q' . 'quit' - ....[/dim]")
        
        while True:
            try:
                user_input = input("\n[...] ....: ").strip().lower()
                
                if user_input in ['r', 'retry']:
                    console.print("[bold yellow]🔄 .........")
                    manual_retry()
                    console.print("[green]✅ .......！")
                    
                elif user_input in ['s', 'status']:
                    console.print("[bold blue]📊 .........")
                    check_retry_status()
                    
                elif user_input in ['h', 'help']:
                    console.print("\n[bold cyan]📖 ....:")
                    console.print("  [bold]r/retry[/bold] - .........，......")
                    console.print("  [bold]s/status[/bold] - ....API....")
                    console.print("  [bold]h/help[/bold] - .......")
                    console.print("  [bold]q/quit[/bold] - .......")
                    console.print("\n[dim]💡 ..: ...'..X....'.，..'r'.....")
                    
                elif user_input in ['q', 'quit']:
                    console.print("[yellow]🛑 .........")
                    break
                    
                elif user_input == '':
                    continue
                    
                else:
                    console.print(f"[red]❌ ....: '{user_input}'，.. 'h' ....")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]🛑 .........")
                break
            except EOFError:
                break
                
    # ........
    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()
    return input_thread

# ......
def load_progress():
    """.........."""
    progress_file = current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('completed_files', []))
        except (json.JSONDecodeError, KeyError):
            console.print("[yellow]Warning: ........，.....[/yellow]")
            return set()
    return set()

# ....
def save_progress(completed_files):
    """..........，............"""
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
        # .......
        with open(temp_progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
        
        # ........
        os.replace(temp_progress_file, progress_file)
        
    except Exception as e:
        # ......
        if os.path.exists(temp_progress_file):
            try:
                os.remove(temp_progress_file)
            except:
                pass
        console.print(f"[red]......: {e}[/red]")

# ............（.....）
def get_progress_filename():
    """............."""
    return current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)

# ............
def is_file_in_current_shard(file_path):
    """..................."""
    filename = os.path.basename(file_path)
    # .............
    file_hash = hash(filename)
    return file_hash % SHARD_COUNT == SHARD_INDEX

# ......
def cleanup_temp_files():
    """..........."""
    cleaned_count = 0
    
    # ...........
    progress_file = current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)
    temp_progress_file = progress_file + '.tmp'
    if os.path.exists(temp_progress_file):
        try:
            os.remove(temp_progress_file)
            cleaned_count += 1
        except Exception as e:
            console.print(f"[yellow]..........: {e}[/yellow]")
    
    # ..JSON.......
    if os.path.exists(current_task.data_dir):
        for filename in os.listdir(current_task.data_dir):
            if filename.endswith('.json.tmp'):
                temp_file_path = os.path.join(current_task.data_dir, filename)
                try:
                    os.remove(temp_file_path)
                    cleaned_count += 1
                except Exception as e:
                    console.print(f"[yellow]........ {temp_file_path}: {e}[/yellow]")
    
    if cleaned_count > 0:
        console.print(f"[cyan]... {cleaned_count} .....[/cyan]")

# ...........
def is_file_processed(file_path, data):
    """...........，........."""
    return current_task.is_file_processed(file_path, data)

# ......... JSON ..（......）
async def process_file(file_path, progress: Progress, task_id: TaskID, completed_files: set):
    async with semaphore:
        max_retries = 3  # ......
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # .. JSON ..（..）
                with open(file_path, mode='r', encoding='utf-8') as f:
                    contents = f.read()
                    data = json.loads(contents)
                
                # .........
                if is_file_processed(file_path, data):
                    progress.update(task_id, advance=1, description=f"[green].....: {os.path.basename(file_path)}")
                    # ..............
                    async with completed_files_lock:
                        completed_files.add(file_path)
                    console.print(f"[dim]⏭️  .....，..: {os.path.basename(file_path)}[/dim]")
                    return data
                
                if retry_count == 0:
                    progress.update(task_id, description=f"[blue]...: {os.path.basename(file_path)}")
                    console.print(f"[dim]🚀 .......: {os.path.basename(file_path)}[/dim]")
                else:
                    progress.update(task_id, description=f"[yellow]...({retry_count}/{max_retries}): {os.path.basename(file_path)}")
                    console.print(f"[dim]🔄 ....: {os.path.basename(file_path)} (.{retry_count}...)[/dim]")
                
                # ....（.....）
                console.print(f"[dim]⚙️  ........: {os.path.basename(file_path)}[/dim]")
                updated_data = await process_json_data(file_path, data)

                # ..........，........（..）
                temp_file_path = file_path + '.tmp'
                console.print(f"[dim]💾 ......: {os.path.basename(file_path)}[/dim]")
                try:
                    # ......（..）
                    with open(temp_file_path, mode='w', encoding='utf-8') as f:
                        json_content = json.dumps(updated_data, indent=2, ensure_ascii=False)
                        f.write(json_content)
                        console.print(f"[dim]📝 ........ (..: {len(json_content)} ..)[/dim]")
                    
                    # ........（..）
                    os.replace(temp_file_path, file_path)
                    console.print(f"[dim]🔄 ......: {os.path.basename(file_path)}[/dim]")
                    
                except Exception as write_error:
                    console.print(f"[red]❌ ......: {os.path.basename(file_path)} - {write_error}[/red]")
                    # ......
                    if os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                            console.print(f"[dim]🧹 .......: {os.path.basename(temp_file_path)}[/dim]")
                        except:
                            pass
                    raise write_error
                
                # .....................
                async with completed_files_lock:
                    completed_files.add(file_path)
                    current_completed_count = len(completed_files)
                    # ......（.10.......）
                    if current_completed_count % 10 == 0:
                        save_progress(completed_files)
                        console.print(f"[dim]💾 .......: {current_completed_count} ......[/dim]")
                
                progress.update(task_id, advance=1, description=f"[green]...: {os.path.basename(file_path)}")
                console.print(f"[dim]✅ ......: {os.path.basename(file_path)}[/dim]")
                
                # ....，......
                break
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                if retry_count < max_retries:
                    console.print(f'[yellow]⚠️  ....，.... ({retry_count}/{max_retries}): {os.path.basename(file_path)}[/yellow]')
                    console.print(f'[yellow]....: {error_msg}[/yellow]')
                    import traceback
                    traceback.print_exc()
                    # .........
                    wait_time = 1 * retry_count
                    console.print(f'[dim]⏳ .. {wait_time} .......[/dim]')
                    await asyncio.sleep(wait_time)  # ......
                else:
                    # ........
                    progress.update(task_id, advance=1, description=f"[red]..: {os.path.basename(file_path)}")
                    console.print(f'[red]❌ ....，......... ({max_retries}): {os.path.basename(file_path)}[/red]')
                    console.print(f'[red]....: {error_msg}[/red]')
                    # ....completed_files，..............

# .........
async def process_json_data(file_path, data): 
    """..JSON..，........."""
    return await current_task.process_json_data(file_path, data)

# ...：...... JSON ..
async def main():
    # ...........
    cleanup_temp_files()
    
    # ..........
    input_thread = start_user_input_listener()
    
    # ........
    completed_files = load_progress()
    
    # ....JSON..
    all_files = []
    for filename in os.listdir(current_task.data_dir):
        if filename.endswith('.json'):
            full_path = os.path.join(current_task.data_dir, filename)
            all_files.append(full_path)
    
    # ............
    shard_files = [f for f in all_files if is_file_in_current_shard(f)]
    
    # ..........（..........）
    files_to_process = [f for f in shard_files if f not in completed_files]
    
    total_files = len(all_files)
    shard_total = len(shard_files)
    already_completed = len(completed_files)
    to_process = len(files_to_process)
    
    console.print(f"[bold cyan]....: . {SHARD_INDEX + 1}/{SHARD_COUNT} .")
    console.print(f"[bold cyan].....: {total_files}")
    console.print(f"[bold blue].......: {shard_total}")
    console.print(f"[bold green]...: {already_completed}")
    console.print(f"[bold yellow]...: {to_process}")
    
    if to_process == 0:
        console.print("[bold green]...............！")
        return
    
    console.print(f"\n[bold magenta]💡 ......: {current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)}")
    console.print("[bold magenta]💡 ..: .......，...:")
    console.print("[dim]  - .. 'r' ......（.API......）")
    console.print("[dim]  - .. 's' ........")
    console.print("[dim]  - ...'..X....'.，......'r'....[/dim]")
    
    # .....
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
            description="[cyan].........", 
            total=to_process
        )
        
        # ......
        tasks = []
        for file_path in files_to_process:
            tasks.append(process_file(file_path, progress, task_id, completed_files))
        
        # ......
        await asyncio.gather(*tasks)
        
        # ......
        save_progress(completed_files)
        
        progress.update(task_id, description="[bold green]........！")
        
    console.print(f"[bold green]✅ ......！... {to_process} ...")
    console.print(f"[bold blue]💾 ...... {current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)}")

if __name__ == '__main__':
    import argparse
    
    # .......
    parser = argparse.ArgumentParser(description='........ - ........')
    parser.add_argument('--task', type=str, default=TASK_NAME, 
                       help=f'........，..: {", ".join(list_available_tasks())}')
    parser.add_argument('--reset', action='store_true', help='...........')
    parser.add_argument('--shard-index', type=int, default=0, help='......（.0..）')
    parser.add_argument('--shard-count', type=int, default=1, help='....')
    parser.add_argument('--list-tasks', action='store_true', help='........')
    
    args = parser.parse_args()
    
    # .......
    if args.list_tasks:
        console.print("[bold cyan]📋 ......:")
        for task_name in list_available_tasks():
            task_instance = get_task(task_name)
            console.print(f"  [bold]{task_name}[/bold] - {task_instance.get_task_description()}")
        sys.exit(0)
    
    # .....
    try:
        current_task = get_task(args.task)
        TASK_NAME = args.task
    except ValueError as e:
        console.print(f"[red]..: {e}[/red]")
        sys.exit(1)
    
    # ......
    SHARD_INDEX = args.shard_index
    SHARD_COUNT = args.shard_count
    
    # ......
    if SHARD_INDEX < 0 or SHARD_INDEX >= SHARD_COUNT:
        console.print(f"[red]..: shard-index ({SHARD_INDEX}) ... 0 . {SHARD_COUNT-1} ..[/red]")
        sys.exit(1)
    
    if SHARD_COUNT <= 0:
        console.print(f"[red]..: shard-count ({SHARD_COUNT}) .... 0[/red]")
        sys.exit(1)
    
    # ......
    if args.reset:
        progress_file = current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)
        if os.path.exists(progress_file):
            os.remove(progress_file)
            console.print(f"[yellow]....... {progress_file}[/yellow]")
        else:
            console.print(f"[yellow].... {progress_file} ...[/yellow]")
        sys.exit(0)
    
    console.print(f"[bold cyan]🚀 ..........")
    console.print(f"[bold cyan]📋 ....: {current_task.task_name}")
    console.print(f"[bold cyan]📁 ....: {current_task.data_dir}")
    if SHARD_COUNT > 1:
        console.print(f"[bold cyan]📊 ....: . {SHARD_INDEX + 1}/{SHARD_COUNT} .")
        console.print(f"[dim]....: {current_task.get_progress_filename(SHARD_INDEX, SHARD_COUNT)}[/dim]")
    console.print("[dim]..: .. --reset .............[/dim]")
    console.print("[dim]..: .. --shard-index . --shard-count ............[/dim]")
    console.print("[dim]..: .. --task ...........[/dim]")
    console.print("[dim]..: .. --list-tasks ........[/dim]")
    console.print("[bold yellow]🆕 ...: ......！..................[/bold yellow]")
    
    asyncio.run(main())
