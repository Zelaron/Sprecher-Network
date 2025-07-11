"""
Modernized batch sweep runner for Sprecher Network experiments.

This script can run multiple experiments in parallel, allows selecting specific
sweeps to run, and provides a rich, dynamic progress display.

Usage examples:
  # List all available sweeps
  python sn_sweeps.py --list

  # Run all sweeps in parallel using all available CPU cores
  python sn_sweeps.py

  # Run two specific sweeps sequentially
  python sn_sweeps.py --sweeps toy_1d_poly feynman_uv --parallel 1

  # Run all 2D sweeps in parallel
  python sn_sweeps.py --sweeps toy_2d toy_2d_vector special_bessel poisson
"""

import subprocess
import sys
import argparse
import time
import os
import re
import threading
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text
from rich.live import Live
from rich.table import Table, Column

# --- Sweep Configurations ---
# Using a dictionary for named, readable, and selectable sweeps.
SWEEPS = {
    # (dataset, arch, phi_knots, Phi_knots, epochs, extra_args)
    "toy_1d_poly": {
        "dataset": "toy_1d_poly", "arch": "15,15,15,15", "phi_knots": 20, "Phi_knots": 20, "epochs": 3000
    },
    "toy_1d_complex": {
        "dataset": "toy_1d_complex", "arch": "10", "phi_knots": 100, "Phi_knots": 100, "epochs": 4000
    },
    "toy_2d": {
        "dataset": "toy_2d", "arch": "10,10,10", "phi_knots": 100, "Phi_knots": 100, "epochs": 4000
    },
    "toy_2d_vector": {
        "dataset": "toy_2d_vector", "arch": "15,15", "phi_knots": 100, "Phi_knots": 100, "epochs": 2000
    },
    "toy_100d": {
        "dataset": "toy_100d", "arch": "100", "phi_knots": 15, "Phi_knots": 15, "epochs": 100,
        "extra_args": ["--norm_first"]  # Enable normalization on first block for high-dim problem
    },
    "special_bessel": {
        "dataset": "special_bessel", "arch": "10,10,10", "phi_knots": 50, "Phi_knots": 50, "epochs": 4000
    },
    "feynman_uv": {
        "dataset": "feynman_uv", "arch": "15,15", "phi_knots": 100, "Phi_knots": 100, "epochs": 8000
    },
    "poisson": {
        "dataset": "poisson", "arch": "15,15", "phi_knots": 25, "Phi_knots": 25, "epochs": 4000
    },
    "toy_4d_to_5d": {
        "dataset": "toy_4d_to_5d", "arch": "20,20,20", "phi_knots": 50, "Phi_knots": 50, "epochs": 2000
    },
}

# Rich console for better printing
console = Console()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Sprecher Network experiment sweeps.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--sweeps", nargs="*", default=list(SWEEPS.keys()),
        help="Names of the sweeps to run (default: all)."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available sweep names and exit."
    )
    parser.add_argument(
        "--parallel", type=int, default=os.cpu_count(),
        help="Number of sweeps to run in parallel (default: all available CPU cores)."
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop all sweeps immediately if one fails."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug output to see raw subprocess output."
    )
    return parser.parse_args()


def parse_progress(line, total_epochs):
    """Parse progress information from training output."""
    # Try to parse tqdm format: "Training Network:  25%|████████▌  | 1000/4000 [00:45<02:15, 22.15it/s]"
    # Handle both with and without Unicode progress bars
    tqdm_match = re.search(r'(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[', line)
    if tqdm_match:
        percentage = int(tqdm_match.group(1))
        current = int(tqdm_match.group(2))
        total = int(tqdm_match.group(3))
        return percentage, current, total
    
    # Try to parse simpler tqdm format without percentage: "| 1000/4000 ["
    tqdm_simple = re.search(r'\|\s*(\d+)/(\d+)\s*\[', line)
    if tqdm_simple:
        current = int(tqdm_simple.group(1))
        total = int(tqdm_simple.group(2))
        percentage = int((current / total) * 100)
        return percentage, current, total
    
    # Try to parse epoch print format: "Epoch 1000: Loss = ..."
    epoch_match = re.search(r'Epoch\s+(\d+):', line)
    if epoch_match and total_epochs:
        current = int(epoch_match.group(1))
        percentage = min(int((current / total_epochs) * 100), 100)
        return percentage, current, total_epochs
    
    return None, None, None


def parse_loss(line):
    """Parse loss information from training output."""
    # Look for patterns like "loss=1.23e-04" (tqdm postfix)
    loss_postfix = re.search(r'loss=([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)', line)
    if loss_postfix:
        return float(loss_postfix.group(1))
    
    # Look for patterns like "Loss = 0.001234" (epoch print statements)
    loss_match = re.search(r'Loss\s*=\s*([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)', line)
    if loss_match:
        return float(loss_match.group(1))
    
    return None


def run_experiment_with_progress(sweep_name, config, progress_dict, debug=False):
    """Worker function to run a single experiment with progress tracking."""
    cmd = [
        sys.executable, "sn_experiments.py",
        "--dataset", config["dataset"],
        "--arch", config["arch"],
        "--phi_knots", str(config["phi_knots"]),
        "--Phi_knots", str(config["Phi_knots"]),
        "--epochs", str(config["epochs"]),
        "--seed", "0",
        "--save_plots",
        "--no_show"
    ]
    
    # Add any extra arguments for this sweep
    if "extra_args" in config:
        cmd.extend(config["extra_args"])
    
    # Set debug flag in progress dict
    if debug:
        progress_dict._debug = True
    
    total_epochs = config["epochs"]
    stdout_lines = []
    stderr_lines = []
    
    try:
        # Start process with pipes
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered for real-time output
            universal_newlines=True
        )
        
        # Function to read from pipe
        def read_pipe(pipe, lines_list, is_stderr=False):
            for line in iter(pipe.readline, ''):
                if line:
                    lines_list.append(line)
                    # Parse both stdout and stderr for progress (tqdm often goes to stderr)
                    # Debug output if enabled
                    if hasattr(progress_dict, '_debug') and progress_dict._debug:
                        print(f"[DEBUG {sweep_name}] {line.strip()}")
                    
                    # Parse progress
                    percentage, current, total = parse_progress(line, total_epochs)
                    if percentage is not None:
                        progress_dict[sweep_name] = {
                            'percentage': percentage,
                            'current': current,
                            'total': total,
                            'status': 'running'
                        }
                    
                    # Parse loss
                    loss = parse_loss(line)
                    if loss is not None and sweep_name in progress_dict:
                        progress_dict[sweep_name]['loss'] = loss
            pipe.close()
        
        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(target=read_pipe, args=(process.stdout, stdout_lines, False))
        stderr_thread = threading.Thread(target=read_pipe, args=(process.stderr, stderr_lines, True))
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Wait for threads to finish reading
        stdout_thread.join()
        stderr_thread.join()
        
        # Update final status
        progress_dict[sweep_name] = {
            'percentage': 100,
            'current': total_epochs,
            'total': total_epochs,
            'status': 'success' if return_code == 0 else 'failed',
            'return_code': return_code
        }
        
    except Exception as e:
        return_code = -1
        stderr_lines.append(str(e))
        progress_dict[sweep_name] = {
            'percentage': 0,
            'current': 0,
            'total': total_epochs,
            'status': 'crashed',
            'return_code': -1
        }
    
    return sweep_name, return_code, ''.join(stdout_lines), ''.join(stderr_lines)


def main():
    """Run all specified sweep configurations with a rich progress display."""
    args = parse_args()

    if args.list:
        console.print("[bold blue]Available sweeps:[/bold blue]")
        for name in SWEEPS:
            console.print(f"  - [yellow]{name}[/yellow]")
        return

    sweeps_to_run = {name: SWEEPS[name] for name in args.sweeps if name in SWEEPS}
    if not sweeps_to_run:
        console.print("[bold red]Error: No valid sweeps selected.[/bold red]")
        return
    
    num_sweeps = len(sweeps_to_run)
    max_workers = min(args.parallel, num_sweeps)
    
    console.print(f"[bold blue]Starting Sprecher Network sweep experiments...[/bold blue]")
    console.print(f"Running {num_sweeps} sweep(s) in parallel on {max_workers} worker(s).\n")

    start_time = time.time()
    results = []

    # Create a shared dictionary for progress updates
    with Manager() as manager:
        progress_dict = manager.dict()
        
        progress_columns = [
            TextColumn("[progress.description]{task.description}", table_column=Column(width=30)),
            BarColumn(bar_width=None, table_column=Column(ratio=2)),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.fields[epoch_info]}", table_column=Column(width=15)),
            TextColumn("{task.fields[loss_info]}", table_column=Column(width=18)),
            TimeElapsedColumn(),
        ]

        with Progress(*progress_columns, console=console, refresh_per_second=4, expand=True) as progress:
            overall_task = progress.add_task("[bold]Overall Progress", total=num_sweeps, epoch_info="", loss_info="")
            
            # Create tasks for each sweep
            task_ids = {}
            for name, config in sweeps_to_run.items():
                task_id = progress.add_task(
                    f"[cyan]{name}[/cyan]",
                    total=100,  # percentage based
                    epoch_info="[dim]waiting...[/dim]",
                    loss_info="",
                    visible=True
                )
                task_ids[name] = task_id
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_experiment_with_progress, name, config, progress_dict, args.debug): name
                    for name, config in sweeps_to_run.items()
                }
                
                # Progress update thread
                stop_updates = threading.Event()
                
                def update_progress_bars():
                    while not stop_updates.is_set():
                        for name in sweeps_to_run:
                            if name in progress_dict:
                                info = dict(progress_dict[name])
                                task_id = task_ids[name]
                                
                                # Update progress
                                progress.update(task_id, completed=info.get('percentage', 0))
                                
                                # Update epoch info
                                if info.get('status') == 'running':
                                    current = info.get('current', 0)
                                    total = info.get('total', 1)
                                    epoch_text = f"[yellow]Epoch {current}/{total}[/yellow]"
                                    progress.update(task_id, epoch_info=epoch_text)
                                    
                                    # Update loss info
                                    if 'loss' in info:
                                        loss_text = f"[dim]loss: {info['loss']:.6f}[/dim]"
                                        progress.update(task_id, loss_info=loss_text)
                                
                                elif info.get('status') == 'success':
                                    progress.update(
                                        task_id,
                                        description=f"[green][OK][/green] {name}",
                                        epoch_info="[green]completed[/green]",
                                        completed=100
                                    )
                                elif info.get('status') == 'failed':
                                    progress.update(
                                        task_id,
                                        description=f"[red][FAIL][/red] {name}",
                                        epoch_info="[red]failed[/red]",
                                        completed=100
                                    )
                                elif info.get('status') == 'crashed':
                                    progress.update(
                                        task_id,
                                        description=f"[red][WARN][/red] {name}",
                                        epoch_info="[red]crashed[/red]",
                                        completed=100
                                    )
                        
                        time.sleep(0.25)  # Update 4 times per second
                
                # Start progress update thread
                update_thread = threading.Thread(target=update_progress_bars)
                update_thread.start()
                
                # Wait for experiments to complete
                completed = 0
                for future in as_completed(futures):
                    name = futures[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        _, return_code, _, _ = result
                        
                        if return_code != 0 and args.fail_fast:
                            console.print("\n[bold red]--fail-fast enabled. Terminating remaining sweeps.[/bold red]")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                    
                    except Exception as e:
                        results.append((name, -1, "", str(e)))
                    
                    completed += 1
                    progress.update(overall_task, advance=1)
                
                # Give the update thread a moment to process final updates
                time.sleep(0.5)
                
                # Stop progress updates
                stop_updates.set()
                update_thread.join()
                
                # Final cleanup: ensure all completed experiments show "completed" status
                for name in sweeps_to_run:
                    if name in progress_dict:
                        info = dict(progress_dict[name])
                        task_id = task_ids[name]
                        
                        if info.get('status') == 'success':
                            progress.update(
                                task_id,
                                description=f"[green][OK][/green] {name}",
                                epoch_info="[green]completed[/green]",
                                completed=100
                            )
                        elif info.get('status') == 'failed':
                            progress.update(
                                task_id,
                                description=f"[red][FAIL][/red] {name}",
                                epoch_info="[red]failed[/red]",
                                completed=100
                            )
                        elif info.get('status') == 'crashed':
                            progress.update(
                                task_id,
                                description=f"[red][WARN][/red] {name}",
                                epoch_info="[red]crashed[/red]",
                                completed=100
                            )

    end_time = time.time()
    
    # --- Final Summary ---
    console.print("\n")
    
    # Create summary table
    summary_table = Table(title="Sweep Execution Summary", box=None)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")
    
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)
    time_str = f"{int(minutes)}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"
    
    success_count = sum(1 for _, rc, _, _ in results if rc == 0)
    failed_count = len(results) - success_count
    
    summary_table.add_row("Total Time", f"[bold]{time_str}[/bold]")
    summary_table.add_row("Total Sweeps", str(num_sweeps))
    summary_table.add_row("[green]Successful[/green]", f"[green]{success_count}[/green]")
    summary_table.add_row("[red]Failed[/red]", f"[red]{failed_count}[/red]")
    summary_table.add_row("Success Rate", f"{(success_count/num_sweeps)*100:.1f}%")
    
    console.print(summary_table)
    
    # Show details for failed sweeps
    failed_results = [res for res in results if res[1] != 0]
    if failed_results:
        console.print("\n[bold red]Failed Sweep Details:[/bold red]\n")
        
        for name, return_code, stdout, stderr in failed_results:
            error_panel = Table(title=f"[yellow]{name}[/yellow] (Exit Code: {return_code})", box=None, padding=(0, 1))
            error_panel.add_column("Output", no_wrap=False)
            
            # Extract relevant error info
            error_lines = []
            if stderr.strip():
                # Look for the most relevant error lines
                stderr_lines = stderr.strip().split('\n')
                for line in stderr_lines[-10:]:  # Last 10 lines usually contain the error
                    if 'Error' in line or 'error' in line or 'Traceback' in line:
                        error_lines.append(f"[red]{line}[/red]")
                    else:
                        error_lines.append(f"[dim]{line}[/dim]")
            
            if error_lines:
                error_panel.add_row('\n'.join(error_lines))
                console.print(error_panel)
                console.print()
    
    console.print("\n[bold green][OK][/bold green] All sweeps completed!")
    console.print("[dim]Check the 'plots' directory for generated visualizations.[/dim]")

if __name__ == "__main__":
    main()