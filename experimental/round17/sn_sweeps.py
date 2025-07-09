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
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

# --- Sweep Configurations ---
# Using a dictionary for named, readable, and selectable sweeps.
SWEEPS = {
    # (dataset, arch, phi_knots, Phi_knots, epochs)
    "toy_1d_poly": {
        "dataset": "toy_1d_poly", "arch": "15,15", "phi_knots": 100, "Phi_knots": 100, "epochs": 4000
    },
    "toy_1d_complex": {
        "dataset": "toy_1d_complex", "arch": "15,15", "phi_knots": 100, "Phi_knots": 100, "epochs": 4000
    },
    "toy_2d": {
        "dataset": "toy_2d", "arch": "10,10,10,10", "phi_knots": 100, "Phi_knots": 100, "epochs": 4000
    },
    "toy_2d_vector": {
        "dataset": "toy_2d_vector", "arch": "15,15", "phi_knots": 100, "Phi_knots": 100, "epochs": 1500
    },
    "toy_100d": {
        "dataset": "toy_100d", "arch": "100", "phi_knots": 15, "Phi_knots": 15, "epochs": 100
    },
    "special_bessel": {
        "dataset": "special_bessel", "arch": "2,2,2", "phi_knots": 50, "Phi_knots": 50, "epochs": 12000
    },
    "feynman_uv": {
        "dataset": "feynman_uv", "arch": "2,2,2", "phi_knots": 100, "Phi_knots": 100, "epochs": 12000
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
    return parser.parse_args()


def run_experiment(sweep_name, config):
    """Worker function to run a single experiment in a separate process."""
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
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return_code = process.returncode
        stdout = process.stdout
        stderr = process.stderr
    except Exception as e:
        return_code = -1
        stdout = ""
        stderr = str(e)
    
    return sweep_name, return_code, stdout, stderr


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

    progress_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        SpinnerColumn(),
        TimeElapsedColumn(),
    ]

    with Progress(*progress_columns, console=console) as progress:
        overall_task = progress.add_task("[bold]Total Progress", total=num_sweeps)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_experiment, name, config): (
                    name,
                    progress.add_task(f"Sweep '{name}'", total=1, visible=True, start=False)
                )
                for name, config in sweeps_to_run.items()
            }

            for future in as_completed(futures):
                name, task_id = futures[future]
                progress.start_task(task_id)
                
                try:
                    result = future.result()
                    results.append(result)
                    _, return_code, _, _ = result

                    if return_code == 0:
                        status = Text("SUCCESS", style="bold green")
                    else:
                        status = Text("FAILED", style="bold red")

                    progress.update(task_id, completed=1, description=f"Sweep '{name}': {status}")
                    progress.stop_task(task_id)
                    
                    if return_code != 0 and args.fail_fast:
                        console.print("\n[bold red]--fail-fast enabled. Terminating remaining sweeps.[/bold red]")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                except Exception as e:
                    results.append((name, -1, "", str(e)))
                    status = Text("CRASHED", style="bold red")
                    progress.update(task_id, completed=1, description=f"Sweep '{name}': {status}")
                    progress.stop_task(task_id)

                progress.update(overall_task, advance=1)

    end_time = time.time()
    
    # --- Final Summary ---
    console.print("\n" + "=" * 60)
    console.print("[bold blue]Sweep Summary[/bold blue]")
    console.print("=" * 60)
    console.print(f"Total time elapsed: {end_time - start_time:.2f} seconds")

    success_count = sum(1 for _, rc, _, _ in results if rc == 0)
    failed_results = [res for res in results if res[1] != 0]

    console.print(f"  [bold green]Successful[/bold green]: {success_count}/{num_sweeps}")
    console.print(f"  [bold red]Failed[/bold red]:     {len(failed_results)}/{num_sweeps}")
    
    if failed_results:
        console.print("\n[bold red]Details for Failed Sweeps:[/bold red]")
        for name, return_code, stdout, stderr in failed_results:
            console.print(f"\n--- [yellow]Sweep: {name}[/yellow] (Exit Code: {return_code}) ---")
            if stdout.strip():
                console.print("[bold]STDOUT:[/bold]")
                console.print(stdout.strip(), style="dim")
            if stderr.strip():
                console.print("[bold]STDERR:[/bold]")
                console.print(stderr.strip(), style="bright_red")
    
    console.print("\nAll sweeps completed!")
    console.print("Check the 'plots' directory for generated figures.")

if __name__ == "__main__":
    main()