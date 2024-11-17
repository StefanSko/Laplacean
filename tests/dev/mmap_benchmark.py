import time
import psutil
import numpy as np
import torch
import jax.numpy as jnp
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@dataclass
class BenchmarkResult:
    peak_memory_mb: float
    execution_time_ms: float
    
def measure_memory_and_time(operation: Callable, size: int) -> BenchmarkResult:
    process = psutil.Process()
    start_mem = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()
    
    operation(size)
    
    end_time = time.time()
    end_mem = process.memory_info().rss / (1024 * 1024)
    
    return BenchmarkResult(
        peak_memory_mb=end_mem - start_mem,
        execution_time_ms=(end_time - start_time) * 1000
    )

def benchmark_operation(operation: Callable, name: str, sizes: List[int], 
                       repetitions: int = 3) -> Dict[int, List[BenchmarkResult]]:
    results = {}
    
    for size in sizes:
        size_results = []
        for _ in range(repetitions):
            result = measure_memory_and_time(operation, size)
            size_results.append(result)
        results[size] = size_results
        
    return results

# Define operations for each backend
def numpy_in_memory(size: int):
    arr = np.random.random((size, size))
    result = arr @ arr.T
    return result

def numpy_mmap(size: int):

    mmap_path = Path("numpy_bench.mmap")
    
    # Create arrays
    arr = np.memmap(mmap_path, dtype='float64', mode='w+', shape=(size, size))
    arr[:] = np.random.random((size, size))
    arr.flush()
    
    # Perform operation
    result = arr @ arr.T
    
    # Cleanup
    del arr
    
    if mmap_path.exists():
        mmap_path.unlink()
    
    return result

def torch_in_memory(size: int):
    arr = torch.rand(size, size)
    result = arr @ arr.T
    return result

def torch_mmap(size: int):
    
    mmap_path = Path("torch_bench.mmap")
    
    # Create using numpy memmap and convert to torch
    arr_np = np.memmap(mmap_path, dtype='float64', mode='w+', shape=(size, size))
    arr_np[:] = np.random.random((size, size))
    arr_np.flush()
    
    arr = torch.from_numpy(arr_np)
    result = arr @ arr.T
    
    # Cleanup
    del arr_np, arr
    if mmap_path.exists():
        mmap_path.unlink()
    
    return result

def jax_in_memory(size: int):
    arr = jnp.array(np.random.random((size, size)))
    result = arr @ arr.T
    return result

def jax_mmap(size: int):
    
    mmap_path = Path("jax_bench.mmap")
    
    # Create using numpy memmap and convert to jax
    arr_np = np.memmap(mmap_path, dtype='float64', mode='w+', shape=(size, size))
    arr_np[:] = np.random.random((size, size))
    arr_np.flush()
    
    arr = jnp.array(arr_np)
    result = arr @ arr.T
    
    # Cleanup
    del arr_np, arr
    if mmap_path.exists():
        mmap_path.unlink()
    
    return result

def run_experiment_suite(operations: Dict[str, Callable], 
                        sizes: List[int], 
                        n_runs: int = 15) -> pd.DataFrame:
    """Run benchmarks N times and collect results in a DataFrame."""
    results_data = []
    
    for run_id in range(n_runs):
        print(f"\nRun {run_id + 1}/{n_runs}")
        
        for name, operation in operations.items():
            print(f"  Benchmarking {name}...")
            
            for size in sizes:
                print(f"    Size {size}x{size}")
                
                # Force garbage collection before measurement
                import gc
                gc.collect()
                
                process = psutil.Process()
                start_mem = process.memory_info().rss / (1024 * 1024)
                peak_mem = start_mem
                start_time = time.time()
                
                # Run operation
                operation(size)
                
                # Get peak memory
                current_mem = process.memory_info().rss / (1024 * 1024)
                peak_mem = max(peak_mem, current_mem)
                
                end_time = time.time()
                
                # Calculate memory usage as peak - baseline
                memory_used = peak_mem - start_mem
                
                results_data.append({
                    'run_id': run_id,
                    'backend': name,
                    'size': size,
                    'memory_mb': max(0, memory_used),  # Ensure non-negative
                    'time_ms': (end_time - start_time) * 1000
                })
                
                # Additional cleanup
                gc.collect()
    
    return pd.DataFrame(results_data)

def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for each backend and size combination."""
    stats_df = df.groupby(['backend', 'size']).agg({
        'memory_mb': ['mean', 'std', 'min', 'max'],
        'time_ms': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    return stats_df

def plot_enhanced_results(df: pd.DataFrame, metric: str = 'memory'):
    """Create enhanced plots with error bars and distributions."""
    plot_dir = Path("plots/")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Line plot with confidence intervals
    metric_col = 'memory_mb' if metric == 'memory' else 'time_ms'
    ylabel = 'Peak Memory Usage (MB)' if metric == 'memory' else 'Execution Time (ms)'
    
    sns.lineplot(data=df, x='size', y=metric_col, hue='backend',
                ci=95, err_style='band', ax=ax1)
    ax1.set_title(f'{ylabel} vs Matrix Size (with 95% CI)')
    ax1.set_xlabel('Matrix Size (N×N)')
    ax1.set_ylabel(ylabel)
    
    # Plot 2: Box plot for distribution
    sns.boxplot(data=df, x='size', y=metric_col, hue='backend', ax=ax2)
    ax2.set_title(f'Distribution of {ylabel} by Matrix Size')
    ax2.set_xlabel('Matrix Size (N×N)')
    ax2.set_ylabel(ylabel)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(plot_dir / f'{metric}_enhanced_benchmark.png')
    plt.close()
    
    # Create violin plots separately
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='size', y=metric_col, hue='backend')
    plt.title(f'Distribution of {ylabel} by Matrix Size')
    plt.xlabel('Matrix Size (N×N)')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_dir / f'{metric}_violin_plot.png')
    plt.close()

def main():
    sizes = [4000, 6000, 8000, 10000]
    n_runs = 20  # Single parameter for number of repetitions
    
    operations = {
        'PyTorch (In-Memory)': torch_in_memory,
        'PyTorch (Mmap)': torch_mmap,
        'JAX (In-Memory)': jax_in_memory,
        'JAX (Mmap)': jax_mmap,
    }
    
    # Run experiments with single N
    results_df = run_experiment_suite(operations, sizes, n_runs)
    
    # Calculate and print statistics
    memory_stats = calculate_statistics(results_df)
    print("\nMemory Usage Statistics (MB):")
    print(memory_stats['memory_mb'])
    
    print("\nExecution Time Statistics (ms):")
    print(memory_stats['time_ms'])
    
    
    # Generate enhanced plots
    plot_enhanced_results(results_df, 'memory')
    plot_enhanced_results(results_df, 'time')

if __name__ == "__main__":
    main()