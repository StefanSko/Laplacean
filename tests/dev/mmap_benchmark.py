import time
import psutil
import numpy as np
import torch
import jax.numpy as jnp
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List
import matplotlib.pyplot as plt

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

def plot_results(all_results: Dict[str, Dict[int, List[BenchmarkResult]]], 
                sizes: List[int], metric: str = 'memory'):
    plt.figure(figsize=(12, 6))
    
    for backend, results in all_results.items():
        if metric == 'memory':
            values = [np.mean([r.peak_memory_mb for r in results[size]]) for size in sizes]
            ylabel = 'Peak Memory Usage (MB)'
        else:
            values = [np.mean([r.execution_time_ms for r in results[size]]) for size in sizes]
            ylabel = 'Execution Time (ms)'
            
        plt.plot(sizes, values, marker='o', label=backend)
    
    plt.xlabel('Matrix Size (N×N)')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs Matrix Size')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f'{metric}_benchmark.png')
    plt.close()

def main():
    # Define matrix sizes to test
    sizes = [1000, 2000, 3000, 4000]
    repetitions = 3
    
    operations = {
        'NumPy (In-Memory)': numpy_in_memory,
        'NumPy (Mmap)': numpy_mmap,
        'PyTorch (In-Memory)': torch_in_memory,
        'PyTorch (Mmap)': torch_mmap,
        'JAX (In-Memory)': jax_in_memory,
        'JAX (Mmap)': jax_mmap,
    }
    
    all_results = {}
    
    for name, operation in operations.items():
        print(f"Benchmarking {name}...")
        results = benchmark_operation(operation, name, sizes, repetitions)
        all_results[name] = results
        
        # Print results for this operation
        for size in sizes:
            mem_avg = np.mean([r.peak_memory_mb for r in results[size]])
            time_avg = np.mean([r.execution_time_ms for r in results[size]])
            print(f"Size {size}x{size}:")
            print(f"  Average Memory Usage: {mem_avg:.2f} MB")
            print(f"  Average Execution Time: {time_avg:.2f} ms")
        print()
    
    # Generate plots
    plot_results(all_results, sizes, 'memory')
    plot_results(all_results, sizes, 'time')

if __name__ == "__main__":
    main()