import time
import tracemalloc

def benchmark(func, *args):
    tracemalloc.start()
    start = time.time()
    result = func(*args)
    end = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n===== Benchmark =====")
    print(f"Execution Time: {end - start:.2f} sec")
    print(f"Peak Memory: {peak / 10**6:.2f} MB")
    print("=====================\n")

    return result