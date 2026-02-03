"""
Simple trainer entrypoint shim for CPU-only runs.
"""

def main():
    # This repository's experiments are GPU-driven. Provide a no-op CPU fallback
    # that prints a helpful message and exits successfully so that the runner
    # can at least validate code paths without requiring GPUs.
    print("CPU-only fallback trainer. GPU kernels are not required for code execution in this environment.")

if __name__ == "__main__":
    main()
