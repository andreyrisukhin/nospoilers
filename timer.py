import time

class Timer:
    def __init__(self, message: str):
        self.start_time = None
        self.elapsed_time = None
        self.message = message

    def __enter__(self):
        print(self.message)
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        print(f"Elapsed time: {self.elapsed_time:.4f} seconds")