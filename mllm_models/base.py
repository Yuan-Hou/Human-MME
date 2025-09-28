import time
import threading
import sys
from functools import wraps
import requests


class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, image, question):
        raise NotImplementedError("Subclasses should implement this method")

    def batch_predict(self, images, questions):
        raise NotImplementedError("Subclasses should implement this method")
    
    def concurrency(self):
        return False

def timeout_retry_decorator(max_retries=10, timeout_exceptions=None):
    """
    Timeout failure retry decorator
    
    Args:
        max_retries: Maximum retry attempts, after which it enters user input waiting mode
        timeout_exceptions: Tuple of exception types that should trigger retries
    """
    if timeout_exceptions is None:
        # Default timeout and connection related exceptions
        from openai import APITimeoutError, APIConnectionError
        timeout_exceptions = (
            TimeoutError, 
            requests.exceptions.Timeout, 
            requests.exceptions.ConnectionError,
            ConnectionError,
            APITimeoutError,
            APIConnectionError,
        )
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            
            while True:
                try:
                    return func(*args, **kwargs)
                except timeout_exceptions as e:
                    retry_count += 1
                    print(f"Request failed, retry attempt {retry_count}... Error type: {type(e).__name__}, Error message: {str(e)}")
                    
                    if retry_count >= max_retries:
                        print(f"Retried {max_retries} times, entering waiting mode...")
                        wait_for_user_input()
                        retry_count = 0  # Reset retry count
                        continue
                    
                    # Wait briefly before retry, with increasing wait time based on retry count
                    wait_time = min(retry_count, 5)  # Maximum wait time of 5 seconds
                    time.sleep(wait_time)
                except Exception as e:
                    # For other types of exceptions, print info but don't retry
                    print(f"Non-timeout exception, not retrying: {type(e).__name__}: {str(e)}")
                    raise
                    
        return wrapper
    return decorator


def wait_for_user_input():
    """
    Block and wait for user input, prompting every second
    """
    print("Waiting for user input to continue retrying... (press Enter to continue)")
    
    user_input = None
    input_received = threading.Event()
    
    def get_input():
        nonlocal user_input
        try:
            user_input = input()
            input_received.set()
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
            sys.exit(1)
    
    # Start input thread
    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True
    input_thread.start()
    
    # Prompt every second until user input is received
    while not input_received.is_set():
        time.sleep(1)
        if not input_received.is_set():
            print("Waiting for user input... (press Enter to continue)")
    
    print("User input received, continuing retries...")