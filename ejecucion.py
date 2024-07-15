import pyautogui

import time
import threading


memoria_accion = { '0': ['space', 1], 
                   '3': ['left', 0.5],
                   '2': ['right', 0.5]}

def debounce(default_wait_time):
    def decorator(func):
        last_called = [0]
        lock = threading.Lock()
        def wrapped(*args, wait_time=None, **kwargs):
            if wait_time is None:
                wait_time = default_wait_time
            with lock:
                now = time.time()
                time_since_last_call = now - last_called[0]
                if time_since_last_call >= wait_time:
                    last_called[0] = now
                    func(*args, **kwargs)
                    return 0
                else:
                    remaining_time = wait_time - time_since_last_call
                    return remaining_time

        return wrapped
    return decorator


@debounce(0)
def ejecutar(accion):
    print("EJECUTAR")
    # pyautogui.press(accion)
    pyautogui.keyDown(accion)
    time.sleep(0.1)
    pyautogui.keyUp(accion)

def controlador_accion(sign):
    sign_string = str(sign)
    accion = memoria_accion.get(sign_string)
    if accion:
        print(accion[1])
        ejecutar(accion[0], wait_time=accion[1])
        
    
    