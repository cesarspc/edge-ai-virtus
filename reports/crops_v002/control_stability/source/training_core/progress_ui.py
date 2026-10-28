"""Console progress and heartbeats, including slow dependency imports."""
from contextlib import contextmanager
import os
import sys
import threading
import time


def enabled():
    setting=os.environ.get("VIRTUS_PROGRESS", "auto")
    return setting=="1" or (setting!="0" and sys.stdout.isatty())


def bar(*args, **kwargs):
    from tqdm import tqdm
    kwargs.setdefault("disable", not enabled())
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("ascii", True)
    kwargs.setdefault("mininterval", .5)
    return tqdm(*args, **kwargs)


@contextmanager
def stage(description, active=True):
    """An indeterminate stage: report elapsed time without inventing an ETA."""
    if not active:
        yield
        return
    start=time.monotonic()
    finished=threading.Event()
    print(f"[INICIO] {description}", flush=True)
    def heartbeat():
        while not finished.wait(5):
            print(f"  {description}... {int(time.monotonic()-start)} s", flush=True)
    worker=threading.Thread(target=heartbeat,daemon=True)
    worker.start()
    try:
        yield
    except BaseException:
        print(f"[INTERRUMPIDO] {description}",flush=True)
        raise
    else:
        print(f"[LISTO] {description} ({time.monotonic()-start:.1f} s)",flush=True)
    finally:
        finished.set()
        worker.join(timeout=1)


def ready_batches(loader, description):
    """Show a heartbeat while Windows starts workers and prepares the first batch."""
    with stage(description):
        iterator = iter(loader)
        first = next(iterator, None)
    if first is not None:
        yield first
    yield from iterator
