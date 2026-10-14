"""Local runs only: cap CPU threads at 4 and lower process priority so the laptop stays usable."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("EMBED_THREADS", "4")
try:
    import psutil

    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
except Exception:
    pass
