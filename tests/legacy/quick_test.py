"""Quick server test."""
import httpx, subprocess, sys, time, os
os.environ["DB_PATH"] = "demo_muninn.db"

from muninn.db import init_db
init_db("demo_muninn.db")

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "muninn.api:app", "--host", "127.0.0.1", "--port", "8901"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
time.sleep(5)

b = "http://127.0.0.1:8901"
r = httpx.get(b + "/")
print("root:", r.status_code, r.text[:200])

r2 = httpx.get(b + "/stats")
print("stats:", r2.status_code, r2.text[:200])

proc.terminate()
proc.wait()
os.unlink("demo_muninn.db")
