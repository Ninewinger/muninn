import sys, sqlite3, struct
sys.stdout.reconfigure(encoding='utf-8')

# Method 1: Direct connection (what check_dims uses)
conn1 = sqlite3.connect(r"D:\github\muninn\muninn_v3.db")
conn1.row_factory = sqlite3.Row
conn1.enable_load_extension(True)
import sqlite_vec
sqlite_vec.load(conn1)
r1 = conn1.execute("SELECT embedding FROM facet_embeddings LIMIT 1").fetchone()
print(f"Direct: {len(r1['embedding'])} bytes = {len(r1['embedding'])//4} floats")
conn1.close()

# Method 2: get_connection (what hook uses)
sys.path.insert(0, r"D:\github\muninn")
from muninn.db import get_connection
conn2 = get_connection(r"D:\github\muninn\muninn_v3.db")
r2 = conn2.execute("SELECT embedding FROM facet_embeddings LIMIT 1").fetchone()
print(f"get_connection: {len(r2['embedding'])} bytes = {len(r2['embedding'])//4} floats")
conn2.close()
