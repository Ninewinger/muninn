import sys, sqlite3, struct
sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect(r"D:\github\muninn\muninn_v3.db")
conn.row_factory = sqlite3.Row
try:
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    r = conn.execute("SELECT embedding FROM facet_embeddings LIMIT 1").fetchone()
    print(f"vec0 OK, dims: {len(struct.unpack('1024f', r['embedding']))}")
except Exception as e:
    print(f"Error: {e}")
conn.close()
