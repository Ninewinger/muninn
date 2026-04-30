import sys, sqlite3
sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect(r"D:\github\muninn\muninn_v3.db")
rows = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' OR type='view'").fetchall()
for r in rows:
    if r[0]:
        print(r[0][:120])
        print()
conn.close()
