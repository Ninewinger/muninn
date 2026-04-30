#!/usr/bin/env python3
"""Inspect Muninn database contents."""
import sqlite3
import sqlite_vec
import os

os.chdir('/mnt/d/github/muninn')

for db_name in ['muninn.db', 'muninn_v3.db']:
    print(f'\n=== {db_name} ===')
    if not os.path.exists(db_name):
        print('  NOT FOUND')
        continue
    
    conn = sqlite3.connect(db_name)
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    except:
        pass
    conn.row_factory = sqlite3.Row
    
    # Tables + counts
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    for t in tables:
        name = t['name']
        try:
            count = conn.execute(f'SELECT COUNT(*) FROM [{name}]').fetchone()[0]
            if count > 0:
                print(f'  {name}: {count} rows')
        except:
            pass
    
    # Config
    try:
        rows = conn.execute('SELECT * FROM embedding_config').fetchall()
        for r in rows:
            print(f'  CONFIG: {r["key"]} = {r["value"]}')
    except Exception as e:
        print(f'  Config: {e}')
    
    # Sample events
    try:
        rows = conn.execute('SELECT id, type, substr(content,1,120) as content FROM events ORDER BY id DESC LIMIT 8').fetchall()
        if rows:
            print('  Recent events:')
            for r in rows:
                print(f'    [{r["id"]}] {r["type"]}: {r["content"]}')
        else:
            print('  No events')
    except Exception as e:
        print(f'  Events: {e}')
    
    # Peers
    try:
        rows = conn.execute('SELECT id, name, type FROM peers ORDER BY type, name').fetchall()
        if rows:
            print('  Peers:')
            for r in rows:
                print(f'    {r["id"]} | {r["name"]} ({r["type"]})')
    except Exception as e:
        print(f'  Peers: {e}')
    
    # Memories
    try:
        rows = conn.execute('SELECT id, substr(content,1,100) as content FROM memories LIMIT 5').fetchall()
        if rows:
            print('  Sample memories:')
            for r in rows:
                print(f'    [{r["id"]}] {r["content"]}')
    except Exception as e:
        print(f'  Memories: {e}')
    
    # Activations
    try:
        count = conn.execute('SELECT COUNT(*) FROM activations').fetchone()[0]
        print(f'  Activations: {count}')
    except Exception as e:
        print(f'  Activations: {e}')
    
    conn.close()

print('\nDone.')
