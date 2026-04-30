#!/usr/bin/env python3
"""Check actual embedding dimensions in Muninn DB."""
import sqlite3
import sqlite_vec
import struct

conn = sqlite3.connect('muninn.db')
conn.enable_load_extension(True)
sqlite_vec.load(conn)

# Get a sample facet embedding and check its dimension
row = conn.execute('SELECT peer_id, embedding FROM facet_embeddings LIMIT 1').fetchone()
if row:
    emb = row[1]
    dim = len(struct.unpack(f'<{len(emb)//4}f', emb))
    print(f'Actual embedding dimension in DB: {dim}')
    print(f'Bytes: {len(emb)}, floats: {dim}')

# Check facet count per peer
rows = conn.execute('''
    SELECT p.id, p.name, p.type, COUNT(f.id) as facet_count 
    FROM peers p LEFT JOIN peer_facets f ON p.id = f.peer_id 
    GROUP BY p.id ORDER BY p.type, p.name
''').fetchall()
print()
print('Peer facets:')
for r in rows:
    print(f'  {r[0]} ({r[1]}) [{r[2]}]: {r[3]} facets')

# Sample facets content
print()
print('Sample facets per peer:')
rows = conn.execute('''
    SELECT p.name, f.facet_type, f.text 
    FROM peer_facets f JOIN peers p ON f.peer_id = p.id
    LIMIT 20
''').fetchall()
for r in rows:
    print(f'  {r[0]} [{r[1]}]: {r[2][:100]}')

conn.close()
print('\nDone.')
