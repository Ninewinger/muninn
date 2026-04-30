#!/usr/bin/env python3
"""Export Muninn DB peers+facets to JSON for benchmarking."""
import sqlite3, sqlite_vec, json, os

os.chdir('/mnt/d/github/muninn')
conn = sqlite3.connect('muninn.db')
conn.enable_load_extension(True)
sqlite_vec.load(conn)

# Get all peers
peers = {}
rows = conn.execute('SELECT id, name, type, domain, description, representation, tags FROM peers').fetchall()
for r in rows:
    peers[r[0]] = {
        'id': r[0], 'name': r[1], 'type': r[2], 'domain': r[3],
        'description': r[4], 'representation': r[5], 'tags': r[6]
    }

# Get all facets
facets = []
rows = conn.execute('''
    SELECT f.id, f.peer_id, f.facet_type, f.text, p.name, p.type
    FROM peer_facets f JOIN peers p ON f.peer_id = p.id
    ORDER BY f.id
''').fetchall()
for r in rows:
    facets.append({
        'id': r[0], 'peer_id': r[1], 'facet_type': r[2],
        'text': r[3], 'peer_name': r[4], 'peer_type': r[5]
    })

data = {'peers': peers, 'facets': facets}
with open('/mnt/d/github/muninn/db_export.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f'Exported {len(peers)} peers, {len(facets)} facets to db_export.json')
for pid, p in peers.items():
    fc = sum(1 for f in facets if f['peer_id'] == pid)
    print(f'  {pid} ({p["name"]}) [{p["type"]}]: {fc} facets')
