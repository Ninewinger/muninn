import sys, os
sys.path.insert(0, 'D:/github/muninn')
os.environ['HF_HOME'] = 'D:/hf_cache'
os.environ['HF_HUB_OFFLINE'] = '1'

from muninn.db import get_connection
from muninn.router_v2 import route

DB = 'D:/github/muninn/muninn.db'

# Check DB state
conn = get_connection(DB)
peers = conn.execute("SELECT name, confidence FROM peers").fetchall()
facets = conn.execute("SELECT COUNT(*) FROM peer_facets").fetchone()[0]
conns = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
print(f"DB: {len(peers)} peers, {facets} facetas, {conns} conexiones")
conn.close()

# Test routing
queries = [
    "escribir codigo python",
    "como me llamo",
    "quien soy",
    "que herramientas tengo",
    "recordar tomar agua",
    "diseñar combate",
    "mi abuela",
    "ir al gym",
    "que skills tienes",
    "como funciona nanobot",
    "miedo a la muerte",
]

print(f"\n{'Query':35s} {'Peer':20s} {'Score':>6s} {'Faceta':>12s}")
print("-" * 80)
for q in queries:
    r = route(q, db_path=DB)
    if r:
        top = r[0]
        print(f"  {q:33s} {top['peer_name']:20s} {top['total_score']:6.3f} {top['facet_type']:>12s}")
    else:
        print(f"  {q:33s} {'SIN ACTIVACION':20s}")
