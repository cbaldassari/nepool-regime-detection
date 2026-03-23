"""
ray_find_project.py
===================
Scopre dove si trova il progetto sui nodi worker del cluster Ray.

Lancia un task Ray su ogni nodo disponibile e cerca step01_preprocessing.py
nelle posizioni più comuni (home, /opt, /mnt, /data, …).

Uso
---
  python ray_find_project.py
"""

import ray
import socket

RAY_ADDRESS = "ray://10.4.4.7:10001"

# ── task Ray ────────────────────────────────────────────────────────────────

@ray.remote
def find_project(marker: str) -> dict:
    """Cerca il file marker nelle posizioni comuni e restituisce il path trovato."""
    import os
    import socket as _socket
    from pathlib import Path

    host = _socket.gethostname()
    cwd  = os.getcwd()
    home = str(Path.home())

    # Cerca il file marker nelle posizioni comuni
    search_roots = [
        home,
        "/opt",
        "/mnt",
        "/data",
        "/workspace",
        "/project",
        "/srv",
        "/tmp",
    ]

    found = []
    for root in search_roots:
        try:
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                if marker in filenames:
                    found.append(dirpath)
                # limita la profondità per non impiegarci troppo
                depth = dirpath[len(root):].count(os.sep)
                if depth >= 5:
                    dirnames.clear()
        except PermissionError:
            continue

    return {
        "host": host,
        "cwd":  cwd,
        "home": home,
        "found": found,
    }


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Connessione a {RAY_ADDRESS}...")
    try:
        ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)
    except ConnectionError:
        print("  cluster non trovato — modalita' locale")
        ray.init(ignore_reinit_error=True)

    nodes = [n for n in ray.nodes() if n.get("Alive")]
    print(f"Nodi attivi: {len(nodes)}\n")

    # Lancia un task per nodo (usando scheduling spread con NodeAffinitySchedulingStrategy
    # non è necessario — bastano N task con num_cpus=1 e Ray li distribuisce)
    n_tasks = max(len(nodes), 1)
    futures = [
        find_project.options(num_cpus=1).remote("step01_preprocessing.py")
        for _ in range(n_tasks)
    ]

    results = ray.get(futures)

    seen_hosts = set()
    for r in results:
        h = r["host"]
        if h in seen_hosts:
            continue
        seen_hosts.add(h)
        print(f"Host : {h}")
        print(f"  cwd  : {r['cwd']}")
        print(f"  home : {r['home']}")
        if r["found"]:
            print(f"  TROVATO in:")
            for p in r["found"]:
                print(f"    {p}")
        else:
            print("  NON TROVATO nei path di ricerca")
        print()

    ray.shutdown()


if __name__ == "__main__":
    main()
