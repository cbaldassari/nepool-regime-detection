import pandas as pd
base = r"C:/Users/crist/OneDrive - Università degli studi della Tuscia di Viterbo/Documenti/Papers/paper/LMP/cluster2/nepool-regime-detection-main/results"

pre = pd.read_parquet(f"{base}/preprocessed.parquet")
print("=== preprocessed.parquet ===")
print("columns:", list(pre.columns))
print("rows:", len(pre))
print("has arcsinh_lmp:", "arcsinh_lmp" in pre.columns)
print("has log_lmp:", "log_lmp" in pre.columns)

emb = pd.read_parquet(f"{base}/embeddings.parquet")
print("\n=== embeddings.parquet ===")
print("shape:", emb.shape)
print("columns[:4]:", list(emb.columns)[:4])

reg = pd.read_parquet(f"{base}/regimes.parquet")
print("\n=== regimes.parquet ===")
print("columns:", list(reg.columns))
print("shape:", reg.shape)
print("regime dist:", dict(reg["regime"].value_counts().sort_index()))
