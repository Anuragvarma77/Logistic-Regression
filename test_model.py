import pickle, joblib, numpy as np, pathlib

pkl = pathlib.Path("model.pkl")
assert pkl.exists(), f"{pkl.resolve()} not found"

# try joblib first
try:
    obj = joblib.load(pkl)
except Exception:
    obj = pickle.load(pkl.open("rb"))

print("Loaded type:", type(obj))
