# check_bm25.py
import os, pickle, sys, traceback

BM25_PATH = "bm25_retriever.pkl"

print("Path exists:", os.path.exists(BM25_PATH))
if not os.path.exists(BM25_PATH):
    sys.exit(0)

print("Size (bytes):", os.path.getsize(BM25_PATH))

try:
    with open(BM25_PATH, "rb") as f:
        obj = pickle.load(f)
    print("Loaded OK. Type:", type(obj))
except Exception as e:
    print("LOAD FAILED:", repr(e))
    traceback.print_exc(limit=5)
    # try to read first 256 bytes to show file header
    try:
        with open(BM25_PATH, "rb") as f:
            header = f.read(256)
        print("File header (hex):", header[:64].hex())
    except Exception as e2:
        print("Could not read header:", e2)
