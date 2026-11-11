
import time
import faiss
import numpy as np


DIMENSION = 384
DATASET_SIZES = [1_000, 5_000, 10_000]
QUERY_COUNT = 50
K = 5

rng = np.random.default_rng(42)


def make_vectors(n):
    vectors = rng.normal(
        size=(n, DIMENSION)
    ).astype("float32")

    faiss.normalize_L2(vectors)
    return vectors


def build_indexes(vectors):

    n, dimension = vectors.shape
    indexes = {}

    # Flat
    start = time.perf_counter()

    flat = faiss.IndexFlatIP(dimension)
    flat.add(vectors)

    indexes["Flat"] = flat
    flat_time = time.perf_counter() - start

    # HNSW
    start = time.perf_counter()

    hnsw = faiss.IndexHNSWFlat(
        dimension, 32, faiss.METRIC_INNER_PRODUCT
    )
    hnsw.hnsw.efConstruction = 40
    hnsw.hnsw.efSearch = 32
    hnsw.add(vectors)

    indexes["HNSW"] = hnsw
    hnsw_time = time.perf_counter() - start

    # IVF
    nlist = min(100, n // 39)

    if nlist >= 2:
        start = time.perf_counter()

        quantizer = faiss.IndexFlatIP(dimension)

        ivf = faiss.IndexIVFFlat(
            quantizer,
            dimension,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )

        ivf.train(vectors)
        ivf.add(vectors)
        ivf.nprobe = min(8, nlist)

        indexes["IVF"] = ivf
        ivf_time = time.perf_counter() - start
    else:
        ivf_time = None

    build_times = {
        "Flat": flat_time,
        "HNSW": hnsw_time,
        "IVF": ivf_time,
    }

    return indexes, build_times


def evaluate(index, flat, queries):

    latencies = []
    recalls = []

    for query in queries:
        query = query.reshape(1, -1)

        _, exact_ids = flat.search(query, K)

        start = time.perf_counter()
        _, result_ids = index.search(query, K)
        elapsed = time.perf_counter() - start

        exact_set = set(exact_ids[0])
        result_set = set(result_ids[0])

        recalls.append(
            len(exact_set & result_set) / K
        )

        latencies.append(elapsed * 1000)

    index_size_mb = (
        faiss.serialize_index(index).nbytes
        / (1024 ** 2)
    )

    return {
        "latency": np.mean(latencies),
        "recall": np.mean(recalls),
        "size": index_size_mb,
    }


for n in DATASET_SIZES:

    print(f"\n{'=' * 45}")
    print(f"DATASET SIZE: {n}")
    print("=" * 45)

    vectors = make_vectors(n)
    queries = make_vectors(QUERY_COUNT)

    indexes, build_times = build_indexes(vectors)
    flat = indexes["Flat"]

    for name, index in indexes.items():

        metrics = evaluate(index, flat, queries)

        print(f"\n{name}")
        print("Build time (s):", round(build_times[name], 4))
        print("Search latency (ms):", round(metrics["latency"], 4))
        print("Recall@5:", round(metrics["recall"], 4))
        print("Index size (MB):", round(metrics["size"], 2))
