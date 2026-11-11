
import time
import faiss
import numpy as np

from assistant.document_loader import load_pdf
from assistant.text_splitter import split_documents
from langchain_huggingface import HuggingFaceEmbeddings


# 1. Load documents
documents = load_pdf(
    "data/research_papers/attention_is_all_you_need.pdf"
)
chunks = split_documents(documents)

# 2. Generate embeddings once
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": False},
)

texts = [chunk.page_content for chunk in chunks]
vectors = np.array(
    embedder.embed_documents(texts),
    dtype="float32",
)

faiss.normalize_L2(vectors)

n, dimension = vectors.shape
k = min(5, n)

print(f"Vectors: {n}, Dimensions: {dimension}")


# 3. Build exact Flat index
flat = faiss.IndexFlatIP(dimension)
flat.add(vectors)


# 4. Build HNSW index
hnsw = faiss.IndexHNSWFlat(
    dimension,
    32,
    faiss.METRIC_INNER_PRODUCT,
)
hnsw.hnsw.efConstruction = 40
hnsw.hnsw.efSearch = 32
hnsw.add(vectors)


# 5. Build IVF index when dataset is large enough
ivf = None

if n >= 20:
    nlist = min(100, max(2, n // 5))

    quantizer = faiss.IndexFlatIP(dimension)

    ivf = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        nlist,
        faiss.METRIC_INNER_PRODUCT,
    )

    ivf.train(vectors)
    ivf.add(vectors)
    ivf.nprobe = min(4, nlist)


# 6. Prepare test queries
queries = [
    "What is the main contribution of this paper?",
    "Which methodology does the research use?",
    "What are the experimental results?",
]

query_vectors = np.array(
    embedder.embed_documents(queries),
    dtype="float32",
)

faiss.normalize_L2(query_vectors)


# 7. Benchmark
def benchmark(name, index):

    latencies = []
    recalls = []

    for query_vector in query_vectors:

        query = query_vector.reshape(1, -1)

        # Exact ground truth
        _, exact_ids = flat.search(query, k)

        # Approximate or exact index search
        start = time.perf_counter()
        _, result_ids = index.search(query, k)
        elapsed = time.perf_counter() - start

        exact_set = set(exact_ids[0])
        result_set = set(result_ids[0])

        recall = len(exact_set & result_set) / k

        latencies.append(elapsed * 1000)
        recalls.append(recall)

    print(f"\n{name}")
    print("Average latency (ms):", round(np.mean(latencies), 3))
    print("Recall@K:", round(np.mean(recalls), 3))


benchmark("Flat", flat)
benchmark("HNSW", hnsw)

print("\nHNSW PARAMETER EXPERIMENT")

for ef in [8, 16, 32, 64, 128]:
    hnsw.hnsw.efSearch = ef
    benchmark(f"HNSW efSearch={ef}", hnsw)

if ivf is not None:
    benchmark("IVF", ivf)
else:
    print("\nIVF skipped: need at least 20 vectors.")
