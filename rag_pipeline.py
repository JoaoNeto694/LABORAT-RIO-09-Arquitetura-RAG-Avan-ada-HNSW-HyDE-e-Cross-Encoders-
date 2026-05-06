import sys
import time
import numpy as np
import faiss

# Configurações
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Hiperparâmetros HNSW
HNSW_M = 16              # nº máximo de conexões por nó em cada camada
HNSW_EF_CONSTRUCTION = 64  # qualidade da construção (mais alto = grafo melhor, mais lento)
HNSW_EF_SEARCH = 32        # qualidade da busca (mais alto = mais preciso, mais lento)

# PASSO 1 — Indexação HNSW
def construir_indice_hnsw(corpus, embedder):
   
    # Vetoriza o corpus e constrói um índice HNSW com FAISS
    # Usa similaridade de cosseno (via produto interno em vetores normalizados).
    embeddings = embedder.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(embeddings)

    return index