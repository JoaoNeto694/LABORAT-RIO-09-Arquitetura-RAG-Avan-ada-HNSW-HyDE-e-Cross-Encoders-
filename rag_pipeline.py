import time
import numpy as np
import faiss
from google.colab import userdata
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from corpus import CORPUS

# Configurações
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Hiperparâmetros HNSW
HNSW_M = 16 # nº máximo de conexões por nó em cada camada
HNSW_EF_CONSTRUCTION = 64 # qualidade da construção
HNSW_EF_SEARCH = 32 # qualidade da busca

TOP_K_RETRIEVE = 10 # funil largo (Bi-Encoder)
TOP_K_FINAL = 3 # funil fino (Cross-Encoder)


# PASSO 1 — Indexação HNSW
def construir_indice_hnsw(corpus, embedder):
    
    # Vetoriza o corpus e constrói um índice HNSW com FAISS.
    # Usa similaridade de cosseno (via produto interno em vetores normalizados).
    embeddings = embedder.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(embeddings)

    return index

# PASSO 2 — Query Transformation com HyDE
# HyDE: pede ao LLM para 'alucinar' um trecho técnico que responderia à query.
def gerar_documento_hipotetico(query):
    prompt = (
        "Você é um médico especialista. O usuário fez a pergunta abaixo em linguagem coloquial. "
        "Escreva um parágrafo curto (3 a 5 frases) que pareceria um trecho de manual médico técnico "
        "respondendo a essa pergunta, usando jargão clínico apropriado (nomes de síndromes, sintomas "
        "em terminologia médica, possíveis tratamentos). Não diga que é hipotético — escreva como "
        "se fosse o próprio manual.\n\n"
        f"Pergunta do usuário: {query}\n\n"
        "Trecho do manual:"
    )

    # Usa a Groq API
    client = Groq(api_key=userdata.get('GROQ_API_KEY'))
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


# PASSO 3 — Busca rápida (Bi-Encoder + HNSW)
def busca_rapida(doc_hipotetico, embedder, index, k=TOP_K_RETRIEVE):
    # Vetoriza o documento hipotético e busca os k mais similares no índice HNSW.
    vetor = embedder.encode([doc_hipotetico], normalize_embeddings=True)
    vetor = np.asarray(vetor, dtype=np.float32)

    # Mede o tempo de busca
    t0 = time.perf_counter()
    scores, idxs = index.search(vetor, k)
    dt = (time.perf_counter() - t0) * 1000

    resultados = []
    # Imprime os resultados com as similaridades e um preview do documento.
    for rank, (idx, score) in enumerate(zip(idxs[0], scores[0]), start=1):
        resultados.append((idx, float(score), CORPUS[idx]))
        preview = CORPUS[idx][:90].replace("\n", " ")
        print(f"#{rank:>2}  cos={score:+.4f}  doc[{idx:>2}]  {preview}")
    print()
    return resultados

# PASSO 4 — Re-ranking com Cross-Encoder
def reranquear(query_original, candidatos, cross_encoder, top_n=TOP_K_FINAL):
    # Prepara os pares (query, doc) para o Cross-Encoder.
    pares = [(query_original, doc) for (_, _, doc) in candidatos]

    t0 = time.perf_counter()
    scores = cross_encoder.predict(pares, show_progress_bar=False)
    dt = (time.perf_counter() - t0) * 1000

    # Rankeia os candidatos com base na pontuação do Cross-Encoder.
    rankeado = sorted(
        zip(candidatos, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # Imprime os resultados finais com as pontuações do Cross-Encoder e do Bi-Encoder.
    finais = []
    for rank, ((idx, cos, doc), ce_score) in enumerate(rankeado[:top_n], start=1):
        finais.append({"idx": idx, "cos": cos, "ce_score": float(ce_score), "doc": doc})
        print(f"TOP {rank}  doc[{idx}]")
        print(f"cross-encoder score: {ce_score:+.4f}   |   cosseno (HNSW): {cos:+.4f}")
        print(f"{doc}\n")
    return finais


# Pipeline orquestrador
def pipeline(query):
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    # Passo 1
    index = construir_indice_hnsw(CORPUS, embedder)

    # Passo 2 — HyDE
    doc_hipotetico = gerar_documento_hipotetico(query)

    # Passo 3 — Bi-Encoder + HNSW
    candidatos = busca_rapida(doc_hipotetico, embedder, index)

    # Passo 4 — Cross-Encoder
    finais = reranquear(query, candidatos, cross_encoder)
 
    return finais


if __name__ == "__main__":
    QUERY = "dor de cabeça latejante e luz incomodando"
    pipeline(QUERY)