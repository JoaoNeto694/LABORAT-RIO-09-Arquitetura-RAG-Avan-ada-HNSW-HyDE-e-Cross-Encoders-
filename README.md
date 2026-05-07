# Laboratório 09 — Arquitetura RAG Avançada (HNSW + HyDE + Cross-Encoder)

Pipeline de **Retrieval-Augmented Generation** de nível de produção, aplicado a um
assistente de busca em manuais médicos. O sistema traduz queries coloquiais do paciente
para o jargão técnico dos manuais antes de recuperar e refinar os documentos.

## Arquitetura

```
Query coloquial do usuário
  │
  │   "dor de cabeça latejante e luz incomodando"
  ▼
[2] HyDE — LLM gera um documento técnico hipotético
  │
  │   "Cefaleia pulsátil unilateral acompanhada de fotofobia..."
  ▼
[Bi-Encoder] → vetor denso (384 dim, normalizado)
  │
  ▼
[3] Busca aproximada no índice HNSW (FAISS)  ← funil largo (Top-10)
  │
  ▼
[4] Re-ranking com Cross-Encoder              ← funil fino (Top-3)
  │
  ▼
Top-3 documentos → contexto do LLM gerador
```

Cada estágio tem um papel distinto:

| Estágio | Função | Custo | Precisão |
|---------|--------|-------|----------|
| HyDE | Ponte semântica entre linguagem coloquial e jargão | 1 chamada LLM | — |
| HNSW (Bi-Encoder) | Triagem rápida em milhares/milhões de docs | O(log N) | Aproximada |
| Cross-Encoder | Refinamento com atenção cruzada query↔doc | O(k) onde k é pequeno | Alta |

## Estrutura

```
lab09_rag/
├── corpus.py           # 20 fragmentos de manuais médicos (jargão técnico)
├── rag_pipeline.py     # Pipeline completo (Passos 1–4)
└── README.md           # Este arquivo
```

## Como executar

```bash
!pip install -q faiss-cpu sentence-transformers groq

export GROQ_API_KEY="sua_chave_groq"

python rag_pipeline.py

```

## Hiperparâmetros do HNSW (configurados em `rag_pipeline.py`)

```python
HNSW_M = 16   # conexões máximas por nó em cada camada
HNSW_EF_CONSTRUCTION = 64   # qualidade da construção do grafo
HNSW_EF_SEARCH = 32   # qualidade da busca em runtime
```

---

## Tarefa Analítica — Impacto de M e ef_construction na memória RAM

O KNN exato armazena apenas os vetores em RAM (`N · D · 4 bytes`) e percorre todos eles a cada busca — custo O(N). O HNSW adiciona um overhead de grafo (`~N · M · 8 bytes`) para guardar os links de vizinhança entre os nós, mas reduz a busca para O(log N).

**M** controla quantos vizinhos cada nó mantém no grafo. Aumentar M melhora o recall, mas o overhead de RAM cresce linearmente — dobrar M dobra o tamanho do grafo. Valores típicos ficam entre 12 e 48.

**ef_construction** controla quantos candidatos são avaliados ao inserir cada vetor durante a construção. Não afeta a RAM final — só o tempo de build e a qualidade das arestas escolhidas. Valores maiores geram um grafo melhor, mas o índice demora mais para ser construído.

Na prática, o HNSW usa entre 5% e 30% mais RAM que o KNN exato, em troca de buscas ordens de grandeza mais rápidas em corpora grandes.

---

## Modelos utilizados

- **Bi-Encoder de embeddings**:
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dim, suporte a português).
- **Cross-Encoder para re-ranking**:
  `cross-encoder/ms-marco-MiniLM-L-6-v2` (treinado em MS MARCO, retorna score de
  relevância query↔passagem).
- **LLM para HyDE**: Llama 3.3 70B via Groq API.

Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por João Antônio
