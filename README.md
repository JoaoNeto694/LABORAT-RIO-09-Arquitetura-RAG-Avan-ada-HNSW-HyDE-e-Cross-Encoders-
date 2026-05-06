## Tarefa Analítica — Impacto de M e ef_construction na memória RAM

O KNN exato armazena apenas os vetores em RAM (`N · D · 4 bytes`) e percorre todos eles a cada busca — custo O(N). O HNSW adiciona um overhead de grafo (`~N · M · 8 bytes`) para guardar os links de vizinhança entre os nós, mas reduz a busca para O(log N).

**M** controla quantos vizinhos cada nó mantém no grafo. Aumentar M melhora o recall, mas o overhead de RAM cresce linearmente — dobrar M dobra o tamanho do grafo. Valores típicos ficam entre 12 e 48.

**ef_construction** controla quantos candidatos são avaliados ao inserir cada vetor durante a construção. Não afeta a RAM final — só o tempo de build e a qualidade das arestas escolhidas. Valores maiores geram um grafo melhor, mas o índice demora mais para ser construído.

Na prática, o HNSW usa entre 5% e 30% mais RAM que o KNN exato, em troca de buscas ordens de grandeza mais rápidas em corpora grandes.