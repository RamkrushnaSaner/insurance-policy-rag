# Scorecard: Neon + pgvector (Qwen3) with the table route

| group | measure | dev | test | all |
|---|---|---|---|---|
| PRIMARY | Fully correct (all questions) | 22/25 = 88.0% (95% CI 70–96%) | 78/88 = 88.6% (95% CI 80–94%) | 100/113 = 88.5% (95% CI 81–93%) |
| Answer correctness | fully correct: answerable | 85.7% | 86.3% | 86.2% |
| Answer correctness | key-fact recall (answerable) | 0.944 | 0.945 | 0.945 |
| Answer correctness | critical error rate | 0.0% | 2.3% | 1.8% |
| Answer correctness | plan-table values | 17/18 correct, 0 wrong | – | 17/18 correct, 0 wrong |
| Answer correctness | list completeness (list questions) | 0.917 | 0.947 | 0.942 |
| Faithfulness | groundedness (supported sentences, HHEM) | 0.835 | 0.885 | 0.873 |
| Faithfulness | hallucination rate (answers with an unsupported sentence) | 54.2% | 26.0% | 32.7% |
| Faithfulness | answers with figures not in the context | 0.0% | 1.1% | 0.9% |
| Faithfulness | answers citing section numbers not in the context | 0.0% | 1.1% | 0.9% |
| Retrieval | context recall (evidence passages reaching the model) | 0.897 | 0.936 | 0.927 |
| Retrieval | search recall (evidence among the candidates) | 0.952 | 0.950 | 0.950 |
| Retrieval | MRR@10 (rank of the first relevant chunk) | 0.640 | 0.857 | 0.808 |
| Retrieval | context precision (share of context tokens that are evidence) | 0.288 | 0.258 | 0.265 |
| Retrieval | evidence present in the chunk file | 0.857 | 0.973 | 0.947 |
| Abstention | correct abstention (not answerable) | 100.0% | 100.0% | 100.0% |
| Abstention | false refusal rate (answerable) | 4.8% | 1.4% | 2.1% |
| Abstention | answerable questions refused by the gate | 0 | 0 | 0 |
| Robustness | everyday-wording gap (policy wording minus everyday wording) | -2.9% | -5.1% | -4.4% |
| Efficiency | seconds per answer, median | 58.5 | 47.2 | 49.2 |
| Efficiency | seconds per answer, 95th percentile | 177 | 113 | 152 |
| Efficiency | prompt tokens, median | 1647 | 1611 | 1619 |
| Efficiency | answer tokens, median | 136 | 111 | 115 |
| Efficiency | thinking tokens, median | 426 | 334 | 340 |
| Reliability | thinking stopped at the budget (told to answer) | 3 | 5 | 8 |
| Reliability | answers cut off at the length limit | 0 | 0 | 0 |
| Reliability | pipeline errors / out-of-memory retries | 0 / 0 | 0 / 0 | 0 / 0 |
| Abstention | gate separability (AUROC of the best reranker score) | 0.702 | 0.954 |  |
| Abstention | evidence removed: answer says it is not shown | 3/3 |  |  |
| Efficiency | tokens per second / peak GPU memory (GB) |  |  | 6.39 / 12.53 |
| Reliability | retrieval identical when repeated / regenerated answer identical |  |  | 113/113 / True |
