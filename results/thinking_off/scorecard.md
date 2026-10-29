# Scorecard: thinking off (configs/experiments/no_thinking.yaml)

| group | measure | dev | test | all |
|---|---|---|---|---|
| PRIMARY | Fully correct (all questions) | 18/25 = 72.0% (95% CI 52–86%) | 72/88 = 81.8% (95% CI 72–88%) | 90/113 = 79.6% (95% CI 71–86%) |
| Answer correctness | fully correct: answerable | 71.4% | 78.1% | 76.6% |
| Answer correctness | key-fact recall (answerable) | 0.853 | 0.884 | 0.877 |
| Answer correctness | critical error rate | 8.0% | 3.4% | 4.4% |
| Answer correctness | plan-table values | 11/18 correct, 0 wrong | – | 11/18 correct, 0 wrong |
| Answer correctness | list completeness (list questions) | 0.750 | 0.970 | 0.936 |
| Faithfulness | groundedness (supported sentences, HHEM) | 0.916 | 0.930 | 0.927 |
| Faithfulness | hallucination rate (answers with an unsupported sentence) | 20.0% | 14.5% | 15.7% |
| Faithfulness | answers with figures not in the context | 0.0% | 0.0% | 0.0% |
| Faithfulness | answers citing section numbers not in the context | 0.0% | 1.1% | 0.9% |
| Retrieval | context recall (evidence passages reaching the model) | 0.889 | 0.932 | 0.922 |
| Retrieval | search recall (evidence among the candidates) | 0.952 | 0.989 | 0.980 |
| Retrieval | MRR@10 (rank of the first relevant chunk) | 0.806 | 0.894 | 0.874 |
| Retrieval | context precision (share of context tokens that are evidence) | 0.391 | 0.280 | 0.305 |
| Retrieval | evidence present in the chunk file | 1.000 | 1.000 | 1.000 |
| Abstention | correct abstention (not answerable) | 75.0% | 100.0% | 94.7% |
| Abstention | false refusal rate (answerable) | 9.5% | 5.5% | 6.4% |
| Abstention | answerable questions refused by the gate | 0 | 0 | 0 |
| Robustness | everyday-wording gap (policy wording minus everyday wording) | -5.8% | 24.9% | 17.5% |
| Efficiency | seconds per answer, median | 10.7 | 9.375 | 9.495 |
| Efficiency | seconds per answer, 95th percentile | 72.9 | 35.3 | 51.8 |
| Efficiency | prompt tokens, median | 1626 | 1623 | 1624 |
| Efficiency | answer tokens, median | 90.5 | 77.0 | 78.5 |
| Reliability | answers cut off at the length limit | 1 | 0 | 1 |
| Reliability | pipeline errors / out-of-memory retries | 0 / 0 | 0 / 0 | 0 / 0 |
| Abstention | gate separability (AUROC of the best reranker score) | 0.619 | 0.952 |  |
| Abstention | evidence removed: answer says it is not shown | 3/4 |  |  |
| Efficiency | tokens per second / peak GPU memory (GB) |  |  | 6.53 / 8.69 |
| Reliability | retrieval identical when repeated / regenerated answer identical |  |  | 113/113 / True |
