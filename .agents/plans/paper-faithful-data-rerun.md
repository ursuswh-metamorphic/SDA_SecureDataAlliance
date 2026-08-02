# Paper-faithful data rerun

Branch: `paper-faithful-data-rerun`

## Frozen upstream data

- Source: `DocAILab/FedE4RAG_Dataset`
- Revision: `398304846743f184d36f2c35a3db58fa9be70a9d`
- Released file: `FEDE4FIN/train_data/data_50000_random.json`
- SHA256: `e10e402a2189948eb11759f3cec65901bfff6471c6577e3726c240827f1f176a`
- Actual records: 43,658
- Schema: `company`, `page`, `index`, `reference`, `question`

Client order and released record counts:

1. AES: 4,842
2. BOEING: 5,302
3. ACTIVISIONBLIZZARD: 6,328
4. PG: 8,382
5. PEPSICO: 18,804

The official `FEDE4FIN/train_corpus.json` is frozen at SHA256
`009a967f9472ec71c42497ac11aae12db82417e9c4279ccaafa689de2d75f165`.
It contains 368 documents and 23,123 pages. The five training companies
account for the 51 documents described by the paper.

## Gates passed locally

- Exact release hash and record counts.
- Exact five-company roster and one company per client.
- No empty query/reference pairs; uniform five-field schema.
- Page-level leakage audit against the repaired frozen evaluator corpus.
- Validation/test qrels coverage remains 100%.
- `test_company_partitioner.py`: 5 passed.

## Evaluation policy

The first rerun uses the existing frozen full-corpus evaluator so results are
directly comparable with the Q1 remediation baselines. The paper reports a
6,656-page validation corpus, while the public release contains one 30,829-page
corpus and the released code caps it at 6,066 before appending references.
These are separate protocol variants and must not be conflated.

## Next run

Run `scripts/vast/job-paper-faithful-data-smoke.sh` on a CUDA server. If the
3-round x 3-step smoke completes and the task metadata remains exact, schedule
the same baseline matrix and tuned B5 configuration at the full budget.
