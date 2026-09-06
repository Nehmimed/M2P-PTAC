
# Data availability



The raw field telemetry used in this study is subject to institutional and deployment-specific restrictions and is therefore not redistributed in this repository.



The repository provides the complete data-quality assessment (DQA) implementation, chronological split logic, controlled-degradation procedures, forecasting and correction models, evaluation code, experimental configurations, seed definitions, and selected derived result artifacts used in the manuscript.



The public experiment pipeline treats the cleaned datasets and their associated observation/fault masks as frozen inputs. The validation stage checks and hashes these inputs before model-side experiments are executed.



Users wishing to reproduce the workflow with their own data should provide timestamp-aligned water-quality measurements compatible with the preprocessing and experiment interfaces documented in the source code.

