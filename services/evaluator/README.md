# Research evaluator — service planned

Sprint 1 scorer is a local trusted module `src/feddisclosure/evaluation/analytics.py`. It is **not** a deployed independent Docker service. A future evaluator service must have privileged synthetic-truth access and no shared import or mounted ground-truth with attack containers.
