# Public-release gates: decisions that cannot be invented

Use this checklist **before** switching the GitHub repository to Public.

- [ ] The rightsholders authorize publication of the original code and
      historical developer documentation.
- [ ] Replace the current `LICENSE` status notice with an explicitly approved,
      complete license text. Do not describe the current archive as open source.
- [ ] Verify that the names/order in `CITATION.cff` accurately reflect **software**
      attribution, not merely paper authorship; add publication metadata/DOI only
      when confirmed.
- [ ] Check whether the manuscript/publisher permits distribution of a PDF or
      DOCX. `paper/` intentionally contains metadata rather than the manuscript.
- [ ] Review source scripts, YAML, dependency configuration, README and Git
      history for secrets, internal machine identifiers, private addresses and
      restricted personal information. A simple textual scan is not proof of safety.
- [ ] Confirm raw WoS/Scopus, arXiv snapshots, patents and other datasets are
      redistributable before including **any** records.
- [ ] Test installation on a fresh machine with Python 3.12; check Windows
      PowerShell instructions separately if Windows support is advertised.
- [ ] Run GitHub Actions CI (the YAML is supplied, not yet run remotely).
- [ ] Execute a real scan on a supported model and record its exact model
      revision, prompt, environment and output checksum.
- [ ] Confirm E1–E5/D1 evidence matrix against the final manuscript and clearly
      identify which study inputs/outputs are not bundled.
- [ ] Confirm whether to retain advanced experimental scripts (`k8s/`, GPU box
      SSH, patent scripts) in the first public release.
- [ ] Confirm owner (`knkknu` personal account/organization or another owner),
      repository URL, collaboration roles and branch protection.

Only after these checks, change visibility to **Public** and create a numbered
GitHub release. A release tag should correspond to a specific audited commit.
