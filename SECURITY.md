# Security and privacy

Do not publish credentials (`GEMINI_API_KEY`, `GOOGLE_API_KEY`, Hugging Face tokens,
SSH material), restricted datasets, private paths containing identities, or personal
data from research sources. Use `.horizon.env` locally; it is ignored by Git.

When reporting a vulnerability, avoid attaching credentials, private corpus
records, or exploit data to a public issue. If the GitHub repository enables
private vulnerability reporting, use **Security → Report a vulnerability**.
Otherwise, contact the repository maintainers through a private channel named
on the repository profile (to be configured before public release).

The code downloads and executes third-party model libraries; review dependencies,
model license terms, and generated output separately. Ingested text is untrusted
input and may contain prompt-injection attempts. Model outputs should not be
treated as executable instructions or factual truth without verification.
