# Vector experiments

Small plots for cosine similarity, angles and distance measures.

Use a separate environment for this project. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`.

```powershell
python vector_similarities/cosine_similarities.py
python vector_similarities/all_similarity_distance_calculations.py
python vector_similarities/3d-visualize.py
```

Run one at a time. Each script opens a Matplotlib window. Edit the sample vectors in the script to compare different inputs. The existing PDF and image are reference outputs.

## Oracle embedding example

`embeddings/generate_embeddings.py` is imported from the July 2025 experiment.
Install `embeddings/requirements.txt`, configure Oracle Instant Client, and set
`ORACLE_USER`, `ORACLE_PASSWORD` and optionally `ORACLE_DSN` in your environment.
The example recreates `TEXT_EMBEDDINGS` and calls an existing
`insert_text_embedding` procedure. It is a database-specific experiment, not a
self-contained setup script. Run it only against a disposable demo database.
