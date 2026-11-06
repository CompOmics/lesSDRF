# lesSDRF

This github repository contains the code used to create the lesSDRF application which is available via https://lessdrf.streamlit.app/
A manual on the tool in pdf format is available here: https://github.com/compomics/lesSDRF/blob/main/lesSDRF_manual.pdf

lesSDRF v0.2.0 follows the [SDRF-Proteomics specification](https://sdrf.quantms.org/) v1.1.0. The column
definitions, allowed values and validation come from the templates bundled with
[sdrf-pipelines](https://github.com/bigbio/sdrf-pipelines) (the official validator), so upgrading `sdrf-pipelines` in
`requirements.txt` keeps the app in line with the specification.

## Running the app

Python 3.10 or newer is required (sdrf-pipelines needs it); the app is tested with Python 3.11.

```bash
pip install -r requirements.txt
streamlit run Home.py
```

This opens the Home screen of the app. The following steps can be found in the pages folder and are numbered accordingly.
On Streamlit Community Cloud, select Python 3.11 (or newer) in the app settings.

Alternatively, during development, you can use docker compose to build and run the app using the `watch` feature locally:

```bash
docker compose watch
```

This will automatically sync file changes from your local machine to the docker container. Streamlit will automatically pick up file changes and will offer to reload the application.

Open your browser and go to http://localhost:8501 to see the app running.

## Code structure

| File | Purpose |
| --- | --- |
| `Home.py`, `pages/` | The Streamlit pages (one per step) |
| `spec.py` | Everything that depends on the SDRF specification: templates, column rules, value formats, column order, export and validation |
| `app_utils.py` | Shared Streamlit building blocks: data loading, ontology search, editors and the sidebar |
| `ParsingModule.py`, `*.ipynb` | Offline helpers and notebooks that turn ontology files into the data files in `data/` |
| `textmining.py` | Finds SDRF metadata in pasted methods text (step 1, *From methods text*) |
| `scripts/build_accession_map.py` | Builds `data/term_accessions.json.gz` (label → PSI-MS/PRIDE accession) |

### Metadata from methods text

Step 1 can read a pasted methods section and suggest SDRF values, without an LLM or any external service:
dictionary matching against the bundled ontologies (diseases, tissues, cell lines, instruments, ...), rules for
common phrasings (enzymes, reagents, modifications and whether they are fixed or variable, TMT/iTRAQ/SILAC channels,
DDA/DIA, HCD, collision energy, mass tolerances, age, sex) and NCBITaxon look-ups for Latin species names. Every
suggestion is highlighted in the text, has a confidence score and must be accepted before it is written to the SDRF.

The confidence scores are calibrated on the 107 manuscripts with reference SDRFs in the
[agentic-metadata](https://github.com/CompOmics/agentic-metadata) benchmark: suggestions scored 0.6–0.7 were correct
about 65% of the time, 0.7–0.8 about 75% and above 0.8 just over 80%. Study-wide parameters are found best (cleavage
agent, organism, dissociation method, modifications, instrument); sample-level columns such as disease and organism
part are often absent from methods text or ambiguous, so their suggestions get low scores.

### Memory and speed

The hosted app has limited memory. To keep it stable with many simultaneous users:

- Ontology data is loaded lazily, one file at a time, and shared between all sessions (`st.cache_resource`); nothing
  large is stored per session.
- Large term lists (e.g. 137k diseases, 1.7M NCBITaxon names) are searched on the server; only matches are sent to the
  browser. Ontology trees are only shown on request and are pruned to ~1.5 MB.
- The SDRF is validated in-process with sdrf-pipelines, only when you click *Validate SDRF*.

## Ontology data

The parsed ontologies used in the app are in the data folder. Every topic contains three types of data:
- all_elements: a list of every term in the ontology subset. Is used to check for ontology compatibility of local metadata.
- dict: a nested dictionary that follows the ontology tree structure
- nodes: a node like version of the nested dictionary according to the format required for the tree-select module: https://github.com/Schluca/streamlit_tree_select

`data/term_accessions.json.gz` maps PSI-MS and PRIDE term labels to accessions, so that comment columns are written as
`NT=<label>;AC=<accession>` as the specification recommends. Regenerate it after updating the ontology files:

```bash
python scripts/build_accession_map.py ontology/psi-ms20251015.obo ontology/pride_cv20251015.obo
```

### Ontology versions

The term lists in `data/` were last refreshed on 2025-10-15 (EFO-based lists such as disease, cell line and
developmental stage); see the git history of `data/` for earlier updates. The accession map is built from PRIDE CV
(release 2025-02-17) and PSI-MS (4.1.205).
