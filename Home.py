import os
import re

import pandas as pd
import streamlit as st

import app_utils as au
import spec

au.setup_page("lesSDRF", icon=None)

st.title("Welcome to lesSDRF")
st.subheader("Spending less time on SDRF creates more time for amazing research")
st.write(
    f"""By providing metadata in a machine-readable format, other researchers can access your data more easily and
    you maximize its impact. The Sample and Data Relationship Format
    ([SDRF](https://www.nature.com/articles/s41467-021-26111-3)) is the HUPO-PSI recognized metadata format within
    proteomics. lesSDRF follows the [SDRF-Proteomics specification](https://sdrf.quantms.org/) {spec.SDRF_SPEC_VERSION}
    and its templates, and validates your file with the official validator
    ([sdrf-pipelines](https://github.com/bigbio/sdrf-pipelines)). This tool is developed by the
    [CompOmics group](https://compomics.com/) and published in
    [Nature Communications](https://www.nature.com/articles/s41467-023-42543-5)."""
)
st.write(
    """On this page, choose the SDRF templates that match your study and provide the raw file names.
Then follow the steps in the sidebar.
- Step 1: If you have a local metadata file, you can upload it to map to the SDRF file
- Step 2: Provide information on labels (TMT, SILAC, ...) in your samples
- Step 3: Fill in the columns that are required for a valid SDRF
- Step 4: Fill in optional columns and factor values to further optimise your SDRF file
- Step 5: Add templates for specific experiment types (DIA, single-cell, immunopeptidomics, ...)

You can download your intermediate file at any time from the sidebar and continue later by uploading it here."""
)

with open(os.path.join(au.APP_DIR, "example_SDRF.tsv"), "rb") as fh:
    st.download_button("Download an example SDRF file", fh.read(), file_name="example.sdrf.tsv",
                       on_click="ignore", help="In need of some inspiration? This is what a finished SDRF looks like.")

# --------------------------------------------------------------------------------------
st.header("1. Choose your templates")
techs = spec.available(spec.TECHNOLOGY_TEMPLATES)
current = spec.selected_templates()
c1, c2 = st.columns(2)
with c1:
    tech = st.radio("Technology", techs, format_func=spec.TECHNOLOGY_TEMPLATES.get,
                    index=next((techs.index(t) for t in current if t in techs), 0))
with c2:
    samples = spec.available(spec.SAMPLE_TEMPLATES)
    sample = st.selectbox("Organism / sample type", samples, format_func=spec.SAMPLE_TEMPLATES.get,
                          index=next((samples.index(t) for t in current if t in samples and t), 0),
                          help="Determines organism-specific columns such as age and sex for human samples.")
experiments = [t for t in spec.available(spec.EXPERIMENT_TEMPLATES)
               if tech == "ms-proteomics" or t not in spec.MS_ONLY_TEMPLATES]
chosen_experiments = st.multiselect(
    "Experiment type(s) (optional)", experiments, format_func=spec.EXPERIMENT_TEMPLATES.get,
    default=[t for t in current if t in experiments])
templates = [tech] + ([sample] if sample else []) + chosen_experiments

with st.expander("Columns in these templates"):
    rules = spec.column_rules(templates)
    st.dataframe(
        pd.DataFrame([{"column": c, "requirement": rules[c].requirement,
                       "templates": ", ".join(t for t in dict.fromkeys(rules[c].templates)
                                              if t not in ("base", "sample-metadata"))}
                      for c in spec.all_columns(templates)]),
        hide_index=True, height=300)
    st.caption("Required and recommended columns are added to your SDRF now; optional ones can be added in step 4.")

if templates != current and au.get_df() is not None:
    st.info("You changed the templates of your current SDRF.")
    if st.button("Apply these templates to my current SDRF (adds any missing required/recommended columns)"):
        st.session_state["templates"] = templates
        au.save_df(au.ensure_columns(au.get_df(), spec.template_columns(templates)))
        st.rerun()

# --------------------------------------------------------------------------------------
st.header("2. Start a new SDRF or continue a saved one")
new_tab, upload_tab = st.tabs(["Start a new SDRF", "Continue with a saved SDRF"])

with new_tab:
    method = st.radio("How do you want to create your SDRF?", ["Enter raw file names", "Create empty rows"],
                      horizontal=True)
    files = []
    if method == "Enter raw file names":
        text = st.text_area(
            "Raw file names, separated by commas, tabs or new lines (you can paste a column from a spreadsheet)",
            height=180,
            help=f"The raw file names go into comment[data file] and are the basis of your SDRF file. "
                 f"Maximum {au.MAX_FILES} files.")
        if text:
            parts = re.split(r"[,\t\r\n]+", text) if re.search(r"[,\t\r\n]", text) else text.split()
            files = [p.strip() for p in parts if p.strip()]
            dupes = sorted({f for f in files if files.count(f) > 1})
            st.write(f"{len(files)} file(s)" + (f" — duplicates: {', '.join(dupes)}" if dupes else ""))
        n_rows = len(files)
    else:
        n_rows = st.number_input("Number of empty rows", min_value=1, max_value=au.MAX_FILES, value=1)

    too_many = n_rows > au.MAX_FILES
    if too_many:
        st.error(f"Too many files, please enter a maximum of {au.MAX_FILES}.")
    overwrite = True
    if au.get_df() is not None:
        overwrite = st.checkbox("Replace my current SDRF (it will be lost unless you downloaded it)")
    if st.button("Create SDRF", type="primary", disabled=not n_rows or too_many or not overwrite):
        columns = spec.template_columns(templates)
        df = pd.DataFrame({c: [None] * int(n_rows) for c in columns})
        if files:
            df["comment[data file]"] = files
        if "technology type" in df and tech in spec.TECHNOLOGY_TYPES:
            df["technology type"] = spec.TECHNOLOGY_TYPES[tech]
        if "characteristics[sample type]" in df:
            df["characteristics[sample type]"] = "study sample"
        st.session_state["templates"] = templates
        au.save_df(df)
        st.rerun()

with upload_tab:
    upload = st.file_uploader("Upload an intermediate SDRF file (.tsv)", type=["tsv", "txt"],
                              help=f"A previously saved SDRF file with at most {au.MAX_ROWS} rows")
    if upload is not None:
        try:
            uploaded = spec.read_sdrf_upload(upload)
        except Exception as exc:  # malformed upload
            st.error(f"Could not read this file as a tab-separated SDRF: {exc}")
            uploaded = None
        if uploaded is not None and len(uploaded) > au.MAX_ROWS:
            st.error(f"Too many rows, please upload a maximum of {au.MAX_ROWS}.")
        elif uploaded is not None:
            found = spec.templates_from_df(uploaded)
            use = found or templates
            st.write(f"{len(uploaded)} rows, {uploaded.shape[1]} columns. Templates: "
                     + ", ".join(use) + (" (from the file)" if found else " (selected above)"))
            st.dataframe(uploaded, hide_index=True, height=200)
            # Older lesSDRF/SDRF versions used other names for these columns and values.
            renames = {"comment[tool metadata]": None, "comment[depletion]": "characteristics[depletion]",
                       "characteristics[strain/breed]": "characteristics[strain or breed]"}
            if st.button("Use this file", type="primary"):
                for old, new in renames.items():
                    for col in [c for c in uploaded.columns if spec.base_name(c) == old]:
                        if new is None:
                            uploaded = uploaded.drop(columns=[col])
                        else:
                            uploaded = uploaded.rename(columns={col: col.replace(old, new)})
                st.session_state["templates"] = use
                au.save_df(uploaded)
                st.rerun()

df = au.get_df()
if df is not None:
    au.show_df(df)
au.sidebar_export()
