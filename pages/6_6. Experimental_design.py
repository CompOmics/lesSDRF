
import streamlit as st
import pandas as pd
import os
import re
import gzip
import json
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from streamlit_tree_select import tree_select

# Step 1: Paste raw file names
raw_files_text = st.text_area("Paste raw file names (one per line):")
raw_files = [f.strip() for f in raw_files_text.splitlines() if f.strip()]

# Step 2: Specify design
cols = st.columns(4)
with cols[0]:
    n_conditions = st.number_input("Conditions", 1, 20, 2)
with cols[1]:
    n_samples = st.number_input("Samples per condition", 1, 50, 1)
with cols[2]:
    n_bioreps = st.number_input("Biological replicates", 1, 10, 1)
with cols[3]:
    n_fractions = st.number_input("Fractions per sample", 1, 10, 1)

# Step 3: Labeling
labeled = st.radio("Is this a labeled experiment (e.g. TMT)?", ["Yes", "No"])
if labeled == "Yes":
    label_set = st.multiselect("Labels used (in order)", options=["TMT126", "TMT127N", "TMT127C", "TMT128N", "TMT128C", "TMT129N", "TMT129C", "TMT130N", "TMT130C", "TMT131"])

# Step 4: Auto-generate a table for user to edit
import pandas as pd

expected_rows = len(raw_files) * len(label_set) if labeled == "Yes" else len(raw_files)
data = []

sample_id = 1
for i, raw in enumerate(raw_files):
    for j, label in enumerate(label_set if labeled == "Yes" else [None]):
        data.append({
            "raw_file": raw,
            "label": label or "label free",
            "sample_id": f"sample{sample_id}",
            "condition": f"condition{(sample_id - 1) // n_samples + 1}",
            "bio_rep": ((sample_id - 1) % n_bioreps) + 1,
            "fraction": (j % n_fractions) + 1
        })
        sample_id += 1

df = pd.DataFrame(data)
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

st.success("✅ This structure can now be used to generate your SDRF JSON.")
