import pandas as pd
import streamlit as st

import app_utils as au
import spec

au.setup_page("Labeling")
st.title("2. Labeling")
st.markdown("""If a raw file contains several labels (e.g. TMT channels or SILAC heavy/light), every label is
annotated on its own row. Select the labels used in your experiment and the files they occur in; the rows of those
files are then duplicated, one per label. For label-free data, select *label free sample*.""")

df = au.require_df()
au.show_df(df)
au.sidebar_export()

if "comment[data file]" not in df.columns or spec.is_empty_column(df, "comment[data file]"):
    st.warning("Fill in comment[data file] first (on the Home page or in step 3).")
    st.stop()

labels = au.term_picker("labeling", "all_label_elements", "label_nodes", label="label")
files = list(dict.fromkeys(df["comment[data file]"].dropna().tolist()))

assignment = {}
if labels:
    st.write("Select the files that contain each label. Choose **ALL** if the label is in every file.")
    for label in labels:
        chosen = st.multiselect(f"Files labeled with **{label}**", ["ALL"] + files, key=f"label_files_{label}")
        assignment[label] = files if "ALL" in chosen else chosen

if assignment and any(assignment.values()):
    labels_of = {f: [lab for lab in labels if f in assignment.get(lab, [])] for f in files}
    n_new = sum(max(len(labels_of.get(f, [])), 1) for f in df["comment[data file]"].tolist())
    unlabeled = [f for f, labs in labels_of.items() if not labs]
    if unlabeled:
        st.info(f"{len(unlabeled)} file(s) have no label selected and keep their current label: "
                + ", ".join(unlabeled[:10]) + (" ..." if len(unlabeled) > 10 else ""))
    st.write(f"Your SDRF will have **{n_new}** rows.")
    if n_new > au.MAX_ROWS:
        st.error(f"That is more than the maximum of {au.MAX_ROWS} rows.")
    elif st.button("Apply labels", type="primary"):
        rows = []
        for _, row in df.iterrows():
            labs = labels_of.get(row["comment[data file]"], [])
            for lab in labs or [None]:
                new = row.copy()
                if lab is not None:
                    new["comment[label]"] = au.with_accession("comment[label]", lab)
                rows.append(new)
        out = pd.DataFrame(rows).reset_index(drop=True)
        out = out.sort_values("comment[data file]", kind="stable").reset_index(drop=True)
        au.save_df(out)
        st.success("Labels added. Remember that source name + assay name + label must be unique per row.")
        st.rerun()
