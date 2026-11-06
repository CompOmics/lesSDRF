import re

import streamlit as st

import app_utils as au
import spec

au.setup_page("Additional columns")
st.title("4. Additional columns")

df = au.require_df()
templates = spec.selected_templates()
rules = spec.column_rules(templates)

optional = [c for c in spec.template_columns(templates, levels=("optional",))]
custom = [c for c in df.columns if spec.base_name(c) == c and c not in rules and not c.startswith("factor value[")]
listed = optional + custom

au.show_df(df)

with st.sidebar:
    selection = st.radio(
        "Optional columns you can add",
        ["start", "factor value", "custom column"] + listed + ["clear a column"],
        format_func=lambda c: {"start": "Start", "factor value": "Factor value(s)",
                               "custom column": "Another column", "clear a column": "Clear a column"}.get(
            c, au.column_label(df, c)),
        key="additional_selection")
au.sidebar_export()

if selection == "start":
    st.write("""The selected templates define more columns that can make your SDRF more useful, for example the
    reduction and alkylation reagents, collision energy or treatment. Select one in the sidebar to fill it in.

    Do not forget the **factor value(s)**: the variable(s) your experiment compares, e.g. the disease or a treatment.
    You can also add a column that is not in any template (**Another column**).""")
elif selection == "factor value":
    st.subheader("Factor values")
    st.write("""A factor value column indicates which experimental variable is tested in your study (e.g. disease,
    treatment, time point). It copies the values of a characteristics column into `factor value[...]` and is placed
    at the end of the SDRF. If there are multiple factor values, take care in assigning biological and technical
    replicates.""")
    candidates = [c for c in df.columns if c.startswith("characteristics[") and not spec.is_empty_column(df, c)]
    existing = [c for c in df.columns if c.startswith("factor value[")]
    if existing:
        st.write("Current factor values: " + ", ".join(f"`{c}`" for c in existing))
    chosen = st.multiselect("Characteristics column(s) to use as factor value", candidates)
    if chosen and st.button("Add factor value(s)", type="primary"):
        for col in chosen:
            name = re.sub(r"^characteristics\[", "factor value[", spec.base_name(col))
            df[name] = df[col]
        au.save_df(df)
        st.rerun()
    if existing and st.button("Remove all factor values"):
        au.save_df(df.drop(columns=existing))
        st.rerun()
elif selection == "custom column":
    st.subheader("Add another column")
    st.write("""Sample properties go into `characteristics[...]`, properties of the data file or the measurement
    into `comment[...]`. Use lowercase names, e.g. `characteristics[treatment]` or `comment[collision energy]`.""")
    c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="bottom")
    with c1:
        kind = st.selectbox("Type", ["characteristics", "comment"])
    with c2:
        name = st.text_input("Name (inside the brackets)")
    with c3:
        add = st.button("Add column", type="primary")
    if add and name:
        col = f"{kind}[{name.strip().lower()}]"
        if col in df.columns:
            st.warning(f"{col} is already in your SDRF.")
        else:
            au.save_df(au.ensure_columns(df, [col]))
            st.success(f"Added {col}. Select it in the sidebar to fill it in.")
            st.rerun()
elif selection == "clear a column":
    au.clear_columns_ui(df, "additional")
else:
    au.annotate_column(selection)
