import streamlit as st

import app_utils as au
import spec

au.setup_page("Required columns")
st.title("3. Required columns")

df = au.require_df()
templates = spec.selected_templates()
rules = spec.column_rules(templates)

# Required and recommended columns of the selected templates, plus any copies (col_1, ...) already in the SDRF.
wanted = spec.template_columns(templates)
df = au.ensure_columns(df, [c for c in wanted if c not in df.columns])
if list(df.columns) != list(au.get_df().columns):
    au.save_df(df)
columns = wanted
required = [c for c in columns if rules[c].requirement == "required"]
recommended = [c for c in columns if rules[c].requirement == "recommended"]
todo = [c for c in columns if spec.is_empty_column(df, c)]

au.show_df(df)

with st.sidebar:
    show_done = st.toggle("Show filled-in columns too", value=False)
    listed = columns if show_done else todo
    labels = {c: au.column_label(df, c) + ("" if rules[c].requirement == "required" else " (recommended)")
              for c in listed}
    filled_required = sum(not spec.is_empty_column(df, c) for c in required)
    st.progress(filled_required / max(len(required), 1),
                text=f"{filled_required} of {len(required)} required columns filled")
    selection = st.radio("Columns to fill in", ["start"] + listed + ["clear a column"],
                         format_func=lambda c: {"start": "Start", "clear a column": "Clear a column"}.get(c, labels.get(c, c)),
                         key="required_selection")
au.sidebar_export()

if selection == "start":
    st.write(f"""In the sidebar you find the empty columns that the selected templates
    (**{', '.join(templates)}**) require or recommend. Select the column you want to annotate first.
    A column disappears from the list once it is filled in; switch on *Show filled-in columns too* to edit it again,
    or use *Clear a column* to start over.

    Required columns must have a value in every row. Recommended columns can be left empty; empty optional or
    recommended columns are dropped when you download the file.""")
    if not todo:
        st.success("All required and recommended columns are filled in. Validate your file in the sidebar, "
                   "or continue with step 4 to add optional columns and factor values.")
elif selection == "clear a column":
    au.clear_columns_ui(df, "required")
else:
    au.annotate_column(selection)
