import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import re
import ParsingModule
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from streamlit_tree_select import tree_select
from collections import defaultdict
from PIL import Image
import base64
import io

st.set_page_config(
    page_title="Labeling",
    layout="wide",
    page_icon="🧪",
    menu_items={
        "Get help": "https://github.com/compomics/lesSDRF/issues",
        "Report a bug": "https://github.com/compomics/lesSDRF/issues",
    },
)

import common
common.inject_sidebar_logo()

# Constants
LABEL_ELEMENTS_KEY = "all_label_elements"
LABEL_NODES_KEY = "label_nodes"
COMMENT_DATA_FILE_COL = "comment[data file]"
COMMENT_LABEL_COL = "comment[label]"
data_dict = st.session_state["data_dict"]

def assign_labels(template_df, label_dict, all_labels):
    new_rows = []
    
    # 1Duplicate rows for labeled files
    for filename, labels in label_dict.items():
        rows_to_duplicate = (
            template_df.copy() if filename == "ALL"
            else template_df[template_df[COMMENT_DATA_FILE_COL] == filename]
        )
        for label in labels:
            new_row = rows_to_duplicate.copy()
            new_row[COMMENT_LABEL_COL] = label
            new_rows.append(new_row)
    
    labeled_df = pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()

    # 2️Find files not included in label_dict
    labeled_files = set(label_dict.keys()) if "ALL" not in label_dict else set(template_df[COMMENT_DATA_FILE_COL].unique())
    unassigned_files = set(template_df[COMMENT_DATA_FILE_COL].unique()) - labeled_files
    
    if unassigned_files:
        unassigned_rows = template_df[template_df[COMMENT_DATA_FILE_COL].isin(unassigned_files)].copy()
        unassigned_rows[COMMENT_LABEL_COL] = "label free sample"  # or np.nan
    else:
        unassigned_rows = pd.DataFrame()
    
    # 3️ Combine labeled and unassigned rows
    combined_df = pd.concat([labeled_df, unassigned_rows], ignore_index=True)
    return combined_df.sort_values(by=COMMENT_DATA_FILE_COL)

def edit_mapping_table_aggrid(df, label_options):
    """
    Shows editable AgGrid table for assigning labels per file.
    Returns updated mapping DataFrame after clicking Update.
    """
    df = df.copy()
    df.fillna("", inplace=True)
    cell_style = {"background-color": "#ffa478"}

    builder = GridOptionsBuilder.from_dataframe(df)
    builder.configure_grid_options(
        pagination=True,
        paginationPageSize=500,
        enableRangeSelection=True,
        enableFillHandle=True,
        suppressMovableColumns=True,
        singleClickEdit=True
    )
    # builder.configure_pagination(paginationAutoPageSize=False, paginationPageSize=500)
    builder.configure_default_column(filterable=True, sortable=True, resizable=True)
    for col in df.columns:
        if col == "File name":
            builder.configure_column(col, editable=False)
        else:
            builder.configure_column(
                col,
                editable=True,
                cellEditor="agSelectCellEditor",
                cellEditorParams={"values": [""] + label_options},
                cellStyle=cell_style
            )



    gridOptions = builder.build()

    grid_return = AgGrid(
        df,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.MANUAL,  # user must click Update
        data_return_mode=DataReturnMode.AS_INPUT,
        fit_columns_on_grid_load=False,
        height=600
    )

    return grid_return["data"]

st.title("2. Labeling")
st.markdown(
    """If a raw file contains multiple labels, every label will need to be annotated on a different row. Here you can map label information to your raw files. 
    As a result, the raw file information will be duplicated with the correct label filled in"""
)


# Get filled in template_df from other page
# if template_df is not in the session state, don't run all the code below
if "template_df" not in st.session_state:
    st.error("Please fill in the template file in the Home page first", icon="🚨")  
    st.stop()
else:
    template_df = st.session_state["template_df"] 
    st.write("**This is your current SDRF file.**")
    st.write(template_df)

#  Initialize label selections in session state if not already present
if "all_selected_labels" not in st.session_state:
    st.session_state["all_selected_labels"] = []

with st.sidebar:
    if st.button("Validate SDRF file before download"):
        # Perform validation or conversion
        converted_data = ParsingModule.convert_df(template_df)
        st.success("SDRF file validated! Ready to download.")
        
        st.download_button(
            label="Download validated SDRF file",
            data=converted_data,
            file_name="intermediate_SDRF.sdrf.tsv",
            mime="text/tab-separated-values"
        )

    st.write("""Please refer to your data and lesSDRF within your manuscript as follows:
                 *The experimental metadata has been generated using lesSDRF and is available through ProteomeXchange with the dataset identifier [PXDxxxxxxx]*""")


# --- Select labels ---
label_options = set(data_dict[LABEL_ELEMENTS_KEY])
label_options.add(" ")  # optional blank for no label

st.write("Input the label that was used in your experiment. If no label was added, indicate this using *label free sample*.")
all_label_elements = data_dict["all_label_elements"]
label_nodes = data_dict["label_nodes"]
with st.form("Select here your ontology terms using the autocomplete function or the ontology-based tree menu", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        return_search = st.multiselect(
        "Select ontology terms (autocomplete):",
        sorted(label_options)
        )
    with col2:
        st.write("Or browse ontology tree:")
        tree_result = tree_select(data_dict[LABEL_NODES_KEY], no_cascade=True, expand_on_click=True, check_model="leaf")
        tree_checked = tree_result["checked"] if tree_result and "checked" in tree_result else []

    selected_labels = [lbl.split(',')[-1] for lbl in (return_search + tree_checked) if lbl]
    submitted = st.form_submit_button("Submit selection")
    st.session_state['selected_labels'] = True

st.write(selected_labels, submitted)
if selected_labels in st.session_state():
    st.session_state["all_selected_labels"] = selected_labels
    st.write(f"Selection contains: {selected_labels}")

    num_label_cols = len(st.session_state["all_selected_labels"])
    label_column_names = [f"Assigned Label {i+1}" for i in range(num_label_cols)]
    file_names = template_df[COMMENT_DATA_FILE_COL].tolist()
    # Create mapping table: each file = row; assigned label = editable dropdown
    if "mapping_table" not in st.session_state:
        initial_data = {
            "File name": file_names
        }
        for col in label_column_names:
            initial_data[col] = [""] * len(file_names)
        st.session_state["mapping_table"] = pd.DataFrame(initial_data)
    column_config = {
    "File name": st.column_config.Column(
        label="File name",
        disabled=True,  # not editable
        required=True
    )
    }
    # 3Show editable table
    edited_mapping_df = edit_mapping_table_aggrid(st.session_state["mapping_table"], st.session_state["all_selected_labels"])
    # Detect grid update by comparing to session_state
    if not edited_mapping_df.equals(st.session_state["mapping_table"]):
        st.session_state["mapping_table"] = edited_mapping_df

        label_assignments = (
            edited_mapping_df
            .melt(id_vars="File name", value_vars=label_column_names)
            .query('value != ""')  # drop rows where value is empty
            .drop_duplicates(subset=["File name", "value"])  # remove duplicates
            .rename(columns={"value": "Label"})
        )

        # Group labels by file into a dictionary
        label_dict = (
            label_assignments
            .groupby("File name")["Label"]
            .apply(list)
            .to_dict()
        )

        updated_df = assign_labels(template_df, label_dict, st.session_state["all_selected_labels"])
        st.session_state["template_df"] = updated_df

        st.success("Labels applied to SDRF.")
        st.write("Updated SDRF file:")
        st.dataframe(updated_df, use_container_width=True)
else:
    st.write('no selection')
    
