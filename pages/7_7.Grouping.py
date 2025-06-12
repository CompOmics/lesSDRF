
import streamlit as st
import pandas as pd
import os
import re
import gzip
import json
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from streamlit_tree_select import tree_select

@st.cache_data
def load_data():
    """Load gzipped JSON data and Unimod CSV"""
    folder_path = os.path.join(local_dir, "data")
    unimod_path = os.path.join(local_dir, "ontology", "unimod.csv")
    data = {}
    for filename in os.listdir(folder_path):
        if re.search(r"archaea|bacteria|eukaryota|virus|unclassified|other sequences", filename):
            continue
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".json.gz"):
            try:
                with gzip.open(file_path, "rb") as f:
                    json_str = f.read().decode('utf-8')
                    try:
                        data[filename.replace(".json.gz", "")] = json.loads(json_str)
                    except json.JSONDecodeError:
                        st.error(f"Error decoding JSON in file {file_path}")
            except gzip.BadGzipFile:
                st.error(f"Error reading {file_path}: not a gzipped file")
        else:
            st.warning(f"Skipping {file_path}: not a gzipped file")

    unimod = pd.read_csv(unimod_path, sep="\t")
    return data, unimod
def merge_shared_and_group_metadata(edited_shared_df: pd.DataFrame, edited_group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge shared and group-specific metadata into one table.
    Group-specific values take precedence over shared values.

    Parameters:
    - edited_shared_df: DataFrame with 1 row of shared metadata (index = 'sample')
    - edited_group_df: DataFrame with one row per sample group (index = 'sample')

    Returns:
    - merged_df: Final merged DataFrame, one row per sample group
    """
    # Reset indices to use 'sample' as column
    shared = edited_shared_df.reset_index()
    group = edited_group_df.reset_index()

    # Broadcast shared metadata to each sample group
    shared_expanded = pd.DataFrame({
        col: shared.at[0, col] for col in shared.columns if col != "sample"
    }, index=group["sample"])
    shared_expanded.reset_index(inplace=True)
    shared_expanded.rename(columns={"index": "sample"}, inplace=True)

    # Merge group-specific metadata
    merged_df = pd.merge(
        shared_expanded,
        group,
        on="sample",
        how="left",
        suffixes=("", "_group")
    )

    return merged_df
def build_condition_descriptions_ui(label: str, parameters: list[str], key_prefix: str = "") -> dict:
    st.subheader(label)
    n_conditions = st.number_input(
        f"How many conditions for {label.lower()}?", 
        min_value=1, value=2, step=1,
        key=f"{key_prefix}_n_conditions"
    )

    use_custom_names = st.checkbox(
        "Name each condition manually", 
        value=True, 
        key=f"{key_prefix}_use_custom_names"
    )

    set_columns_per_group = st.checkbox(
        "Set defining metadata columns per condition", 
        value=True, 
        key=f"{key_prefix}_set_columns_per_group"
    )

    global_columns = []
    if not set_columns_per_group:
        global_columns = st.multiselect(
            "Defining metadata column across all conditions", 
            parameters, 
            key=f"{key_prefix}_global_columns"
        )
    col1, col2 = st.columns(2)
    descriptions = {}
    for i in range(n_conditions):
        if use_custom_names:
            with col1:
                name = st.text_input(
                    f"Condition name #{i+1}", 
                    value="", 
                    key=f"{key_prefix}_name_{i}"
                )
                name = name.strip() or f"{key_prefix}_group_{i+1}"
        else:
            name = f"{key_prefix}_group_{i+1}"

        if set_columns_per_group:
            with col2: 
                cols = st.multiselect(
                    f"Columns for {name}", 
                    parameters, 
                    key=f"{key_prefix}_cols_{i}"
                )
        else:
            cols = global_columns

        descriptions[name] = {
            "name": name,
            "column": cols
        }

    if st.checkbox("Save and preview", key=f"{key_prefix}_save_preview"):
        st.session_state[f"{key_prefix}_descriptions"] = descriptions
        st.success("✔️ Metadata saved.")

        preview_df = pd.DataFrame([
            {"Group": v["name"], "Metadata columns": ", ".join(v["column"])}
            for v in descriptions.values()
        ])
        st.dataframe(preview_df, use_container_width=True)

    return descriptions

def expand_columns_with_duplicates(final_chars, factor_values, dup_chars):
    for char in dup_chars:
        is_factor = char in factor_values
        i = 2
        while True:
            new_col = f"{char}_{i}"
            if is_factor and new_col not in factor_values:
                factor_values.add(new_col)
            if new_col not in final_chars:
                final_chars.append(new_col)
                break
            i += 1
    return final_chars, factor_values

def build_metadata_table(df: pd.DataFrame, char_list: list[str], data_dict: dict, label: str, key_prefix: str = ""):
    column_config = {}
    for char in char_list:
        base_char = re.sub(r'_\d+$', '', char)
        ontology_key = f"all_{base_char.replace(' ', '_')}_elements"
        options = set(data_dict.get(ontology_key, []))
        options = [opt for opt in options if opt is not None]
        if options:
            column_config[char] = st.column_config.SelectboxColumn(label=char, options=sorted(options), help="Ontology terms")
        else:
            column_config[char] = st.column_config.TextColumn(label=char)

    form_key = f"{key_prefix}_form_{label}"

    with st.form(form_key):
        edited_df = st.data_editor(df, column_config=column_config, use_container_width=True)
        if st.form_submit_button(f"✅ Save {label} metadata"):
            st.session_state[f"{key_prefix}_{label}_metadata_df"] = edited_df
            st.session_state[f"{key_prefix}_{label}_done"] = True
    return edited_df

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
st.set_page_config(page_title="Flexible SDRF Generator", layout="wide")
st.title("🧪 Flexible SDRF Generator")

# Load data dictionary and validators
if "data_dict" not in st.session_state:
    st.info("🔄 Loading ontology data dictionary for the first time...")
    data_dict, unimod = load_data()  # <-- from @st.cache_data
    st.session_state["data_dict"] = data_dict
    st.session_state["unimod"] = unimod
else:
    data_dict = st.session_state["data_dict"]
    unimod = st.session_state["unimod"]



# --- Step 0: Choose Template ---
local_dir = os.path.dirname(os.path.dirname(__file__))
species = ["", "human", "cell-line", "default", "invertebrates", "plants", "vertebrates"]
selected_species = st.selectbox(
    "Select a species for the SDRF template:",
    species,
    help="This selection impacts the default columns in your SDRF template."
)

if selected_species:
    folder_path = os.path.join(local_dir, "templates")
    template_path = os.path.join(folder_path, f"sdrf-{selected_species}.sdrf.tsv")
    template_df = pd.read_csv(template_path, sep="\t")
    REQUIRED_COLS = template_df.columns.tolist()
    st.session_state["required_columns"] = REQUIRED_COLS
else:
    REQUIRED_COLS = []

VALIDATORS = {
    "age": {
        "pattern": re.compile(r"^\d+(Y(\d+M(\d+D)?)?|M(\d+D)?|W|\d+Y-\d+Y)$"),
        "example": "40Y, 40Y5M2D, 8W, 40Y-85Y"
    },
    "sex": {
        "pattern": re.compile(r"^(M|F|X)$"),
        "example": "M, F, or X"
    },
    "individual": {
        "pattern": re.compile(r"^\d+$"),
        "example": "A positive integer (e.g., 1, 12, 345)"
    }
}

# Sample-level columns
SAMPLE_ONTOLOGY_PARAMETERS = ["organism", "ancestry category", "cell", "cell line", "disease", "developmental stage", "enrichment process", "organism part"]
SAMPLE_FREE_PARAMETERS = ["age", "sex", "individual", "compound", "compound concentration", "synthetic peptide"]
ALL_SAMPLE_PARAMETERS = SAMPLE_ONTOLOGY_PARAMETERS + SAMPLE_FREE_PARAMETERS
# Run-level columns
RUN_ONTOLOGY_PARAMETERS = ["alkylation reagent", "dissociation method", "instrument", "fractionation method", "reduction reagent"]
RUN_FREE_PARAMETERS = ["collision energy", "fragment mass tolerance", "precursor mass tolerance"]
RUN_LIST_PARAMETERS = ["cleavage agent details", "depletion"]
ALL_RUN_PARAMETERS = RUN_ONTOLOGY_PARAMETERS + RUN_FREE_PARAMETERS + RUN_LIST_PARAMETERS
ALL_RUN_PARAMETERS = [param for param in ALL_RUN_PARAMETERS if param != "data file"]
OTHER_PARAMS = ['data file', 'biological replicate', 'technical replicate', 'fraction identifier', 'label']
ALL_PARAMETERS = ALL_SAMPLE_PARAMETERS + ALL_RUN_PARAMETERS + OTHER_PARAMS
# Step: Describe Biological Conditions
with st.expander("Define Biological Conditions"):
    condition_descriptions = build_condition_descriptions_ui("Define Biological Conditions", ALL_SAMPLE_PARAMETERS, key_prefix="bio")

with st.expander("Add sample metadata"):
    if "bio_descriptions" not in st.session_state:
        st.warning("Please define biological conditions first.")
        st.stop()

    factor_values = set()
    bio_descriptions = st.session_state["bio_descriptions"]
    for v in bio_descriptions.values():
        factor_values.update(set(v["column"]))

    group_names = list(bio_descriptions.keys())

    filtered_chars = [char for char in ALL_SAMPLE_PARAMETERS if f"characteristics[{char}]" not in REQUIRED_COLS]
    sel_char = st.multiselect("Select additional metadata columns to annotate", options=filtered_chars, key="bio_sel_char_sample")
    final_chars = sel_char.copy()
    for col in REQUIRED_COLS:
        if 'replicate' in col:
            continue
        if col.startswith("characteristics[") and col.endswith("]"):
            inner = col[len("characteristics["):-1]
            if inner not in final_chars:
                final_chars.append(inner)

    dup_chars = st.multiselect("Is there any parameter you would need to annotate multiple times like e.g. two organisms within one sample?", options=final_chars, key="bio_dup_chars")
    final_chars, factor_values = expand_columns_with_duplicates(final_chars, factor_values, dup_chars)

    # Shared metadata table
    sample_chars = set(final_chars) - factor_values
    shared_df = pd.DataFrame({char: [""] for char in sorted(sample_chars)})
    shared_df["sample"] = ["shared"]
    shared_df = shared_df.set_index("sample")
    st.write('Annotate metadata shared amongst all biological conditions')
    edited_shared_df = build_metadata_table(shared_df, sample_chars, data_dict, "shared", key_prefix="sample")

    # Group-specific table
    group_df = pd.DataFrame({char: ["" for _ in group_names] for char in factor_values})
    group_df["sample"] = group_names
    group_df = group_df.set_index("sample")
    st.write('Annotate metadata specific to each biological condition')
    edited_group_df = build_metadata_table(group_df, factor_values, data_dict, "group", key_prefix="sample")

    # Final merge + validation
    if st.session_state.get("sample_shared_done") and st.session_state.get("sample_group_done"):
        bio_merged_df = merge_shared_and_group_metadata(st.session_state["sample_shared_metadata_df"], st.session_state["sample_group_metadata_df"])
        st.dataframe(bio_merged_df)
        st.session_state["sample_shared_done"] = False
        st.session_state["sample_group_done"] = False
        st.session_state["bio_merged_df"] = bio_merged_df
        for df, label in zip([st.session_state["sample_shared_metadata_df"], st.session_state["sample_group_metadata_df"]], ["Shared", "Group"]):
            for field, validator in VALIDATORS.items():
                if field in df.columns:
                    values = df[field].dropna().astype(str).unique()
                    invalid = [v for v in values if not validator["pattern"].match(v)]
                    if invalid:
                        st.warning(f"🚨 Invalid values in **{field}** ({label}):")
                        for v in invalid:
                            st.write(f"- `{v}` → expected: {validator['example']}")
                    else:
                        st.success(f"✅ All values in **{field}** ({label}) are valid!")

with st.expander('Sample structure'):
    #how many samples, how many bioreps, to which condition?
    conditions = condition_descriptions.keys()
    col1, col2 = st.columns(2)
    with col1:
    #how many smaples per condition do you have?
        n_samples = st.number_input(
            "How many samples per condition?",
            min_value=1, value=2, step=1,
            help="This will determine how many sample nodes are created under each condition."
        )
    with col2:
        n_bioreps = st.number_input(
            "How many biological replicates per sample?",
            min_value=1, value=1, step=1,
            help="This will determine how many biological replicate nodes are created under each sample."
        )
        # Initial design rows
    rows = []
    for cond in conditions:
        for s in range(1, n_samples + 1):
            sample_id = f"{cond}_S{s}"
            for b in range(1, n_bioreps + 1):
                    rows.append({
                        "Condition": cond,
                        "Sample ID": sample_id,
                        "BioRep ID": f"{sample_id}_B{b}"
                })

    bio_experimental_design = pd.DataFrame(rows)

    column_config = {
        "Condition": st.column_config.SelectboxColumn(
            label="Condition",
            options=conditions,
            help="Choose one of the predefined biological conditions."
        ),
        "Samples": st.column_config.NumberColumn(min_value=1, step=1),
        "BioReplicates per Sample": st.column_config.NumberColumn(min_value=1, step=1)
    }

    st.caption('Visual representation of your experimental design, feel free to edit the table below. ')
    edited_experimental_design = st.data_editor(
        bio_experimental_design,
        column_config=column_config,
        use_container_width=True,
        num_rows="dynamic"
    )
    if st.button("✅ Confirm Design Table"):
        st.session_state["experimental_design_df"] = edited_experimental_design
        st.success("Design saved! Here's your final experimental design table:")
        st.dataframe(edited_experimental_design, use_container_width=True)
with st.expander("Define Technical Conditions"):
    technical_descriptions = build_condition_descriptions_ui('Define Technical Conditions', ALL_RUN_PARAMETERS, key_prefix="tech")

with st.expander("Add technical metadata"):
    if "tech_descriptions" not in st.session_state:
        st.warning("Please define technical conditions first.")
        st.stop()
    tech_descriptions = st.session_state["tech_descriptions"]
    factor_values = set()
    for v in tech_descriptions.values():
        factor_values.update(set(v["column"]))

    group_names = list(tech_descriptions.keys())
    filtered_chars = [char for char in ALL_RUN_PARAMETERS if f"comment[{char}]" not in REQUIRED_COLS]
    sel_char = st.multiselect("Select additional metadata", options=filtered_chars, key="tech_sel_char_sample")
    final_chars = sel_char.copy()
    for col in REQUIRED_COLS:
        if any(keyword in col for keyword in ['replicate', 'data file', 'fraction', 'label']):
            continue
        if col.startswith("comment[") and col.endswith("]"):
            inner = col[len("comment["):-1]
            if inner not in final_chars:
                final_chars.append(inner)

    dup_chars = st.multiselect("Allow multiple values for these columns?", options=final_chars, key="tech_dup_chars")
    final_chars, factor_values = expand_columns_with_duplicates(final_chars, factor_values, dup_chars)

    # Shared metadata table
    sample_chars = set(final_chars) - factor_values
    shared_df = pd.DataFrame({char: [""] for char in sorted(sample_chars)})
    shared_df["sample"] = ["shared"]
    shared_df = shared_df.set_index("sample")
    edited_shared_df = build_metadata_table(shared_df, sample_chars, data_dict, "shared", key_prefix="run")

    # Group-specific table
    group_df = pd.DataFrame({char: ["" for _ in group_names] for char in factor_values})
    group_df["sample"] = group_names
    group_df = group_df.set_index("sample")
    edited_group_df = build_metadata_table(group_df, factor_values, data_dict, "group", key_prefix="run")

    if st.session_state.get("run_shared_done") and st.session_state.get("run_group_done"):
        tech_merged_df = merge_shared_and_group_metadata(st.session_state["run_shared_metadata_df"], st.session_state["run_group_metadata_df"])
        st.dataframe(tech_merged_df)
        st.session_state["tech_merged_df"] = tech_merged_df
        st.session_state["run_shared_done"] = False
        st.session_state["run_group_done"] = False

with st.expander('Run structure'):
    raw_files_text = st.text_area("Paste raw file names (one per line):", placeholder="file1.raw\nfile2.raw\n...")
    raw_files = [f.strip() for f in raw_files_text.splitlines() if f.strip()]
    st.write(f"Detected {len(raw_files)} raw files.")
    st.write("We encourage informative filenames to ease the assignment of metadata. Is there any part of your raw file names that maps to a specific metadata field?")
    #take one example file name, try splitting it on various delimiters like ; _ " ", and then ask the user to select the parts that correspond to sample, fraction, technical replicate, etc.
    if raw_files:
        example_filename = raw_files[0]
        st.write(f"Example raw file name: `{example_filename}`")
        st.write("We will try to extract metadata from the raw file names and use your first raw file as example")
        sections = re.split(r'[;_.-/\ ]', example_filename)
        #make a dataframe with the sections as one column, the possible parameters with a dropdown as the second column
        sections_df = pd.DataFrame(sections, columns=["Section"])
        sections_df["Parameter"] = ""
        sections_column_config = {
            "Parameter": st.column_config.SelectboxColumn(options=["", "Sample", "Fraction", "Technical replicate", "Biological replicate", "Label"])
        }

        edited_sections_df = st.data_editor(sections_df, use_container_width=True, column_config=sections_column_config, num_rows="dynamic")
        
        if st.button("✅ Save Section Parameters"):
            st.session_state["edited_sections_df"] = edited_sections_df
            st.success("Section parameters saved! You can now assign them to your raw files.")

        # Step 2: Specify design
        cols = st.columns(4)
        conditions = condition_descriptions.keys()
        with cols[0]:
            n_samples = st.number_input("Samples per condition", 1, 50, 1)
        with cols[1]:
            n_bioreps = st.number_input("Biological replicates", 1, 10, 1)
        with cols[2]:
            n_techreps = st.number_input("Technical replicates", 1, 10, 1)
        with cols[3]:
            n_fractions = st.number_input("Fractions per sample", 1, 10, 1)

        # Step 3: Labeling
        labeled = st.radio("Is this a labeled experiment (e.g. TMT)?", ["Yes", "No"])
        if labeled == "Yes":
            label_set = st.multiselect("Labels used (in order)", options=["TMT126", "TMT127N", "TMT127C", "TMT128N", "TMT128C", "TMT129N", "TMT129C", "TMT130N", "TMT130C", "TMT131"])

        # Step 4: Auto-generate a table for user to edit
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





cola, colb, colc = st.columns(3)
with cola:
    sort_option = st.multiselect('According to which level are your raw files ordered? Click in order', options=['Sample', 'Biological replicate', 'Technical replicate', 'Fraction', 'Label'])
with colb:
    labels = st.multiselect('Which labels are in your raw files?', options=sorted(set(data_dict.get("all_label_elements", []))))
    if labels != ['label free sample']:
        label_param = st.multiselect('Based on which parameter(s) were your samples labeled?', options=ALL_PARAMETERS)
        runs = st.selectbox('Are the same labels used in all runs?', options=['Yes', 'No'], index=0)
with colc:
    pooling = st.selectbox('Were any samples pooled into the same raw file?', options=['Yes', 'No'], index=1)
    if pooling == 'Yes':
        pooling_param = st.multiselect('Based on which parameter(s) were your samples pooled?', options=ALL_PARAMETERS)

if labels==['label free sample'] and pooling=='No':
    singleshot = True
    st.success('You have a one sample to one raw file experiment, we will merge your raw files to the experimental design based on your ordering')



conditions = condition_descriptions.keys()
col1, col2, col3, col4 = st.columns(4)

with col3:
    n_techreps = st.number_input(
        "How many technical replicates per biological replicate?",
        min_value=1, value=1, step=1,
        help="This will determine how many technical replicate nodes are created under each biological replicate."
    )
with col4:
    n_fractions = st.number_input(
        "How many fractions?",
        min_value=1, value=1, step=1,
        help="This will determine how many fraction nodes are created under each sample."
    )
# Initial design rows
rows = []
for cond in conditions:
    for s in range(1, n_samples + 1):
        sample_id = f"{cond}_S{s}"
        for b in range(1, n_bioreps + 1):
            bio_id = f"{sample_id}_B{b}"
            for t in range(1, n_techreps + 1):
                tech_id = f"{bio_id}_T{t}"
                for f in range(1, n_fractions + 1):
                    rows.append({
                        "Condition": cond,
                        "Sample ID": sample_id,
                        "BioRep ID": bio_id,
                        "TechRep ID": tech_id,
                        "Frac ID": f"{tech_id}_F{f}"
                })

experimental_design = pd.DataFrame(rows)

column_config = {
    "Condition": st.column_config.SelectboxColumn(
        label="Condition",
        options=conditions,
        help="Choose one of the predefined biological conditions."
    ),
    "Samples": st.column_config.NumberColumn(min_value=1, step=1),
    "BioReplicates per Sample": st.column_config.NumberColumn(min_value=1, step=1),
    "TechReplicates per BioRep": st.column_config.NumberColumn(min_value=1, step=1)
}

st.caption('Visual representation of your experimental design, feel free to edit the table below. ')
edited_experimental_design = st.data_editor(
    experimental_design,
    column_config=column_config,
    use_container_width=True,
    num_rows="dynamic"
)
st.write(label_param, labels)
if st.button("✅ Confirm Design Table"):
    st.session_state["experimental_design_df"] = edited_experimental_design
    st.success("Design saved! Here's your final experimental design table:")
    if singleshot:
        sort_mapping = {
            "Sample": "Sample ID",
            "Biological replicate": "BioRep ID",
            "Technical replicate": "TechRep ID",
            "Fraction": "Frac ID",
            "Label": "Label"
        }
        sort_columns = [sort_mapping.get(col, col) for col in sort_option if sort_mapping.get(col, col) in edited_experimental_design.columns]
        edited_experimental_design = edited_experimental_design.sort_values(by=sort_columns).reset_index(drop=True)
        edited_experimental_design['data file'] = raw_files
        st.dataframe(edited_experimental_design, use_container_width=True)
    else:
        st.write('We will automatically assign your raw files to the experimental design based on your ordering and the number of samples, biological replicates, technical replicates, fractions and labels you have selected.')
        #first get label distribution
        grouped_label = edited_experimental_design.groupby(label_param)