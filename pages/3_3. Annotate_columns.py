import re
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import json
import gzip
import ParsingModule
import os
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from streamlit_tree_select import tree_select
from PIL import Image
import base64
import io
import common

st.set_page_config(
    page_title="Annotate columns",
    layout="wide",
    page_icon="🧪",
    menu_items={
        "Get help": "https://github.com/compomics/lesSDRF/issues",
        "Report a bug": "https://github.com/compomics/lesSDRF/issues",
    },
)


import common
common.inject_sidebar_logo()

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

@st.cache_data
def load_organism_data():
    #find dir one up from current dir
    data_dir = os.path.dirname(os.path.dirname(__file__))
    folder_path = os.path.join(data_dir, "data")
    data = {}
    for filename in os.listdir(folder_path):
        # only load the files containing the following names: archae, bacteria, eukaryota, virus, unclassified, other sequences
        if re.search(r"archaea|bacteria|eukaryota|virus|unclassified|other_sequences", filename):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith(".json.gz"):
                try:
                    with gzip.open(file_path, "rb") as f:
                        file_data = json.load(f)
                        filename_key = filename.replace(".json.gz", "")
                        data[filename_key] = file_data
                except gzip.BadGzipFile:
                    st.write(f"Error reading file {file_path}: not a gzipped file")
            else:
                st.write(f"Skipping file {file_path}: not a gzipped file")
        else:
            continue
    return data

def organism_selection(species):
    lem = organism_data[f"all_{species}_elements"]
    search_term = st.text_input(f"Search for an {species} species here", "")
    ret = ParsingModule.autocomplete_species_search(lem, search_term)
    if ret != None:
        return ret

def clean_final_str(final_str):
    final_str  = final_str.replace(';;', ';')
    #if it ends on TA=, not followed by anything remove TA=
    if final_str.endswith("TA="):
        final_str = final_str[:-3]
    #add a space before each ;
    final_str = final_str.replace(';', '; ')
    return final_str

data_dict = st.session_state["data_dict"]

# unimod = data_dict["unimod_dict"]
# === CATEGORY: Free text input ===
FREE_TEXT_COLUMNS = [
    "source name",
    "assay name",
    "characteristics[age]", #requires validation,
    "characteristics[sex]", #validation
    "characteristics[individual]" #needs validation, Only numbers?
]

FREE_TEXT_WITH_QUESTION_COLUMNS = [
    "comment[collision energy]", #multipel within one experiment?
    "comment[precursor mass tolerance]",#indicate unit, validate
    "comment[fragment mass tolerance]", #indicate unit, validate
    "characteristics[compound]",#multipel within one experiment?
    "characteristics[concentration of compound]"# multipel within one experiment?
]

# === CATEGORY: Ontology selection ===
ONTOLOGY_COLUMNS = [
    "comment[alkylation reagent]",
    "characteristics[ancestry category]",
    "characteristics[cell type]",
    "characteristics[cell line]",
    "characteristics[disease]",
    "characteristics[developmental stage]",
    "comment[dissociation method]",
    "characteristics[enrichment process]",
    "characteristics[organism part]",
    "comment[instrument]",
    "comment[fractionation method]",
    "comment[reduction reagent]"
]

# === CATEGORY: Multiple options from list ===
LIST_COLUMNS = [
    "comment[cleavage agent details]", #options are in data_dict["cleavage_list"]
    "technology type", #options are "proteomic profiling by mass spectrometry","metabolomics profiling by mass spectrometry","other"
    "comment[depletion]", #only options are depleted fraction, bound fraction, not depleted
    "characterics[synthetic peptide]" #only options are synthetic vs not synthetic
]
LIST_OPTIONS = {
                "comment[cleavage agent details]": data_dict.get("cleavage_list", []),
                "technology type": [
                    "proteomic profiling by mass spectrometry",
                    "metabolomics profiling by mass spectrometry",
                    "other"
                ],
                "comment[depletion]": [
                    "depleted fraction",
                    "bound fraction",
                    "not depleted"
                ],
                "characterics[synthetic peptide]": [
                    "synthetic",
                    "not synthetic"
                ]
            }

# === CATEGORY: Structured modification annotation ===
MODIFICATION_COLUMNS = [
    "comment[modification parameters]" #needs special
]

# === CATEGORY: Organism ===
ORGANISM_COLUMN = [
    "characteristics[organism]"
]

# === CATEGORY: replicate info ===
REP_COLUMNS = [
    "comment[fraction identifier]", #chose within a range? or free text?
    "comment[technical replicate]", #range
    "characteristics[biological replicate]" #range
]

ADDITIONAL_COLUMNS = [
    "factor value",
    "characteristics[age]",
    "characteristics[ancestry category]",
    "characteristics[biological replicate]",
    "characteristics[cell line]",
    "characteristics[cell type]",
    "characteristics[developmental stage]",
    "characteristics[disease]",
    "characteristics[individual]",
    "characteristics[organism part]",
    "characteristics[organism]",
    "characteristics[sex]",
    "characteristics[enrichment process]",
    "characteristics[compound]",
    "characteristics[concentration of compound]",
    "comment[modification parameters]",
    "comment[cleavage agent details]",
    "comment[data file]",
    "comment[fraction identifier]",
    "comment[instrument]",
    "comment[label]",
    "comment[technical replicate]",
    "comment[fragment mass tolerance]",
    "comment[precursor mass tolerance]",
    "comment[dissociation method]",
    "comment[depletion]", 
    "comment[collision energy]"
]



COLUMN_EXPLANATIONS = {
    "source name": "The source name is the unique sample name (it can be present multiple times if the same sample is used several times in the same dataset) eg. healthy_patient_1, diseased_patient_1. If you did not add it using the previous mapping function, you can input it here manually.",
    "assay name": "The assay name is a name for the run file, this has to be a uniqe value eg. run 1 - run 2 - run 3_fraction1. If you did not add it using the previous mapping function, you can input it here manually.",
    "factor value":"You can choose one or multiple columns that define the **factor value** in your experiment. This column indicates which experimental factor/variable is used as the hypothesis to perform the data analysis. If there are multiple factor values, we ask the user to take care in assigning biological and technical replicates",
    "technology type": "Choose the technology type from a set of options:",
    "characteristics[age]": "Input the ages of your samples using the Years Months Days format, e.g. 1Y 2M 3D. Age ranges should be formatted as e.g. 1Y-3Y",
    "comment[alkylation reagent]": "Input the alkylation reagent that was used in your experiment",
    "characteristics[ancestry category]": "Input the ancestry of your samples",
    "characteristics[cell type]": "Input the cell type of your sample",
    "characteristics[cell line]": "Input the cell line of your sample if one was used",
    "comment[cleavage agent details]": "Input the cleavage agent details of your sample",
    "characteristics[compound]": "If a compound was added to your sample, input the name here",
    "characteristics[concentration of compound]": "Input the concentration with which the compound was added to your sample",
    "comment[collision energy]": "Input the collision energy that was used in your experiment",
    "characteristics[developmental stage]": "Input the developmental stage of your sample",
    "characteristics[disease]": "If you have healthy and control samples, indicate healthy samples using *normal*. Input the disease for the other samples using the ontology",
    "comment[dissociation method]": "Input the dissociation method that was used in your experiment",
    "characteristics[enrichment process]": "Input the enrichment process that was used in your experiment",
    "comment[instrument]": "Input the instrument that was used in your experiment",
    "characteristics[individual]": "Do you have multiple individuals in your data?",
    "comment[label]": "Input the label that was used in your experiment. If no label was added, indicate this using *label free sample*.",
    "characteristics[organism]": "Select the species that is present in your sample",
    "characteristics[organism part]": "Select the part of the organism that is present in your sample",
    "comment[reduction reagent]": "Input the reduction reagent that was used in your experiment",
    "characteristics[sex]": "Are there multiple sexes in your data?",
    "comment[technical replicate]": "Do you have technical replicates?",
    "characteristics[biological replicate]": "Do you have biological replicates?",
    "comment[fragment mass tolerance]": "Input the fragment mass tolerance and **the unit (ppm or Da)**. click Update twice when finished",
    "comment[precursor mass tolerance]": "Input the precursor mass tolerance and **the unit (ppm or Da)**. click Update twice when finished",
    "characterics[synthetic peptide]": "If the sample is a synthetic peptide library, indicate this by selecting *synthetic* or *not synthetic*",
    "comment[depletion]": "Input depleted fractions",
    "comment[modification parameters]": "First select the modifications that are in your sample using the drop down autocomplete menu. After selection you will need to choose the modification type, position and target amino acid. Modification type can be fixed, variable or annotated. Annotated is used to search for all the occurrences of the modification into an annotated protein database file like UNIPROT XML or PEFF. Position can be anywhere, protein N-term, protein C-term, any N-term or any C-term. Target amino acid can be any amino acid or X if it's not in the list. You can also select multiple amino acids. If the modification of your choice is not available in the drop down list, select \"Other\" and input the modification name, chemical formula and mass of your custom modification. The final modification will be formatted following the SDRF guidelines after which you can click on \"Submit modifications\"."
}

# Define the default button color (you can adjust this as desired)
default_color = "#ff4b4b"

# Define the button CSS styles
button_styles = f"""
    background-color: white;
    color: {default_color};
    border-radius: 20px;
    padding: 10px 20px;
    border: 1px solid {default_color};
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
"""
st.title("""3. Annotate columns""")
# if df is not in the session state, don't run all the code below
if "template_df" not in st.session_state:
    st.error("Please fill in the template file in the Home page first", icon="🚨")  
    st.stop()
else:
    df = st.session_state["template_df"]


empty_columns = [col for col in df.columns if df[col].isnull().all() or (df[col] == "None").all()]
filled_columns = [col for col in df.columns if col not in empty_columns]
total_columns = len(df.columns)
filled_count = len(filled_columns)
progress = filled_count / total_columns


st.subheader("Current SDRF File")
st.progress(progress)
st.write(f"{filled_count}/{total_columns} columns annotated")
# ---- STEP 1: Build AgGrid with dynamic column config
# We'll rebuild builder every time to respect selected column
checkbox_values = {}

with st.expander(f"✅ Filled columns: {len(filled_columns)}"):
    for col in filled_columns:
        col1, col2 = st.columns([3,1])
        with col1:
            st.checkbox(col, value=True, disabled=True)
        with col2:
            if st.button(f"Clear {col}", key=f"clear_{col}"):
                df[col] = np.nan  # or "empty"
                if col not in empty_columns:
                    empty_columns.append(col)
                if col in filled_columns:
                    filled_columns.remove(col)
                st.success(f"Cleared column {col}. It’s now marked as empty.")
                st.session_state["template_df"] = df
                st.rerun()

with st.expander(f"➕ Optional columns (add if relevant)"):
    for col in ADDITIONAL_COLUMNS:
        if col not in df.columns:
            if st.button(f"Add {col}", key=f"add_{col}"):
                df[col] = np.nan  # or "empty"
                empty_columns.append(col)
                st.success(f"Added optional column: {col}")
                st.session_state["template_df"] = df
                st.rerun()

# ---- STEP 1: Build AgGrid with dynamic column config
# We'll rebuild builder every time to respect selected column
selected_column = st.selectbox(
    f"Required columns: select to annotate: {len(empty_columns)}",
    [""] + empty_columns
)

if selected_column:
    st.info(COLUMN_EXPLANATIONS.get(selected_column, f"Please annotate {selected_column}."))
    builder = GridOptionsBuilder.from_dataframe(df)
    builder.configure_grid_options(enableRangeSelection=True, enableFillHandle=True, singleClickEdit=True)

    extracted_name = re.search(r"\[(.*?)\]", selected_column)
    extracted_name = extracted_name.group(1) if extracted_name else selected_column
    extracted_name = extracted_name.replace(' ', '_')

    if selected_column in FREE_TEXT_COLUMNS:
        st.write(f"Selected column type: {type(selected_column)}")
        st.write(f"{extracted_name} requires free text input annotation")
        builder.configure_column(selected_column, editable=True,cellEditor="agTextCellEditor",cellStyle={"background-color": "#ffa478"})

    elif selected_column in ONTOLOGY_COLUMNS:
        st.write(f"Column '{extracted_name}' requires ontology-based annotation.")
        df, columns_to_edit, num_required = ParsingModule.structure_questions(df, selected_column)
        #exception for cell type
        if extracted_name == "cell_type":
            extracted_name = "cell"
        ontology_elements = data_dict.get(f"all_{extracted_name}_elements", [])
        ontology_tree = data_dict.get(f"{extracted_name}_nodes", {})
        selected_terms = ParsingModule.select_ontology_terms(
            selected_column,
            ontology_elements,
            ontology_tree,
            num_required=num_required
        )
        if selected_terms:
            if len(selected_terms) == 1:
                for col in columns_to_edit:
                    df[col] = selected_terms[0]
                    st.session_state["template_df"] = df  
            else:
                # === STEP 3: configure dropdown in AgGrid
                builder.configure_columns(
                    columns_to_edit,
                    editable=True,
                    cellEditor="agSelectCellEditor",
                    cellEditorParams={"values": selected_terms},
                    cellStyle={"background-color": "#ffa478"}
                )
                st.session_state["template_df"] = df 
        

    elif selected_column in LIST_COLUMNS:
        st.write(f"Column {extracted_name} has a designated list of options within the drop down")
        df, columns_to_edit, num_required = ParsingModule.structure_questions(df, selected_column)
        full_options = LIST_OPTIONS.get(selected_column, [])
        selected_terms = st.multiselect(
            f"Select allowed options for {extracted_name}:", 
            full_options,
            max_selections = num_required
        )
        if len(selected_terms) == num_required:
            if len(selected_terms) == 1:
                st.success(f"Selected: {selected_terms}. Only one option, will be automatically filled in. Just click Update")
                for col in columns_to_edit:
                    df[col] = selected_terms[0]
                    st.session_state["template_df"] = df
            else:
                # === STEP 3: configure dropdown in AgGrid
                st.success(f"Selected: {selected_terms}. Multiple options, use drop down menu within the table and click Update when finished")
                builder.configure_columns(
                    columns_to_edit,
                    editable=True,
                    cellEditor="agSelectCellEditor",
                    cellEditorParams={"values": selected_terms},
                    cellStyle={"background-color": "#ffa478"}
                )

    elif selected_column in FREE_TEXT_WITH_QUESTION_COLUMNS:
        st.write(f"Column {extracted_name} has a designated list of options within the drop down")
        df, columns_to_edit, num_required = ParsingModule.structure_questions(df, selected_column)
        # Fill in num_required

        builder.configure_columns(
            columns_to_edit,
            editable=True,
            cellEditor="agTextCellEditor",
            cellStyle={"background-color": "#ffa478"}
        )

    elif selected_column in ORGANISM_COLUMN:
        # Ensure base column exists
        if selected_column not in df.columns:
            df[selected_column] = np.nan

        # STEP 1: structure dataframe (multiple/multiple-in-one)
        df, columns_to_edit, num_required = ParsingModule.structure_questions(df, selected_column)

        classical_model_organisms = [
            "Homo sapiens", "Mus musculus", "Drosophila Melanogaster",
            "Arabidopsis thaliana", "Xenopus laevis", "Xenopus tropicalis",
            "Saccharomyces cerevisiae", "Caenorhabditis elegans",
            "Danio rerio", "Escherichia coli", "Cavia porcellus"
        ]

        st.session_state.setdefault("selected_species", set())

        model_type = st.radio("Are all used organisms a classical model organism?", ["Yes", "No"])

        selected_species = set()

        if model_type == "Yes":
            selected_species = set(st.multiselect(
                f"Select {num_required} classical model organism(s):",
                classical_model_organisms,
                max_selections=num_required
            ))

        elif model_type == "No":
            st.write("Loading NCBITaxon ontology... (may take a few seconds)")
            organism_data = load_organism_data()

            if organism_data:
                st.success("Ontology loaded!")
            else:
                st.error("Failed to load ontology data.")

            url = "https://www.ebi.ac.uk/ols/ontologies/ncbitaxon"
            st.markdown(f"[Open OLS ontology tree]({url})", unsafe_allow_html=True)

            tabs = st.tabs(['Eukaryota', 'Archaea', 'Bacteria', 'Viruses', 'Unclassified', 'Other'])
            categories = ['eukaryota', 'archaea', 'bacteria', 'virus', 'unclassified', 'other_sequences']

            for tab, category in zip(tabs, categories):
                with tab:
                    ret = organism_selection(category)
                    if ret:
                        selected_species.update(ret if isinstance(ret, list) else [ret])

        st.write(f"Selected species so far: {selected_species}")

        # Allow user to adjust count
        if len(selected_species) > num_required:
            st.warning(f"Selected {len(selected_species)} species but need {num_required}. Please deselect extra species below.")
            deselect = st.multiselect("Deselect species:", list(selected_species))
            selected_species -= set(deselect)

        elif len(selected_species) < num_required:
            st.warning(f"Selected {len(selected_species)} species but need {num_required}.")

        ready = st.checkbox("Confirm selected species")

        if ready and len(selected_species) == num_required:
            selected_terms = list(selected_species)

            if len(selected_terms) == 1:
                for col in columns_to_edit:
                    df[col] = selected_terms[0]
                # st.success(f"Auto-filled columns {columns_to_edit} with: {selected_terms[0]}")
            else:
                builder.configure_columns(
                    columns_to_edit,
                    editable=True,
                    cellEditor="agSelectCellEditor",
                    cellEditorParams={"values": selected_terms},
                    cellStyle={"background-color": "#ffa478"}
                )

            # Clean up session state
            del st.session_state["selected_species"]

    elif selected_column in MODIFICATION_COLUMNS:
        st.write("Select modifications and their details for annotation.")

        unimod = data_dict["unimod_dict"]
        mod_options = sorted(list(unimod.keys()))
        mod_options.append("Other")

        modification_types = ["Fixed", "Variable", "Annotated"]
        positions = ["Anywhere", "Protein N-term", "Protein C-term", "Any N-term", "Any C-term"]
        target_amino_acids = ["X","G","A","L","M","F","W","K","Q","E","S","P","V","I","C","Y","H","R","N","D","T"]

        selected_mods = st.multiselect("Select modifications:", mod_options)

        final_modifications = []
        for mod in selected_mods:
            st.markdown(f"**Configuration for: {mod}**")
            col1, col2, col3 = st.columns(3)

            with col1:
                mt_sel = st.selectbox("Modification type:", modification_types, key=f"mt_{mod}")
            with col2:
                pos_sel = st.selectbox("Position:", positions, key=f"pos_{mod}")
            with col3:
                aa_sel = st.multiselect("Target amino acids:", target_amino_acids, key=f"aa_{mod}")
                aa_str = ",".join(aa_sel)

            if mod == "Other":
                col4, col5, col6 = st.columns(3)
                with col4:
                    name = st.text_input("Name:", key=f"name_{mod}")
                with col5:
                    formula = st.text_input("Formula:", key=f"formula_{mod}")
                with col6:
                    mass = st.text_input("Mass:", key=f"mass_{mod}")

                mod_string = f"NT={name};MT={mt_sel};PP={pos_sel};TA={aa_str};CF={formula};MM={mass}"
            else:
                mod_string = f"{unimod[mod]};MT={mt_sel};PP={pos_sel};TA={aa_str}"

            mod_string = clean_final_str(mod_string)
            final_modifications.append(mod_string)
            st.success(f"Final annotation: {mod_string}")

        st.write(f"Selected modifications: {final_modifications}")

        if final_modifications:
            confirm = st.checkbox("Click to add selected modifications to SDRF")

            if confirm:
                index = df.columns.get_loc(selected_column) if selected_column in df.columns else len(df.columns)
                for idx, mod in enumerate(final_modifications):
                    new_col_name = selected_column if idx == 0 else f"{selected_column}_{idx}"
                    if new_col_name not in df.columns:
                        st.write(new_col_name, idx, index)
                        df.insert(index + idx, new_col_name, mod)
                    else:
                        df[new_col_name] = mod
                st.success(f"Added {len(final_modifications)} modification columns.")

    elif selected_column in REP_COLUMNS:    
        st.write(f"{extracted_name} requires numerical annotation")
        replicate_presence = st.selectbox(f"Do you have {extracted_name}?", ("", "Yes", "No"))
        if replicate_presence == "No":
            st.success(f"All samples will be assigned {extracted_name}=1")
            df[selected_column] = 1
        elif replicate_presence == "Yes":
            st.info(f"You can now annotate the {extracted_name} number per sample (numeric values only).")     
            builder.configure_column(
                selected_column,
                editable=True,
                cellEditor="agTextCellEditor",
                cellStyle={"background-color": "#ffa478"}
        )

    else:
        # fallback: free text input
        builder.configure_column(
            selected_column,
            editable=True,
            cellEditor="agTextCellEditor",
            cellStyle={"background-color": "#ffa478"}
        )



    gridOptions = builder.build()

    grid_return = AgGrid(
        df,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.MANUAL,
        data_return_mode=DataReturnMode.AS_INPUT,
        fit_columns_on_grid_load=False
    )

    # st.session_state["df"] = grid_return["data"]
    empty_columns = [col for col in df.columns if df[col].isnull().all() or (df[col] == "None").all()]

    if grid_return["data"] is not None and not grid_return["data"].equals(st.session_state.get("template_df")):
        # User clicked update manually
        st.session_state["template_df"] = grid_return["data"]

    df = st.session_state["template_df"]