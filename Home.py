import streamlit as st
import pandas as pd
import re
import numpy as np

import ParsingModule
import warnings
warnings.filterwarnings("ignore")
import os
import json
import gzip
local_dir = os.path.dirname(__file__)
from PIL import Image
import base64
import io
st.set_page_config(
    page_title="lesSDRF",
    layout="wide",
    menu_items={
        "Get help": "https://github.com/compomics/lesSDRF/issues",
        "Report a bug": "https://github.com/compomics/lesSDRF/issues",
    },
)

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

def get_base64_image(image):
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    return img_str

my_logo = add_logo(logo_path="final_logo.png", width=149, height=58)

st.markdown(
    f"""
    <style>
        [data-testid="stSidebarNav"] {{
            background-image: url('data:image/png;base64,{get_base64_image(my_logo)}');
            background-repeat: no-repeat;
            background-position: 40px 20px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

#get local directory using os, and add the data folder to the path



# use streamlit cache data to load gzipped jsons files from folder
@st.cache_data
def load_data():
    local_dir = os.path.dirname(__file__)
    folder_path = os.path.join(local_dir, "data")
    unimod_path = os.path.join(local_dir, "ontology", "unimod.csv")
    data = {}
    for filename in os.listdir(folder_path):
        # do not load the files containing the following names: archae, bacteria, eukaryota, virus, unclassified, other sequences
        if re.search(r"archaea|bacteria|eukaryota|virus|unclassified|other sequences", filename):
            continue
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".json.gz"):
            try:
                with gzip.open(file_path, "rb") as f:
                    try:
                        json_bytes = f.read()
                        json_str = json_bytes.decode('utf-8')
                        file_data = json.loads(json_str)
                        filename_key = filename.replace(".json.gz", "")
                        data[filename_key] = file_data
                    except json.JSONDecodeError:
                        st.write(f"Error decoding JSON in file {file_path}")
            except gzip.BadGzipFile:
                st.write(f"Error reading file {file_path}: not a gzipped file")
        else:
            st.write(f"Skipping file {file_path}: not a gzipped file")

    unimod = pd.read_csv(unimod_path, sep="\t")
    return data, unimod


data_dict, unimod = load_data()
if "data_dict" not in st.session_state:
    st.session_state["data_dict"] = data_dict
if "unimod" not in st.session_state:
    st.session_state["unimod"] = unimod
st.title("Welcome to lesSDRF")
st.subheader("Spending less time on SDRF creates more time for amazing research")
st.write(
    """By providing metadata in a machine-readable format, other researchers can access your data more easily and you maximize its impact. 
    The Sample and Data Relationship Format ([SDRF](https://www.nature.com/articles/s41467-021-26111-3)) is the HUPO-PSI recognized metadata format within proteomics. lesSDRF will streamline this annotation process for you. This tool is developed by the [CompOmics group](https://compomics.com/) and published in [Nature Communications](https://www.nature.com/articles/s41467-023-42543-5). \n""")
st.write("""
    On this homepage, select the species-specific default SDRF file that matches your study and provide the raw file names. 
    Then, follow the steps in the sidebar.  
- Step 1: If you have a local metadata file, you can upload it to map to the SDRF file
- Step 2: Provide information on potential labels in your sample
- Step 3: Fill in the columns that are required for a valid SDRF
- Step 4: Fill in columns with additional information to further optimise your SDRF file  
- Step 5: For atypical experiment types, you can check community suggested columns \n
"""
)

st.markdown("""You are able to download your intermediate file at any given time, so you can come back to the other steps whenever suits you.  
Upload your intermediate SDRF file here:""")

upload_df = st.file_uploader(
    "Upload intermediate SDRF file", type=["tsv"], accept_multiple_files=False, help='Upload a previously saved SDRF file. It should be in tsv format and should not contain more than 500 samples'
)
if upload_df is not None:
    template_df = pd.read_csv(upload_df, sep='\t')
    if template_df.shape[0]>500:
        st.error('Too many samples, please upload a maximum of 500 samples')
    else:
        st.write(template_df)
        st.session_state["template_df"] = template_df

st.markdown("""In need of some inspiration? Download this example SDRF file to get an idea of the required output""")
with open(f'{local_dir}/example_SDRF.tsv', 'rb') as f:
    st.download_button("Download example SDRF", f, file_name="example.sdrf.tsv")

st.subheader("Start here with a completely new SDRF file")
species = ["","human", "cell-line", "default", "invertebrates", "plants", "vertebrates"]
selected_species = st.selectbox("""Select a species for the SDRF template which will contain the basic colummns to fill in for this specific species. 
If your species is not in the drop down list, you can always use the default template.""", 
species, help="This species selection will impact the default columns present in your SDRF template. You can always add more columns in step *Additional columns*.")

if selected_species != "":
    folder_path = os.path.join(local_dir, "templates")
    # Load the corresponding CSV file based on the selected species
    template_df = pd.read_csv(
        f"{folder_path}/sdrf-{selected_species}.sdrf.tsv",
        sep="\t",
    )
    template_df["comment[modification parameters]"] = np.nan
    template_df["comment[fragment mass tolerance]"] = np.nan
    template_df["comment[precursor mass tolerance]"] = np.nan

    # Add radio button to choose between methods
    input_method = st.radio(
        "Choose how to create your template:",
        ["Input raw file names", "Create empty rows"]
    )

    if input_method == "Input raw file names":
        # Ask user to upload filenames of their samples
        filenames = []
        uploaded_names = st.text_input(
            "Input raw file names as a comma or tab separated list", 
            help="The raw file names will be input in the comment[data file] column and are the basis of your SDRF file. Input maximum 500 raw files"
        )
        if uploaded_names is not None:
            #if comma separated, split on comma, if tab separated, split on tab
            if "," in uploaded_names:
                uploaded_names = uploaded_names.split(",")
            elif "\t" in uploaded_names:
                uploaded_names = uploaded_names.split("\t")
            elif " " in uploaded_names:
                uploaded_names = uploaded_names.split(" ")
            else:
                uploaded_names = [uploaded_names]
            #remove trailing and leading spaces
            uploaded_names = [name.strip() for name in uploaded_names]
            filenames.append(uploaded_names)

        if filenames and len(filenames[0]) > 500:
            st.error('Too many samples, please upload a maximum of 500 samples')
        elif filenames:
            st.write(f"Added filenames: {filenames[0]}")
            ## Store filenames in the dataframe
            template_df["comment[data file]"] = filenames[0]
            st.session_state["template_df"] = template_df

    else:  # Create empty rows
        n_rows = st.number_input(
            "Number of empty rows to create",
            min_value=1,
            max_value=500,
            value=1,
            help="Choose how many empty rows to create in your template (maximum 500)"
        )
        if st.button("Create template"):
            # Create empty DataFrame with n rows
            template_df = pd.DataFrame(
                np.nan, 
                index=range(n_rows),
                columns=template_df.columns
            )
            st.write(f"Created template with {n_rows} empty rows")
            st.session_state["template_df"] = template_df

    ## Show the data in a table
    st.write(template_df)
    if "template_df" not in st.session_state:
        st.session_state["template_df"] = template_df
    with st.sidebar:
        download = st.download_button("Press to download SDRF file",ParsingModule.convert_df(template_df), "intermediate_SDRF.sdrf.tsv", help="download your SDRF file")
        st.write("""Please refer to your data and lesSDRF within your manuscript as follows:
                 *The experimental metadata has been generated using lesSDRF and is available through ProteomeXchange with the dataset identifier [PXDxxxxxxx]*""")