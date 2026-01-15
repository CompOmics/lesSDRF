import gzip
import json
import os
import pickle
import re
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import pronto
import streamlit as st
from pronto import Ontology
from st_aggrid import (AgGrid, DataReturnMode, GridOptionsBuilder,
                       GridUpdateMode)
from streamlit_tree_select import tree_select


def help():
    """This module contains all parsable functions necessary to build the SDRF GUI"""
    print("This module contains all parsable functions necessary to build the SDRF GUI")
    print(
        "The get_json_subclasses function returns a nested dictionary of all subclasses of a given term in a json ontology"
    )
    print(
        "The get_obo_subclasses function returns a nested dictionary of all subclasses of a given term in an obo ontology"
    )
    print("The flatten function returns a list of all values in a nested dictionary")
    print(
        "The transform_nested_dict_to_tree function returns a list of dictionaries that can be used to build a tree in streamlit"
    )
    print(
        "The store_as_gzipped_json function stores a dictionary/list as a gzipped json file"
    )
    print(
        "The open_gzipped_json function opens a gzipped json file and returns the dictionary/list"
    )


def get_json_subclasses(ontology, term_id, term_label, d, nodes_dict=None, data=None):
    """This function takes the path to the ontology file in json format, the desired term id from the root node (e.g. http://www.ebi.ac.uk/efo/EFO_0000635) and the term label (e.g. 'organism part')
    and returns a nested dictionary of all subclasses of the given term.
    """
    if nodes_dict is None:  # load the json file only once
        with open(ontology) as f:
            data = json.load(f)
        nodes_dict = {
            node["id"]: node["lbl"]
            for node in data["graphs"][0]["nodes"]
            if all(key in node for key in ["id", "lbl"])
        }

    if term_id not in nodes_dict:
        return f"{term_id} node not in ontology"  # node not found in ontology, return early

    if term_label not in d:
        d[term_label] = {}  # add the parent to the dictionary

    for term in data["graphs"][0]["edges"]:  # iterate through the edges
        if (term["obj"] == term_id) and (
            term["pred"] in ["http://purl.obolibrary.org/obo/BFO_0000050", "is_a"]
        ):
            parent = term["sub"]
            if parent == "http://purl.obolibrary.org/obo/MONDO_0011876":
                continue  # skip MONDO_0011876
            parent_label = nodes_dict.get(parent)
            if parent_label is not None:
                if parent_label in d:
                    d[term_label][parent_label] = d[parent_label]
                    del d[parent_label]
                else:
                    d[term_label][parent_label] = {}
                get_json_subclasses(
                    ontology, parent, parent_label, d[term_label], nodes_dict, data
                )

    return d


def remove_duplicate_values(d):
    for k, v in d.items():
        if isinstance(v, dict):
            remove_duplicate_values(v)
        if k in v:
            del v[k]

    return d


def get_obo_subclasses(onto, obo_id, obo_label, d=None, distance=1):
    if d is None:
        d = defaultdict(dict)
    """This function is built on pronto.
    It takes the path to the ontology file in obo format, the desired term id from the root node (e.g. MS:1000031) and the term label (e.g. 'instrument model') 
    and returns a nested dictionary of all subclasses of the given term. To only get the direct subclasses, the distance is set to 1
    """

    subclasses = list(onto[obo_id].subclasses(distance=1))
    if len(subclasses) > 1:
        d[obo_label] = {}
        for i in subclasses[1:]:
            obo_id = i.id
            obo_label = i.name
            d[obo_label] = get_obo_subclasses(
                onto, obo_id, obo_label, defaultdict(dict), distance=1
            )
    else:
        d = {}

    d = remove_dupcliate_values(d)
    return d


def flatten(d):
    """This function takes a nested dictionary and returns all unique elements in the dictionary as a list"""
    if not isinstance(d, dict):
        print("Input is not a dictionary")
    items = []
    for k, v in d.items():  # iterate through the dictionary
        items.append(k)  # add the key to the list
        if isinstance(
            v, dict
        ):  # if the value is a dictionary, call the function recursively
            items.extend(flatten(v))
        else:
            items.append(v)
    items = list(set(items))
    return items


def transform_nested_dict_to_tree(d, parent_label=None, parent_value=None):
    """This function takes a nested dictionary and returns a tree like dictionary that can be used in streamlit streamlit_tree_select"""
    if not isinstance(d, dict):
        print("Input is not a dictionary")
    result = []
    for key, value in d.items():
        label = key
        if parent_label:
            label = f"{parent_label} , {key}"
        children = []
        if value:
            children = transform_nested_dict_to_tree(value, label, key)
        if children:
            result.append({"label": key, "value": label, "children": children})
        else:
            result.append({"label": key, "value": label})
    return result


def store_as_gzipped_json(data, filename):
    """ "Given a datatype to store and the filename, this function stores the data as a gzipped json file in .\\data"""
    path = (
        ".\\data\\"
        + filename
        + ".json.gz"
    )
    with gzip.open(path, "wt") as f:
        json.dump(data, f)
    return f"Stored {filename} as gzipped json"


def open_gzipped_json(filename):
    """ "Given a filename, this function opens the data that was stored as a gzipped json in .\\data"""
    path = (
        ".\\data\\"
        + filename
        + ".json.gz"
    )
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return data

def fill_in_from_list(df, column, values_list=None, multiple_in_one=False):
    """provide dataframe, column and optional a list of values. 
    reates an editable dataframe in which only that column can be modified possibly with the values from the list
    If the list is empty, the column is freely editable
    If the list contains only one value, the column is filled with that value
    If the list contains more than one value, a dropdown menu is created with the values from the list
    If multiple_in_one is True, multiple columns are created with the same dropdown menu"""
    columns_to_adapt = [column]
    df.fillna("empty", inplace=True)
    cell_style = {"background-color": "#ffa478"}
    builder = GridOptionsBuilder.from_dataframe(df)

    if values_list and (len(values_list)==1): # if there is only one value, fill in the column with that value
        df[column] = values_list[0]
        df.replace("empty", np.nan, inplace=True)
    elif values_list and (len(values_list)>1): # if there is a list of values, add a dropdown menu to the column
        # add '' to the beginning of values list so it starts with an empty input
        values_list.insert(0, "")
        values_list.insert(1, "NA") #add NA
        if multiple_in_one: # add columns based on number of values in values_list
            for i in range(len(values_list)-1):
                df[f"{column}_{i}"] = ""
                columns_to_adapt.append(f"{column}_{i}")
            builder.configure_columns(columns_to_adapt, editable=True, cellEditor="agSelectCellEditor", cellEditorParams={"values": values_list}, cellStyle = cell_style)
        else: # if not multiple_in_one, just add the column
            builder.configure_column(column,editable=True,cellEditor="agSelectCellEditor",cellEditorParams={"values": values_list}, cellStyle = cell_style)
        builder.configure_grid_options(enableRangeSelection=True, enableFillHandle=True, suppressMovableColumns=True, singleClickEdit=True)
        gridOptions = builder.build()
        grid_return = AgGrid(
            df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.MANUAL,
            data_return_mode=DataReturnMode.AS_INPUT)
        df = grid_return["data"]
        df.replace("empty", np.nan, inplace=True)
              
    elif values_list is None: # if there is no list of values, make the column editable
        builder.configure_column(column, editable=True, cellStyle = cell_style)
        builder.configure_grid_options(enableRangeSelection=True, enableFillHandle=True, suppressMovableColumns=True, singleClickEdit=True)
        gridOptions = builder.build()
        grid_return = AgGrid(
            df,
            gridOptions=gridOptions,
            update_mode=GridUpdateMode.MANUAL,
            data_return_mode=DataReturnMode.AS_INPUT)
        df = grid_return["data"]
        df.replace("empty", np.nan, inplace=True)
    return df

def multiple_ontology_tree(column, element_list, nodes, df, multiple_in_one = False):
    """
    This function asks the column name, all the elements for the drop down menu and the nodes for the tree.
    It asks for the number of inputs and then creates the input dataframe with in-cell drop down menus with the chosen values.
    """
    #get index of column based on name
    if column not in df.columns:
        df[column] = np.nan
    index = df.columns.get_loc(column)
    col1, col2, col3 = st.columns(3)
    columns_to_adapt = [column]
    with col1:
        multiple = st.radio(f"Are there multiple {column} in your data?", ("No", "Yes"))
        if multiple == "Yes":
            with col2:
                number = st.number_input(
                    f"How many different {column} are in your data?",
                    min_value=0,
                    step=1)
            with col3:
                if multiple_in_one:
                    multiple_in_one_sel = st.radio(f"Are there multiple {column} within one sample?", ("No", "Yes"))     
                    if multiple_in_one_sel == "Yes":
                        for i in range(number-1):
                            # add column next to the original column if it is not already there
                            if f"{column}_{i+1}" not in df.columns:
                                df.insert(index+1, f"{column}_{i+1}", "empty")
                            columns_to_adapt.append(f"{column}_{i+1}")
        else:
            number = 1

    with st.form("Select here your ontology terms using the autocomplete function or the ontology-based tree menu", clear_on_submit=True):
        col4, col5 = st.columns(2)
        with col4:
            # selectbox with search option
            element_list.append(" ")
            element_list = set(element_list)
            return_search = st.multiselect(
                "Select your matching ontology term using this autocomplete function",
                element_list,
                max_selections=number,
            )

        with col5:
            st.write("Or follow the ontology based drop down menu below")
            return_select = tree_select(
                nodes, no_cascade=True, expand_on_click=True, check_model="leaf"
            )
        all = return_search + return_select["checked"]
        all = [i.split(',')[-1] for i in all if i is not None]
        if (len(all) >= 1) & (len(all) != number):
            st.error(f"You need to select a total of {number}.")
        s = st.form_submit_button("Submit selection")
        if s:
            st.write(f"Selection contains: {all}")
 
    if s & (len(all) == 1) & number == 1:
        df[column] = all[0]
        st.experimental_rerun()

    else:
        df.fillna("empty", inplace=True)
        st.write("If all cells are correctly filled in click twice on the update button")
        cell_style = {"background-color": "#ffa478"}
        builder = GridOptionsBuilder.from_dataframe(df)
        builder.configure_columns(columns_to_adapt,editable=True,cellEditor="agSelectCellEditor",cellEditorParams={"values": all},cellStyle=cell_style)
        builder.configure_grid_options(enableRangeSelection=True, enableFillHandle=True, suppressMovableColumns=True, singleClickEdit=True)
        go = builder.build()
        grid_return = AgGrid(df,gridOptions=go,update_mode=GridUpdateMode.MANUAL,data_return_mode=DataReturnMode.AS_INPUT)
        df = grid_return["data"]
    df.replace("empty", np.nan, inplace=True)   
    return df


def validate_sdrf(file_path):
    """Runs SDRF validation and returns success status & messages."""
    try:
        result = subprocess.run(
            ["parse_sdrf", "validate-sdrf", "--sdrf_file", file_path],
            capture_output=True,
            text=True,
            check=False  # Prevent exception on failure
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stdout.strip()
    except FileNotFoundError:
        return None, "SDRF validation tool not found. Install `sdrf-pipelines` via `pip install sdrf-pipelines`."




# function check_df_for_ontology_terms
# checks if the dataframe contains ontology terms

def check_df_for_ontology_terms(df, columns_to_check, column_ontology_dict):
    clear_columns = []
    for i in columns_to_check:
        name = (i.split('[')[-1].split(']')[0]).replace(' ', '_')
        name = 'all_' + name + '_elements'
        #if the column is an ontology column
        if name in column_ontology_dict.keys():
            onto_elements = column_ontology_dict[name]
            elements = df[i].unique()
            elements = [i for i in elements if i is not np.nan]
            # check if elements are all in onto_elements
            # if not, return the elements that are not in the ontology
            if not set(elements).issubset(set(onto_elements)):
                not_in_onto = set(elements) - set(onto_elements)
                st.error(f'The following elements are not in the ontology: {not_in_onto}')
                clear_columns.append(i)
            elif set(elements).issubset(set(onto_elements)) and len(elements) >= 1:
                st.success(f'The column {i} contains only ontology terms')
        if i == 'characteristics[age]':
            if not check_age_format(df, 'characteristics[age]'):
                st.error(f'The age format is not correct. Please use the following format: 1Y 2M 3D')
                clear_columns.append(i)
        if i == 'characteristics[sex]':
            uniques = np.unique(df[i].values())
            accepted = ['M', 'F', 'unknown']
            # check if uniques contain value that is not in accepted
            if not set(uniques).issubset(set(accepted)):
                not_in_onto = set(uniques) - set(accepted)
                st.error(f'{not_in_onto} are not accepted in the characteristics[sex] column. Please use M, F or unknown')
                clear_columns.append(i)
    
    # if there are columns that are not in the ontology, ask if the user wants to clear them
    if len(clear_columns) >= 1:
        st.error(f'The following columns contain elements that are not in the ontology: {clear_columns}')
        st.write('Do you want to clear these columns?')
        y = st.checkbox("Yes")
        n = st.checkbox("No")
        if y:
            for i in clear_columns:
                df[i] = np.nan
                st.success(f'Column {i} has been cleared')

def check_age_format(df, column):
    """
    Check if the data in a column in a pandas dataframe follows the age formatting of Y M D.
    If a range, this should be formatted as e.g. 48Y-84Y.
    Parameters:
    df (pandas.DataFrame): The pandas dataframe to check.
    column (str): The name of the column to check.


    Returns:
    tuple: (bool, list) where bool indicates if all data in the column follows the age formatting 
           and list contains the wrong parts (if any).
    """
    wrong_parts = []
    for index, row in df.iterrows():
        if row[column] not in ["", "empty", "None", "Not available"]:
            if not re.match(r"^(\s*\d+\s*Y)?(\s*\d+\s*M)?(\s*\d+\s*D)?(|\s*-\s*\d+\s*Y)?(|\s*-\s*\d+\s*M)?(|\s*-\s*\d+\s*D)?(/)?(|\s*\d+\s*Y)?(|\s*\d+\s*M)?(|\s*\d+\s*D)?$", str(row[column])):
                wrong_parts.append(row[column])
    return False if wrong_parts else True, wrong_parts

def convert_df(df):
    """This function requires a dataframe and sorts its columns as source name - characteristics - others - comment. 
    Leading and trailing whitespaces are removed from all columns
    It then converts the dataframe to a tsv file and downloads it
    It also adds an comment[tool metadata] to indicate it was built with lesSDRF and ontology versioning"""
    df["comment[tool metadata]"] = "lesSDRF v0.1.0"
    #sort dataframe so that "source name" is the first column
    cols = df.columns.tolist()
    #get all elements from the list that start with "characteristic" and sort them alphabetically
    characteristic_cols = sorted([i for i in cols if i.startswith("characteristic")])
    #first elements in characteric_cols should always be characteristics[organism]	characteristics[organism part]
    if "characteristics[organism]" in characteristic_cols:
        characteristic_cols.remove("characteristics[organism]")
        characteristic_cols.insert(0, "characteristics[organism]")
    if "characteristics[organism part]" in characteristic_cols:
        characteristic_cols.remove("characteristics[organism part]")
        characteristic_cols.insert(1, "characteristics[organism part]")
    comment_cols = sorted([i for i in cols if i.startswith("comment")])
    factor_value_cols = sorted([i for i in cols if i.startswith("factor")])
    #get all columns that don't start with "characteristic" or "comment"
    other_cols = [i for i in cols if i not in characteristic_cols and i not in comment_cols and i not in factor_value_cols and i not in ["source name"]]
    #reorder the columns, add "source name" if it is missing
    if "source name" not in df:
        df["source name"] = ""
    new_cols = ["source name"] + characteristic_cols + other_cols + comment_cols + factor_value_cols
    df = df[new_cols]
    #if a column name contains _ followed by a number, remove the underscore and the number
    df.columns = [re.sub(r"(_\d+)", "", i) for i in df.columns]
    #remove leading and trailing whitespaces from all columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    _, file_path = tempfile.mkstemp()
    df.to_csv(file_path, index=False, sep="\t", encoding="utf-8")
    is_valid, message = validate_sdrf(file_path)
    os.remove(file_path)

    if is_valid is None:
        st.error(message)
    elif is_valid:
        st.success("✅ SDRF validation passed!")
    else:
        # Filter out 'nan' errors in 'characteristics[organism]'
        filtered_message = "\n".join(
            line for line in message.split("\n") 
            if '"nan"' not in line and 'characteristics[organism]' not in line
        ).strip()

        if filtered_message:  # Show warning only if there are remaining errors
            st.warning(f"⚠️ SDRF validation failed! You can still download the file, but it may be invalid.\n\n**Details:**\n{filtered_message}")


    return df.to_csv(index=False, sep="\t").encode("utf-8")


def autocomplete_species_search(taxum_list, search_term):
    col1, col2 = st.columns(2)
    if (search_term != "") and (search_term != None):
        # Use the filter method to dynamically filter the list of options
        filtered_options = list(filter(lambda x: search_term.lower() in x.lower(), taxum_list))
        exact_match = list(filter(lambda x: search_term.lower() == x.lower(), taxum_list))
        if exact_match:
            with col1:
                st.write(f"An exact match was found: **{exact_match[0]}**")
            with col2:
                use_exact_match  = st.checkbox("Use exact match", key=f"exact_{search_term}")
                if use_exact_match:
                    return exact_match[0]
        # if length is between 0 and 500, display the options
        if len(filtered_options) > 0 and len(filtered_options) < 500:
            with col1:
                selected_options = st.multiselect("Some options closely matching your search time could be found", filtered_options)      
            # Display the selected options
            with col2:
                if selected_options:
                    st.write("You selected:", selected_options)
                    use_options = st.checkbox("Use selected options", key=f"selected_{search_term}")
                    if use_options:
                        return selected_options
        if len(filtered_options) > 500:
            st.write("Too many closely related options to display (>500). Please refine your search.")
        if len(filtered_options) == 0:
            st.write("No options found. Please refine your search.")
