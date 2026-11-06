"""Offline helpers used by the ontology-parsing notebooks (EFO_parser, OBO_parser, OWL_parser, unimod).

These functions turn ontology files into the gzipped JSON files stored in ``data/``.
They are not imported by the Streamlit app itself (see ``app_utils.py`` for that), so this
module deliberately has no Streamlit dependency.
"""
import gzip
import json
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def help():
    """This module contains all parsable functions necessary to build the SDRF GUI data files"""
    print(__doc__)


def get_json_subclasses(ontology, term_id, term_label, d, nodes_dict=None, data=None):
    """This function takes the path to the ontology file in json format, the desired term id from the root node
    (e.g. http://www.ebi.ac.uk/efo/EFO_0000635) and the term label (e.g. 'organism part')
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
    """This function is built on pronto.
    It takes a loaded pronto Ontology, the desired term id from the root node (e.g. MS:1000031) and the term label
    (e.g. 'instrument model') and returns a nested dictionary of all subclasses of the given term.
    To only get the direct subclasses, the distance is set to 1
    """
    if d is None:
        d = defaultdict(dict)

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

    d = remove_duplicate_values(d)
    return d


def flatten(d):
    """This function takes a nested dictionary and returns all unique elements in the dictionary as a list"""
    if not isinstance(d, dict):
        print("Input is not a dictionary")
    items = []
    for k, v in d.items():  # iterate through the dictionary
        items.append(k)  # add the key to the list
        if isinstance(v, dict):  # if the value is a dictionary, call the function recursively
            items.extend(flatten(v))
        else:
            items.append(v)
    items = list(set(items))
    return items


def transform_nested_dict_to_tree(d, parent_label=None, parent_value=None):
    """This function takes a nested dictionary and returns a tree like dictionary that can be used in
    streamlit_tree_select"""
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
    """Given a datatype to store and the filename, this function stores the data as a gzipped json file in ./data"""
    path = os.path.join(DATA_DIR, filename + ".json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    return f"Stored {filename} as gzipped json"


def open_gzipped_json(filename):
    """Given a filename, this function opens the data that was stored as a gzipped json in ./data"""
    path = os.path.join(DATA_DIR, filename + ".json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return data
