"""Shared Streamlit building blocks for the lesSDRF pages.

Memory and speed rules followed here (the hosted app used to run out of memory):
- Ontology data is loaded lazily, one file at a time, with ``st.cache_resource``: one shared, read-only
  copy per server process instead of a deep copy per session (``st.cache_data``) or per rerun.
- Nothing large is stored in ``st.session_state``; sessions only hold the SDRF table and small selections.
- Large term lists are searched on the server; only matching terms are sent to the browser.
- Ontology trees are only rendered on request and are pruned to a size the browser can handle.
- The SDRF is only validated when the user asks for it.
"""
from __future__ import annotations

import gzip
import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

import spec

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
LOGO_PATH = os.path.join(APP_DIR, "final_logo.png")
ICON_PATH = os.path.join(APP_DIR, "partial_logo.png")

MAX_ROWS = 5000          # rows in the SDRF (after label expansion)
MAX_FILES = 1000         # raw files entered on the Home page
SEARCH_LIMIT = 300       # search hits sent to the browser
TREE_BUDGET = 1_500_000  # max JSON size (bytes) of an ontology tree sent to the browser

CITATION = """Please refer to your data and lesSDRF within your manuscript as follows:
*The experimental metadata has been generated using lesSDRF and is available through ProteomeXchange
with the dataset identifier [PXDxxxxxxx]*"""

# Ontology-backed columns: column -> (term list file, tree file)
ONTOLOGY_COLUMNS = {
    "characteristics[organism part]": ("all_organism_part_elements", "organism_part_nodes"),
    "characteristics[tissue supergroup]": ("all_organism_part_elements", "organism_part_nodes"),
    "characteristics[sampling site]": ("all_organism_part_elements", "organism_part_nodes"),
    "characteristics[cell type]": ("all_cell_elements", "cell_nodes"),
    "characteristics[cell line]": ("all_cell_line_elements", "cell_line_nodes"),
    "characteristics[disease]": ("all_disease_elements", "disease_nodes"),
    "characteristics[developmental stage]": ("all_developmental_stage_elements", "developmental_stage_nodes"),
    "characteristics[ancestry category]": ("all_ancestry_category_elements", "ancestry_category_nodes"),
    "characteristics[enrichment process]": ("all_enrichment_elements", "enrichment_nodes"),
    "comment[instrument]": ("all_instrument_elements", "instrument_nodes"),
    "comment[label]": ("all_label_elements", "label_nodes"),
    "comment[dissociation method]": ("all_dissociation_elements", "dissociation_nodes"),
    "comment[fractionation method]": ("all_fractionation_method_elements", "fractionation_nodes"),
    "comment[reduction reagent]": ("all_reduction_reagent_elements", "reduction_nodes"),
    "comment[alkylation reagent]": ("all_alkylation_elements", "alkylation_nodes"),
}

MODEL_ORGANISMS = [
    "Homo sapiens", "Mus musculus", "Rattus norvegicus", "Danio rerio", "Drosophila melanogaster",
    "Caenorhabditis elegans", "Saccharomyces cerevisiae", "Arabidopsis thaliana", "Escherichia coli",
    "Xenopus laevis", "Xenopus tropicalis", "Cavia porcellus", "Sus scrofa", "Bos taurus",
    "Gallus gallus", "Oryza sativa", "Zea mays", "Bacillus subtilis",
]
ORGANISM_ALIASES = {
    "human": "Homo sapiens", "mouse": "Mus musculus", "rat": "Rattus norvegicus", "zebrafish": "Danio rerio",
    "fruit fly": "Drosophila melanogaster", "fruitfly": "Drosophila melanogaster", "drosophila": "Drosophila melanogaster",
    "c. elegans": "Caenorhabditis elegans", "worm": "Caenorhabditis elegans", "yeast": "Saccharomyces cerevisiae",
    "brewer's yeast": "Saccharomyces cerevisiae", "arabidopsis": "Arabidopsis thaliana", "thale cress": "Arabidopsis thaliana",
    "e. coli": "Escherichia coli", "pig": "Sus scrofa", "cow": "Bos taurus", "chicken": "Gallus gallus",
    "guinea pig": "Cavia porcellus", "rice": "Oryza sativa", "maize": "Zea mays",
}
TAXON_GROUPS = {
    "Eukaryota": "all_eukaryota_elements",
    "Bacteria": "all_bacteria_elements",
    "Archaea": "all_archaea_elements",
    "Viruses": "all_virus_elements",
    "Unclassified": "all_unclassified_elements",
    "Other sequences": "all_other_sequences_elements",
}

CLEAVAGE_AGENTS = [
    "Trypsin", "Trypsin/P", "Lys-C", "Lys-C/P", "Lys-N", "Arg-C", "Asp-N", "Asp-N ambic", "Chymotrypsin",
    "TrypChymo", "Glu-C", "V8-DE", "V8-E", "PepsinA", "CNBr", "Formic acid", "LysargiNase",
    "leukocyte elastase", "proline endopeptidase", "glutamyl endopeptidase", "2-iodobenzoate",
    "unspecific cleavage", "no cleavage",
]
MOD_TYPES = ["Fixed", "Variable", "Annotated"]
MOD_POSITIONS = ["Anywhere", "Protein N-term", "Protein C-term", "Any N-term", "Any C-term"]
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]


# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
def setup_page(title: str, icon: str | None = "🧪"):
    st.set_page_config(
        page_title=title,
        layout="wide",
        page_icon=icon,
        menu_items={
            "Get help": "https://github.com/compomics/lesSDRF/issues",
            "Report a bug": "https://github.com/compomics/lesSDRF/issues",
        },
    )
    st.logo(LOGO_PATH, size="large", icon_image=ICON_PATH)


def get_df() -> pd.DataFrame | None:
    return st.session_state.get("template_df")


def save_df(df: pd.DataFrame):
    st.session_state["template_df"] = spec.normalize_df(df).reset_index(drop=True)


def require_df() -> pd.DataFrame:
    """The current SDRF, or stop the page with a pointer to the Home page."""
    df = get_df()
    if df is None:
        st.error("Please create or upload an SDRF file on the Home page first.", icon="🚨")
        st.page_link("Home.py", label="Go to the Home page")
        st.stop()
    return df


def show_df(df: pd.DataFrame, title="**This is your current SDRF file.**"):
    st.write(title)
    st.dataframe(df.fillna(""), hide_index=True, height=min(35 * (len(df) + 1) + 3, 280))  # blank, not "None"


# --------------------------------------------------------------------------------------
# Shared, read-only ontology data
# --------------------------------------------------------------------------------------
def _read_json(name: str):
    with gzip.open(os.path.join(DATA_DIR, f"{name}.json.gz"), "rt", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource(show_spinner=False)
def load_terms(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A term list as (terms, lower-cased terms). Shared by all sessions; never mutate it."""
    values = _read_json(name)
    if isinstance(values, dict):
        values = list(values)
    terms = tuple(dict.fromkeys(str(v) for v in values if v))
    return terms, tuple(t.lower() for t in terms)


@st.cache_resource(show_spinner=False)
def load_unimod() -> dict:
    """Unimod name -> accession number (as a string)."""
    out = {}
    for name, value in _read_json("unimod_dict").items():
        m = re.search(r"AC=(?:UNIMOD:)?(\d+)", str(value))
        if m and name and not name.startswith("#"):
            out[name] = m.group(1)
    return out


@st.cache_resource(show_spinner=False)
def accession_map() -> dict:
    return _read_json("term_accessions")


def _prune(nodes, depth):
    out = []
    for node in nodes:
        item = {"label": node["label"], "value": node["value"]}
        if depth > 1 and node.get("children"):
            item["children"] = _prune(node["children"], depth - 1)
        out.append(item)
    return out


@st.cache_resource(show_spinner=False)
def load_tree(name: str) -> tuple[list, int | None]:
    """An ontology tree pruned to the deepest level that fits TREE_BUDGET. Returns (nodes, depth or None)."""
    nodes = _read_json(name)
    if len(json.dumps(nodes)) <= TREE_BUDGET:
        return nodes, None
    depth = 2
    while depth < 20 and len(json.dumps(_prune(nodes, depth + 1))) <= TREE_BUDGET:
        depth += 1
    return _prune(nodes, depth), depth


def search_terms(name: str, query: str, limit: int = SEARCH_LIMIT) -> list[str]:
    """Case-insensitive substring search; exact matches first, then prefix matches, then the rest."""
    q = query.strip().lower()
    if not q:
        return []
    terms, lower = load_terms(name)
    exact, prefix, other = [], [], []
    for term, low in zip(terms, lower):
        if q in low:
            if low == q:
                exact.append(term)
            elif low.startswith(q):
                prefix.append(term)
            else:
                other.append(term)
    prefix.sort(key=len)
    other.sort(key=len)
    return (exact + prefix + other)[:limit]


def canonical_term(name: str, value: str) -> str | None:
    """The ontology spelling of a value (case-insensitive match), or None."""
    terms, lower = load_terms(name)
    try:
        return terms[lower.index(str(value).strip().lower())]
    except ValueError:
        return None


@st.cache_resource(show_spinner="Loading NCBITaxon names (only needed once)...")
def _taxon_index(name: str) -> tuple[str, str]:
    """All names of a taxon group as one newline-separated string (+ lower-case copy) for fast search.

    Two flat strings take far less memory than a list of 1.7 million Python strings.
    """
    names = _read_json(name)
    text = "\n" + "\n".join(names) + "\n"
    return text, text.lower()


def search_taxa(group: str, query: str, limit: int = SEARCH_LIMIT) -> list[str]:
    q = query.strip().lower()
    if len(q) < 3:
        return []
    text, low = _taxon_index(TAXON_GROUPS[group])
    hits, start = [], 0
    while len(hits) < 20 * limit:  # collect a generous batch, then rank
        i = low.find(q, start)
        if i < 0:
            break
        s = low.rfind("\n", 0, i) + 1
        e = low.find("\n", i)
        hits.append(text[s:e])
        start = e
    hits.sort(key=lambda h: (h.lower() != q, not h.lower().startswith(q), len(h), h))
    return hits[:limit]


def find_organism(value: str) -> str | None:
    """Match a free-text organism name to NCBITaxon (aliases, model organisms, then eukaryotes/bacteria)."""
    v = str(value).strip()
    if v.lower() in ORGANISM_ALIASES:
        return ORGANISM_ALIASES[v.lower()]
    for org in MODEL_ORGANISMS:
        if org.lower() == v.lower():
            return org
    for group in ("Eukaryota", "Bacteria", "Archaea", "Viruses"):
        hits = search_taxa(group, v, limit=1)
        if hits and hits[0].lower() == v.lower():
            return hits[0]
    return None


def with_accession(column: str, label: str) -> str:
    """Write ontology terms in comment columns as NT=<label>;AC=<accession>, as the spec recommends."""
    ontologies = spec.ACCESSION_COLUMNS.get(spec.base_name(column))
    if not ontologies or not label or "=" in label or label in (
            spec.NOT_AVAILABLE, spec.NOT_APPLICABLE, spec.POOLED, spec.ANONYMIZED):
        return label
    entry = accession_map().get(label.strip().lower())
    if not entry:
        return label
    for ontology in ontologies:
        if ontology in entry:
            accession, name = entry[ontology]
            return f"NT={name};AC={accession}"
    return label


# --------------------------------------------------------------------------------------
# Writing values into the SDRF
# --------------------------------------------------------------------------------------
def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add missing columns (copies `col_1`, `col_2` go right after `col`)."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            continue
        base = spec.base_name(col)
        existing = [c for c in df.columns if spec.base_name(c) == base]
        if existing:
            df.insert(df.columns.get_loc(existing[-1]) + 1, col, None)
        else:
            df[col] = None
    return df


def fill_column(column: str, value, rows=None, rerun=True):
    """Write a value into a column, for all rows or only where the boolean list `rows` is True.

    `value` may also be a list with one value per row.
    """
    df = ensure_columns(require_df(), [column])
    if rows is None:
        df[column] = value
    else:
        df[column] = [value if keep else old for keep, old in zip(rows, df[column].tolist())]
    save_df(df)
    if rerun:
        st.rerun()


def row_scope(key: str) -> list[bool] | None:
    """Let the user pick which rows a fill applies to: all rows, a name pattern or selected files.

    This is how a group of samples (e.g. all tumour files) gets its own values without editing row by row.
    Returns None for all rows, otherwise one boolean per row.
    """
    df = require_df()
    id_col = "comment[data file]" if not spec.is_empty_column(df, "comment[data file]") else "source name"
    if spec.is_empty_column(df, id_col):
        return None
    names = [str(v) if v is not None else "" for v in df[id_col].tolist()]
    mode = st.radio("Apply to", ["all rows", f"rows whose {id_col} contains ...", "selected rows"],
                    horizontal=True, key=f"{key}_scope")
    if mode == "all rows":
        return None
    if mode.startswith("rows whose"):
        text = st.text_input(f"Text in {id_col} (not case-sensitive)", key=f"{key}_scope_text",
                             placeholder="e.g. tumour, _F1_, batch2")
        rows = [bool(text) and text.lower() in n.lower() for n in names]
    else:
        chosen = st.multiselect(f"Select {id_col}(s)", list(dict.fromkeys(names)), key=f"{key}_scope_files")
        rows = [n in chosen for n in names]
    st.caption(f"{sum(rows)} of {len(rows)} rows selected")
    return rows


def copy_columns(column: str, n: int) -> list[str]:
    return [column] + [f"{column}_{i}" for i in range(1, n)]


def edit_cells(columns: list[str], options: list[str] | None = None, key: str = "editor",
               transform=None, check=None):
    """Spreadsheet-like editor for some columns of the SDRF, saved with one click.

    options: restrict cells to these values (dropdown); None means free text.
    transform: function applied to every non-empty value on save (e.g. formatting).
    check: function(value) -> bool; values failing it block the save.
    """
    df = ensure_columns(require_df(), columns)
    context = [c for c in ("source name", "assay name", "comment[data file]", "comment[label]")
               if c in df.columns and c not in columns]
    view = df[context + columns].copy()
    if options:
        existing = [v for col in columns for v in view[col].tolist() if not spec.is_empty_value(v)]
        options = list(dict.fromkeys(list(options) + existing))  # never blank out existing values
        config = {c: st.column_config.SelectboxColumn(c, options=options) for c in columns}
    else:
        config = {c: st.column_config.TextColumn(c) for c in columns}
    with st.form(f"form_{key}", border=False):
        st.caption("Click a cell to edit it. You can paste a column of values copied from a spreadsheet (Ctrl+V).")
        edited = st.data_editor(view, column_config=config, disabled=context, hide_index=True,
                                key=f"de_{key}", num_rows="fixed")
        saved = st.form_submit_button("Save changes", type="primary")
    if saved:
        new = df.copy()
        bad = []
        for col in columns:
            values = []
            for v in edited[col].tolist():
                if spec.is_empty_value(v):
                    values.append(None)
                    continue
                v = str(v).strip()
                if transform:
                    v = transform(v)
                if check and not check(v):
                    bad.append(v)
                values.append(v)
            new[col] = values
        if bad:
            st.error(f"These values are not valid: {', '.join(sorted(set(bad))[:20])}")
            return
        save_df(new)
        st.rerun()


def assign_values(column: str, values: list[str], key: str, allow_copies: bool | None = None):
    """Put chosen values into a column: fill every row with one value, or pick per row from a dropdown.

    allow_copies: offer extra copies of the column (several values per sample), e.g. two organism parts.
    """
    rule = spec.rule_for(column, spec.selected_templates())
    if allow_copies is None:
        allow_copies = rule.multiple
    values = [with_accession(column, v) for v in values]
    options = list(dict.fromkeys(values + rule.reserved_words))
    if not options:
        return
    n_copies = 1
    if allow_copies and len(values) > 1:
        n_copies = st.number_input(
            "How many of these values can a single sample have? Each extra value goes into its own column",
            min_value=1, max_value=len(values), value=1, step=1, key=f"{key}_copies",
            help="For example a sample containing two organism parts gets two characteristics[organism part] columns.")
    columns = copy_columns(column, int(n_copies))
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        with c1:
            # Keyed on the chosen terms, so picking new terms resets the default to the first of them.
            value = st.selectbox("Fill with", options, key=f"{key}_bulk_{abs(hash(tuple(values)))}")
        with c2:
            target = st.selectbox("Column", columns, key=f"{key}_bulk_col") if len(columns) > 1 else column
        rows = row_scope(key)
        if st.button("Fill", type="primary", key=f"{key}_bulk_btn", disabled=rows is not None and not any(rows)):
            fill_column(target, value, rows)
    st.write("Or choose the value per row:")
    edit_cells(columns, options=options, key=key)


# --------------------------------------------------------------------------------------
# Pickers
# --------------------------------------------------------------------------------------
def _add_to_selection(sel_key, widget_key):
    picked = st.session_state.get(widget_key) or []
    current = st.session_state.setdefault(sel_key, [])
    for term in picked:
        if term not in current:
            current.append(term)
    st.session_state[widget_key] = []


def term_picker(key: str, terms_name: str | None, nodes_name: str | None = None,
                label: str = "ontology term", static_options: list[str] | None = None,
                allow_custom: bool = False) -> list[str]:
    """Collect ontology terms by searching (server-side) or browsing the tree. Returns the selection."""
    sel_key = f"sel_{key}"
    selection = st.session_state.setdefault(sel_key, [])
    c1, c2 = st.columns(2)
    with c1:
        if static_options is not None:
            st.multiselect(f"Select {label}s", [o for o in static_options if o not in selection],
                           key=f"pick_{key}", on_change=_add_to_selection, args=(sel_key, f"pick_{key}"))
        else:
            query = st.text_input(f"Search for a {label}", key=f"q_{key}", placeholder="Type part of the term name")
            if query and len(query.strip()) >= 2:
                hits = search_terms(terms_name, query)
                if hits:
                    more = " (showing the best matches; refine your search for more)" if len(hits) >= SEARCH_LIMIT else ""
                    st.multiselect(f"{len(hits)} matching terms{more}: select to add", hits,
                                   key=f"pick_{key}", on_change=_add_to_selection, args=(sel_key, f"pick_{key}"))
                else:
                    st.info("No matching terms. Try another spelling or a broader term.")
        if allow_custom:
            custom = st.text_input("Not in the list? Type the exact term name and press Enter", key=f"custom_{key}")
            if custom and custom.strip() not in selection:
                if st.button("Add this term", key=f"custom_btn_{key}"):
                    selection.append(custom.strip())
                    st.rerun()
    with c2:
        if nodes_name:
            if st.toggle("Browse the ontology tree", key=f"tree_on_{key}"):
                from streamlit_tree_select import tree_select

                nodes, depth = load_tree(nodes_name)
                if depth:
                    st.caption(f"The tree is shown {depth} levels deep; use the search box to find deeper terms.")
                checked = tree_select(nodes, no_cascade=True, expand_on_click=True, check_model="leaf",
                                      key=f"tree_{key}").get("checked", [])
                terms = [c.split(" , ")[-1].strip() for c in checked if c]
                if terms and st.button(f"Add {len(terms)} checked term(s)", key=f"tree_add_{key}"):
                    for t in terms:
                        if t not in selection:
                            selection.append(t)
                    st.rerun()
    if selection:
        c3, c4 = st.columns([4, 1], vertical_alignment="bottom")
        with c3:
            st.write("**Selected:** " + " · ".join(f"`{s}`" for s in selection))
        with c4:
            if st.button("Clear selection", key=f"clear_{key}"):
                st.session_state[sel_key] = []
                st.rerun()
    return list(selection)


def organism_picker(key: str) -> list[str]:
    sel_key = f"sel_{key}"
    selection = st.session_state.setdefault(sel_key, [])
    st.multiselect("Common model organisms", [o for o in MODEL_ORGANISMS if o not in selection],
                   key=f"pick_{key}", on_change=_add_to_selection, args=(sel_key, f"pick_{key}"))
    with st.expander("Search all of NCBITaxon"):
        c1, c2 = st.columns([1, 3])
        with c1:
            group = st.selectbox("Taxonomic group", list(TAXON_GROUPS), key=f"grp_{key}")
        with c2:
            query = st.text_input("Species or strain name (at least 3 characters)", key=f"q_{key}")
        if query and len(query.strip()) >= 3:
            hits = search_taxa(group, query)
            if hits:
                st.multiselect(f"{len(hits)} matching names: select to add", hits, key=f"pickt_{key}",
                               on_change=_add_to_selection, args=(sel_key, f"pickt_{key}"))
            else:
                st.info("No matching names in this group.")
        st.markdown("Browse the taxonomy on [OLS NCBITaxon](https://www.ebi.ac.uk/ols4/ontologies/ncbitaxon).")
    if selection:
        c3, c4 = st.columns([4, 1], vertical_alignment="bottom")
        with c3:
            st.write("**Selected:** " + " · ".join(f"`{s}`" for s in selection))
        with c4:
            if st.button("Clear selection", key=f"clear_{key}"):
                st.session_state[sel_key] = []
                st.rerun()
    return list(selection)


# --------------------------------------------------------------------------------------
# Column annotation (shared by the Required and Additional columns pages)
# --------------------------------------------------------------------------------------
def _describe(column: str, rule: spec.ColumnRule):
    level = {"required": "Required", "recommended": "Recommended", "optional": "Optional"}[rule.requirement]
    templates = ", ".join(t for t in dict.fromkeys(rule.templates) if t not in ("base", "sample-metadata"))
    st.subheader(column)
    st.caption(f"{level}{' · templates: ' + templates if templates else ''}")
    if rule.description:
        st.write(rule.description)
    notes = []
    if rule.examples:
        notes.append("Examples: " + ", ".join(f"`{e}`" for e in rule.examples[:5]))
    if rule.reserved_words:
        notes.append("Allowed when a value does not exist or is unknown: "
                     + ", ".join(f"`{w}`" for w in rule.reserved_words))
    else:
        notes.append("Every row needs a real value (`not available` is not allowed in this column).")
    st.caption("  \n".join(notes))


def _number_within(df, group_cols, count_col):
    """Number the distinct values of count_col 1, 2, 3, ... within each group, in order of appearance."""
    counters, seen, out = {}, {}, []
    for _, row in df.iterrows():
        group = tuple(row.get(c) for c in group_cols)
        item = row.get(count_col)
        if (group, item) not in seen:
            counters[group] = counters.get(group, 0) + 1
            seen[(group, item)] = counters[group]
        out.append(str(seen[(group, item)]))
    return out


def _integer_column(column, rule, key, extra_ok=()):
    df = require_df()
    base = spec.base_name(column)
    st.write("Use whole numbers starting at 1.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Set every row to 1", key=f"{key}_ones", help="No replicates / no fractionation"):
            fill_column(column, "1")
    with c2:
        if base == "characteristics[biological replicate]":
            condition_cols = [c for c in df.columns if c.startswith(("characteristics[", "factor value["))
                              and c != column and not spec.is_empty_column(df, c)]
            conds = st.multiselect("Number the samples (source names) within each condition defined by",
                                   condition_cols, key=f"{key}_cond",
                                   help="E.g. choose characteristics[disease]: the normal samples get 1, 2, 3, ... "
                                        "and so do the diseased samples.")
            if st.button("Number biological replicates", key=f"{key}_seq",
                         disabled=spec.is_empty_column(df, "source name")):
                fill_column(column, _number_within(df, conds, "source name"))
        elif not spec.is_empty_column(df, "source name") and "comment[data file]" in df:
            if base == "comment[technical replicate]":
                what, group = "runs", ["source name"]
            else:
                what = "fractions (data files)"
                group = ["source name"] + (["comment[technical replicate]"]
                                           if not spec.is_empty_column(df, "comment[technical replicate]") else [])
            if st.button(f"Number the {what} of each sample: 1, 2, 3, ...", key=f"{key}_seq",
                         help="Rows with the same " + " and ".join(group)
                              + " are numbered in the order of their data files."):
                fill_column(column, _number_within(df, group, "comment[data file]"))
    edit_cells([column], key=key,
               check=lambda v: v.isdigit() or v in extra_ok or v in rule.reserved_words)


def _free_text_column(column, rule, key, transform=None, check=None, placeholder=""):
    with st.container(border=True):
        value = st.text_input("Fill with this value", key=f"{key}_all", placeholder=placeholder)
        rows = row_scope(key)
        buttons = st.columns(1 + len(rule.reserved_words))
        with buttons[0]:
            if st.button("Fill", type="primary", key=f"{key}_all_btn") and value:
                v = transform(value.strip()) if transform else value.strip()
                if check and not check(v):
                    st.error(f"`{v}` is not valid for this column.")
                else:
                    fill_column(column, v, rows)
        for col_ui, word in zip(buttons[1:], rule.reserved_words):
            with col_ui:
                if st.button(f"Fill with: {word}", key=f"{key}_{word}"):
                    fill_column(column, word, rows)
    st.write("Or edit the values per row:")
    edit_cells([column], key=key, transform=transform, check=check)


def _modification_column(column, key):
    unimod = load_unimod()
    st.write("""Select the modifications searched for in your data, then set the type, position and target amino
    acid(s) of each. Every modification gets its own column, with the same value in every row.""")
    mods = st.multiselect("Modifications (Unimod)", sorted(unimod, key=str.lower) + ["Other (not in Unimod)"],
                          key=f"{key}_mods")
    results = []
    for mod in mods:
        st.markdown(f"**{mod}**")
        c1, c2, c3 = st.columns(3)
        with c1:
            mt = st.selectbox("Type", MOD_TYPES, index=1, key=f"{key}_mt_{mod}")
        with c2:
            pp = st.selectbox("Position", MOD_POSITIONS, key=f"{key}_pp_{mod}")
        with c3:
            ta = st.multiselect("Target amino acid(s)", AMINO_ACIDS, key=f"{key}_ta_{mod}")
        if mod.startswith("Other"):
            c4, c5, c6 = st.columns(3)
            with c4:
                name = st.text_input("Name", key=f"{key}_nt_{mod}")
            with c5:
                formula = st.text_input("Chemical formula (optional)", key=f"{key}_cf_{mod}")
            with c6:
                mass = st.text_input("Monoisotopic mass (optional)", key=f"{key}_mm_{mod}")
            if name:
                results.append(spec.modification_string(name, None, mt, pp, ",".join(ta), formula, mass))
        else:
            results.append(spec.modification_string(mod, f"UNIMOD:{unimod[mod]}", mt, pp, ",".join(ta)))
    if results:
        st.write("**SDRF notation:**")
        for r in results:
            st.code(r, language=None)
        if st.button("Write modifications to the SDRF", type="primary", key=f"{key}_write"):
            df = require_df()
            df = df.drop(columns=[c for c in df.columns if spec.base_name(c) == column])
            df = ensure_columns(df, copy_columns(column, len(results)))
            for col, value in zip(copy_columns(column, len(results)), results):
                df[col] = value
            save_df(df)
            st.rerun()


def _mass_tolerance_column(column, rule, key):
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            value = st.number_input("Tolerance", min_value=0.0, value=None, step=0.01, format="%g", key=f"{key}_v")
        with c2:
            unit = st.selectbox("Unit", ["ppm", "Da", "mmu"], key=f"{key}_u")
        rows = row_scope(key)
        if st.button("Fill", type="primary", key=f"{key}_fill") and value is not None:
            fill_column(column, f"{value:g} {unit}", rows)
    st.write("Or edit per row (number, a space and the unit, e.g. `10 ppm` or `0.02 Da`):")
    edit_cells([column], key=key, check=lambda v: bool(re.match(r"^\d+(\.\d+)?\s(ppm|Da|mmu)$", v))
               or v in rule.reserved_words)


def annotate_column(column: str):
    """Show the input widgets for one SDRF column."""
    templates = spec.selected_templates()
    rule = spec.rule_for(column, templates)
    key = re.sub(r"\W+", "_", column)
    _describe(column, rule)
    base = spec.base_name(column)

    if base == "source name":
        st.write("""A unique name for each biological sample, e.g. `patient_1_tumour`. The same source name appears
        on several rows when a sample was measured in several runs or fractions.""")
        if "comment[data file]" in require_df() and st.button("Use the data file names (without extension)"):
            df = require_df()
            df["source name"] = [re.sub(r"\.[^.]+$", "", str(v)) if v else None for v in df["comment[data file]"]]
            save_df(df)
            st.rerun()
        edit_cells([column], key=key)
    elif base == "assay name":
        st.write("A unique name for each data acquisition run, e.g. `run 1`, `run 2_fraction 1`.")
        if st.button("Number the runs: run 1, run 2, ... (one per data file)"):
            df = require_df()
            files = df["comment[data file]"].tolist() if "comment[data file]" in df else list(range(len(df)))
            run_of = {f: f"run {i + 1}" for i, f in enumerate(dict.fromkeys(files))}
            df["assay name"] = [run_of[f] for f in files]
            save_df(df)
            st.rerun()
        edit_cells([column], key=key)
    elif base == "comment[data file]":
        st.write("The raw file name, exactly as uploaded to the repository (e.g. `sample1.raw`).")
        edit_cells([column], key=key)
    elif base == "technology type":
        assign_values(column, list(rule.values or [spec.TECHNOLOGY_TYPES["ms-proteomics"]]), key)
    elif base == "characteristics[organism]":
        assign_values(column, organism_picker(key), key, allow_copies=True)
    elif base == "characteristics[sex]":
        assign_values(column, list(rule.values or ["male", "female", "intersex"]), key)
    elif base == "characteristics[age]":
        st.write("Use years (Y), months (M), weeks (W) and days (D) without spaces, e.g. `45Y`, `30Y6M`, `8W`. "
                 "Ranges use a dash (`40Y-50Y`), bounds use `>18Y`.")
        _free_text_column(column, rule, key, transform=spec.clean_age,
                          check=lambda v: spec.check_age(v) or v in rule.reserved_words, placeholder="e.g. 45Y")
    elif base in ("characteristics[individual]",):
        _free_text_column(column, rule, key, placeholder="e.g. 1, patient_A")
    elif base == "characteristics[biological replicate]":
        _integer_column(column, rule, key, extra_ok=(spec.POOLED,))
    elif base in ("comment[technical replicate]", "comment[fraction identifier]"):
        _integer_column(column, rule, key)
    elif base == "comment[cleavage agent details]":
        st.write("For top-down proteomics without digestion, use `not applicable`.")
        assign_values(column, term_picker(key, None, static_options=CLEAVAGE_AGENTS, label="cleavage agent"), key)
    elif base == "comment[proteomics data acquisition method]":
        methods = [m for m in spec.ACQUISITION_METHODS if not rule.values or m in rule.values]
        assign_values(column, term_picker(key, None, static_options=methods, label="acquisition method"), key)
    elif base == "comment[modification parameters]":
        _modification_column(column, key)
    elif base in ("comment[precursor mass tolerance]", "comment[fragment mass tolerance]"):
        _mass_tolerance_column(column, rule, key)
    elif base == "characteristics[disease]":
        st.write("Use `normal` for healthy or control samples.")
        terms_name, nodes_name = ONTOLOGY_COLUMNS[base]
        assign_values(column, term_picker(key, terms_name, nodes_name, label="disease"), key)
    elif base in ONTOLOGY_COLUMNS:
        terms_name, nodes_name = ONTOLOGY_COLUMNS[base]
        assign_values(column, term_picker(key, terms_name, nodes_name, label="term", allow_custom=True), key)
    elif base == "characteristics[sample type]":
        options = rule.examples or ["study sample", "negative control sample", "positive control sample",
                                    "quality control sample", "calibration sample", "blank sample"]
        assign_values(column, term_picker(key, None, static_options=options, label="sample type",
                                          allow_custom=True), key)
    elif rule.values:
        assign_values(column, list(rule.values), key)
    else:
        _free_text_column(column, rule, key, placeholder=rule.examples[0] if rule.examples else "")


def clear_columns_ui(df: pd.DataFrame, key: str):
    st.write("Select columns to empty so you can annotate them again. Extra copies of a column are removed.")
    columns = st.multiselect("Columns to clear", [c for c in df.columns if not spec.is_empty_column(df, c)],
                             key=f"{key}_clear")
    if columns and st.button("Clear", type="primary", key=f"{key}_clear_btn"):
        df = df.copy()
        for col in columns:
            if col != spec.base_name(col):
                df = df.drop(columns=[col])
            else:
                df[col] = None
        save_df(df)
        st.rerun()


def column_label(df: pd.DataFrame, column: str) -> str:
    return column if spec.is_empty_column(df, column) else f"{column} (filled)"


# --------------------------------------------------------------------------------------
# Sidebar: download + validation
# --------------------------------------------------------------------------------------
def sidebar_export(df: pd.DataFrame | None = None):
    df = get_df() if df is None else df
    with st.sidebar:
        if df is None:
            return
        templates = spec.selected_templates()
        tsv = spec.to_tsv(df, templates)
        st.download_button("Download SDRF file", tsv.encode("utf-8"), "lessdrf.sdrf.tsv", mime="text/tab-separated-values",
                           on_click="ignore", type="primary",
                           help="Download your (intermediate) SDRF. You can upload it again on the Home page later.")
        if st.button("Validate SDRF",
                     help="Check the file with sdrf-pipelines, the official SDRF validator"):
            st.session_state["validate"] = True
        if st.session_state.get("validate"):
            results = spec.validate_tsv(tsv, tuple(templates))
            errors = [r for r in results if r["level"] == "error"]
            warnings = [r for r in results if r["level"] == "warning"]
            if not errors and not warnings:
                st.success("✅ Valid SDRF")
            elif not errors:
                st.warning(f"Valid, with {len(warnings)} warning(s)")
            else:
                st.error(f"{len(errors)} error(s), {len(warnings)} warning(s)")
            if results:
                with st.expander("Details", expanded=bool(errors)):
                    for r in results[:200]:
                        st.markdown(f"**{r['level'].capitalize()}:** {r['message']}")
                    if len(results) > 200:
                        st.caption(f"... and {len(results) - 200} more")
            st.caption("Ontology terms are not checked online here. For a full check run "
                       "`parse_sdrf validate-sdrf` (sdrf-pipelines).")
            if st.button("Hide validation"):
                st.session_state["validate"] = False
                st.rerun()
        st.caption(f"Templates: {', '.join(f'{t} v{spec.template_version(t)}' for t in templates)}")
        st.markdown(CITATION)
