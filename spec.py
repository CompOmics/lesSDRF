"""SDRF-Proteomics specification layer for lesSDRF.

Everything that depends on the SDRF-Proteomics specification lives here: which templates exist,
which columns they define, how values must be written and how a finished file is ordered and validated.

Column definitions are read from the templates bundled with the installed ``sdrf-pipelines`` package,
the official validator. When the spec moves on, upgrading ``sdrf-pipelines`` updates the app's
column lists, allowed values and validation together.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

APP_NAME = "lesSDRF"
APP_VERSION = "0.2.0"
SDRF_SPEC_VERSION = "v1.1.0"

# Reserved words (SDRF-Proteomics v1.1.0, "Reserved words"); always lowercase.
NOT_AVAILABLE = "not available"
NOT_APPLICABLE = "not applicable"
POOLED = "pooled"
ANONYMIZED = "anonymized"

# Template layers offered in the app. Names are sdrf-pipelines template names.
TECHNOLOGY_TEMPLATES = {
    "ms-proteomics": "Mass spectrometry proteomics",
    "affinity-proteomics": "Affinity proteomics (Olink, SomaScan)",
}
SAMPLE_TEMPLATES = {
    "": "No organism-specific template (any organism)",
    "human": "Human",
    "vertebrates": "Other vertebrates (mouse, rat, zebrafish, ...)",
    "invertebrates": "Invertebrates (Drosophila, C. elegans, ...)",
    "plants": "Plants",
    "metaproteomics": "Metaproteomics (microbial communities)",
    "human-gut": "Metaproteomics: human gut",
    "soil": "Metaproteomics: soil",
    "water": "Metaproteomics: water",
}
EXPERIMENT_TEMPLATES = {
    "cell-lines": "Cell lines",
    "dia-acquisition": "Data-independent acquisition (DIA)",
    "single-cell": "Single-cell proteomics",
    "immunopeptidomics": "Immunopeptidomics",
    "crosslinking": "Crosslinking MS (XL-MS)",
    "clinical-metadata": "Clinical metadata",
    "oncology-metadata": "Oncology metadata",
}
# Templates that only make sense on top of ms-proteomics.
MS_ONLY_TEMPLATES = {"dia-acquisition", "single-cell", "immunopeptidomics", "crosslinking"}

# File-level metadata columns that lesSDRF fills in itself on export.
METADATA_COLUMNS = (
    "comment[sdrf version]",
    "comment[sdrf template]",
    "comment[sdrf annotation tool]",
    "comment[sdrf validation hash]",
)

REQUIREMENT_RANK = {"required": 0, "recommended": 1, "optional": 2}

# Columns whose ontology terms are written as NT=<label>;AC=<accession>, with the ontologies to look in.
ACCESSION_COLUMNS = {
    "comment[instrument]": ("MS", "PRIDE"),
    "comment[label]": ("PRIDE", "MS"),
    "comment[cleavage agent details]": ("MS",),
    "comment[dissociation method]": ("MS", "PRIDE"),
    "comment[fractionation method]": ("PRIDE",),
    "comment[proteomics data acquisition method]": ("PRIDE",),
    "comment[reduction reagent]": ("PRIDE", "MS"),
    "comment[alkylation reagent]": ("PRIDE", "MS"),
    "comment[ms2 mass analyzer]": ("MS",),
}

# Children of PRIDE:0000659 (proteomics data acquisition method).
ACQUISITION_METHODS = [
    "Data-dependent acquisition",
    "Data-independent acquisition",
    "diaPASEF",
    "SWATH MS",
    "Parallel reaction monitoring",
    "Selected reaction monitoring",
    "MSE",
    "HDMSE",
    "All-ion fragmentation",
    "SONAR",
    "PAcIFIC",
]

TECHNOLOGY_TYPES = {
    "ms-proteomics": "proteomic profiling by mass spectrometry",
    "affinity-proteomics": "protein expression profiling by antibody array",
}

# Age: e.g. 45Y, 6M, 30Y6M, 40Y-50Y, >18Y (no spaces).
_AGE_UNIT = r"(\d+[Yy](\d+[Mm](\d+[Ww](\d+[Dd])?)?)?|\d+[Mm](\d+[Ww](\d+[Dd])?)?|\d+[Ww](\d+[Dd])?|\d+[Dd])"
AGE_PATTERN = re.compile(rf"^(>=?|<=?)?{_AGE_UNIT}(-{_AGE_UNIT})?$")


@dataclass
class ColumnRule:
    """The merged rules for one column across all selected templates."""

    name: str
    requirement: str = "optional"
    description: str = ""
    templates: list[str] = field(default_factory=list)
    allow_not_available: bool = True
    allow_not_applicable: bool = True
    allow_pooled: bool = True
    allow_anonymized: bool = True
    multiple: bool = False
    values: list[str] | None = None
    examples: list[str] = field(default_factory=list)
    default: str | None = None
    ontologies: list[str] = field(default_factory=list)

    @property
    def reserved_words(self) -> list[str]:
        words = []
        if self.allow_not_available:
            words.append(NOT_AVAILABLE)
        if self.allow_not_applicable:
            words.append(NOT_APPLICABLE)
        if self.allow_pooled:
            words.append(POOLED)
        if self.allow_anonymized:
            words.append(ANONYMIZED)
        return words


# --------------------------------------------------------------------------------------
# Template registry (loaded once per server process and shared by all sessions)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _registry():
    from sdrf_pipelines.sdrf.schemas import SchemaRegistry

    return SchemaRegistry()


def template_version(name: str) -> str:
    schema = _registry().get_schema(name)
    return schema.version if schema else "?"


def template_description(name: str) -> str:
    schema = _registry().get_schema(name)
    return schema.description if schema else ""


def available(names) -> list[str]:
    known = set(_registry().get_schema_names())
    return [n for n in names if n == "" or n in known]


@st.cache_resource(show_spinner=False)
def _column_rules(templates: tuple[str, ...]) -> tuple[list[str], dict[str, ColumnRule]]:
    """Ordered column names and merged rules for a combination of templates.

    The validator checks a file against every selected template separately, so a column takes the
    strictest requirement and the narrowest reserved-word permissions of all templates that define it.
    """
    registry = _registry()
    ordered, _ = registry.compile_columns_from_schemas(list(templates))
    # Note: templates.yaml lets metaproteomics exclude sample-metadata, but sdrf-pipelines 0.1.7 does not
    # apply exclusions when validating, so neither does lesSDRF: the columns match what the validator checks.
    rules: dict[str, ColumnRule] = {}
    for name in templates:
        schema = registry.get_schema(name)
        if schema is None:
            continue
        for col in schema.columns:
            rule = rules.get(col.name)
            if rule is None:
                rule = rules[col.name] = ColumnRule(name=col.name, description=col.description)
            rule.templates.append(name)
            req = col.requirement.value
            if REQUIREMENT_RANK[req] < REQUIREMENT_RANK[rule.requirement]:
                rule.requirement = req
            rule.allow_not_available &= col.allow_not_available
            rule.allow_not_applicable &= col.allow_not_applicable
            rule.allow_pooled &= col.allow_pooled
            rule.allow_anonymized &= col.allow_anonymized
            rule.multiple |= (col.cardinality == "multiple")
            for v in col.validators:
                params = v.params or {}
                if v.validator_name == "values" and params.get("values"):
                    vals = [str(x) for x in params["values"] if not str(x).startswith("NT=")]
                    rule.values = vals if rule.values is None else ([x for x in rule.values if x in vals] or vals)
                if params.get("examples") and not rule.examples:
                    rule.examples = [str(x) for x in params["examples"]]
                if v.validator_name == "ontology" and params.get("ontologies"):
                    rule.ontologies = list(params["ontologies"])
            if getattr(col, "default", None):
                rule.default = col.default
    ordered = [c for c in ordered if c in rules]
    return ordered, rules


def column_rules(templates) -> dict[str, ColumnRule]:
    return _column_rules(tuple(templates))[1]


def all_columns(templates) -> list[str]:
    """Every column the templates define, in SDRF order."""
    return list(_column_rules(tuple(templates))[0])


def template_columns(templates, levels=("required", "recommended")) -> list[str]:
    """Columns defined by the templates at the given requirement levels, in SDRF order."""
    ordered, rules = _column_rules(tuple(templates))
    return [c for c in ordered if rules[c].requirement in levels and c not in METADATA_COLUMNS]


def rule_for(column: str, templates) -> ColumnRule:
    """The merged rule for a column, or a permissive default for columns no template defines."""
    base = base_name(column)
    rules = column_rules(templates)
    if base in rules:
        return rules[base]
    return ColumnRule(name=base, description="Column not defined in the selected templates")


def selected_templates() -> list[str]:
    """The templates chosen on the Home page (stored in the session)."""
    return list(st.session_state.get("templates") or ["ms-proteomics"])


def leaf_templates(templates) -> list[str]:
    """Templates worth listing in comment[sdrf template] (drop parents of other selected templates)."""
    registry = _registry()
    parents = set()
    for name in templates:
        schema = registry.get_schema(name)
        while schema and schema.extends:
            parent = schema.extends.split("@")[0]
            parents.add(parent)
            schema = registry.get_schema(parent)
    return [t for t in templates if t not in parents]


def templates_from_df(df: pd.DataFrame) -> list[str] | None:
    """Recover the template selection from comment[sdrf template] columns in an uploaded file."""
    names = []
    for col in df.columns:
        if base_name(col) != "comment[sdrf template]":
            continue
        for value in df[col].dropna().unique():
            m = re.match(r"^(?:NT=)?([\w-]+)(?:;VV=| )v?[\d.]+", str(value))
            if m:
                names.append(m.group(1))
    names = available(names)
    if not names:
        return None
    if not any(n in TECHNOLOGY_TEMPLATES for n in names):
        names.insert(0, "ms-proteomics")
    return list(dict.fromkeys(names))


# --------------------------------------------------------------------------------------
# Column and value helpers
# --------------------------------------------------------------------------------------
_COPY_SUFFIX = re.compile(r"_(\d+)$")


def base_name(column: str) -> str:
    """lesSDRF stores repeated SDRF columns as `col`, `col_1`, `col_2`; this returns `col`."""
    return _COPY_SUFFIX.sub("", column)


def copies_of(df: pd.DataFrame, column: str) -> list[str]:
    return [c for c in df.columns if base_name(c) == column]


def is_empty_value(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return isinstance(v, str) and v.strip() in ("", "empty", "nan", "None")


def is_empty_column(df: pd.DataFrame, column: str) -> bool:
    return column not in df.columns or all(is_empty_value(v) for v in df[column].tolist())


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Object dtype everywhere, stripped strings, None for empty cells."""
    df = df.copy()
    for col in df.columns:
        df[col] = pd.Series(
            [None if is_empty_value(v) else str(v).strip() for v in df[col].tolist()],
            index=df.index,
            dtype=object,
        )
    return df


def read_sdrf_upload(uploaded) -> pd.DataFrame:
    """Read an uploaded SDRF/TSV, keeping repeated columns as `col`, `col_1`, ..."""
    df = pd.read_csv(uploaded, sep="\t", dtype=str, keep_default_na=False)
    seen: dict[str, int] = {}
    new_cols = []
    for col in df.columns:
        base = re.sub(r"\.\d+$", "", col).strip()  # pandas renames duplicates to col.1
        n = seen.get(base, 0)
        new_cols.append(base if n == 0 else f"{base}_{n}")
        seen[base] = n + 1
    df.columns = new_cols
    return normalize_df(df)


def check_age(value: str) -> bool:
    v = str(value).strip()
    return v in (NOT_AVAILABLE, ANONYMIZED, POOLED) or bool(AGE_PATTERN.match(v))


def clean_age(value: str) -> str:
    """Remove spaces and normalise unit case: '12y 3m' -> '12Y3M'."""
    v = re.sub(r"\s+", "", str(value))
    return re.sub(r"[ymwd]", lambda m: m.group(0).upper(), v)


SEX_ALIASES = {
    "m": "male", "male": "male", "man": "male",
    "f": "female", "female": "female", "woman": "female",
    "intersex": "intersex", "hermaphrodite": "hermaphrodite",
    "na": NOT_AVAILABLE, "unknown": NOT_AVAILABLE, "not available": NOT_AVAILABLE,
    "anonymized": ANONYMIZED, "pooled": POOLED,
}


def modification_string(name, accession=None, mod_type="Variable", position="Anywhere",
                        targets="", formula=None, mass=None) -> str:
    """comment[modification parameters] value, keys in spec order: NT, AC, then the rest."""
    parts = [f"NT={name}"]
    if accession:
        parts.append(f"AC={accession}")
    if formula:
        parts.append(f"CF={formula}")
    parts.append(f"MT={mod_type}")
    parts.append(f"PP={position}")
    if targets:
        parts.append(f"TA={targets}")
    if mass:
        parts.append(f"MM={mass}")
    return ";".join(parts)


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------
def _section(col: str) -> int:
    b = base_name(col)
    if b == "source name":
        return 0
    if b.startswith("characteristics["):
        return 1
    if b == "material type":
        return 2
    if b == "assay name":
        return 3
    if b == "technology type":
        return 4
    if b.startswith("factor value["):
        return 7
    if b in METADATA_COLUMNS:
        return 6
    if b.startswith("comment["):
        return 5
    return 2


def order_columns(columns, templates) -> list[str]:
    """SDRF column order: source name, characteristics, assay name, technology type, comments, factor values.

    Within a section, columns follow the template order; copies of a column stay together.
    """
    template_order = {c: i for i, c in enumerate(_column_rules(tuple(templates))[0])}
    first_seen = {}
    for i, c in enumerate(columns):
        first_seen.setdefault(base_name(c), i)

    def key(col):
        b = base_name(col)
        m = _COPY_SUFFIX.search(col)
        return (_section(col), template_order.get(b, 10_000 + first_seen[b]), b, int(m.group(1)) if m else 0)

    return sorted(columns, key=key)


def export_df(df: pd.DataFrame, templates) -> pd.DataFrame:
    """The SDRF as it will be written: metadata columns added, empty optional columns dropped, ordered."""
    df = normalize_df(df)
    rules = column_rules(templates)
    keep = [c for c in df.columns
            if not is_empty_column(df, c) or rules.get(base_name(c), ColumnRule(c)).requirement == "required"]
    df = df[keep]
    df = df.drop(columns=[c for c in df.columns if base_name(c) in METADATA_COLUMNS[:3]])
    if "source name" not in df.columns:
        df.insert(0, "source name", None)
    df["comment[sdrf version]"] = SDRF_SPEC_VERSION
    for i, name in enumerate(leaf_templates(templates)):
        col = "comment[sdrf template]" if i == 0 else f"comment[sdrf template]_{i}"
        df[col] = f"{name} v{template_version(name)}"
    df["comment[sdrf annotation tool]"] = f"{APP_NAME} v{APP_VERSION}"
    df = df[order_columns(list(df.columns), templates)]
    df.columns = [base_name(c) for c in df.columns]
    return df


def to_tsv(df: pd.DataFrame, templates) -> str:
    return export_df(df, templates).to_csv(index=False, sep="\t", lineterminator="\n")


# --------------------------------------------------------------------------------------
# Validation (sdrf-pipelines, in-process, structural checks only)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=20)
def validate_tsv(tsv: str, templates: tuple[str, ...]) -> list[dict]:
    """Validate SDRF text with sdrf-pipelines and return de-duplicated messages.

    Ontology look-ups are skipped: they call the OLS web service and are slow. lesSDRF already restricts
    ontology columns to terms from its bundled ontologies.
    """
    from sdrf_pipelines.sdrf.schemas import SchemaValidator
    from sdrf_pipelines.sdrf.sdrf import read_sdrf

    sdrf_df = read_sdrf(io.StringIO(tsv))
    validator = SchemaValidator(_registry())
    seen, out = set(), []
    for name in templates:
        for err in validator.validate(sdrf_df, name, use_ols_cache_only=True, skip_ontology=True):
            level = {logging.ERROR: "error", logging.WARNING: "warning"}.get(err.error_type, "info")
            if (level, err.message) in seen:
                continue
            seen.add((level, err.message))
            out.append({"level": level, "message": err.message})
    out.sort(key=lambda e: {"error": 0, "warning": 1, "info": 2}[e["level"]])
    return out
