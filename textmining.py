"""Find SDRF metadata in free text (e.g. the methods section of a manuscript) without an LLM.

Two techniques, both deterministic and light on memory:
- Dictionary matching: word n-grams of the text are looked up in the bundled ontology term lists (longest match wins,
  hyphens/punctuation and simple plurals are tolerated, short or mixed-case names must match their exact spelling).
- Rules: regular expressions for values that are not ontology terms (mass tolerances, collision energy, age, ...)
  and for common phrasings of ontology terms (e.g. "carbamidomethylation of cysteines", "HCD", "DIA").

Every hit keeps the sentence it was found in, so the user can judge it before anything is written to the SDRF.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field

import streamlit as st

import app_utils as au
import spec

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\[])|\n\s*\n")
MAX_NGRAM = 8

# Ontology-backed columns searched by dictionary matching: column -> term list.
# Ontology-backed columns searched by dictionary matching: column -> (term list, exact spelling, excluded list).
# Fractionation, reduction/alkylation reagents and enrichment are covered by the PHRASES rules instead.
DICTIONARY_COLUMNS = {
    "characteristics[disease]": ("all_disease_elements", False, None),
    "characteristics[organism part]": ("all_organism_part_elements", False, "all_cell_elements"),
    "characteristics[cell type]": ("all_cell_elements", False, None),
    "characteristics[cell line]": ("all_cell_line_elements", False, "all_cell_elements"),
    "characteristics[developmental stage]": ("all_developmental_stage_elements", False, None),
    "characteristics[ancestry category]": ("all_ancestry_category_elements", False, None),
    "comment[instrument]": ("all_instrument_elements", True, None),
}

# Terms that are too generic to be useful hits on their own.
STOP_TERMS = {
    "disease", "disorder", "syndrome", "cell", "tissue", "organ", "body", "protein", "sample", "control", "system",
    "entity", "structure", "region", "part", "material", "organism", "anatomical entity", "native cell", "cell line",
    "cell type", "cultured cell", "phenotype", "stage", "development", "group", "line", "culture", "solution",
    "column", "head", "back", "face", "hand", "foot", "spot", "chain", "trunk", "unit", "layer", "space", "fluid",
    "surface", "membrane", "matrix", "lumen", "wall", "zone", "process", "sample enrichment", "enrichment process",
    "not available", "not applicable", "fractionation method", "second dimension details", "ph low value",
    "ph high value", "ipg strip length cm", "lift", "neck", "arm", "leg", "lead", "lens", "fold", "joint",
    "embryo", "infection", "injury", "lesion", "obesity", "carcinoma", "cancer", "tumor", "tumour", "neoplasm",
    # common in methods sections, rarely meant as the sample's anatomy/cell line
    "pore", "scale", "capillary", "calf", "milk", "vessel", "root", "tail", "sem", "complication", "intoxication",
    "septum", "plate", "well", "bead", "beads", "trap", "spray", "tip", "needle", "hood", "film", "gel", "mass",
}

# Phrases (regex, case-insensitive) -> canonical SDRF value, per column.
PHRASES = {
    "comment[proteomics data acquisition method]": [
        (r"\bdiaPASEF\b", "diaPASEF"),
        (r"\bSWATH", "SWATH MS"),
        (r"data[- ]independent acquisition|\bDIA\b", "Data-independent acquisition"),
        (r"parallel reaction monitoring|\bPRM\b", "Parallel reaction monitoring"),
        (r"selected reaction monitoring|multiple reaction monitoring|\bSRM\b|\bMRM\b", "Selected reaction monitoring"),
        (r"data[- ]dependent (?:acquisition|mode|analysis)|\bDDA\b|\btop[- ]?\d+ (?:method|mode|most|precursors?|ions?|peaks?)|ddaPASEF", "Data-dependent acquisition"),
    ],
    "comment[dissociation method]": [
        (r"higher[- ]energy (?:collisional|collision)[- ](?:induced )?dissociation|\bHCD\b", "HCD"),
        (r"\bEThcD\b", "EThcD"),
        (r"electron[- ]transfer dissociation|\bETD\b", "electron transfer dissociation"),
        (r"electron[- ]capture dissociation", "electron capture dissociation"),
        (r"(?<!higher energy )(?<!beam-type )collision[- ]induced dissociation|\bCID\b", "collision-induced dissociation"),
        (r"ultraviolet photodissociation|\bUVPD\b", "ultraviolet photodissociation"),
    ],
    "comment[cleavage agent details]": [
        (r"trypsin/P\b", "Trypsin/P"),
        (r"\btrypsin\b|\btryptic\b|trypsini[sz]ed|trypsin digest", "Trypsin"),
        (r"\bLys[- ]?C\b|endoproteinase Lys", "Lys-C"),
        (r"\bLys[- ]?N\b", "Lys-N"),
        (r"\bArg[- ]?C\b", "Arg-C"),
        (r"\bAsp[- ]?N\b", "Asp-N"),
        (r"\bGlu[- ]?C\b|\bV8 protease", "Glu-C"),
        (r"\bchymotrypsin\b", "Chymotrypsin"),
        (r"\bpepsin\b", "PepsinA"),
        (r"\bproteinase K\b", "unspecific cleavage"),
        (r"\bCNBr\b|cyanogen bromide", "CNBr"),
    ],
    "comment[reduction reagent]": [
        (r"\bDTT\b|dithiothreitol", "Dithiothreitol (DTT)"),
        (r"\bTCEP\b", "Tris(2-carboxyethyl)phosphine (TCEP)"),
        (r"mercaptoethanol|\bBME\b", "Beta-mercaptoethanol (BME)"),
    ],
    "comment[alkylation reagent]": [
        (r"iodoacetamide|\bIAA\b|\bIAM\b", "Iodoacetamide (IAA)"),
        (r"chloroacetamide|\bCAA\b|\bClAA\b", "Chloroacetamide (ClAA)"),
        (r"\bMMTS\b|methyl methanethiosulfonate", "Methyl methanethiosulfonate (MMTS)"),
        (r"N-ethylmaleimide|\bNEM\b", "N-ethylmaleimide (NEM)"),
        (r"4-vinylpyridine", "4-Vinylpyridine (4-VP)"),
        (r"iodoacetic acid", "Iodoacetic acid (IAC)"),
    ],
    "comment[fractionation method]": [
        (r"high[- ]pH (?:reversed?|RP)|basic (?:pH )?reversed?[- ]phase|\bhpRP\b|\bbRP\b",
         "High-pH reversed-phase chromatography (hpHRP)"),
        (r"strong cation[- ]exchange|\bSCX\b", "Strong cation-exchange chromatography (SCX)"),
        (r"strong anion[- ]exchange|\bSAX\b", "Strong anion-exchange chromatography (SAX)"),
        (r"\bHILIC\b|hydrophilic interaction", "Hydrophilic interaction chromatography (HILIC)"),
        (r"\bSDS[- ]PAGE\b|gel[- ]?(?:based )?fractionat|in-gel digest|gel slices|gel bands", "SDS PAGE"),
        (r"off[- ]gel", "Off-gel electrophoresis"),
        (r"isoelectric focusing|\bIEF\b", "Isoelectric focusing"),
        (r"\bERLIC\b", "ERLIC"),
        (r"size[- ]exclusion chromatography|\bSEC\b", "Size-exclusion chromatography (SEC)"),
        (r"weak anion[- ]exchange|\bWAX\b", "Weak anion-exchange chromatography (WAX)"),
        (r"weak cation[- ]exchange|\bWCX\b", "Weak cation-exchange chromatography (WCX)"),
        (r"\bHiRIEF\b", "High-resolution isoelectric focusing (HiRIEF)"),
    ],
    "characteristics[disease]": [
        (r"\bhealthy (?:donors?|controls?|individuals?|volunteers?|subjects?|tissues?|samples?)\b|"
         r"\bnon-?diseased\b|\bnormal (?:tissues?|controls?)\b", "normal"),
    ],
    "characteristics[enrichment process]": [
        (r"phosphopeptide enrichment|\bIMAC\b|\bTiO2\b|titanium dioxide|enrich\w* (?:for |of )?phospho",
         "enrichment of phosphorylated protein"),
    ],
}

ORGANISM_PATTERNS = [
    (r"\b(?:humans?|patients?|donors?|volunteers?|individuals|subjects|Homo sapiens|H\. sapiens)\b", "Homo sapiens"),
    (r"\b(?:mice|mouse|murine|Mus musculus|M\. musculus)\b", "Mus musculus"),
    (r"\b(?:rats?|Rattus norvegicus|R\. norvegicus)\b", "Rattus norvegicus"),
    (r"\b(?:zebrafish|Danio rerio|D\. rerio)\b", "Danio rerio"),
    (r"\b(?:fruit ?fl(?:y|ies)|Drosophila(?: melanogaster)?|D\. melanogaster)\b", "Drosophila melanogaster"),
    (r"\b(?:C\. elegans|Caenorhabditis elegans|nematodes?)\b", "Caenorhabditis elegans"),
    (r"\b(?:yeast|Saccharomyces cerevisiae|S\. cerevisiae|budding yeast)\b", "Saccharomyces cerevisiae"),
    (r"\b(?:Arabidopsis(?: thaliana)?|A\. thaliana|thale cress)\b", "Arabidopsis thaliana"),
    (r"\b(?:E\. ?coli|Escherichia coli)\b", "Escherichia coli"),
    (r"\b(?:Bacillus subtilis|B\. subtilis)\b", "Bacillus subtilis"),
    (r"\b(?:pigs?|porcine|Sus scrofa)\b", "Sus scrofa"),
    (r"\b(?:bovine|cows?|cattle|Bos taurus)\b", "Bos taurus"),
    (r"\b(?:chickens?|Gallus gallus)\b", "Gallus gallus"),
    (r"\b(?:Xenopus laevis|X\. laevis)\b", "Xenopus laevis"),
    (r"\b(?:rice|Oryza sativa)\b", "Oryza sativa"),
    (r"\b(?:maize|Zea mays)\b", "Zea mays"),
]
# Well-known human cell lines imply Homo sapiens.
HUMAN_CELL_LINES = {"HeLa", "HEK293", "HEK293T", "MCF7", "MCF-7", "A549", "Jurkat", "K562", "U2OS", "HCT116",
                    "HepG2", "SH-SY5Y", "Caco-2", "THP-1", "PC-3", "LNCaP", "MDA-MB-231", "U937", "HL-60"}

# Modifications: (regex, Unimod name, default targets, default type, default position)
MODIFICATIONS = [
    (r"carbamidomethyl", "Carbamidomethyl", "C", "Fixed", "Anywhere"),
    (r"propionamide", "Propionamide", "C", "Fixed", "Anywhere"),
    (r"N-?ethylmaleimide|\bNEM\b", "Nethylmaleimide", "C", "Fixed", "Anywhere"),
    (r"methylthio|\bMMTS\b", "Methylthio", "C", "Fixed", "Anywhere"),
    (r"oxidi[sz]ed methionine|methionine oxidation|oxidation", "Oxidation", "M", "Variable", "Anywhere"),
    (r"(?:protein )?N[- ]?term\w* acetyl|acetyl\w* \(?(?:of |on )?(?:the )?(?:protein )?N[- ]?term", "Acetyl", "",
     "Variable", "Protein N-term"),
    (r"lysine acetylation|acetylation (?:of|on) lysine|acetyl \(K\)", "Acetyl", "K", "Variable", "Anywhere"),
    (r"phospho", "Phospho", "S,T,Y", "Variable", "Anywhere"),
    (r"deamidat", "Deamidated", "N,Q", "Variable", "Anywhere"),
    (r"pyro-?glu", "Gln->pyro-Glu", "Q", "Variable", "Any N-term"),
    (r"\bGlyGly\b|di-?glycine|\bGG\b remnant|ubiquitination remnant|K-?ε-?GG", "GG", "K", "Variable", "Anywhere"),
    (r"\bTMTpro\b|TMT ?16|TMT ?18", "TMTpro", "K", "Fixed", "Anywhere"),
    (r"\bTMT(?:6|10|11)?[- ]?(?:plex)?\b", "TMT6plex", "K", "Fixed", "Anywhere"),
    (r"iTRAQ ?8", "iTRAQ8plex", "K", "Fixed", "Anywhere"),
    (r"\biTRAQ\b", "iTRAQ4plex", "K", "Fixed", "Anywhere"),
    (r"dimethyl", "Dimethyl", "K", "Fixed", "Anywhere"),
    (r"(?<!di)methylation", "Methyl", "K,R", "Variable", "Anywhere"),
]
MOD_CONTEXT = re.compile(r"modification|fixed|variable|static|dynamic|search|database|MaxQuant|Mascot|Sequest|"
                         r"Andromeda|Proteome Discoverer|FragPipe|MSFragger|Spectronaut|DIA-NN|PEAKS|"
                         r"carbamidomethyl|alkylat|label", re.I)
AMINO_ACIDS = {"alanine": "A", "cysteine": "C", "aspartate": "D", "aspartic acid": "D", "glutamate": "E",
               "glutamic acid": "E", "phenylalanine": "F", "glycine": "G", "histidine": "H", "isoleucine": "I",
               "lysine": "K", "leucine": "L", "methionine": "M", "asparagine": "N", "proline": "P",
               "glutamine": "Q", "arginine": "R", "serine": "S", "threonine": "T", "valine": "V",
               "tryptophan": "W", "tyrosine": "Y"}

TMT_CHANNELS = {
    6: ["TMT126", "TMT127", "TMT128", "TMT129", "TMT130", "TMT131"],
    10: ["TMT126", "TMT127N", "TMT127C", "TMT128N", "TMT128C", "TMT129N", "TMT129C", "TMT130N", "TMT130C", "TMT131"],
    11: ["TMT126", "TMT127N", "TMT127C", "TMT128N", "TMT128C", "TMT129N", "TMT129C", "TMT130N", "TMT130C",
         "TMT131N", "TMT131C"],
    16: ["TMT126", "TMT127N", "TMT127C", "TMT128N", "TMT128C", "TMT129N", "TMT129C", "TMT130N", "TMT130C",
         "TMT131N", "TMT131C", "TMT132N", "TMT132C", "TMT133N", "TMT133C", "TMT134N"],
}
ITRAQ_CHANNELS = {4: ["ITRAQ114", "ITRAQ115", "ITRAQ116", "ITRAQ117"],
                  8: ["ITRAQ113", "ITRAQ114", "ITRAQ115", "ITRAQ116", "ITRAQ117", "ITRAQ118", "ITRAQ119",
                      "ITRAQ121"]}


@dataclass
class Hit:
    column: str
    value: str
    found: str
    evidence: str
    count: int = 1
    source: str = "rule"  # rule | dictionary | taxonomy | inferred
    spans: list = field(default_factory=list)  # (start, end) of every mention in the cleaned text
    confidence: float = 0.0
    extra: dict = field(default_factory=dict)


class _Sentence(str):
    """A sentence that remembers where it starts in the cleaned text."""
    offset: int = 0


class _Evidence(str):
    """An evidence snippet that remembers the exact span of the match in the cleaned text."""
    span: tuple[int, int] | None = None


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _norm(text: str) -> str:
    return " ".join(_TOKEN.findall(text.lower()))


def _singular(word: str) -> str:
    if word == "mice":
        return "mouse"
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _is_strict(term: str) -> bool:
    """Short names and names with capitals/digits inside (HeLa, LTQ, U2OS) must match their exact spelling."""
    core = term.replace(" ", "")
    return len(core) <= 4 or any(c.isupper() for c in core[1:]) or any(c.isdigit() for c in core)


def clean_text(text: str) -> str:
    """The text as it is searched and displayed: no carriage returns, single spaces."""
    return re.sub(r"[ \t]+", " ", text.replace("\r", "")).strip()


def split_sentences(clean: str) -> list[_Sentence]:
    out, pos = [], 0
    for m in list(_SENTENCE.finditer(clean)) + [None]:
        end = m.start() if m else len(clean)
        seg = clean[pos:end]
        if seg.strip():
            s = _Sentence(seg.strip())
            s.offset = pos + (len(seg) - len(seg.lstrip()))
            out.append(s)
        if m:
            pos = m.end()
    return out


def _short(sentence: str, start: int, end: int, width: int = 110) -> _Evidence:
    a, b = max(0, start - width), min(len(sentence), end + width)
    ev = _Evidence(("..." if a else "") + sentence[a:b].strip() + ("..." if b < len(sentence) else ""))
    offset = getattr(sentence, "offset", None)
    ev.span = (offset + start, offset + end) if offset is not None else None
    return ev


@st.cache_resource(show_spinner=False)
def _phrase_index(name: str, always_strict: bool = False, exclude: str | None = None
                  ) -> dict[str, tuple[str, str | None]]:
    """normalised phrase -> (term, exact spelling required or None). Shared by all sessions.

    always_strict: every term must match its exact spelling (instrument names such as "impact", "explorer").
    exclude: another term list whose terms are left out (e.g. cell types in the cell line list).
    """
    terms, _ = au.load_terms(name)
    skip = set(au.load_terms(exclude)[1]) if exclude else set()
    index: dict[str, tuple[str, str | None]] = {}

    def add(phrase, term):
        key = _norm(phrase)
        if len(key) < 3 or key in STOP_TERMS or key.count(" ") >= MAX_NGRAM:
            return
        strict = " ".join(_TOKEN.findall(phrase)) if always_strict or _is_strict(phrase) else None
        index.setdefault(key, (term.strip(), strict))

    for term in terms:
        low = term.lower().strip()
        if low in STOP_TERMS or low in skip or low.startswith("obsolete") or re.search(r"cell lines?$", low):
            continue
        if always_strict and term == low and " " not in low:
            continue  # all-lowercase one-word "names" ("impact", "explorer") are ordinary English words

        m = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", term)
        if m and re.fullmatch(r"[A-Za-z0-9+/-]{2,10}", m.group(2)) and any(c.isupper() for c in m.group(2)):
            # "Dithiothreitol (DTT)": the name and the abbreviation both point to the term.
            add(term, term)
            add(m.group(1), term)
            add(m.group(2), term)
        elif m:
            # "hypoblast (generic)", "... (Homo sapiens)": the bracketed qualifier is not part of the name.
            add(m.group(1), term)
        else:
            add(term, term)
    return index


class _Collector:
    def __init__(self):
        self.hits: "OrderedDict[tuple[str, str], Hit]" = OrderedDict()

    def add(self, column, value, found, evidence, source="rule", **extra):
        key = (column, value)
        span = getattr(evidence, "span", None)
        if key in self.hits:
            hit = self.hits[key]
            if span in hit.spans:
                return  # the same words matched twice (e.g. two rules)
            hit.count += 1
        else:
            hit = self.hits[key] = Hit(column, value, str(found).strip(), str(evidence), 0, source, extra=dict(extra))
            hit.count = 1
        if span:
            hit.spans.append(span)


# --------------------------------------------------------------------------------------
# Extractors
# --------------------------------------------------------------------------------------
def _dictionary(sentences, col: _Collector):
    for column, (name, always_strict, exclude) in DICTIONARY_COLUMNS.items():
        index = _phrase_index(name, always_strict, exclude)
        for sent in sentences:
            toks = list(_TOKEN.finditer(sent))
            low = [t.group(0).lower() for t in toks]
            i = 0
            while i < len(toks):
                matched = 0
                for n in range(min(MAX_NGRAM, len(toks) - i), 0, -1):
                    words = low[i:i + n]
                    entry = index.get(" ".join(words))
                    if entry is None and len(words[-1]) > 3:
                        entry = index.get(" ".join(words[:-1] + [_singular(words[-1])]))
                    if entry is None:
                        continue
                    term, strict = entry
                    original = " ".join(t.group(0) for t in toks[i:i + n])
                    if strict and original not in (strict, strict + "s"):
                        continue
                    start, end = toks[i].start(), toks[i + n - 1].end()
                    col.add(column, term, sent[start:end], _short(sent, start, end), "dictionary")
                    matched = n
                    break
                i += matched or 1


def _phrases(sentences, col: _Collector):
    for column, rules in PHRASES.items():
        for sent in sentences:
            taken = []
            for pattern, value in rules:
                for m in re.finditer(pattern, sent, flags=re.I):
                    if any(a <= m.start() < b for a, b in taken):
                        continue
                    taken.append((m.start(), m.end()))
                    col.add(column, value, m.group(0), _short(sent, m.start(), m.end()))


def _organisms(sentences, col: _Collector):
    for sent in sentences:
        for pattern, value in ORGANISM_PATTERNS:
            for m in re.finditer(pattern, sent):
                col.add("characteristics[organism]", value, m.group(0), _short(sent, m.start(), m.end()))
        for line in HUMAN_CELL_LINES:
            m = re.search(rf"(?<![\w-]){re.escape(line)}(?![\w-])", sent)
            if m:
                col.add("characteristics[organism]", "Homo sapiens", m.group(0), _short(sent, m.start(), m.end()))
    _binomials(sentences, col)


_BINOMIAL = re.compile(r"\b([A-Z][a-z]{2,})\s([a-z]{3,})\b")
_ABBREVIATED = re.compile(r"\b([A-Z])\.\s?([a-z]{3,})\b")
_NOT_EPITHET = {"were", "was", "and", "with", "from", "for", "the", "that", "this", "using", "are", "has", "had",
                "have", "into", "onto", "per", "via", "after", "before", "from", "then", "than", "all", "both"}


TAXON_SEARCH_GROUPS = ("Eukaryota", "Bacteria", "Archaea", "Viruses")


@st.cache_resource(show_spinner="Loading NCBITaxon genus names (only needed once)...")
def _genera() -> frozenset:
    """Every genus name in NCBITaxon, to skip English word pairs ("Samples were") without a full lookup."""
    out = set()
    for group in TAXON_SEARCH_GROUPS:
        _, low = au._taxon_index(au.TAXON_GROUPS[group])
        out.update(m.group(1) for m in re.finditer(r"\n([a-z]{3,}) ", low))
    return frozenset(out)


def _binomials(sentences, col: _Collector, max_lookups: int = 60):
    """Latin species names (Plasmodium falciparum, P. falciparum) looked up in NCBITaxon."""
    known = {h.value for h in col.hits.values() if h.column == "characteristics[organism]"}
    found: dict[str, str] = {}  # lower-case name -> NCBITaxon spelling
    genera = _genera()
    candidates = OrderedDict()
    for sent in sentences:
        for m in _BINOMIAL.finditer(sent):
            if m.group(2) not in _NOT_EPITHET and m.group(1).lower() in genera:
                candidates.setdefault(f"{m.group(1)} {m.group(2)}", (sent, m.start(), m.end()))
    lookups = 0
    for name, (sent, a, b) in candidates.items():
        if name in known or lookups >= max_lookups:
            continue
        lookups += 1
        for group in TAXON_SEARCH_GROUPS:
            text, low = au._taxon_index(au.TAXON_GROUPS[group])
            i = low.find(f"\n{name.lower()}\n")
            if i >= 0:
                found[name.lower()] = text[i + 1:i + 1 + len(name)]
                col.add("characteristics[organism]", found[name.lower()], name, _short(sent, a, b), "taxonomy")
                break
    # "P. falciparum": expand with a full name found in the text, else look the epithet up in NCBITaxon.
    for sent in sentences:
        for m in _ABBREVIATED.finditer(sent):
            initial, epithet = m.group(1).lower(), m.group(2).lower()
            full = next((v for k, v in found.items() if k[0] == initial and k.endswith(" " + epithet)), None)
            if full is None and lookups < max_lookups and epithet not in _NOT_EPITHET:
                lookups += 1
                text, low = au._taxon_index(au.TAXON_GROUPS["Eukaryota"])
                hit = re.search(rf"\n({initial}[a-z]+ {re.escape(epithet)})\n", low)
                if hit:
                    full = text[hit.start(1):hit.end(1)]
                    found[full.lower()] = full
            if full:
                col.add("characteristics[organism]", full, m.group(0), _short(sent, m.start(), m.end()), "taxonomy")


def _tolerances(sentences, col: _Collector):
    value_re = re.compile(r"(?:±|\+/-|\+-)?\s*(\d+(?:\.\d+)?)\s*(ppm|Da|mmu|Th)\b")
    precursor = re.compile(r"precursor|MS1|parent|peptide mass|peptide tolerance|first search|main search|"
                           r"\bMS\b(?!/)", re.I)
    fragment = re.compile(r"fragment|MS/MS|MS2|product ion|MSMS|MS\^2", re.I)
    for sent in sentences:
        if not re.search(r"toleran|accuracy|deviation|window", sent, re.I):
            continue
        for m in value_re.finditer(sent):
            before = sent[:m.start()]
            p = max((x.end() for x in precursor.finditer(before)), default=-1)
            f = max((x.end() for x in fragment.finditer(before)), default=-1)
            if p < 0 and f < 0:
                continue
            column = "comment[fragment mass tolerance]" if f > p else "comment[precursor mass tolerance]"
            unit = "Da" if m.group(2) == "Th" else m.group(2)
            col.add(column, f"{float(m.group(1)):g} {unit}", m.group(0).strip(), _short(sent, m.start(), m.end()))


def _collision_energy(sentences, col: _Collector):
    nce = r"(?:normali[sz]ed collision(?:al)?(?:[- ]induced dissociation)? energy|\bNCE\b)"
    patterns = [
        (re.compile(rf"(\d+(?:\.\d+)?)\s*(?:%|V|eV)?\s*(?:of\s+)?(?:an?\s+)?{nce}", re.I), "NCE"),
        (re.compile(rf"{nce}\s*(?:\(NCE\)\s*)?(?:of|at|was|=|:|set to|set at)\s*(?:set to\s*)?(\d+(?:\.\d+)?)\s*%?",
                    re.I), "NCE"),
        (re.compile(r"collision(?:al)? energy\s*(?:of|at|was|=|:|set to)?\s*(\d+(?:\.\d+)?)\s*eV", re.I), "eV"),
    ]
    for sent in sentences:
        for pattern, unit in patterns:
            for m in pattern.finditer(sent):
                value = float(m.group(1))
                if 5 <= value <= 100:
                    col.add("comment[collision energy]", f"{value:g} {unit}", m.group(0).strip(),
                            _short(sent, m.start(), m.end()))


_MOD_KIND = re.compile(r"\b(fixed|static|variable|dynamic)\b", re.I)
_MOD_KIND_AS = re.compile(r"\bas (?:an? )?(fixed|static|variable|dynamic)\b", re.I)


def _modification_type(sent: str, start: int, end: int) -> str | None:
    """Fixed or Variable, from the wording around a modification mention.

    Handles "X was set as fixed modification", "X and Y as variable modifications" and
    "Fixed modifications: X; variable modifications: Y".
    """
    def kind(word):
        return "Fixed" if word.lower() in ("fixed", "static") else "Variable"

    near = _MOD_KIND.search(sent[end:end + 40])          # "carbamidomethyl (C) was fixed"
    if near:
        return kind(near.group(1))
    later = _MOD_KIND_AS.search(sent, end)               # "oxidation (M) and acetylation as variable"
    if later:
        return kind(later.group(1))
    before = list(_MOD_KIND.finditer(sent[:start]))      # "variable modifications: oxidation"
    if before:
        return kind(before[-1].group(1))
    return None


def _modifications(sentences, col: _Collector):
    for sent in sentences:
        if not MOD_CONTEXT.search(sent):
            continue
        taken = []
        for pattern, name, targets, mod_type, position in MODIFICATIONS:
            for m in re.finditer(pattern, sent, re.I):
                if any(a <= m.start() < b for a, b in taken):
                    continue
                taken.append((m.start(), m.end()))
                mod_type = _modification_type(sent, m.start(), m.end()) or mod_type
                after = sent[m.end():m.end() + 60]
                letters = (re.match(r"\s*\(([A-Z](?:,?\s?[A-Z]){0,3})\)", after)
                           or re.match(r"\w*\s+(?:of|on|at)\s+(?:the\s+)?([A-Z](?:\s*(?:,|and|/)\s*[A-Z]){0,3})\b(?!\w)",
                                       after))
                if letters and position == "Anywhere":
                    targets = ",".join(re.findall(r"[A-Z]", letters.group(1)))
                else:
                    named = [AMINO_ACIDS[a] for a in AMINO_ACIDS if re.search(rf"\b{a}s?\b", after.lower())]
                    if named and name not in ("Acetyl",):
                        targets = ",".join(dict.fromkeys(named))
                value = spec.modification_string(name, f"UNIMOD:{au.load_unimod().get(name, '')}" if
                                                 au.load_unimod().get(name) else None, mod_type, position, targets)
                col.add("comment[modification parameters]", value, m.group(0), _short(sent, m.start(), m.end()))


def _labels(sentences, col: _Collector):
    for sent in sentences:
        for m in re.finditer(r"label[- ]free|\bLFQ\b", sent, re.I):
            col.add("comment[label]", "label free sample", m.group(0), _short(sent, m.start(), m.end()))
        for m in re.finditer(r"\bTMT\s?(?:pro)?\s?[- ]?(\d{1,2})?[- ]?plex|\bTMTpro\b|\bTMT\b", sent):
            plex = int(m.group(1)) if m.group(1) else (16 if "pro" in m.group(0).lower() else None)
            channels = TMT_CHANNELS.get(plex, ["TMT"])
            for ch in channels:
                col.add("comment[label]", ch, m.group(0), _short(sent, m.start(), m.end()))
        for m in re.finditer(r"\biTRAQ\b(?:\s*(\d)[- ]?plex)?", sent, re.I):
            plex = int(m.group(1)) if m.group(1) else 4
            for ch in ITRAQ_CHANNELS.get(plex, ITRAQ_CHANNELS[4]):
                col.add("comment[label]", ch, m.group(0), _short(sent, m.start(), m.end()))
        for m in re.finditer(r"\bSILAC\b", sent):
            for ch in ["SILAC light", "SILAC heavy"] + (["SILAC medium"] if re.search(r"medium", sent, re.I) else []):
                col.add("comment[label]", ch, m.group(0), _short(sent, m.start(), m.end()))


def _sex_and_age(sentences, col: _Collector):
    for sent in sentences:
        for m in re.finditer(r"\b(female|male|women|men|girls|boys)\b", sent, re.I):
            sex = "female" if m.group(1).lower() in ("female", "women", "girls") else "male"
            col.add("characteristics[sex]", sex, m.group(0), _short(sent, m.start(), m.end()))
        for m in re.finditer(r"(?:aged?\s*(?:of\s*)?)?(\d+(?:\.\d+)?)\s*(?:(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*)?[- ]?"
                             r"(years?|yrs?|months?|weeks?|days?)[- ](?:old|of age)", sent, re.I):
            unit = {"y": "Y", "m": "M", "w": "W", "d": "D"}[m.group(3)[0].lower()]
            a = f"{m.group(1)}{unit}"
            value = f"{a}-{m.group(2)}{unit}" if m.group(2) else a
            if spec.check_age(value):
                col.add("characteristics[age]", value, m.group(0), _short(sent, m.start(), m.end()))


# --------------------------------------------------------------------------------------
# Base confidence per (column, source), calibrated on the agentic-metadata benchmark (107 manuscripts with
# reference SDRFs): roughly the share of such suggestions that were correct. "*" is the default for a source.
# The most-mentioned suggestion of a column gets a bonus on top (see _confidence), which is where most of the
# correct ones are. The reference SDRFs are incomplete, so these numbers err on the low side.
BASE_CONFIDENCE = {
    ("*", "rule"): 0.6,
    ("*", "taxonomy"): 0.35,
    ("*", "dictionary"): 0.3,
    ("*", "inferred"): 0.5,
    ("characteristics[organism]", "rule"): 0.4,
    ("characteristics[organism part]", "dictionary"): 0.2,
    ("characteristics[disease]", "dictionary"): 0.15,
    ("characteristics[cell type]", "dictionary"): 0.1,
    ("characteristics[cell line]", "dictionary"): 0.45,
    ("characteristics[developmental stage]", "dictionary"): 0.2,
    ("characteristics[ancestry category]", "dictionary"): 0.2,
    ("characteristics[sex]", "rule"): 0.75,
    ("characteristics[age]", "rule"): 0.3,
    ("comment[instrument]", "dictionary"): 0.45,
    ("comment[cleavage agent details]", "rule"): 0.65,
    ("comment[dissociation method]", "rule"): 0.7,
    ("comment[modification parameters]", "rule"): 0.7,
    ("comment[label]", "rule"): 0.5,
    ("comment[label]", "inferred"): 0.75,
    ("comment[collision energy]", "rule"): 0.65,
    ("comment[precursor mass tolerance]", "rule"): 0.3,
    ("comment[fragment mass tolerance]", "rule"): 0.4,
    ("comment[fractionation method]", "rule"): 0.45,
    ("comment[proteomics data acquisition method]", "rule"): 0.7,
}


def _confidence(hit: Hit, top_count: int, n_candidates: int) -> float:
    base = BASE_CONFIDENCE.get((hit.column, hit.source), BASE_CONFIDENCE[("*", hit.source)])
    score = base + min(0.15, 0.05 * (hit.count - 1))
    if hit.count == top_count:
        score += 0.1 if n_candidates > 1 else 0.15
    return round(max(0.05, min(0.97, score)), 2)


@st.cache_data(show_spinner=False, max_entries=10)
def extract(text: str) -> dict:
    """Metadata suggestions found in a text.

    Returns {"text": the cleaned text that spans refer to, "hits": [hit dicts, best first within a column]}.
    """
    clean = clean_text(text)
    sentences = split_sentences(clean)
    col = _Collector()
    _organisms(sentences, col)
    _phrases(sentences, col)
    _modifications(sentences, col)
    _labels(sentences, col)
    _tolerances(sentences, col)
    _collision_energy(sentences, col)
    _sex_and_age(sentences, col)
    _dictionary(sentences, col)
    if not any(h.column == "comment[label]" for h in col.hits.values()) and \
            any(h.column == "comment[instrument]" for h in col.hits.values()):
        col.add("comment[label]", "label free sample", "",
                "No labelling (TMT, iTRAQ, SILAC, ...) is mentioned in the text.", "inferred")
    hits = list(col.hits.values())
    by_column: dict[str, list[Hit]] = {}
    for h in hits:
        by_column.setdefault(h.column, []).append(h)
    for column_hits in by_column.values():
        top = max(h.count for h in column_hits)
        for h in column_hits:
            h.confidence = _confidence(h, top, len(column_hits))
    order = {c: i for i, c in enumerate(by_column)}
    hits.sort(key=lambda h: (order[h.column], -h.confidence, -h.count))
    return {"text": clean, "hits": [dict(h.__dict__, id=i) for i, h in enumerate(hits)]}
