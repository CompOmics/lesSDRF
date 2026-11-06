import html
import io
import re

import pandas as pd
import streamlit as st

import app_utils as au
import spec
import textmining as tm

au.setup_page("Map local metadata")
st.title("1. Map local metadata to SDRF")
st.markdown("""If you have a local metadata sheet or a methods section describing your experiment, you can use it to
fill in the SDRF. Values are checked against the SDRF rules: ontology columns must contain ontology terms, age must be
written like `45Y`, sex as `male`/`female`, and so on.""")

df = au.require_df()
au.show_df(df)
au.sidebar_export()


# ======================================================================================
# From a metadata sheet
# ======================================================================================
@st.cache_data(show_spinner=False, max_entries=3)
def read_sheet(data: bytes, name: str) -> pd.DataFrame:
    ext = name.rsplit(".", 1)[-1].lower()
    if ext == "xlsx":
        return pd.read_excel(io.BytesIO(data), dtype=str)
    sep = "," if ext == "csv" else "\t"
    return pd.read_csv(io.BytesIO(data), sep=sep, dtype=str, keep_default_na=False)


def check_values(column: str, values: list) -> tuple[list, list]:
    """Normalise values for an SDRF column. Returns (new values, invalid values)."""
    rule = spec.rule_for(column, spec.selected_templates())
    base = spec.base_name(column)
    reserved = {w.lower(): w for w in rule.reserved_words}
    out, bad = [], []
    for v in values:
        if spec.is_empty_value(v):
            out.append(None)
            continue
        v = str(v).strip()
        new = reserved.get(v.lower())
        if new is None:
            if base == "characteristics[organism]":
                new = au.find_organism(v)
            elif base == "characteristics[sex]":
                new = spec.SEX_ALIASES.get(v.lower())
                new = new if new and (not rule.values or new in rule.values or new in reserved.values()) else None
            elif base == "characteristics[age]":
                new = spec.clean_age(v) if spec.check_age(spec.clean_age(v)) else None
            elif base in au.ONTOLOGY_COLUMNS:
                new = au.canonical_term(au.ONTOLOGY_COLUMNS[base][0], v)
                new = au.with_accession(base, new) if new else None
            elif rule.values:
                new = next((x for x in rule.values if x.lower() == v.lower()), None)
            elif base in ("characteristics[biological replicate]", "comment[technical replicate]",
                          "comment[fraction identifier]"):
                new = v if v.isdigit() else None
            else:
                new = v
        if new is None:
            bad.append(v)
        out.append(new)
    return out, bad


def sheet_tab():
    sheet = st.file_uploader("Upload your local metadata file", type=["csv", "tsv", "txt", "xlsx"])
    if sheet is None:
        return
    try:
        meta = spec.normalize_df(read_sheet(sheet.getvalue(), sheet.name))
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        return
    st.write("Your metadata file:")
    st.dataframe(meta, hide_index=True, height=200)

    # How do metadata rows correspond to SDRF rows?
    key_options = ["(same row order as the SDRF)"] + list(meta.columns)
    key_col = st.selectbox("Which column in your file contains the raw file names?", key_options,
                           help="Rows are matched on comment[data file]. Without such a column, rows are matched by order.")
    if key_col == key_options[0]:
        if len(meta) != len(df):
            st.error(f"Your file has {len(meta)} rows and the SDRF has {len(df)}; "
                     "select the column with raw file names.")
            return
        aligned = meta.reset_index(drop=True)
    else:
        lookup = meta.drop_duplicates(key_col).set_index(key_col)
        files = df.get("comment[data file]", pd.Series([None] * len(df)))
        missing = [f for f in files if f not in lookup.index]
        if missing:
            st.warning(f"{len(missing)} SDRF row(s) have no match in your file, e.g. {missing[:3]}; "
                       "they stay unchanged.")
        aligned = lookup.reindex(files.tolist()).reset_index(drop=True)
        aligned[key_col] = files.tolist()

    st.subheader("Map columns")
    targets = list(dict.fromkeys(spec.all_columns(spec.selected_templates()) + list(df.columns)))
    targets = [t for t in targets if t not in spec.METADATA_COLUMNS]
    pairs = st.session_state.setdefault("mapping_pairs", [])
    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    with c1:
        src = st.selectbox("Column in your file", [""] + list(meta.columns))
    with c2:
        dst = st.selectbox("SDRF column", [""] + targets)
    with c3:
        check = st.button("Check", type="primary", disabled=not (src and dst))

    if check:
        values, bad = check_values(dst, aligned[src].tolist())
        st.session_state["mapping_check"] = {"src": src, "dst": dst, "values": values, "bad": sorted(set(bad))}

    result = st.session_state.get("mapping_check")
    if result and result["src"] in aligned.columns:
        if result["bad"]:
            st.error(f"{len(result['bad'])} value(s) of **{result['src']}** are not valid for **{result['dst']}**: "
                     + ", ".join(f"`{b}`" for b in result["bad"][:20])
                     + ". Fix them in your file, or fill this column in step 3/4 by picking terms from the ontology.",
                     icon="❌")
        else:
            preview = sorted({v for v in result["values"] if v})[:10]
            st.success(f"All values of **{result['src']}** are valid for **{result['dst']}**, e.g. "
                       + ", ".join(f"`{p}`" for p in preview), icon="✅")
            if st.button(f"Write to {result['dst']}"):
                new = au.ensure_columns(au.get_df(), [result["dst"]])
                old = new[result["dst"]].tolist()
                new[result["dst"]] = [v if v is not None else o for v, o in zip(result["values"], old)]
                au.save_df(new)
                pairs.append(f"{result['src']} → {result['dst']}")
                st.session_state["mapping_check"] = None
                st.rerun()

    if pairs:
        st.write("Mapped so far: " + " · ".join(pairs))


# ======================================================================================
# From methods text
# ======================================================================================
STATUS_LABELS = {"pending": "to review", "accepted": "accepted", "declined": "declined"}
# Columns whose values are picked from a list in step 3; several accepted values are pre-selected there.
PICKER_COLUMNS = set(au.ONTOLOGY_COLUMNS) | {"characteristics[organism]", "comment[cleavage agent details]",
                                             "comment[proteomics data acquisition method]"}

HIGHLIGHT_CSS = """
<style>
  body { margin: 0; font-family: "Source Sans Pro", "Segoe UI", sans-serif; font-size: 15px; line-height: 1.65;
         color: #262730; background: #ffffff; }
  .text { white-space: pre-wrap; padding: 12px 16px; }
  mark { border-radius: 3px; padding: 0 2px; cursor: help; }
  mark.pending { background: #fff1b8; }
  mark.accepted { background: #c8ecd0; }
  mark.declined { background: #ececec; color: #8a8a8a; text-decoration: line-through; }
  mark.focus { background: #f79646; color: #ffffff; font-weight: 600; }
</style>
"""


def render_highlights(text: str, hits: list[dict], status: dict, focus: set) -> str:
    """The text as HTML, with every suggestion's mentions highlighted (focused suggestions in orange)."""
    by_span: dict[tuple, list[dict]] = {}
    for h in hits:
        for a, b in h["spans"]:
            by_span.setdefault((a, b), []).append(h)
    # Place focused spans first so overlapping matches never hide them, then longer before shorter.
    order = sorted(by_span, key=lambda s: (not any(h["id"] in focus for h in by_span[s]), -(s[1] - s[0]), s[0]))
    placed: list[tuple[int, int]] = []
    for a, b in order:
        if all(b <= x or a >= y for x, y in placed):
            placed.append((a, b))
    placed.sort()
    out, pos = [], 0
    for a, b in placed:
        group = by_span[(a, b)]
        if any(h["id"] in focus for h in group):
            cls = "focus"
        else:
            states = {status.get(h["id"], "pending") for h in group}
            cls = "accepted" if "accepted" in states else "pending" if "pending" in states else "declined"
        title = "\n".join(f"{h['column']}: {h['value']} ({h['confidence']:.0%}, "
                          f"{STATUS_LABELS[status.get(h['id'], 'pending')]})" for h in group)
        out.append(html.escape(text[pos:a]))
        out.append(f'<mark class="{cls}" title="{html.escape(title)}">{html.escape(text[a:b])}</mark>')
        pos = b
    out.append(html.escape(text[pos:]))
    # Scroll only the text panel itself (scrollIntoView would also scroll the page around it).
    script = ("<script>const f=document.querySelector('mark.focus');"
              "if(f){window.scrollTo(0, Math.max(0, f.offsetTop - window.innerHeight / 3));}</script>")
    return HIGHLIGHT_CSS + f'<div class="text">{"".join(out)}</div>' + script


def apply_accepted(hits: list[dict], overwrite: bool) -> list[str]:
    """Write accepted suggestions into the SDRF. Returns a summary of what happened."""
    df = au.get_df()
    messages = []
    grouped: dict[str, list[str]] = {}
    for h in hits:
        grouped.setdefault(h["column"], []).append(h["value"])
    for column, values in grouped.items():
        values = list(dict.fromkeys(values))
        filled = [c for c in df.columns if spec.base_name(c) == column and not spec.is_empty_column(df, c)]
        if filled and not overwrite:
            messages.append(f"Skipped **{column}**: already filled in (tick *Overwrite* to replace it).")
            continue
        if column == "comment[modification parameters]":
            df = df.drop(columns=[c for c in df.columns if spec.base_name(c) == column])
            cols = au.copy_columns(column, len(values))
            df = au.ensure_columns(df, cols)
            for c, v in zip(cols, values):
                df[c] = v
            messages.append(f"Wrote {len(values)} modification(s) to **{column}**.")
        elif column == "comment[collision energy]":
            df = au.ensure_columns(df, [column])
            df[column] = ";".join(values)
            messages.append(f"Filled **{column}** with `{';'.join(values)}`.")
        elif column == "comment[label]" and values != ["label free sample"]:
            st.session_state["sel_labeling"] = [v for v in values if v != "label free sample"]
            messages.append(f"Pre-selected {len(values)} label(s) in **step 2 (Labeling)**, where every file is "
                            "split into one row per label.")
        elif len(values) == 1:
            df = au.ensure_columns(df, [column])
            df[column] = au.with_accession(column, values[0])
            messages.append(f"Filled **{column}** with `{values[0]}`.")
        elif column in PICKER_COLUMNS:
            key = re.sub(r"\W+", "_", column)
            st.session_state[f"sel_{key}"] = values
            messages.append(f"Pre-selected {len(values)} values for **{column}** in step 3/4: assign them to the "
                            "right rows there (e.g. with *Apply to: rows whose comment[data file] contains ...*).")
        else:
            messages.append(f"**{column}** has {len(values)} accepted values ({', '.join(values)}); "
                            "fill it per row in step 3/4.")
    au.save_df(df)
    return messages


def text_tab():
    st.write("""Paste the methods section of your manuscript (or any description of the experiment). lesSDRF looks for
    organisms, tissues, diseases, cell lines, instruments, enzymes, modifications, labels, mass tolerances and more,
    using the ontologies built into the app. Nothing leaves the app and nothing is written to your SDRF until you
    accept it.""")
    upload = st.file_uploader("Or upload a text file", type=["txt", "md"], key="tm_file")
    default = upload.getvalue().decode("utf-8", errors="ignore") if upload is not None else ""
    text = st.text_area("Methods text", value=default, height=200, key=f"tm_text_{upload.name if upload else ''}",
                        placeholder="e.g. Proteins were reduced with DTT, alkylated with iodoacetamide and digested "
                                    "with trypsin. Peptides were analysed on a Q Exactive HF ...")
    if st.button("Find metadata", type="primary", disabled=not (text and text.strip())):
        with st.spinner("Searching the text..."):
            result = tm.extract(text)
        st.session_state["tm_result"] = result
        st.session_state["tm_status"] = {}
        st.session_state["tm_messages"] = []

    result = st.session_state.get("tm_result")
    if not result:
        return
    hits = result["hits"]
    status: dict = st.session_state.setdefault("tm_status", {})
    if not hits:
        st.info("No metadata found in this text.")
        return

    # ---- filters and bulk actions ------------------------------------------------------
    st.subheader(f"{len(hits)} suggestions")
    c1, c2, c3 = st.columns([2, 2, 2], vertical_alignment="bottom")
    with c1:
        columns = list(dict.fromkeys(h["column"] for h in hits))
        only = st.multiselect("Show SDRF columns", columns, placeholder="All columns", key="tm_filter_cols")
    with c2:
        show = st.segmented_control("Show", ["all", "to review", "accepted", "declined"], default="all",
                                    key="tm_filter_status")
    with c3:
        threshold = st.slider("Accept all suggestions with a confidence of at least", 0.0, 1.0, 0.7, 0.05,
                              format="%.2f", key="tm_threshold")
    visible = [h for h in hits
               if (not only or h["column"] in only)
               and (show in (None, "all") or STATUS_LABELS[status.get(h["id"], "pending")] == show)]

    table = pd.DataFrame([{
        "status": STATUS_LABELS[status.get(h["id"], "pending")],
        "confidence": h["confidence"],
        "SDRF column": h["column"],
        "value": h["value"],
        "found as": h["found"] or "(inferred)",
        "mentions": h["count"],
    } for h in visible])

    left, right = st.columns([5, 4])
    with left:
        st.caption("Select rows to highlight them in the text, then accept or decline them.")
        # The selection is stored per row number: a new key whenever the rows change (filter, accept, decline)
        # makes sure an old selection never lands on other suggestions.
        table_key = "tm_table_" + str(abs(hash(tuple((h["id"], status.get(h["id"], "")) for h in visible))))
        event = st.dataframe(
            table, hide_index=True, on_select="rerun", selection_mode="multi-row", key=table_key,
            height=min(35 * (len(table) + 1) + 3, 460),
            column_config={
                "status": st.column_config.TextColumn("status", width="small"),
                "confidence": st.column_config.ProgressColumn("confidence", min_value=0.0, max_value=1.0,
                                                              format="percent", color="#f79646", width="small"),
                "mentions": st.column_config.NumberColumn("mentions", width="small"),
            })
        selected = [visible[i]["id"] for i in event.selection.rows if i < len(visible)]
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("Accept", type="primary", disabled=not selected, key="tm_accept"):
                status.update({i: "accepted" for i in selected})
                st.rerun()
        with b2:
            if st.button("Decline", disabled=not selected, key="tm_decline"):
                status.update({i: "declined" for i in selected})
                st.rerun()
        with b3:
            if st.button(f"Accept all ≥ {threshold:.0%}", key="tm_accept_all"):
                status.update({h["id"]: "accepted" for h in visible if h["confidence"] >= threshold
                               and status.get(h["id"], "pending") == "pending"})
                st.rerun()
        with b4:
            if st.button("Reset", key="tm_reset"):
                status.clear()
                st.rerun()
        if selected:
            for h in [h for h in hits if h["id"] in selected][:3]:
                st.caption(f"**{h['column']}: {h['value']}** — {h['evidence']}")
    with right:
        st.caption("Highlights: yellow = to review, green = accepted, grey = declined, orange = selected. "
                   "Hover a highlight for details.")
        # All user text is HTML-escaped in render_highlights, so pasted text cannot inject markup.
        st.iframe(render_highlights(result["text"], visible, status, set(selected)), height=520)

    # ---- write to the SDRF ---------------------------------------------------------
    accepted = [h for h in hits if status.get(h["id"]) == "accepted"]
    st.divider()
    c4, c5 = st.columns([3, 2], vertical_alignment="center")
    with c4:
        overwrite = st.checkbox("Overwrite columns that are already filled in", key="tm_overwrite")
    with c5:
        if st.button(f"Write {len(accepted)} accepted suggestion(s) to the SDRF", type="primary",
                     disabled=not accepted, key="tm_apply"):
            st.session_state["tm_messages"] = apply_accepted(accepted, overwrite)
            st.rerun()
    for message in st.session_state.get("tm_messages", []):
        st.write("- " + message)


sheet_ui, text_ui = st.tabs(["From a metadata sheet", "From methods text"])
with sheet_ui:
    sheet_tab()
with text_ui:
    text_tab()
