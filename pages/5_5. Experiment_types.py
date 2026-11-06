import streamlit as st

import app_utils as au
import spec

au.setup_page("Experiment types")
st.title("5. Experiment types")
st.write("""Some experiment types need extra columns. The SDRF-Proteomics community maintains a template for each of
them ([sdrf-templates](https://github.com/bigbio/sdrf-templates)). Adding a template adds its required and
recommended columns to your SDRF; fill them in under step 3, and its optional columns under step 4.""")

df = au.require_df()
au.show_df(df)
au.sidebar_export()

templates = spec.selected_templates()
tech = next((t for t in templates if t in spec.TECHNOLOGY_TEMPLATES), "ms-proteomics")
options = [t for t in spec.available(spec.EXPERIMENT_TEMPLATES)
           if tech == "ms-proteomics" or t not in spec.MS_ONLY_TEMPLATES]

for name in options:
    active = name in templates
    with st.container(border=True):
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        with c1:
            st.markdown(f"**{spec.EXPERIMENT_TEMPLATES[name]}** · `{name} v{spec.template_version(name)}`"
                        + (" · in use" if active else ""))
            st.caption(spec.template_description(name))
            own = [c for c in spec.template_columns([name], levels=("required", "recommended"))
                   if c not in spec.template_columns([tech], levels=("required", "recommended", "optional"))]
            if own:
                st.caption("Adds: " + ", ".join(f"`{c}`" for c in own))
        with c2:
            if not active and st.button("Add", key=f"add_{name}", type="primary"):
                new_templates = templates + [name]
                st.session_state["templates"] = new_templates
                au.save_df(au.ensure_columns(df, spec.template_columns(new_templates)))
                st.rerun()
            if active and st.button("Remove", key=f"rm_{name}"):
                st.session_state["templates"] = [t for t in templates if t != name]
                st.rerun()

st.caption("Removing a template keeps the columns you already filled in; clear them in step 3 or 4 if needed.")
st.markdown("Missing an experiment type? Join the discussion at "
            "[proteomics-sample-metadata](https://github.com/bigbio/proteomics-sample-metadata/issues).")
