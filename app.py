"""
app.py -- Streamlit UI for the EMB-CAR ID Card Generator.

Run with:  streamlit run app.py

This intentionally contains NO pipeline logic itself -- everything real lives
in id_pipeline.py. This file only wires buttons to it and displays progress.
"""

import streamlit as st
import id_pipeline as pipe

st.set_page_config(page_title="EMB-CAR ID Card Generator", layout="centered")
st.title("EMB-CAR ID Card Generator")

if "df_loaded" not in st.session_state:
    st.session_state.df_loaded = False
if "cards_generated" not in st.session_state:
    st.session_state.cards_generated = False


def run_stage(generator_func, **kwargs):
    """Consumes a pipeline generator, showing each yielded line live in a
    scrolling log box instead of waiting silently for the whole stage to finish."""
    log_lines = []
    log_box = st.empty()
    result = None
    with st.spinner("Working..."):
        for item in generator_func(**kwargs):
            if isinstance(item, tuple) and item and item[0] == "__RESULT__":
                result = item[1:]
                continue
            log_lines.append(str(item))
            log_box.code("\n".join(log_lines), language=None)
    return result


st.header("1. Fetch Data")
force_full = st.checkbox("Force full reprocess (ignore last-run filter)", value=False)
if st.button("Fetch Data from Google Sheets + Drive"):
    run_stage(pipe.fetch_data, force_full_reprocess=force_full)
    st.session_state.df_loaded = True

if st.session_state.df_loaded:
    st.success(f"{len(pipe._state['df'])} total response(s) loaded.")
    with st.expander("Preview data"):
        st.dataframe(pipe._state["df"])

st.divider()

st.header("2. Generate Cards")
force_reprint_all = st.checkbox("Force reprint EVERYONE (use after a template/layout fix)", value=False)
reprint_ids_raw = st.text_input("Force reprint specific ID number(s), comma-separated (optional)")
if st.button("Generate Cards"):
    reprint_ids = {s.strip() for s in reprint_ids_raw.split(",") if s.strip()}
    run_stage(pipe.generate_cards, force_reprint_all=force_reprint_all, force_reprint_ids=reprint_ids)
    st.session_state.cards_generated = True

st.divider()

st.header("3. Export PDF")
if st.button("Export Print-Ready PDF"):
    result = run_stage(pipe.export_pdf)
    if result and result[0]:
        pdf_bytes, archive_path = result
        st.success(f"PDF ready: {archive_path.name}")
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=archive_path.name,
            mime="application/pdf",
        )
    else:
        st.warning("No PDF was produced -- check the log above.")

st.divider()
with st.expander("Setup notes"):
    st.markdown("""
    - **Shared folder auto-drop:** set `SHARED_PRINT_DIR` at the top of `id_pipeline.py`
      to your shared folder path (e.g. `Path(r"C:\\ID_Card_Printouts")`) to also copy
      every export there automatically, in addition to the download button above.
    - **Network access from other PCs:** run this with
      `streamlit run app.py --server.address 0.0.0.0`, then other PCs on the same
      office network can reach it at `http://<this-pc-ip>:8501`.
    """)