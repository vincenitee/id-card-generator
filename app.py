"""
app.py -- Streamlit UI for the EMB-CAR ID Card Generator.

Run with:  streamlit run app.py --server.address 0.0.0.0

No pipeline logic lives here -- everything real is in id_pipeline.py.
This file only wires the UI to it.
"""

import streamlit as st
import id_pipeline as pipe

st.set_page_config(
    page_title="EMB-CAR ID Card Generator",
    page_icon="\U0001FAAA",
    layout="wide",
)

# --- minimal styling polish ---
st.markdown("""
    <style>
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

if "df_loaded" not in st.session_state:
    st.session_state.df_loaded = False
if "cards_generated" not in st.session_state:
    st.session_state.cards_generated = False
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = None


def run_stage(label, generator_func, **kwargs):
    """Runs a pipeline generator inside a native Streamlit status widget --
    shows a spinner while running, a live-updating log, and settles into a
    green 'complete' or red 'error' state when done, instead of a plain
    scrolling text box with no clear finished/failed signal."""
    result = None
    with st.status(label, expanded=True) as status:
        try:
            for item in generator_func(**kwargs):
                if isinstance(item, tuple) and item and item[0] == "__RESULT__":
                    result = item[1:]
                    continue
                st.write(str(item))
            status.update(label=f"{label} \u2014 done", state="complete")
        except Exception as e:
            st.error(f"Error: {e}")
            status.update(label=f"{label} \u2014 failed", state="error")
    return result


def show_metrics():
    df = pipe._state["df"]
    if len(df) == 0:
        return
    total = len(df)
    missing = int(df["id_missing"].sum()) if "id_missing" in df.columns else 0
    flagged = sum(
        1 for r in pipe._state["all_results"].values()
        if r["photo_flags"].get("error") or r["photo_flags"].get("needs_review") or r["signature_error"]
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total responses", total)
    c2.metric("Missing ID number", missing, delta=None, delta_color="off" if missing == 0 else "inverse")
    c3.metric("Flagged for review", flagged, delta=None, delta_color="off" if flagged == 0 else "inverse")


def show_review_grid():
    """Actual image thumbnails in a responsive grid, instead of a static
    matplotlib plot -- crisper, and lets you visually scan a whole batch fast."""
    flagged = {
        id_num: r for id_num, r in pipe._state["all_results"].items()
        if r["photo_flags"].get("error") or r["photo_flags"].get("needs_review") or r["signature_error"]
    }
    if not flagged:
        st.success("Nothing flagged for review.")
        return

    st.caption(f"{len(flagged)} row(s) need a manual look before printing")
    cols = st.columns(4)
    for i, (id_num, r) in enumerate(flagged.items()):
        with cols[i % 4]:
            if r["photo"]:
                st.image(r["photo"], use_container_width=True)
            else:
                st.error("No photo")
            st.caption(f"**{id_num}**")
            if r["photo_flags"].get("needs_review"):
                st.caption(":orange[Auto-recropped -- verify]")
            if r["photo_flags"].get("error"):
                st.caption(f":red[Photo error: {r['photo_flags']['error'][:40]}]")
            if r["signature_error"]:
                st.caption(f":red[Signature error: {r['signature_error'][:40]}]")


# =========================================================
# Sidebar -- advanced options kept out of the main flow
# =========================================================
with st.sidebar:
    st.header("\U0001FAAA EMB-CAR")
    st.caption("ID Card Generator")
    st.divider()

    st.subheader("Advanced options")
    force_full = st.checkbox("Force full re-fetch", value=False,
                              help="Ignore the last-run filter and re-check every response, not just new ones.")
    force_reprint_all = st.checkbox("Force reprint everyone", value=False,
                                     help="Use after a template/layout fix that affects every previously printed card.")
    reprint_ids_raw = st.text_input("Force reprint specific ID(s)", placeholder="EMBB-027-2026, EMBB-028-2026")

    st.divider()
    with st.expander("Setup notes"):
        st.markdown("""
        - **Shared folder:** set `SHARED_PRINT_DIR` in `id_pipeline.py`
        - **Network access:** run with `--server.address 0.0.0.0`,
          then other PCs reach this at `http://<this-pc-ip>:8501`
        """)


# =========================================================
# Main area
# =========================================================
st.title("EMB-CAR ID Card Generator")
show_metrics()
st.divider()

tab_fetch, tab_generate, tab_export = st.tabs(["\U0001F4E5 1. Fetch Data", "\U0001FAAA 2. Generate Cards", "\U0001F4C4 3. Export & Print"])

with tab_fetch:
    st.write("Pulls responses from the Google Form, cleans the data, and fetches employee photos/signatures from Drive.")
    if st.button("Fetch Data", type="primary", use_container_width=True):
        run_stage("Fetching data", pipe.fetch_data, force_full_reprocess=force_full)
        st.session_state.df_loaded = True

    if st.session_state.df_loaded and len(pipe._state["df"]) > 0:
        st.divider()
        st.subheader("Data preview")
        st.dataframe(
            pipe._state["df"].drop(columns=["e_signature_path", "id_picture_path"], errors="ignore"),
            use_container_width=True,
            column_config={
                "id_missing": st.column_config.CheckboxColumn("Missing ID"),
            },
        )
        st.subheader("Review queue")
        show_review_grid()

with tab_generate:
    st.write("Composites photo, signature, and data onto the official card templates.")
    st.caption("Rows already printed in a previous run are skipped automatically, unless forced above.")
    if st.button("Generate Cards", type="primary", use_container_width=True, disabled=not st.session_state.df_loaded):
        reprint_ids = {s.strip() for s in reprint_ids_raw.split(",") if s.strip()}
        run_stage("Generating cards", pipe.generate_cards,
                   force_reprint_all=force_reprint_all, force_reprint_ids=reprint_ids)
        st.session_state.cards_generated = True
    if not st.session_state.df_loaded:
        st.caption(":gray[Fetch data first.]")

with tab_export:
    st.write("Combines this run's cards into one print-ready PDF, correctly ordered front-then-back per employee.")
    if st.button("Export PDF", type="primary", use_container_width=True, disabled=not st.session_state.cards_generated):
        result = run_stage("Exporting PDF", pipe.export_pdf)
        st.session_state.pdf_ready = result

    if not st.session_state.cards_generated:
        st.caption(":gray[Generate cards first.]")

    if st.session_state.pdf_ready and st.session_state.pdf_ready[0]:
        pdf_bytes, archive_path = st.session_state.pdf_ready
        st.divider()
        st.success(f"Ready: **{archive_path.name}**")
        st.download_button(
            "\u2b07 Download PDF",
            data=pdf_bytes,
            file_name=archive_path.name,
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
        if pipe.SHARED_PRINT_DIR:
            st.caption(f"Also copied to the shared folder for the printer PC: `{pipe.SHARED_PRINT_DIR}`")