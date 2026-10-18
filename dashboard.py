import csv
import json
import os
import re
from datetime import datetime
from html import escape
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

BASE_DIR = Path(__file__).resolve().parent
HISTORY_DIR = BASE_DIR / "analysis_history"
LATEST_JSON = BASE_DIR / "analysis_results.json"
LATEST_CSV = BASE_DIR / "analysis_results.csv"
load_dotenv(BASE_DIR / ".env")

COLORS = {"Positive": "#34d399", "Negative": "#fb7185", "Feature Request": "#a78bfa", "Other": "#94a3b8"}

st.set_page_config(page_title="FeedbackIQ | Customer Intelligence", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp {background:#0b1120;color:#f3f4f6}
[data-testid="stHeader"] {background:#0b1120}
[data-testid="stSidebar"] {background:#111827;border-right:1px solid #334155;color:#f3f4f6!important}
[data-testid="stSidebar"] * {color:#e5e7eb}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] strong {color:#fff!important}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#cbd5e1!important}
[data-testid="stSidebar"] button,[data-testid="stSidebar"] button p,[data-testid="stSidebar"] button span {color:#fff!important}
[data-testid="stSidebar"] hr {border-color:#334155!important}
.block-container {max-width:1500px;padding-top:1.8rem;padding-bottom:3rem}
.hero {background:linear-gradient(120deg,#1e3a8a,#111827 80%);border:1px solid #3b528b;border-radius:18px;padding:25px 29px;margin-bottom:22px}
.hero h1 {color:#fff!important;font-size:2rem;margin:0}
.hero p {color:#dbeafe!important;margin:8px 0 0}
.metric-card {background:#111c2e;border:1px solid #334155;border-radius:14px;padding:19px;min-height:116px}
.metric-label {color:#cbd5e1!important;font-size:.84rem;letter-spacing:.04em}
.metric-value {font-size:2rem;font-weight:750;margin-top:8px}
.section-heading {color:#fff!important;font-size:1.22rem;font-weight:700;margin:14px 0}
.stApp p,.stApp label {color:#e5e7eb}
[data-testid="stMetric"] {background:#111c2e;border:1px solid #334155;border-radius:12px;padding:14px}
[data-testid="stMetricLabel"] {color:#cbd5e1!important}
[data-testid="stMetricValue"] {color:#fff!important}
div[data-testid="stPlotlyChart"] {background:#111827;border:1px solid #334155;border-radius:14px;padding:8px}
.stButton button,.stDownloadButton button {min-height:42px;border-radius:9px;font-weight:650}
[data-testid="stExpander"] {border-color:#334155}
input,textarea,[data-baseweb="select"] * {color:#111827!important}
</style>
""", unsafe_allow_html=True)


def normalize(data):
    if not isinstance(data, dict):
        data = {}
    data.setdefault("metadata", {})
    data.setdefault("analytics", {})
    data.setdefault("results", [])
    data.setdefault("failures", [])
    return data


def read_data(path):
    return normalize(json.loads(Path(path).read_text(encoding="utf-8")))


def get_history():
    entries = []
    if HISTORY_DIR.exists():
        for folder in HISTORY_DIR.iterdir():
            path = folder / "analysis_results.json"
            if folder.is_dir() and path.exists():
                try:
                    data = read_data(path)
                    meta = data["metadata"]
                    created = meta.get("created_at", "")
                    try:
                        sort_value = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        sort_value = folder.stat().st_mtime
                    entries.append({
                        "id": folder.name,
                        "path": path,
                        "name": meta.get("analysis_name", folder.name),
                        "created": created,
                        "total": meta.get("total_processed", len(data["results"])),
                        "sort": sort_value,
                    })
                except Exception:
                    pass
    return sorted(entries, key=lambda x: x["sort"], reverse=True)


def csv_content(results):
    buffer = StringIO()
    fields = ["id", "feedback", "summary", "classification", "recommendation"]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    return buffer.getvalue()


def save_run(results, failures, source_name):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source_name).stem).strip("_")[:35] or "feedback"
    run_id = f"{stamp}_{name}"
    folder = HISTORY_DIR / run_id
    folder.mkdir()
    counts = {}
    for row in results:
        category = row.get("classification", "Other")
        counts[category] = counts.get(category, 0) + 1
    data = {
        "metadata": {
            "run_id": run_id,
            "analysis_name": Path(source_name).stem or "Feedback analysis",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            "total_submitted": len(results) + len(failures),
            "total_processed": len(results),
            "total_failed": len(failures),
        },
        "analytics": {
            "total_feedback": len(results),
            "positive": counts.get("Positive", 0),
            "negative": counts.get("Negative", 0),
            "feature_requests": counts.get("Feature Request", 0),
            "other": counts.get("Other", 0),
        },
        "results": results,
        "failures": failures,
    }
    json_text = json.dumps(data, indent=2, ensure_ascii=False)
    (folder / "analysis_results.json").write_text(json_text, encoding="utf-8")
    (folder / "analysis_results.csv").write_text(csv_content(results), encoding="utf-8-sig")
    LATEST_JSON.write_text(json_text, encoding="utf-8")
    LATEST_CSV.write_text(csv_content(results), encoding="utf-8-sig")
    return run_id, data


def business_pdf(data):
    """Create the executive-ready PDF report for the currently selected run."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.7*cm, bottomMargin=1.7*cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="FI_Title", parent=styles["Title"], fontSize=23, leading=29, textColor=colors.HexColor("#172554"), alignment=1, spaceAfter=10))
    styles.add(ParagraphStyle(name="FI_Heading", parent=styles["Heading2"], textColor=colors.HexColor("#1D4ED8"), spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="FI_Body", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="FI_Small", parent=styles["BodyText"], fontSize=7.2, leading=9.4, wordWrap="CJK"))

    def safe(value):
        return escape(str(value if value is not None else ""))

    results = data.get("results", [])
    meta = data.get("metadata", {})
    counts = {}
    for row in results:
        cat = row.get("classification", "Other")
        counts[cat] = counts.get(cat, 0) + 1
    total = len(results)
    pos, neg = counts.get("Positive", 0), counts.get("Negative", 0)
    feat, other = counts.get("Feature Request", 0), counts.get("Other", 0)

    story = [
        Paragraph("FeedbackIQ", styles["FI_Title"]),
        Paragraph("Customer Feedback Intelligence Report", styles["Heading2"]),
        Paragraph(f"Analysis: {safe(meta.get('analysis_name', 'Feedback analysis'))}", styles["FI_Body"]),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles["FI_Body"]),
        Spacer(1, 10),
        Paragraph("1. Executive Summary", styles["FI_Heading"]),
    ]
    if total:
        summary = (
            f"This report covers {total} analyzed records: {pos} positive ({pos/total*100:.1f}%), "
            f"{neg} negative ({neg/total*100:.1f}%), {feat} feature requests ({feat/total*100:.1f}%), "
            f"and {other} other ({other/total*100:.1f}%). "
        )
        summary += ("Negative feedback exceeds positive feedback. Review complaints and validate their impact."
                    if neg > pos else
                    "Positive feedback exceeds negative feedback. Preserve valued product qualities while addressing issues."
                    if pos > neg else
                    "Positive and negative counts are equal. Review individual records before setting priorities.")
        summary += " Findings describe this dataset and may not represent the entire customer population."
    else:
        summary = "No analyzed feedback records are available."
    story.append(Paragraph(safe(summary), styles["FI_Body"]))
    story.append(Paragraph("2. Feedback Overview", styles["FI_Heading"]))
    overview = [
        ["Metric", "Count", "Percentage"],
        ["Total feedback", str(total), "100%" if total else "0%"],
        ["Positive", str(pos), f"{pos/total*100:.1f}%" if total else "0%"],
        ["Negative", str(neg), f"{neg/total*100:.1f}%" if total else "0%"],
        ["Feature requests", str(feat), f"{feat/total*100:.1f}%" if total else "0%"],
        ["Other", str(other), f"{other/total*100:.1f}%" if total else "0%"],
    ]
    table = Table(overview, colWidths=[8*cm, 3*cm, 4*cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#172554")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F1F5F9")]),
        ("PADDING",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(table)
    story.append(Paragraph("3. Customer Issues and Feature Requests", styles["FI_Heading"]))
    issues = [r for r in results if r.get("classification") in ("Negative","Feature Request")]
    if not issues:
        story.append(Paragraph("No negative feedback or feature requests were identified.", styles["FI_Body"]))
    for i, row in enumerate(issues, 1):
        story.append(Paragraph(f"<b>{i}. {safe(row.get('classification'))}</b>", styles["FI_Body"]))
        story.append(Paragraph(f"<b>Feedback:</b> {safe(row.get('feedback'))}", styles["FI_Body"]))
        story.append(Paragraph(f"<b>Summary:</b> {safe(row.get('summary'))}", styles["FI_Body"]))
    story.append(Paragraph("4. Recommended Business Actions", styles["FI_Heading"]))
    for i, row in enumerate(results, 1):
        story.append(Paragraph(f"<b>{i}. {safe(row.get('classification'))}</b>", styles["FI_Body"]))
        story.append(Paragraph(safe(row.get("recommendation")), styles["FI_Body"]))
    story.extend([PageBreak(), Paragraph("5. Detailed Feedback Register", styles["FI_Heading"])])
    rows = [["Category","Customer Feedback","AI Summary","Recommended Action"]]
    for row in results:
        rows.append([
            Paragraph(safe(row.get("classification")), styles["FI_Small"]),
            Paragraph(safe(row.get("feedback")), styles["FI_Small"]),
            Paragraph(safe(row.get("summary")), styles["FI_Small"]),
            Paragraph(safe(row.get("recommendation")), styles["FI_Small"]),
        ])
    detail = Table(rows, colWidths=[2.5*cm,4.3*cm,4.1*cm,5.0*cm], repeatRows=1)
    detail.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#172554")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("GRID",(0,0),(-1,-1),.4,colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F1F5F9")]),
        ("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),6),
    ]))
    story.append(detail)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawString(1.5*cm, 1.0*cm, "FeedbackIQ | AI-assisted customer feedback analysis")
        canvas.drawRightString(A4[0]-1.5*cm, 1.0*cm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------- Sidebar and history ----------------
history = []
if HISTORY_DIR.exists():
    for folder in HISTORY_DIR.iterdir():
        file = folder / "analysis_results.json"
        if folder.is_dir() and file.exists():
            try:
                d = read_data(file)
                meta = d["metadata"]
                created = meta.get("created_at", "")
                try:
                    sort_key = datetime.fromisoformat(created.replace("Z","+00:00")).timestamp()
                except Exception:
                    sort_key = folder.stat().st_mtime
                history.append({
                    "id": folder.name, "path": file,
                    "name": meta.get("analysis_name", folder.name),
                    "created": created,
                    "total": meta.get("total_processed", len(d["results"])),
                    "sort": sort_key,
                })
            except Exception:
                pass
history.sort(key=lambda x: x["sort"], reverse=True)

with st.sidebar:
    st.markdown("# 📊 FeedbackIQ")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")
    st.divider()
    if st.button("＋ Start New Analysis", type="primary", use_container_width=True):
        st.session_state["selected_run"] = "__new__"
        st.rerun()
    st.markdown("### Analysis History")
    if history:
        labels = {}
        for run in history:
            try:
                when = datetime.fromisoformat(run["created"].replace("Z","+00:00")).astimezone().strftime("%d %b %Y, %I:%M %p")
            except Exception:
                when = run["id"]
            labels[f"{run['name']} · {when} · {run['total']} items"] = run["id"]
        options = list(labels.keys())
        current = st.session_state.get("selected_run")
        if current == "__new__":
            # Keep Start New Analysis selected while still listing history entries.
            for label, run_id in labels.items():
                if st.button(label, key=f"history_button_{run_id}", use_container_width=True):
                    st.session_state["selected_run"] = run_id
                    st.rerun()
        else:
            index = next((i for i, label in enumerate(options) if labels[label] == current), 0)
            selected_label = st.radio(
                "Open saved analysis",
                options,
                index=index,
                key="history_choice",
                label_visibility="collapsed",
            )
            st.session_state["selected_run"] = labels[selected_label]
    else:
        st.caption("No saved analyses yet. Start a new analysis to create history.")
    st.divider()
    st.caption("Text is styled for high contrast.")
    if st.button("↻ Refresh history", use_container_width=True):
        st.rerun()

selected_run = st.session_state.get("selected_run")
selected_entry = next((r for r in history if r["id"] == selected_run), None)
if selected_run == "__new__":
    data = None
elif selected_entry:
    data = read_data(selected_entry["path"])
elif LATEST_JSON.exists():
    data = read_data(LATEST_JSON)
else:
    data = None

st.markdown("""
<div class="hero">
<h1>Customer Feedback Intelligence</h1>
<p>Turn customer opinions into actionable product insights.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- New analysis workflow ----------------
if selected_run == "__new__" or data is None:
    st.subheader("Start a New Analysis")
    st.write("Upload JSON or CSV. Each feedback item uses one Gemini request; requests run concurrently.")
    uploaded = st.file_uploader("Upload customer feedback", type=["json","csv"], key="upload_new")
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".json"):
                raw = json.load(uploaded)
                if isinstance(raw, dict):
                    raw = raw.get("feedback", [])
                if not isinstance(raw, list):
                    st.error("JSON must be a list or contain a 'feedback' list.")
                    st.stop()
                items = []
                for row in raw:
                    value = (row.get("feedback") or row.get("text") or row.get("comment") or row.get("message") or "") if isinstance(row, dict) else str(row)
                    if str(value).strip():
                        items.append(str(value).strip())
            else:
                csv_df = pd.read_csv(uploaded)
                lookup = {str(c).strip().lower(): c for c in csv_df.columns}
                column = next((lookup[k] for k in ("feedback","text","comment","message") if k in lookup), None)
                if column is None:
                    st.error("CSV needs a feedback, text, comment, or message column.")
                    st.stop()
                items = [str(v).strip() for v in csv_df[column].dropna().tolist() if str(v).strip()]

            st.success(f"Loaded {len(items)} feedback items.")
            st.dataframe(pd.DataFrame({"Feedback": items}), use_container_width=True, hide_index=True)
            limit = st.number_input("Maximum items to analyze", min_value=1, max_value=500, value=min(10,max(1,len(items))), step=1)
            workers = st.select_slider("Parallel Gemini requests", options=[1,2,3,4], value=3, help="2–3 is a sensible starting point for a free API tier.")
            if st.button("Analyze Feedback", type="primary", use_container_width=True, disabled=not items):
                from agents import analyze_feedback_batch
                chosen = items[:int(limit)]
                progress = st.progress(0)
                status = st.empty()

                def on_progress(done, total):
                    progress.progress(done / total if total else 1)
                    status.write(f"Completed {done} of {total} items...")

                with st.spinner("Analyzing feedback with Gemini..."):
                    results, failures = analyze_feedback_batch(chosen, max_workers=int(workers), progress_callback=on_progress)
                status.empty()
                if results:
                    run_id, saved = save_run(results, failures, uploaded.name)
                    st.session_state["selected_run"] = run_id
                    if failures:
                        st.warning(f"{len(failures)} item(s) failed; successful items were saved.")
                    st.success(f"Analysis saved: {len(results)} successful, {len(failures)} failed.")
                    st.rerun()
                else:
                    st.error("No items succeeded. Check your API key, model, quota, and terminal error messages.")
        except (json.JSONDecodeError, pd.errors.ParserError) as exc:
            st.error(f"Could not read upload: {exc}")
        except UnicodeDecodeError:
            st.error("The uploaded file must use a supported text encoding.")
        except Exception as exc:
            st.error(f"Could not prepare analysis: {exc}")


# ---------------- Selected analysis dashboard ----------------
else:
    data = normalize(data)
    rows = data.get("results", [])
    meta = data.get("metadata", {})
    if not rows:
        st.warning("This analysis has no successful records.")
    else:
        df = pd.DataFrame(rows)
        if "classification" not in df:
            df["classification"] = "Other"
        df["classification"] = df["classification"].fillna("Other")
        st.subheader(f"Analysis: {meta.get('analysis_name', 'Feedback analysis')}")
        st.caption(f"Created: {meta.get('created_at','Unknown')} · Model: {meta.get('model','Gemini')}")
        counts = df["classification"].value_counts()
        total = len(df)
        metrics = [
            ("TOTAL FEEDBACK", total, "#60a5fa"),
            ("POSITIVE", int(counts.get("Positive",0)), "#34d399"),
            ("NEGATIVE", int(counts.get("Negative",0)), "#fb7185"),
            ("FEATURE REQUESTS", int(counts.get("Feature Request",0)), "#a78bfa"),
        ]
        cols = st.columns(4)
        for col, (label, value, color) in zip(cols, metrics):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="color:{color}">{value}</div></div>', unsafe_allow_html=True)

        chart_df = pd.DataFrame([{"Category": c, "Count": int(counts.get(c,0))} for c in ("Positive","Negative","Feature Request","Other") if counts.get(c,0)>0])
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-heading">Feedback Distribution</div>', unsafe_allow_html=True)
            fig = px.pie(chart_df, names="Category", values="Count", hole=.65, color="Category", color_discrete_map=COLORS)
            fig.update_traces(textposition="inside", textinfo="percent")
            fig.add_annotation(text=f"<b>{total}</b><br>Records", x=.5, y=.5, showarrow=False, font=dict(size=20,color="#fff"))
            fig.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827", font=dict(color="#f3f4f6"), margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown('<div class="section-heading">Category Breakdown</div>', unsafe_allow_html=True)
            fig = px.bar(chart_df, x="Category", y="Count", color="Category", text="Count", color_discrete_map=COLORS)
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Number of records", paper_bgcolor="#111827", plot_bgcolor="#111827", font=dict(color="#f3f4f6"), yaxis=dict(gridcolor="#334155",rangemode="tozero"), xaxis=dict(gridcolor="#334155"), margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(fig, use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["Feedback Explorer","Business Actions","Failed Items"])
        with tab1:
            c1,c2 = st.columns([1,2])
            with c1:
                cats = ["All"] + sorted(df["classification"].unique().tolist())
                cat = st.selectbox("Filter category", cats, key=f"cat_{selected_run}")
            with c2:
                term = st.text_input("Search this analysis", placeholder="Search feedback, summary, or recommendation...", key=f"term_{selected_run}")
            filtered = df.copy()
            if cat != "All":
                filtered = filtered[filtered["classification"] == cat]
            if term.strip():
                needle = term.lower().strip()
                mask = filtered[["feedback","summary","recommendation"]].fillna("").astype(str).apply(lambda col: col.str.lower().str.contains(needle, regex=False)).any(axis=1)
                filtered = filtered[mask]
            st.caption(f"{len(filtered)} matching records")
            for i, row in enumerate(filtered.to_dict("records"),1):
                with st.container(border=True):
                    l,r = st.columns([5,1])
                    with l:
                        st.markdown(f"**Feedback #{row.get('id',i)}**")
                        st.write(row.get("feedback",""))
                    with r:
                        st.markdown(f"**{row.get('classification','Other')}**")
                    with st.expander("View AI analysis"):
                        st.markdown("**Summary**")
                        st.write(row.get("summary",""))
                        st.markdown("**Recommended action**")
                        st.write(row.get("recommendation",""))
            st.download_button("Download filtered CSV", data=csv_content(filtered.to_dict("records")).encode("utf-8-sig"), file_name="filtered_feedback.csv", mime="text/csv")
        with tab2:
            st.caption("These are AI-generated suggestions, not completed tasks.")
            action_cat = st.selectbox("Action category", ["All","Negative","Feature Request","Positive","Other"], key=f"actions_{selected_run}")
            action_rows = rows if action_cat == "All" else [r for r in rows if r.get("classification") == action_cat]
            for i,row in enumerate(action_rows,1):
                with st.container(border=True):
                    st.markdown(f"**{i}. {row.get('classification','Other')}**")
                    st.write(row.get("recommendation",""))
                    with st.expander("Original feedback and summary"):
                        st.write(row.get("feedback",""))
                        st.caption(row.get("summary",""))
        with tab3:
            failures = data.get("failures", [])
            if failures:
                st.warning(f"{len(failures)} item(s) failed.")
                st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
            else:
                st.success("No failed items in this run.")

        st.divider()
        st.subheader("Download this analysis")
        st.caption("These downloads belong to the selected history entry.")
        d1,d2,d3 = st.columns(3)
        json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        csv_bytes = csv_content(rows).encode("utf-8-sig")
        pdf_bytes = business_pdf(data)
        run_prefix = meta.get("run_id", "FeedbackIQ")
        with d1:
            st.download_button("Download Results JSON", data=json_bytes, file_name=f"{run_prefix}_results.json", mime="application/json", use_container_width=True)
        with d2:
            st.download_button("Download Results CSV", data=csv_bytes, file_name=f"{run_prefix}_results.csv", mime="text/csv", use_container_width=True)
        with d3:
            st.download_button("Download Business Report PDF", data=pdf_bytes, file_name=f"{run_prefix}_business_report.pdf", mime="application/pdf", type="primary", use_container_width=True)
        if meta.get("total_failed",0):
            st.caption(f"Submitted: {meta.get('total_submitted',total)} · Processed: {meta.get('total_processed',total)} · Failed: {meta.get('total_failed',0)}")

st.divider()

