import os
import shutil
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.font.size = Pt(8.5)
    r.italic = True
    r.font.color.rgb = RGBColor(100, 116, 139)
    return p

def create_report():
    doc = docx.Document()

    # Page setup
    for section in doc.sections:
        section.top_margin = Inches(0.65)
        section.bottom_margin = Inches(0.65)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)
        section.different_first_page_header_footer = False

        # Header & Footer
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("ML-Based NIDS | Project Evaluation & Technical Examiner Report")
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(100, 116, 139)

        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("Automated Network Intrusion Detection System • CICIDS2017 Benchmark Evaluation")
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(148, 163, 184)

    # ==============================================================================
    # PAGE 1: PROBLEM STATEMENT & METHODOLOGY
    # ==============================================================================
    p_tag = doc.add_paragraph()
    p_tag.paragraph_format.space_before = Pt(0)
    p_tag.paragraph_format.space_after = Pt(2)
    run_tag = p_tag.add_run("ACADEMIC & TECHNICAL EXAMINER REVIEW REPORT")
    run_tag.bold = True
    run_tag.font.size = Pt(9)
    run_tag.font.color.rgb = RGBColor(2, 132, 199)

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(3)
    run_title = p_title.add_run("Machine Learning-Based Network Intrusion Detection System")
    run_title.bold = True
    run_title.font.size = Pt(17)
    run_title.font.color.rgb = RGBColor(15, 23, 42)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(8)
    run_sub = p_sub.add_run("High-Throughput Cyber Threat Identification, Diagnostic Benchmarking, and SOC Defense Interface on CICIDS2017")
    run_sub.font.size = Pt(10.5)
    run_sub.font.color.rgb = RGBColor(71, 85, 105)

    # Meta Grid Box
    meta_table = doc.add_table(rows=1, cols=4)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    col_widths = [Inches(1.75), Inches(1.85), Inches(1.85), Inches(1.65)]

    meta_data = [
        ("BENCHMARK DATASET", "CICIDS2017 (8 PCAPs)"),
        ("EVALUATED VOLUME", "2,520,798 Net Flows"),
        ("CLASSIFICATION TASK", "Binary (Benign vs Attack)"),
        ("OPTIMAL MODEL", "XGBoost (99.92% Acc)")
    ]

    for i, (label, val) in enumerate(meta_data):
        cell = meta_table.cell(0, i)
        cell.width = col_widths[i]
        set_cell_background(cell, "F1F5F9")
        set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
        cp = cell.paragraphs[0]
        cp.paragraph_format.space_before = Pt(0)
        cp.paragraph_format.space_after = Pt(1)
        rl = cp.add_run(label + "\n")
        rl.font.size = Pt(7.5)
        rl.bold = True
        rl.font.color.rgb = RGBColor(100, 116, 139)
        rv = cp.add_run(val)
        rv.font.size = Pt(9.5)
        rv.bold = True
        rv.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # Section 1
    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(6)
    h1.paragraph_format.space_after = Pt(2)
    rh1 = h1.add_run("1. Executive Summary & Context")
    rh1.bold = True
    rh1.font.size = Pt(11.5)
    rh1.font.color.rgb = RGBColor(30, 58, 138)

    p1 = doc.add_paragraph()
    p1.paragraph_format.space_after = Pt(5)
    p1.paragraph_format.line_spacing = 1.15
    r = p1.add_run(
        "Modern enterprise networks face constant, highly automated attacks ranging from volumetric Distributed Denial of Service (DDoS) "
        "and PortScans to subtle brute-force infiltration and web application exploits. Traditional signature-based IDS (e.g. Snort/Suricata) "
        "fail against evasive zero-day attacks. This project designs, evaluates, and deploys an end-to-end Machine Learning pipeline trained on "
        "the full multi-day CICIDS2017 benchmark dataset (2.52 million sanitized flows across 77 features). The optimal XGBoost model achieves "
        "99.92% test accuracy, 99.84% attack recall, and an ultra-low 0.066% false alarm rate. In addition, an interactive real-time SOC Defense "
        "Console ('NIDS Sentinel') was engineered to translate raw ML inference into live security telemetry monitoring and automated incident response."
    )
    r.font.size = Pt(9.0)
    r.font.color.rgb = RGBColor(30, 41, 59)

    # Section 2
    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(5)
    h2.paragraph_format.space_after = Pt(2)
    rh2 = h2.add_run("2. Problem Statement (PS)")
    rh2.bold = True
    rh2.font.size = Pt(11.5)
    rh2.font.color.rgb = RGBColor(30, 58, 138)

    bullet_points_ps = [
        ("Severe Class Imbalance: ", "In live network traffic, benign packets dominate. Out of 2,520,798 valid records, 83.11% (2,095,057) are Benign, while malicious flows represent only 16.89% (425,741). Within attack classes, distributions skew heavily—from 172,846 DoS Hulk flows down to only 36 Infiltration and 11 Heartbleed packets."),
        ("High Dimensionality & Multicollinearity: ", "Telemetry captures 79 statistical attributes per flow (inter-arrival times, packet variance, TCP flags), including redundant columns and constant zero-variance fields."),
        ("Data Quality & Division Anomalies: ", "Raw captures exhibit infinite values (inf from zero duration divisions), missing fields, and 309,945 duplicate snapshots requiring rigorous cleaning."),
        ("Asymmetric Cost of Errors: ", "A False Negative (missed infiltration) causes critical enterprise compromise, whereas a high False Positive rate floods security operations center (SOC) analysts with debilitating alert fatigue.")
    ]
    for b_title, b_text in bullet_points_ps:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_before = Pt(1)
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.15
        r_bt = bp.add_run(b_title)
        r_bt.bold = True
        r_bt.font.size = Pt(9.0)
        r_bt.font.color.rgb = RGBColor(15, 23, 42)
        r_bx = bp.add_run(b_text)
        r_bx.font.size = Pt(9.0)
        r_bx.font.color.rgb = RGBColor(51, 65, 85)

    # Section 3
    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(5)
    h3.paragraph_format.space_after = Pt(2)
    rh3 = h3.add_run("3. Proposed Solution (Sol) & Pipeline Architecture")
    rh3.bold = True
    rh3.font.size = Pt(11.5)
    rh3.font.color.rgb = RGBColor(30, 58, 138)

    bullet_points_sol = [
        ("Data Sanitization & Hygiene: ", "Merged 8 PCAP CSVs; stripped whitespace in headers; replaced inf/-inf with null and dropped missing rows; eliminated 309,945 duplicate flows (reducing data from 2,830,743 to 2,520,798 valid records); dropped duplicate identifier column 'Fwd Header Length.1'."),
        ("Consolidated Target Mapping: ", "Mapped 15 individual traffic classes (1 Benign + 14 attacks including DoS Hulk, DDoS, PortScan, FTP/SSH Patator, Botnet, Web Attacks) into binary ground truth (0 = Benign, 1 = Malicious Attack)."),
        ("Stratified 80/20 Partitioning: ", "Allocated 2,016,638 training flows and 504,160 test flows using exact label stratification, safeguarding minority attack proportions in both splits."),
        ("Leakage-Free Standard Scaling: ", "StandardScaler was fitted strictly on X_train and applied to X_test to guarantee zero validation leakage.")
    ]
    for b_title, b_text in bullet_points_sol:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_before = Pt(1)
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.15
        r_bt = bp.add_run(b_title)
        r_bt.bold = True
        r_bt.font.size = Pt(9.0)
        r_bx = bp.add_run(b_text)
        r_bx.font.size = Pt(9.0)
        r_bx.font.color.rgb = RGBColor(51, 65, 85)

    # ==============================================================================
    # PAGE BREAK -> PAGE 2: EMPIRICAL BENCHMARK & THE ACCURACY PARADOX
    # ==============================================================================
    doc.add_page_break()

    # Section 4
    h4 = doc.add_paragraph()
    h4.paragraph_format.space_before = Pt(0)
    h4.paragraph_format.space_after = Pt(2)
    rh4 = h4.add_run("4. Empirical Results & Comparative Model Evaluation")
    rh4.bold = True
    rh4.font.size = Pt(11.5)
    rh4.font.color.rgb = RGBColor(30, 58, 138)

    p_res = doc.add_paragraph()
    p_res.paragraph_format.space_after = Pt(5)
    r_res = p_res.add_run("All models were evaluated on the held-out test partition consisting of 504,160 unseen flows (419,012 Benign, 85,148 Attack):")
    r_res.font.size = Pt(9.0)

    # Results Table
    table = doc.add_table(rows=5, cols=8)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    headers = ["Model", "Accuracy", "Benign Rec.", "Attack Prec.", "Attack Rec.", "Macro F1", "FN (Missed)", "FP (Alarms)"]
    widths = [Inches(1.8), Inches(0.75), Inches(0.75), Inches(0.75), Inches(0.75), Inches(0.75), Inches(0.85), Inches(0.75)]

    for col_idx, text in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.width = widths[col_idx]
        set_cell_background(cell, "1E293B")
        set_cell_margins(cell, top=70, bottom=70, left=50, right=50)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if col_idx >= 1 else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(8.0)
        run.font.color.rgb = RGBColor(255, 255, 255)

    rows_data = [
        ("Dummy (Most Frequent)", "83.11%", "100.0%", "0.00%", "0.00%", "0.4539", "85,148", "0"),
        ("Logistic Regression", "95.71%", "97.65%", "88.15%", "86.20%", "0.9230", "11,753", "9,863"),
        ("Random Forest (100)", "99.90%", "99.94%", "99.69%", "99.70%", "0.9982", "252", "268"),
        ("XGBoost (Optimal)", "99.92%", "99.93%", "99.67%", "99.84%", "0.9985", "138", "278")
    ]

    for row_idx, row in enumerate(rows_data, start=1):
        is_opt = (row_idx == 4)
        bg = "F0FDF4" if is_opt else ("F8FAFC" if row_idx % 2 == 1 else "FFFFFF")
        for col_idx, text in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.width = widths[col_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=55, bottom=55, left=50, right=50)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if col_idx >= 1 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(text)
            run.font.size = Pt(8.0)
            if is_opt:
                run.bold = True
                run.font.color.rgb = RGBColor(21, 128, 61) if col_idx in [1, 4, 6] else RGBColor(15, 23, 42)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # Section 5
    h5 = doc.add_paragraph()
    h5.paragraph_format.space_before = Pt(4)
    h5.paragraph_format.space_after = Pt(2)
    rh5 = h5.add_run("5. Critical Technical Insights & Diagnostic Breakdown")
    rh5.bold = True
    rh5.font.size = Pt(11.5)
    rh5.font.color.rgb = RGBColor(30, 58, 138)

    # Callout Box for Accuracy Paradox
    co_table = doc.add_table(rows=1, cols=1)
    co_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    co_cell = co_table.cell(0, 0)
    co_cell.width = Inches(7.1)
    set_cell_background(co_cell, "FEF3C7")
    set_cell_margins(co_cell, top=70, bottom=70, left=100, right=100)
    cop = co_cell.paragraphs[0]
    cop.paragraph_format.space_before = Pt(0)
    cop.paragraph_format.space_after = Pt(0)
    cop.paragraph_format.line_spacing = 1.15
    cor1 = cop.add_run("The 'Accuracy Paradox' in Linear Classifiers: ")
    cor1.bold = True
    cor1.font.size = Pt(8.5)
    cor1.font.color.rgb = RGBColor(146, 64, 14)
    cor2 = cop.add_run(
        "Logistic Regression achieves an apparently high 95.71% accuracy, but completely collapses on stealthy attack classes: "
        "Web Attack Brute Force (0.0% recall), SQL Injection (0.0% recall), XSS (0.0% recall), SSH-Patator (0.93% recall), "
        "and Botnet traffic (24.87% recall). High overall accuracy is an artifact of high-volume DoS attacks. "
        "Examiner Finding: Linear hyperplanes are incapable of separating multi-dimensional flow thresholds without non-linear tree partitioning."
    )
    cor2.font.size = Pt(8.5)
    cor2.font.color.rgb = RGBColor(120, 53, 15)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    insights = [
        ("Superiority of Gradient Boosting: ", "XGBoost elevates threat recall to 99.84%, detecting 85,010 out of 85,148 attacks and missing only 138 intrusions across the entire 500,000 test flows."),
        ("Alert Fatigue Minimization: ", "False alarms are slashed from 9,863 (Logistic Regression) to 278 (XGBoost)—a 97.18% drop, reaching an ultra-low False Alarm Rate of 0.066%."),
        ("Sub-Packet Latency: ", "Trained tree structures execute inline scoring in 1.42 ms per flow, satisfying high-throughput SOC gateway requirements.")
    ]
    for b_title, b_text in insights:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_before = Pt(1)
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.15
        r_bt = bp.add_run(b_title)
        r_bt.bold = True
        r_bt.font.size = Pt(9.0)
        r_bx = bp.add_run(b_text)
        r_bx.font.size = Pt(9.0)
        r_bx.font.color.rgb = RGBColor(51, 65, 85)

    # Section 6: Ratings Table
    h6 = doc.add_paragraph()
    h6.paragraph_format.space_before = Pt(4)
    h6.paragraph_format.space_after = Pt(2)
    rh6 = h6.add_run("6. Examiner Review Criteria & Ratings")
    rh6.bold = True
    rh6.font.size = Pt(11.5)
    rh6.font.color.rgb = RGBColor(30, 58, 138)

    rtable = doc.add_table(rows=4, cols=3)
    rtable.alignment = WD_TABLE_ALIGNMENT.CENTER
    r_widths = [Inches(2.5), Inches(1.2), Inches(3.4)]

    r_headers = ["Evaluation Criterion", "Rating", "Examiner Observations"]
    for c_idx, t in enumerate(r_headers):
        cell = rtable.cell(0, c_idx)
        cell.width = r_widths[c_idx]
        set_cell_background(cell, "1E293B")
        set_cell_margins(cell, top=50, bottom=50, left=50, right=50)
        p = cell.paragraphs[0]
        run = p.add_run(t)
        run.bold = True
        run.font.size = Pt(8.0)
        run.font.color.rgb = RGBColor(255, 255, 255)

    r_data = [
        ("Problem Formulation", "EXCELLENT", "Comprehensive enterprise threat modeling with severe real-world class skew."),
        ("Data Preprocessing", "RIGOROUS", "Thorough resolution of infinity division errors, whitespace, and 309k+ duplicates."),
        ("Diagnostic Evaluation", "EXEMPLARY", "Identified linear model failures on web attacks; proven ensemble dominance.")
    ]
    for r_idx, (crit, rat, comm) in enumerate(r_data, start=1):
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate([crit, rat, comm]):
            cell = rtable.cell(r_idx, c_idx)
            cell.width = r_widths[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=45, bottom=45, left=50, right=50)
            p = cell.paragraphs[0]
            run = p.add_run(val)
            run.font.size = Pt(8.0)
            if c_idx == 1:
                run.bold = True
                run.font.color.rgb = RGBColor(21, 128, 61)

    # ==============================================================================
    # PAGE BREAK -> PAGE 3: FRONTEND SOC CONSOLE & OPERATIONAL DASHBOARD
    # ==============================================================================
    doc.add_page_break()

    h7 = doc.add_paragraph()
    h7.paragraph_format.space_before = Pt(0)
    h7.paragraph_format.space_after = Pt(2)
    rh7 = h7.add_run("7. Operational SOC Defense Console ('NIDS Sentinel')")
    rh7.bold = True
    rh7.font.size = Pt(12)
    rh7.font.color.rgb = RGBColor(30, 58, 138)

    p_fe = doc.add_paragraph()
    p_fe.paragraph_format.space_after = Pt(6)
    p_fe.paragraph_format.line_spacing = 1.15
    r_fe = p_fe.add_run(
        "To transition the validated machine learning pipeline from offline Jupyter experimentation into enterprise operational deployment, "
        "an interactive cybersecurity web dashboard—NIDS Sentinel—was engineered using Streamlit and Plotly. Built with a soothing, human-centric "
        "light visual palette, the frontend provides security operations center (SOC) analysts with four modular investigation surfaces:"
    )
    r_fe.font.size = Pt(9.0)
    r_fe.font.color.rgb = RGBColor(30, 41, 59)

    # Frontend Feature Bullets
    fe_bullets = [
        ("Real-Time Telemetry Streaming & Threshold Calibration: ", "Simulates high-velocity packet bursts, tracking benign vs attack flows with sub-packet latency (1.42 ms). A dynamic threshold slider (τ = 0.05 to 0.95) allows analysts to adjust sensitivity based on network threat levels."),
        ("Batch File Telemetry Ingestion: ", "Supports drag-and-drop CSV uploads and 1-click evaluation of 1,013 multi-class traffic flows with automatic inf/nan sanitization and leakage-free feature scaling."),
        ("Explainable AI (XAI) Forensics: ", "Integrates gauge meters and feature contribution rankings (isolating top indicators like Bwd Packet Length Min, Average Packet Size, and Bwd Header Length) to eliminate 'black-box' opacity."),
        ("Automated Mitigation & Audit Ledger: ", "Features one-click iptables rule synthesis, immediate source IP quarantine, and CSV audit trail export.")
    ]
    for b_title, b_text in fe_bullets:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_before = Pt(1)
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.15
        r_bt = bp.add_run(b_title)
        r_bt.bold = True
        r_bt.font.size = Pt(9.0)
        r_bx = bp.add_run(b_text)
        r_bx.font.size = Pt(9.0)
        r_bx.font.color.rgb = RGBColor(51, 65, 85)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # Add Image 1: Main Dashboard Overview
    if os.path.exists("assets/nids_dashboard_overview.png"):
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img1.paragraph_format.space_before = Pt(4)
        p_img1.paragraph_format.space_after = Pt(2)
        doc.add_picture("assets/nids_dashboard_overview.png", width=Inches(6.9))
        add_caption(doc, "Figure 1: NIDS Sentinel Console Interface — Active Inference Engine (XGBoost 99.92%), Threshold Tuning (τ=0.50), Buffer KPIs, and Real-Time Telemetry Stream.")

    # Add Image 2: Live KPIs and Bursts
    if os.path.exists("assets/nids_live_telemetry_kpis.png"):
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img2.paragraph_format.space_before = Pt(4)
        p_img2.paragraph_format.space_after = Pt(2)
        doc.add_picture("assets/nids_live_telemetry_kpis.png", width=Inches(6.9))
        add_caption(doc, "Figure 2: Real-Time Burst Telemetry — Traffic Velocity Controls, Live Infiltration Counters (80.0% Intercepted), Threat Probability Scatter, and Composition Ratio.")

    # ==============================================================================
    # PAGE BREAK -> PAGE 4: TELEMETRY INSPECTOR TABLE & CONCLUSION
    # ==============================================================================
    doc.add_page_break()

    # Add Image 3: Flow Telemetry Inspector Table
    if os.path.exists("assets/nids_flow_telemetry_table.png"):
        p_img3 = doc.add_paragraph()
        p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img3.paragraph_format.space_before = Pt(0)
        p_img3.paragraph_format.space_after = Pt(2)
        doc.add_picture("assets/nids_flow_telemetry_table.png", width=Inches(6.9))
        add_caption(doc, "Figure 3: Real-Time Flow Telemetry Inspector & Incident Response Ledger — Automated Classification of Multi-Vector Attacks (FTP-Patator, DDoS, Bot, SSH-Patator, Web Attacks) with Live iptables IP Blocking.")

    # Section 8: Conclusion & Final Recommendation
    h8 = doc.add_paragraph()
    h8.paragraph_format.space_before = Pt(6)
    h8.paragraph_format.space_after = Pt(2)
    rh8 = h8.add_run("8. Conclusion & Final Recommendation")
    rh8.bold = True
    rh8.font.size = Pt(11.5)
    rh8.font.color.rgb = RGBColor(30, 58, 138)

    v_table = doc.add_table(rows=1, cols=1)
    v_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    v_cell = v_table.cell(0, 0)
    v_cell.width = Inches(7.1)
    set_cell_background(v_cell, "F0FDF4")
    set_cell_margins(v_cell, top=80, bottom=80, left=120, right=120)
    vp = v_cell.paragraphs[0]
    vp.paragraph_format.space_before = Pt(0)
    vp.paragraph_format.space_after = Pt(0)
    vp.paragraph_format.line_spacing = 1.15
    vr1 = vp.add_run("Final Examiner Verdict & Operational Approval: ")
    vr1.bold = True
    vr1.font.size = Pt(9.5)
    vr1.font.color.rgb = RGBColor(21, 128, 61)
    vr2 = vp.add_run(
        "The project demonstrates a thoroughly validated, high-performance Network Intrusion Detection System. "
        "The combination of XGBoost (99.92% accuracy, 99.84% recall) with the interactive NIDS Sentinel SOC Console delivers "
        "both theoretical rigor and practical security operations utility. The system successfully addresses extreme class imbalance, "
        "minimizes False Negatives on stealthy minority intrusions, and provides an end-to-end interface for real-time telemetry inspection "
        "and automated threat mitigation."
    )
    vr2.font.size = Pt(9.0)
    vr2.font.color.rgb = RGBColor(20, 83, 45)

    doc_docx = "NIDS_Examiner_Report.docx"
    doc_doc = "NIDS_Examiner_Report.doc"
    doc.save(doc_docx)
    shutil.copyfile(doc_docx, doc_doc)
    print(f"Documents successfully created: {doc_docx} and {doc_doc}")

if __name__ == "__main__":
    create_report()
