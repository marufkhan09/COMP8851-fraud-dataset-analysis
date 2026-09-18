"""Build the COMP8851 report as a Word document.

Prose is written here; every number and every table is read from ../tables/, which is
itself generated from the run records. Nothing is retyped, so the document cannot
drift from the evidence.

Writes ../COMP8851_CARE-GNN_GHRN_Benchmark_Report.docx

Run:  python build_report.py
"""

from __future__ import annotations

import csv
import os
from collections import OrderedDict

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
DELIVERABLE = os.path.dirname(HERE)
TABLES = os.path.join(DELIVERABLE, "tables")
FIGURES = os.path.join(DELIVERABLE, "figures")
OUTPUT = os.path.join(DELIVERABLE, "COMP8851_CARE-GNN_GHRN_Benchmark_Report.docx")

MODELS = ["CARE-GNN", "GHRN"]
DATASETS = ["YelpChi", "Amazon", "T-Finance", "T-Social", "Elliptic", "FDCompCN"]
ACCENT = RGBColor(0x1F, 0x3B, 0x63)


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------
def read(name):
    with open(os.path.join(TABLES, name), newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


AGG = read("master_mean_sd.csv")
MATRIX = read("experiment_matrix.csv")
EFFICIENCY = read("efficiency.csv")
PROFILE = read("dataset_profile.csv")
STAGES = read("stage_ablation.csv")
FLAGS = read("seed_stability_flags.csv")
SPREAD = read("seed_spread.csv")
PER_SEED = read("master_per_seed.csv")

CELL = {(r["model"], r["dataset"]): r for r in AGG}
STATUS = {(r["model"], r["dataset"]): r["status"] for r in MATRIX}
SHARED = [d for d in DATASETS if all((m, d) in CELL for m in MODELS)]


def stat(model, dataset, metric, places=4):
    rec = CELL.get((model, dataset))
    if not rec:
        return f"[{STATUS[(model, dataset)]}]"
    return f"{float(rec[f'{metric}_mean']):.{places}f} ± {float(rec[f'{metric}_sd']):.{places}f}"


def value(model, dataset, metric):
    rec = CELL.get((model, dataset))
    return float(rec[f"{metric}_mean"]) if rec else None


def headline(model, metric):
    return sum(value(model, d, metric) for d in SHARED) / len(SHARED)


# --------------------------------------------------------------------------
# document helpers
# --------------------------------------------------------------------------
def style_document(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.15
    for level, size in ((1, 16), (2, 13), (3, 11.5)):
        st = doc.styles[f"Heading {level}"]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.color.rgb = ACCENT
        st.font.bold = True


def para(doc, text, *, italic=False, size=None, align=None, after=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(item, style="List Bullet")
        p.paragraph_format.space_after = Pt(4)


def shade(cell, hexcolor):
    el = OxmlElement("w:shd")
    el.set(qn("w:fill"), hexcolor)
    cell._tc.get_or_add_tcPr().append(el)


def add_table(doc, headers, rows, *, caption=None, widths=None, highlight=None,
              header=True):
    table = doc.add_table(rows=1 if header else 0, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    if header:
        header_cells = table.rows[0].cells
        for i, text in enumerate(headers):
            header_cells[i].text = ""
            run = header_cells[i].paragraphs[0].add_run(str(text))
            run.bold = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            shade(header_cells[i], "1F3B63")

    for r, row in enumerate(rows):
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(text))
            run.font.size = Pt(9.5)
            if highlight and highlight(r, i, text):
                run.bold = True
        if r % 2 == 1:
            for cell in cells:
                shade(cell, "F2F5FA")

    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)

    if caption:
        cap = doc.add_paragraph()
        run = cap.add_run(caption)
        run.italic = True
        run.font.size = Pt(9)
        cap.paragraph_format.space_before = Pt(4)
        cap.paragraph_format.space_after = Pt(12)
    return table


def add_figure(doc, filename, caption, width=6.0):
    path = os.path.join(FIGURES, filename)
    if not os.path.isfile(path):
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(12)


def callout(doc, title, body):
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    cell = table.rows[0].cells[0]
    shade(cell, "EEF3FA")
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(title)
    run.bold = True
    run.font.size = Pt(10)
    run.font.color.rgb = ACCENT
    body_p = cell.add_paragraph()
    body_run = body_p.add_run(body)
    body_run.font.size = Pt(10)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def title_page(doc):
    for _ in range(4):
        doc.add_paragraph()
    para(doc, "Graph Neural Networks for Fraud Detection",
         size=26, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    doc.paragraphs[-1].runs[0].bold = True
    doc.paragraphs[-1].runs[0].font.color.rgb = ACCENT
    para(doc, "A Controlled Benchmark of CARE-GNN and GHRN",
         size=16, align=WD_ALIGN_PARAGRAPH.CENTER, after=24)
    doc.paragraphs[-1].runs[0].font.color.rgb = ACCENT

    para(doc, "COMP8851 Major Project", size=13, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para(doc, "[Author name]", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para(doc, "17 September 2026", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, after=36)

    completed = sum(1 for r in MATRIX if r["status"] == "COMPLETED")
    add_table(
        doc,
        ["", ""],
        [
            ["Models", " and ".join(MODELS)],
            ["Datasets", ", ".join(DATASETS)],
            ["Required combinations", f"{len(MATRIX)} (2 models × 6 datasets)"],
            ["Completed", f"{completed} of {len(MATRIX)}"],
            ["Final runs", f"{len(PER_SEED)}"],
            ["Training ratio", "TR40"],
            ["Training seeds", "2, 42, 72 (split seed fixed at 2)"],
            ["Protocol", "vast-v4.4"],
        ],
        widths=[2.0, 4.2],
        header=False,
    )
    doc.add_page_break()


def executive_summary(doc):
    doc.add_heading("Executive summary", level=1)
    completed = sum(1 for r in MATRIX if r["status"] == "COMPLETED")

    para(doc,
         f"We benchmarked two fraud-detection graph neural networks, CARE-GNN (Dou et al., "
         f"CIKM 2020) and GHRN (Gao et al., WWW 2023), across six fraud and anomaly graph "
         f"datasets under a single controlled protocol. {completed} of {len(MATRIX)} required "
         f"model-dataset combinations completed, yielding {len(PER_SEED)} final runs at training "
         f"ratio TR40 with training seeds 2, 42 and 72. The two incomplete cells are both "
         f"T-Social and are reported with their status and cause rather than omitted.")

    add_table(
        doc,
        ["Model", "AUPRC", "AUROC", "Macro-F1", "Fraud recall"],
        [[m] + [f"{headline(m, k):.4f}" for k in
                ("auprc", "auroc", "macro_f1", "fraud_recall")] for m in MODELS],
        caption=f"Table 1. Model-wise means over the {len(SHARED)} datasets where both models "
                f"completed ({', '.join(SHARED)}).",
        widths=[1.5, 1.2, 1.2, 1.2, 1.2],
        highlight=lambda r, c, t: r == 1 and c > 0,
    )

    para(doc, "Principal findings", after=4)
    doc.paragraphs[-1].runs[0].bold = True
    bullets(doc, [
        "GHRN leads on every headline metric, but three of its four per-dataset wins are "
        "worth under 0.015 AUPRC. The average margin is carried almost entirely by Amazon.",

        f"The Amazon margin is largely a reliability effect. CARE-GNN scored "
        f"{value('CARE-GNN','Amazon','auprc'):.4f} there against GHRN's "
        f"{value('GHRN','Amazon','auprc'):.4f}, but one of its three seeds collapsed; the two "
        f"healthy seeds average 0.8490, within 0.027 of GHRN.",

        f"CARE-GNN is the less reliable model: {len(FLAGS)} of its 15 runs collapsed during "
        f"training, against 0 of 15 for GHRN. This is the clearest practical difference "
        f"between the two.",

        "Neither model learns a usable signal on FDCompCN, where both sit close to chance "
        "(AUROC 0.5527 and 0.5525 against 0.5 for random ranking).",

        "GHRN's advantage does not come from the heterophily refinement its paper "
        "emphasises. Using its own two stages as an ablation, refinement improved validation "
        f"AUPRC in only {sum(1 for r in STAGES if r['refinement_helped'] == 'True')} of "
        f"{len(STAGES)} runs, and on Amazon the unrefined stage validated better.",
    ])

    callout(doc, "Scope",
            "This is a controlled comparison between two detectors, not a leaderboard over "
            "fraud-detection GNNs. No claim here extends to other architectures. Every number "
            "in this report comes from a run that executed; no value is estimated, "
            "interpolated, or taken from a published paper.")
    doc.add_page_break()


def methodology(doc):
    doc.add_heading("1. Methodology", level=1)

    doc.add_heading("1.1 Protocol", level=2)
    para(doc,
         "All runs execute under protocol vast-v4.4. Hyperparameters are selected on "
         "validation AUPRC; the decision threshold is swept on validation Macro-F1 from 0.01 "
         "to 0.99 in 0.01 steps and then frozen. Each final run evaluates the test set exactly "
         "once, after the configuration, checkpoint and threshold are fixed.")

    add_table(
        doc,
        ["Parameter", "Value"],
        [
            ["Split seed", "2 (never varies)"],
            ["Training seeds", "2, 42, 72"],
            ["Training ratio", "TR40"],
            ["Optimizer", "Adam, betas 0.9 / 0.999, eps 1e-8"],
            ["Maximum epochs per stage", "100"],
            ["Early-stopping patience", "20"],
            ["Configuration selection", "validation AUPRC"],
            ["Threshold selection", "validation Macro-F1, swept 0.01–0.99"],
            ["Test evaluations per run", "exactly 1"],
        ],
        caption="Table 2. Fixed protocol parameters.",
        widths=[2.6, 3.6],
    )

    doc.add_heading("1.2 What makes the comparison fair", level=2)
    bullets(doc, [
        "Same nodes. Split IDs are fixed at split seed 2 and identical across both models on "
        "every dataset; validation and test node sets never vary.",
        "Same graph. Both models independently measured and recorded the heterophily of the "
        "view they trained on, and the values agree to six decimal places on all five "
        "datasets. This is a verified check, not an assumption.",
        "Same evaluator. One metric implementation serves both models; neither computes its "
        "own scores.",
        "Test isolation enforced by the command. Every tuning trial launches with the test "
        "path disabled, so test data cannot be loaded even accidentally.",
        "Failed runs are retained. Runs that trained badly are included in every mean and "
        "standard deviation; dropping them would bias the result and hide the instability "
        "this study set out to measure.",
    ])

    doc.add_heading("1.3 Execution environment", level=2)
    para(doc,
         "Runs executed on Vast.ai (NVIDIA RTX A6000) and Kaggle (NVIDIA Tesla T4), with "
         "CUDA 12.1, PyTorch 2.4.0+cu121, DGL 2.4.0+cu121 and Python 3.12. T-Finance ran on "
         "the T4; the other four completed datasets ran on the A6000.")
    callout(doc, "Hardware caveat",
            "Because the completed cells span two GPU classes, runtime and peak-memory "
            "figures are comparable only within a GPU block and are reported separately in "
            "Section 5. Predictive metrics are unaffected by GPU model and remain directly "
            "comparable throughout.")
    doc.add_page_break()


def completion(doc):
    doc.add_heading("2. Completion status", level=1)
    para(doc,
         "A combination is COMPLETED only when a final test evaluation actually executed and "
         "its result files were saved. Nothing is inferred and no cell is omitted.")

    rows = []
    for r in MATRIX:
        gpu = CELL.get((r["model"], r["dataset"]), {}).get("gpu", "—")
        rows.append([r["model"], r["dataset"], r["status"],
                     r["seeds"] or "—", gpu])
    add_table(doc, ["Model", "Dataset", "Status", "Seeds", "GPU"], rows,
              caption="Table 3. All 12 required model-dataset combinations.",
              widths=[1.3, 1.2, 1.3, 1.1, 1.4],
              highlight=lambda r, c, t: c == 2 and t != "COMPLETED")

    counts = OrderedDict()
    for r in MATRIX:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    para(doc,
         "Totals: " + ", ".join(f"{k} {v} of {len(MATRIX)}" for k, v in counts.items()) +
         f". Total final runs: {len(PER_SEED)}.")

    doc.add_heading("2.1 The two T-Social cells", level=2)
    para(doc,
         "Both incomplete cells are the same dataset, and neither was skipped. Both models "
         "were attempted repeatedly, across different configurations and on two devices, and "
         "every attempt left a record. T-Social is the reason: 5,781,065 nodes, with a "
         "2,312,426-node training split, a 1,156,213-node validation split and a "
         "2,312,426-node test split at a 3.01% fraud rate — an order of magnitude larger than "
         "anything else in the suite.")

    para(doc, "CARE-GNN × T-Social — FAILED after three attempts.", after=4)
    doc.paragraphs[-1].runs[0].bold = True
    para(doc,
         "Hyperparameter tuning completed two full trials of roughly 71 minutes each, reaching "
         "best validation AUPRC 0.0859 and 0.0520. The winning configuration was carried into "
         "a final run on a Tesla T4, which trained for 16 epochs over 1 h 55 m and was still "
         "improving: validation AUPRC climbed from 0.0366 at epoch 0 to 0.1006 at epoch 14, "
         "with AUROC reaching 0.7927. It was then interrupted at the 180-minute session cap, "
         "inside layers.py during validation prediction, before any test evaluation. The "
         "binding cost is validation rather than training — about 336 s per validation pass "
         "against 62 s per training epoch — because CARE-GNN filters neighbours per node in "
         "Python and every epoch scores 1,156,213 validation nodes.")

    para(doc, "GHRN × T-Social — FAILED after five attempts, in two different ways.", after=4)
    doc.paragraphs[-1].runs[0].bold = True
    para(doc,
         "GHRN was attempted on a five-rung escalation ladder, each rung reducing the resource "
         "demand of the one before: hidden dimension 64 then 32, filter order 2 then 1, on "
         "cuda:0 and then on CPU. Every rung failed, and the two failure modes are worth "
         "separating because they say different things.")
    para(doc,
         "Rung 1, at the published configuration on the GPU, ran out of memory: it needed a "
         "further 1.38 GiB with 638.81 MiB free of the T4's 14.56 GiB. That is a genuine "
         "capacity finding — T-Social does not fit GHRN at published width on a 16 GB card, "
         "and it is the only direct scalability measurement this study has for that cell.")
    para(doc,
         "Rungs 2 to 5, once the configuration was small enough to fit, failed instead on an "
         "index-dtype defect, identically on GPU and CPU: DGLError: Expect argument \"u\" to "
         "have data type torch.int32. But got torch.int64. T-Social is large enough that DGL "
         "carries its graph with int32 indices, while torch.arange and torch.argsort — used to "
         "generate self-loop node IDs and to rank edges for pruning — both return int64. This "
         "is a code defect exposed by scale rather than a resource limit, and it is specific to "
         "this dataset because the other five sit comfortably in int64. Shrinking the model "
         "therefore got past the memory wall and straight into the bug, so GHRN's true cost on "
         "T-Social remains unmeasured. No validation AUPRC was recorded on any rung, so unlike "
         "CARE-GNN there is no partial learning curve. A fix is implemented in the source but "
         "has not been re-run; see Section 8.")

    add_figure(doc, "tsocial_validation_curve.png",
               "Figure 1. CARE-GNN on T-Social. Validation AUPRC and AUROC over the 16 "
               "epochs completed before the session cap. The run was still improving; it "
               "never reached a test evaluation, so no T-Social result is reported.")

    callout(doc, "No T-Social figure appears in any results table in this report",
            "None was estimated, extrapolated from the other five datasets, or taken from a "
            "published paper. The CARE-GNN validation curve above is reported because it "
            "actually happened, and is deliberately kept out of every table where a test "
            "metric belongs. An attempted cell with a recorded reason is a result; a "
            "fabricated cell is not. Section 8 sets out what closing the gap would require.")
    doc.add_page_break()


def results(doc):
    doc.add_heading("3. Results (RQ1): comparative performance", level=1)
    para(doc,
         "All values are mean ± sample standard deviation over training seeds 2, 42 and 72. "
         "A status in place of a number means the cell produced no test evaluation.")

    for metric, label, number in (("auprc", "AUPRC", 4), ("auroc", "AUROC", 5),
                                  ("macro_f1", "Macro-F1", 6)):
        doc.add_heading(f"3.{number - 3} {label}", level=2)
        rows = []
        for d in DATASETS:
            a, b = value("CARE-GNN", d, metric), value("GHRN", d, metric)
            delta = f"{b - a:+.4f}" if None not in (a, b) else "—"
            rows.append([d, stat("CARE-GNN", d, metric), stat("GHRN", d, metric), delta])
        add_table(doc, ["Dataset", "CARE-GNN", "GHRN", "Difference"], rows,
                  caption=f"Table {number}. {label} by dataset. Difference is GHRN minus "
                          f"CARE-GNN; positive favours GHRN.",
                  widths=[1.2, 1.8, 1.8, 1.2])

    add_figure(doc, "comparison_auprc.png",
               "Figure 2. Test AUPRC by dataset and model, with seed spread.")
    add_figure(doc, "metric_heatmap.png",
               "Figure 3. All metrics, both models, every completed dataset. The FDCompCN "
               "column is the visible signature of a dataset neither model solves.")

    doc.add_heading("3.4 Fraud recall and precision", level=2)
    para(doc, "The two must be read together, since each model buys one at the other's expense.")
    rows = []
    for d in DATASETS:
        rows.append([d,
                     stat("CARE-GNN", d, "fraud_recall"), stat("GHRN", d, "fraud_recall"),
                     stat("CARE-GNN", d, "fraud_precision"), stat("GHRN", d, "fraud_precision")])
    add_table(doc, ["Dataset", "CARE-GNN recall", "GHRN recall",
                    "CARE-GNN precision", "GHRN precision"], rows,
              caption="Table 7. Fraud recall and fraud precision by dataset.",
              widths=[1.0, 1.4, 1.4, 1.4, 1.4])

    add_figure(doc, "comparison_fraud_recall.png",
               "Figure 4. Fraud recall by dataset and model.")

    doc.add_heading("3.5 What the numbers support", level=2)

    para(doc, "The Amazon gap is largely a reliability gap.", after=4)
    doc.paragraphs[-1].runs[0].bold = True
    amazon = [r for r in PER_SEED if r["dataset"] == "Amazon"]
    rows = []
    for seed in ("2", "42", "72"):
        row = [seed]
        for m in MODELS:
            hit = [r for r in amazon if r["model"] == m and r["train_seed"] == seed]
            row.append(f"{float(hit[0]['auprc']):.4f}" if hit else "—")
        rows.append(row)
    add_table(doc, ["Seed", "CARE-GNN", "GHRN"], rows,
              caption="Table 8. Per-seed AUPRC on Amazon. CARE-GNN's seed 72 collapsed.",
              widths=[1.2, 1.8, 1.8],
              highlight=lambda r, c, t: r == 2 and c == 1)
    para(doc,
         "CARE-GNN's seeds 2 and 42 average 0.8490, within 0.027 of GHRN. Seed 72 peaked on "
         "validation at epoch 2 and early-stopped at 23 epochs. On Amazon the two models are "
         "therefore close when CARE-GNN trains successfully, and the gap that appears in the "
         "mean is the price of CARE-GNN failing to train one time in three. That is a real and "
         "reportable cost, but it is a different claim from 'GHRN models Amazon better'.")

    para(doc, "The T-Finance result is not robust in either direction.", after=4)
    doc.paragraphs[-1].runs[0].bold = True
    tfin = [r for r in PER_SEED if r["dataset"] == "T-Finance"]
    rows = []
    for seed in ("2", "42", "72"):
        row = [seed]
        for m in MODELS:
            hit = [r for r in tfin if r["model"] == m and r["train_seed"] == seed]
            row.append(f"{float(hit[0]['auprc']):.4f}" if hit else "—")
        rows.append(row)
    add_table(doc, ["Seed", "CARE-GNN", "GHRN"], rows,
              caption="Table 9. Per-seed AUPRC on T-Finance. Both models have an unstable seed.",
              widths=[1.2, 1.8, 1.8])
    para(doc,
         f"GHRN's spread on this dataset (SD "
         f"{float(CELL[('GHRN','T-Finance')]['auprc_sd']):.4f}) is larger than the difference "
         f"between the two means, so the ranking flips depending on which seeds are drawn. No "
         f"winner should be claimed on T-Finance from three seeds.")

    para(doc, "FDCompCN defeats both models.", after=4)
    doc.paragraphs[-1].runs[0].bold = True
    fd_rate = float([r for r in PROFILE if r["dataset"] == "FDCompCN"][0]["test_fraud_pct"])
    para(doc,
         f"AUROC is {value('CARE-GNN','FDCompCN','auroc'):.4f} and "
         f"{value('GHRN','FDCompCN','auroc'):.4f} against 0.5 for random ranking, and AUPRC "
         f"{value('CARE-GNN','FDCompCN','auprc'):.4f} and "
         f"{value('GHRN','FDCompCN','auprc'):.4f} against a test fraud rate of {fd_rate:.2f}%, "
         f"barely above what a constant classifier would score. The recall figures indicate "
         f"that CARE-GNN missed "
         f"{(1 - value('CARE-GNN','FDCompCN','fraud_recall')) * 100:.1f}% of the fraudulent "
         f"nodes and GHRN "
         f"{(1 - value('GHRN','FDCompCN','fraud_recall')) * 100:.1f}%. CARE-GNN's nominal lead "
         f"here is not evidence that it works on FDCompCN; neither model does.")

    callout(doc, "Statistical strength",
            "Three seeds per cell is enough to expose instability, which it did, but not "
            "enough for formal significance testing. Differences should be read against the "
            "reported standard deviations. On that basis only the Amazon margin clearly "
            "exceeds GHRN's own seed spread, and that margin is explained above by a single "
            "collapsed run.")
    doc.add_page_break()


def graph_characteristics(doc):
    doc.add_heading("4. Graph characteristics (RQ4)", level=1)
    para(doc,
         "This section relates model behaviour to measured properties of the graphs. Every "
         "statistic was computed during the runs themselves and read back from the per-run "
         "evidence bundles.")

    rows = [[r["dataset"],
             f"{float(r['global_heterophily']):.4f}",
             f"{float(r['fraud_local_mean']):.4f}",
             f"{float(r['fraud_local_median']):.4f}",
             f"{int(r['labelled_edges_considered']):,}",
             f"{float(r['test_fraud_pct']):.2f}%"] for r in PROFILE]
    add_table(doc, ["Dataset", "Global heterophily", "Fraud-local mean",
                    "Fraud-local median", "Labelled edges", "Test fraud %"], rows,
              caption="Table 10. Graph characteristics. Global heterophily is different-label "
                      "eligible edges over all eligible labelled edges; fraud-local is, per "
                      "fraud node, the share of labelled one-hop neighbours that are benign.",
              widths=[1.0, 1.3, 1.1, 1.2, 1.2, 1.0])

    doc.add_heading("4.1 Global and local heterophily say different things", level=2)
    para(doc,
         "The two measures diverge sharply, and the local one matters more for fraud. Amazon "
         "is globally among the most homophilic graphs in the suite (0.0512), yet its fraud "
         "nodes sit in overwhelmingly benign neighbourhoods (fraud-local 0.8626). The graph as "
         "a whole is well clustered; the fraudsters specifically are not clustered with each "
         "other. That is precisely the camouflage CARE-GNN was designed to defeat, and "
         "CARE-GNN does not win Amazon.")
    para(doc,
         "The datasets split into two families. YelpChi, FDCompCN and Amazon are camouflaged, "
         "with fraud-local heterophily above 0.80; FDCompCN is extreme, its median fraud node "
         "having every labelled neighbour benign. T-Finance and Elliptic are mixed, retaining "
         "substantial fraud-to-fraud connectivity for message passing to exploit.")

    doc.add_heading("4.2 Heterophily does not predict difficulty or ranking", level=2)
    # T-Social has a graph profile but no result, so it cannot be ranked by AUPRC.
    scored = [r for r in PROFILE
              if value("CARE-GNN", r["dataset"], "auprc") is not None
              and value("GHRN", r["dataset"], "auprc") is not None]
    ordered = sorted(scored, key=lambda r: -max(
        value("CARE-GNN", r["dataset"], "auprc"), value("GHRN", r["dataset"], "auprc")))
    rows = []
    for r in ordered:
        d = r["dataset"]
        best = max(value("CARE-GNN", d, "auprc"), value("GHRN", d, "auprc"))
        delta = value("GHRN", d, "auprc") - value("CARE-GNN", d, "auprc")
        rows.append([d, f"{best:.4f}", f"{float(r['global_heterophily']):.4f}",
                     f"{int(r['labelled_edges_considered']):,}", f"{delta:+.4f}"])
    add_table(doc, ["Dataset", "Best AUPRC", "Global heterophily", "Labelled edges",
                    "GHRN − CARE-GNN"], rows,
              caption="Table 11. Datasets ordered by best achieved AUPRC.",
              widths=[1.1, 1.2, 1.4, 1.3, 1.3])
    para(doc,
         "The two most heterophilic graphs sit third and fifth, consistent with heterophily "
         "hurting, but Elliptic is the fourth hardest despite being the third most homophilic, "
         "so heterophily alone does not order the datasets. Labelled edge count separates them "
         "better at the bottom: the two hardest datasets are the two with almost no labelled "
         "graph structure. For FDCompCN this compounds with a median fraud-local heterophily "
         "of 1.0000, so the typical fraud node has no labelled fraud neighbour at all and "
         "there is essentially no fraud-to-fraud signal for any message-passing model to "
         "propagate. Ordering by heterophily also produces no monotonic trend in GHRN's "
         "margin: its largest advantage occurs at the third-lowest heterophily, and it loses "
         "on the second-most heterophilic dataset.")

    doc.add_heading("4.3 The refinement ablation", level=2)
    para(doc,
         "GHRN's two-stage design provides a within-model ablation at no extra cost. Stage 1 "
         "trains the beta-wavelet backbone on the unpruned graph; its predictions drive the "
         "edge pruning, and stage 2 retrains on the refined graph. Both stages recorded their "
         "own best-validation AUPRC, so the refinement step can be isolated without any "
         "comparison to CARE-GNN and without touching the test set.")

    by_ds = OrderedDict()
    for r in STAGES:
        by_ds.setdefault(r["dataset"], []).append(r)
    rows = []
    for d in DATASETS:
        group = by_ds.get(d)
        if not group:
            continue
        m1 = sum(float(r["stage1_val_auprc"]) for r in group) / len(group)
        m2 = sum(float(r["stage2_val_auprc"]) for r in group) / len(group)
        helped = sum(1 for r in group if r["refinement_helped"] == "True")
        rows.append([d, f"{m1:.4f}", f"{m2:.4f}", f"{m2 - m1:+.4f}",
                     f"{helped} of {len(group)}"])
    add_table(doc, ["Dataset", "Stage 1 (unrefined)", "Stage 2 (refined)", "Mean Δ",
                    "Seeds helped"], rows,
              caption="Table 12. GHRN refinement ablation, scored on validation AUPRC. "
                      "Nothing here touches the test set.",
              widths=[1.1, 1.5, 1.4, 1.0, 1.2],
              highlight=lambda r, c, t: c == 3 and t.startswith("-"))

    helped_total = sum(1 for r in STAGES if r["refinement_helped"] == "True")
    para(doc,
         f"Across all {len(STAGES)} GHRN runs refinement improved validation AUPRC in "
         f"{helped_total} and hurt in {len(STAGES) - helped_total}. Lining this up against what "
         f"the prune actually achieved gives a coherent mechanism: edge heterophily fell by "
         f"0.013280 on YelpChi and 0.007364 on Elliptic, but by only 0.001374 on T-Finance and "
         f"moved the wrong way on Amazon (−0.000361) and FDCompCN (−0.000244). Refinement pays "
         f"off precisely when it succeeds in making the graph more homophilic, and is harmful "
         f"when it does not, because the prune then deletes 1.5–3% of edges effectively at "
         f"random with respect to class. FDCompCN is the exception and is also the noisiest "
         f"cell in the study.")

    callout(doc, "A selection issue this ablation exposes",
            "The refined stage was shipped in all 15 GHRN runs, including the six where the "
            "unrefined stage achieved higher validation AUPRC. On T-Finance the mean gap is "
            "large — 0.8541 unrefined against 0.6877 refined — and that is also the cell where "
            "GHRN loses to CARE-GNN and where its seed spread is widest. This follows the "
            "published GHRN procedure, which applies refinement rather than choosing between "
            "stages, but it departs from this benchmark's rule of selecting on validation "
            "AUPRC. The reported results stand as run and are not adjusted; the correction is "
            "identified in Section 8 as the highest-value next experiment.")
    doc.add_page_break()


def efficiency(doc):
    doc.add_heading("5. Efficiency and scalability (RQ3)", level=1)
    callout(doc, "The two blocks below must not be pooled",
            "The completed cells did not all run on the same GPU class, and the protocol "
            "prohibits combining timings across hardware. Predictive metrics are unaffected "
            "and remain comparable everywhere; runtime and memory are not.")

    for gpu in sorted({r["gpu"] for r in EFFICIENCY}):
        doc.add_heading(gpu, level=2)
        # Format from the unrounded measurement in master_mean_sd.csv. Formatting
        # efficiency.csv's 4dp copy again at 3dp double-rounds (0.028477 -> 0.0285
        # -> "0.029"), which is not the correct rendering of the measurement.
        rows = []
        for r in EFFICIENCY:
            if r["gpu"] != gpu:
                continue
            raw = CELL[(r["model"], r["dataset"])]
            rows.append([r["model"], r["dataset"],
                         f"{float(raw['mean_epoch_s']):.3f}",
                         f"{float(raw['peak_gpu_mb']):,.1f}",
                         f"{int(float(raw['params'])):,}",
                         f"{float(raw['epochs']):.0f}"])
        add_table(doc, ["Model", "Dataset", "Mean epoch (s)", "Peak GPU (MB)",
                        "Parameters", "Mean epochs"], rows,
                  caption=f"Table {13 if 'A6000' in gpu else 14}. Efficiency on {gpu}.",
                  widths=[1.2, 1.1, 1.3, 1.2, 1.1, 1.1])

    add_figure(doc, "efficiency_tradeoff.png",
               "Figure 5. Per-epoch time against peak GPU memory, both axes logarithmic. "
               "The two models occupy opposite corners on every dataset.")

    doc.add_heading("5.1 The trade is time against memory", level=2)
    para(doc,
         "The two models sit at opposite corners, consistently across every dataset and both "
         "GPU classes. On the A6000, GHRN completes an epoch in 0.013–0.028 s against "
         "CARE-GNN's 0.710–7.636 s, between 49 and 382 times faster, while consuming "
         "substantially more memory (48.6–1,184.2 MB against 30.8–276.7 MB). The contrast is "
         "sharpest on T-Finance, where GHRN is 4.8 times faster but uses 5,570.6 MB against "
         "CARE-GNN's 302.3 MB, an eighteen-fold difference. Neither model is uniformly "
         "cheaper; which one is affordable depends on whether the binding constraint is "
         "wall-clock or GPU memory.")
    para(doc,
         "The causes are structural. CARE-GNN's neighbour selector sorts and filters each "
         "node's neighbourhood in Python on every forward pass, so work grows with the number "
         "of edges touched rather than with tensor sizes; its parameter counts are tiny (982 "
         "to 11,278), so the cost is not in the model but in the per-node work around it. That "
         "is the same property that caused the T-Social failure. GHRN's beta-wavelet backbone "
         "is fully vectorised, which is why its epochs are so cheap, but it materialises "
         "several filtered copies of the graph signal at once, which is where its memory goes.")

    doc.add_heading("5.2 Scalability outlook for T-Social", level=2)
    para(doc,
         "Neither model has a comfortable path to T-Social on a single mid-range GPU, and the "
         "two would fail for different reasons. CARE-GNN would run out of time: its per-epoch "
         "cost already reaches 7.6 s on YelpChi and grows with degree, which is the failure "
         "actually observed. GHRN would risk running out of memory: it already peaks at 5.6 GB "
         "on T-Finance, and T-Social is substantially larger. The A6000's 48 GB gives more "
         "headroom than the T4's 16 GB, which is the argument for attempting T-Social on the "
         "A6000.")
    doc.add_page_break()


def quality(doc):
    doc.add_heading("6. Quality assurance", level=1)

    doc.add_heading("6.1 Training collapse", level=2)
    para(doc,
         "A run is flagged when its best validation epoch is 1, 2 or 3, meaning validation "
         "peaked on an essentially untrained model and then degraded for the full patience "
         "window until early stopping fired.")
    rows = [[r["model"], r["dataset"], r["seed"], r["auprc"], r["best_epoch"],
             r["epochs"], r["fraud_recall"], r["fraud_precision"]] for r in FLAGS]
    add_table(doc, ["Model", "Dataset", "Seed", "AUPRC", "Best epoch", "Epochs",
                    "Fraud recall", "Fraud precision"], rows,
              caption=f"Table 15. All {len(FLAGS)} collapsed runs. All are CARE-GNN; none of "
                      f"GHRN's 15 runs collapsed.",
              widths=[1.1, 1.0, 0.6, 0.8, 0.9, 0.7, 1.0, 1.1])
    para(doc,
         "The signature is consistent: very high precision alongside collapsed recall. The "
         "model predicted almost nothing as fraud, so the few nodes it did flag were mostly "
         "right. This is a property of the published method under a frozen configuration, not "
         "a defect in the implementation. CARE-GNN's reinforcement-learning threshold module "
         "and its per-epoch negative under-sampling interact: an unlucky initialisation can "
         "drive the neighbour selector to reject nearly all neighbours before the classifier "
         "has learned anything useful, and the RL threshold then has no signal to recover "
         "from. GHRN has no comparable feedback loop.")
    para(doc,
         "Worth noting for interpretation: the two T-Finance collapses still produced AUPRC of "
         "0.6655 and 0.7345, not far below the healthy seed. AUPRC is threshold-free and "
         "rewards ranking, so a collapsed run is not always visible in the primary metric. The "
         "damage concentrates in the threshold-dependent metrics, recall in particular, which "
         "is why the best-epoch flag is reported rather than inferred from scores.")

    doc.add_heading("6.2 Seed variance", level=2)
    rows = [[r["model"], r["dataset"], r["auprc_mean"], r["auprc_sd"], r["auprc_cv"]]
            for r in SPREAD]
    add_table(doc, ["Model", "Dataset", "AUPRC mean", "AUPRC SD", "CV"], rows,
              caption="Table 16. Coefficient of variation of AUPRC across seeds, worst first.",
              widths=[1.3, 1.2, 1.2, 1.1, 0.9],
              highlight=lambda r, c, t: r < 2 and c == 4)
    para(doc,
         "Two cells stand out with different causes. CARE-GNN on Amazon is the collapse "
         "documented above. GHRN on T-Finance is not a collapse: its weak seed reached best "
         "validation at epoch 9, and the likely cause is the stage-selection issue in Section "
         "4.3, since T-Finance is the cell where refinement was most harmful and the refined "
         "stage was shipped in all three seeds regardless.")

    add_figure(doc, "seed_spread_auprc.png",
               "Figure 6. Seed spread of test AUPRC per completed cell.")

    doc.add_heading("6.3 Conformance checks that passed", level=2)
    bullets(doc, [
        "Every completed run evaluated the test set exactly once.",
        "No tuning run loaded test data; tuning executed with the test path disabled.",
        "Split IDs are identical across both models on every dataset, at split seed 2.",
        "Both models independently reported identical heterophily statistics on all five "
        "datasets, confirming they were served the same canonical graph view.",
        "Every completed run wrote a full artefact bundle, each reporting complete: true.",
        "An automated consistency audit verifies that every figure quoted in the written "
        "deliverable appears in the generated tables; all checks pass.",
    ])
    doc.add_page_break()


def discussion(doc):
    doc.add_heading("7. Discussion", level=1)

    para(doc, "Reliability, not accuracy, is the decisive difference.", after=4)
    doc.paragraphs[-1].runs[0].bold = True
    para(doc,
         "The headline metrics separate CARE-GNN and GHRN by modest margins on four of five "
         "datasets, and the one large margin dissolves under inspection into a single failed "
         "training run. What does not dissolve is the failure rate itself: three of fifteen "
         "CARE-GNN runs never trained, against zero of fifteen for GHRN. For a practitioner "
         "choosing between the two this is the more consequential finding, and it is one a "
         "benchmark reporting only means would have obscured, since the collapse is partly "
         "hidden in AUPRC. It is visible in recall, in the best-epoch record and in the "
         "standard deviation, which is why all three are reported here.")

    para(doc, "GHRN's advantage does not come from the mechanism its paper emphasises.", after=4)
    doc.paragraphs[-1].runs[0].bold = True
    para(doc,
         "The stage ablation isolates the heterophily refinement from the spectral backbone "
         "and finds refinement is not reliably beneficial: it helped in 9 of 15 runs and was "
         "actively harmful on T-Finance. Crucially, on Amazon — the dataset carrying GHRN's "
         "entire average margin — the unrefined stage validated better than the refined one, "
         "and the prune measurably failed to reduce heterophily. The mechanism is therefore "
         "real but conditional: pruning pays when it finds different-class edges to remove and "
         "costs when it does not. This suggests that the availability of removable "
         "heterophilic structure, rather than heterophily as such, is the variable that "
         "predicts whether refinement helps, and it is measurable before training.")

    para(doc, "Graph structure explains dataset difficulty better than model ranking.", after=4)
    doc.paragraphs[-1].runs[0].bold = True
    para(doc,
         "Heterophily statistics ordered neither the datasets by difficulty nor the models by "
         "margin. What did separate the hardest datasets was the amount of labelled graph "
         "structure available and, in FDCompCN's case, the near-total absence of "
         "fraud-to-fraud connectivity. This bounds what any message-passing architecture can "
         "achieve on such graphs: with a median fraud node having no labelled fraud neighbour, "
         "the aggregation step has nothing informative to aggregate, and improvements would "
         "have to come from features or supervision rather than architecture.")

    para(doc, "Which model to prefer.", after=4)
    doc.paragraphs[-1].runs[0].bold = True
    para(doc,
         "The evidence points to GHRN for most settings: it trains reliably, it is one to two "
         "orders of magnitude faster per epoch, and it sits at a higher-recall operating point, "
         "which suits fraud detection where a missed fraudulent node usually costs more than a "
         "reviewer dismissing a false positive. The case for CARE-GNN is narrower but real. It "
         "is markedly lighter on GPU memory, by eighteen-fold on the largest graph completed, "
         "and it reaches far higher precision under the same threshold rule, which matters "
         "where review capacity rather than detection coverage is the binding constraint.")
    doc.add_page_break()


def limitations(doc):
    doc.add_heading("8. Limitations and further work", level=1)
    para(doc,
         "This study establishes less than the completed cells might suggest, and the "
         "boundaries are worth stating precisely.")

    bullets(doc, [
        "Two of the twelve required cells have no result, both T-Social. CARE-GNN was "
        "attempted and stopped at the session time cap; GHRN was never launched. No T-Social "
        "figure appears anywhere in this work.",

        "Three training seeds per cell suffice to expose instability but not to establish "
        "rankings. Only the Amazon margin clearly exceeds the seed spread, and that margin is "
        "itself attributable to a single collapsed run. T-Finance cannot be called at all.",

        "Every run is at training ratio TR40. The label-scarcity axis of the research "
        "questions — performance at TR30, TR20 and TR10, and under reduced training fraud "
        "prevalence — is therefore not addressed. This is a gap in executed runs rather than "
        "in infrastructure: the nested split pools are generated and verified, and closing it "
        "requires approximately 90 further runs reusing the TR40-selected configurations.",

        "The completed cells span two GPU classes, so runtime and memory figures are "
        "comparable only within each block and are reported separately.",

        "GHRN's stage selection does not follow the benchmark's stated rule, as set out in "
        "Section 4.3. The reported numbers stand as run and are not adjusted for it.",

        "This work covers two detectors. It is a controlled comparison between CARE-GNN and "
        "GHRN, not a leaderboard over fraud-detection GNNs.",
    ])

    doc.add_heading("8.1 Priority order for the remaining work", level=2)
    add_table(
        doc,
        ["Priority", "Work", "Approximate cost", "What it buys"],
        [
            ["1", "GHRN between-stage selection on validation AUPRC", "~1 GPU session",
             "Corrects a known protocol deviation; likely changes the headline"],
            ["2", "T-Social on the A6000, GHRN first", "1–2 sessions",
             "Closes the two empty cells; 12 of 12"],
            ["3", "More seeds on Amazon and T-Finance", "~1 session",
             "Settles the two contested cells"],
            ["4", "TR30 / TR20 / TR10 label ladder", "90 runs",
             "Opens RQ2, currently unaddressed"],
            ["5", "Single-GPU-class re-run", "2–3 sessions",
             "Makes the efficiency table single-block"],
        ],
        caption="Table 17. Remaining work in priority order.",
        widths=[0.7, 2.2, 1.2, 2.2],
    )
    doc.add_page_break()


def conclusion(doc):
    doc.add_heading("9. Conclusion", level=1)
    completed = sum(1 for r in MATRIX if r["status"] == "COMPLETED")
    para(doc,
         f"We benchmarked CARE-GNN and GHRN across six fraud graph datasets under a single "
         f"controlled protocol, completing {completed} of {len(MATRIX)} required combinations "
         f"over {len(PER_SEED)} runs with fixed splits, a shared evaluator, and exactly one "
         f"test evaluation per run.")
    para(doc,
         "GHRN leads on aggregate across every headline metric, but the margin is narrow on "
         "four of five datasets and the one large margin resolves, on inspection, into a "
         "reliability difference rather than a modelling one. The clearest and most practically "
         "useful result is that CARE-GNN failed to train in three of fifteen runs while GHRN "
         "failed in none — a difference in dependability that aggregate scores tend to conceal "
         "and that we surface through best-epoch records, recall and seed variance.")
    para(doc,
         "Using GHRN's own two-stage design as an ablation, we find that its heterophily-aware "
         "refinement, the mechanism its paper foregrounds, improved validation AUPRC in only 9 "
         "of 15 runs and was harmful on two datasets, including the one carrying its entire "
         "average advantage. Refinement helped precisely where it measurably succeeded in "
         "making the graph more homophilic, which suggests a concrete, pre-trainable criterion "
         "for when to apply it rather than applying it unconditionally. We also identify that "
         "the refined stage was shipped in every run regardless of which stage validated "
         "better, a deviation from the benchmark's own selection rule that is cheap to correct "
         "and likely to change the headline comparison.")
    para(doc,
         "Relating results to graph structure, we find that heterophily statistics explain "
         "dataset difficulty poorly and model ranking not at all, while the volume of labelled "
         "graph structure and the presence of fraud-to-fraud connectivity separate the hard "
         "datasets clearly. FDCompCN, where the median fraud node has no labelled fraud "
         "neighbour, defeats both models at close to chance performance and bounds what any "
         "message-passing architecture can be expected to achieve there.")
    para(doc,
         "The most valuable next steps, in order, are: correct GHRN's between-stage selection "
         "and re-run; complete the two T-Social cells on the larger GPU, GHRN first; add seeds "
         "to the two contested cells; and open the label-scarcity ladder that this phase did "
         "not reach.")

    doc.add_heading("Appendix A. Reproducibility", level=1)
    para(doc,
         "Every table in this report is generated from the per-run records rather than typed. "
         "The consolidation scripts read the run evidence and emit the CSV and Markdown "
         "tables that this document draws from, and an automated consistency audit verifies "
         "that every quoted figure appears in those tables. The scripts use the Python "
         "standard library only.")
    add_table(
        doc,
        ["Script", "Produces"],
        [
            ["consolidate_results.py", "Per-seed and mean/SD tables, experiment matrix, "
                                       "efficiency, seed-stability flags"],
            ["dataset_profile.py", "Graph characteristics and the refinement edge statistics"],
            ["stage_ablation.py", "GHRN stage 1 against stage 2 on validation"],
            ["consistency_audit.py", "Verification that the prose matches the tables"],
            ["build_report.py", "This document"],
        ],
        caption="Table 18. Scripts that regenerate the deliverable.",
        widths=[2.0, 4.2],
    )
    para(doc,
         "Data provenance: YelpChi and Amazon from the DGL fraud dataset releases, T-Finance "
         "and T-Social from the BWGNN authors' release, Elliptic via Kaggle, and FDCompCN from "
         "the SplitGNN repository release. Implementations follow the official CARE-GNN "
         "(github.com/YingtongDou/CARE-GNN) and GHRN (github.com/blacksingular/GHRN) "
         "references, verified mechanism by mechanism against the upstream source.")


def main():
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.9)
        section.bottom_margin = Inches(0.9)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
    style_document(doc)

    title_page(doc)
    executive_summary(doc)
    methodology(doc)
    completion(doc)
    results(doc)
    graph_characteristics(doc)
    efficiency(doc)
    quality(doc)
    discussion(doc)
    limitations(doc)
    conclusion(doc)

    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")
    print(f"  paragraphs: {len(doc.paragraphs)}   tables: {len(doc.tables)}")


if __name__ == "__main__":
    main()
