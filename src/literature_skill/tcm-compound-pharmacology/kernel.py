"""Helpers for the tcm-compound-pharmacology skill.

Auto-loaded into the python kernel when the skill is loaded. Provides input
parsing, PubMed query construction, the effect-family query map, and the
CSV + markdown output writer. See SKILL.md for the workflow these support.
"""

# Effect families -> PubMed field-restricted fragment. Keys are the effect
# labels; values are the boolean [tiab] fragment ANDed with the compound query
# to gauge how well-documented each effect is. Open-ended: add effects the
# literature supports even if not listed here.
EFFECT_QUERIES = {
    "Anti-inflammatory": '(anti-inflammatory[tiab] OR antiinflammatory[tiab] OR inflammation[tiab])',
    "Anti-cancer": '(anti-cancer[tiab] OR anticancer[tiab] OR antitumor[tiab] OR antitumour[tiab] OR antineoplastic[tiab] OR apoptosis[tiab])',
    "Neuroprotective": '(neuroprotective[tiab] OR neuroprotection[tiab] OR neurodegenerat*[tiab])',
    "Antimicrobial": '(antimicrobial[tiab] OR antibacterial[tiab] OR antiviral[tiab] OR antifungal[tiab] OR antiparasitic[tiab])',
    "Antioxidant": '(antioxidant[tiab] OR "oxidative stress"[tiab] OR "radical scavenging"[tiab])',
    "Hepatoprotective": '(hepatoprotective[tiab] OR hepatoprotection[tiab] OR "liver injury"[tiab])',
    "Cardioprotective": '(cardioprotective[tiab] OR cardioprotection[tiab] OR "myocardial"[tiab])',
    "Antidiabetic": '(antidiabetic[tiab] OR hypoglycemic[tiab] OR "insulin resistance"[tiab] OR "blood glucose"[tiab])',
    "Lipid-lowering / anti-obesity": '(hypolipidemic[tiab] OR "lipid-lowering"[tiab] OR anti-obesity[tiab] OR antiobesity[tiab])',
    "Analgesic": '(analgesic[tiab] OR antinociceptive[tiab] OR "pain relief"[tiab])',
    "Immunomodulatory": '(immunomodulat*[tiab] OR immunostimulat*[tiab] OR immunosuppress*[tiab])',
    "Antidepressant/anxiolytic": '(antidepressant[tiab] OR anxiolytic[tiab] OR "depressive-like"[tiab])',
}

TOXICITY_QUERY = ('(toxicity[tiab] OR toxic[tiab] OR hepatotoxic*[tiab] OR nephrotoxic*[tiab] '
                  'OR cardiotoxic*[tiab] OR genotoxic*[tiab] OR poisoning[tiab] OR "adverse effect*"[tiab])')

# Single OR-fragment spanning the common beneficial-effect terms. Used by the
# indirect-link step (workflow section "Indirect-link screening") to test
# whether a PARENT scaffold / structural analog of an otherwise-Low compound has
# ANY documented pharmacological effect, before any Low->Middle upgrade.
ALL_EFFECTS_FRAGMENT = (
    '(anti-inflammatory[tiab] OR antiinflammatory[tiab] OR anticancer[tiab] '
    'OR antitumor[tiab] OR antitumour[tiab] OR antineoplastic[tiab] OR antioxidant[tiab] '
    'OR neuroprotective[tiab] OR antimicrobial[tiab] OR antibacterial[tiab] OR antiviral[tiab] '
    'OR antifungal[tiab] OR hepatoprotective[tiab] OR cardioprotective[tiab] OR antidiabetic[tiab] '
    'OR hypoglycemic[tiab] OR analgesic[tiab] OR antinociceptive[tiab] OR immunomodulat*[tiab] '
    'OR antidepressant[tiab] OR pharmacological[tiab] OR therapeutic[tiab])'
)


def build_parent_effect_query(term, field="tiab"):
    """Query to test whether a parent scaffold / analog `term` has ANY
    documented beneficial effect: '"term"[tiab] AND (<effect OR-fragment>)'.
    Use in the indirect-link step; a nonzero hit count means READ the abstracts
    to confirm the relative's effect before upgrading a Low compound to Middle.
    """
    t = (term or "").strip().replace('"', " ")
    if not t:
        return ""
    return '"%s"[%s] AND %s' % (t, field, ALL_EFFECTS_FRAGMENT)


VALID_SCORES = ("High", "Middle", "Low")


def parse_compound_list(text):
    """Parse raw user input into compounds. One compound per line (or per
    comma-separated item if single line); within an entry, ';' separates
    SYNONYMS of the same compound. Returns
    [{"canonical": str, "synonyms": [str, ...]}].
    """
    if text is None:
        return []
    raw = text.strip()
    if not raw:
        return []
    # Prefer newline splitting; fall back to comma if it's clearly one line.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) == 1 and ";" not in lines[0] and "," in lines[0]:
        lines = [p.strip() for p in lines[0].split(",") if p.strip()]
    out = []
    for ln in lines:
        # Strip a leading list bullet/number ONLY when it is clearly a list
        # marker followed by whitespace — never chew into names that legitimately
        # start with a digit or hyphen (e.g. "3-keto-cinobufagin", "beta-carotene").
        import re
        bm = re.match(r"^\s*(?:[-*\u2022]|\d{1,2}[.)])\s+(.*)$", ln)
        if bm:
            ln = bm.group(1).strip()
        syns = [s.strip() for s in ln.split(";") if s.strip()]
        if not syns:
            continue
        out.append({"canonical": syns[0], "synonyms": syns})
    return out


def build_synonym_query(synonyms, field="tiab"):
    """OR-join compound synonyms into a PubMed subquery, phrase-quoted and
    field-restricted, e.g. ("ginsenoside rg1"[tiab] OR "panaxatriol rg1"[tiab]).
    """
    if isinstance(synonyms, str):
        synonyms = [synonyms]
    parts = []
    seen = set()
    for s in synonyms:
        s = s.strip()
        key = s.lower()
        if not s or key in seen:
            continue
        seen.add(key)
        parts.append('"%s"[%s]' % (s, field))
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "(" + " OR ".join(parts) + ")"


def join_items(items):
    return "; ".join(str(x) for x in (items or []))


def write_screen_outputs(rows, out_dir=".", csv_name="tcm_pharmacology_screen.csv",
                         md_name="tcm_pharmacology_report.md"):
    """Write the CSV supplementary table and the markdown report from a list of
    per-compound row dicts (see SKILL.md 'Row object shape'). Returns
    (csv_path, md_path).
    """
    import os
    import csv as _csv

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, csv_name)
    md_path = os.path.join(out_dir, md_name)

    cols = ["compound_canonical", "synonyms", "score", "toxicity", "effects",
            "effect_summary", "evidence_basis", "parent_related", "n_references",
            "references", "notes"]

    def norm(r):
        score = str(r.get("score", "Low")).strip().capitalize()
        if score not in VALID_SCORES:
            score = "Low"
        refs = r.get("references", []) or []
        # evidence_basis: Direct (compound itself), Indirect (parent/analog), or None.
        basis = r.get("evidence_basis")
        if not basis:
            if r.get("parent_related") or any("[parent/analog]" in str(x) for x in refs):
                basis = "Indirect (parent/analog)"
            elif score in ("High", "Middle"):
                basis = "Direct"
            else:
                basis = "None"
        return {
            "compound_canonical": r.get("compound_canonical", ""),
            "synonyms": join_items(r.get("synonyms", [])),
            "score": score,
            "toxicity": r.get("toxicity", "None") or "None",
            "effects": join_items(r.get("effects", [])),
            "effect_summary": join_items(r.get("effect_summary", [])),
            "evidence_basis": basis,
            "parent_related": r.get("parent_related", "") or "",
            "n_references": len(refs),
            "references": join_items(refs),
            "notes": r.get("notes", ""),
        }

    nrows = [norm(r) for r in rows]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for nr in nrows:
            w.writerow(nr)

    badge = {"High": "🟢 High", "Middle": "🟡 Middle", "Low": "⚪ Low"}
    lines = []
    lines.append("# TCM compound pharmacology screen\n")
    n = len(nrows)
    n_high = sum(1 for r in nrows if r["score"] == "High")
    n_mid = sum(1 for r in nrows if r["score"] == "Middle")
    n_low = sum(1 for r in nrows if r["score"] == "Low")
    n_tox = sum(1 for r in nrows if r["toxicity"] and r["toxicity"] != "None")
    lines.append("Screened **%d** compound(s): %d High, %d Middle, %d Low. "
                 "%d carry a toxicity warning.\n" % (n, n_high, n_mid, n_low, n_tox))

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Compound | Score | Evidence | Toxicity | Headline effect |")
    lines.append("|---|---|---|---|---|")
    for r, orig in zip(nrows, rows):
        headline = (orig.get("effects") or ["—"])[0] if orig.get("effects") else "None found"
        tox = "⚠️ " + r["toxicity"] if r["toxicity"] and r["toxicity"] != "None" else "—"
        ev = "indirect" if r["evidence_basis"].startswith("Indirect") else (
             "direct" if r["evidence_basis"] == "Direct" else "—")
        lines.append("| %s | %s | %s | %s | %s |" % (
            r["compound_canonical"], badge.get(r["score"], r["score"]), ev, tox, headline))
    lines.append("")

    # Per-compound detail
    lines.append("## Detail\n")
    for r, orig in zip(nrows, rows):
        lines.append("### %s  %s" % (r["compound_canonical"], badge.get(r["score"], r["score"])))
        if r["synonyms"] and r["synonyms"] != r["compound_canonical"]:
            lines.append("*Synonyms:* %s\n" % r["synonyms"])
        if r["toxicity"] and r["toxicity"] != "None":
            lines.append("> ⚠️ **Toxicity:** %s\n" % r["toxicity"])
        if r["evidence_basis"].startswith("Indirect"):
            pr = r["parent_related"]
            lines.append("> ↪️ **Indirect link** — no direct pharmacology for the compound "
                         "itself; effect(s) below are documented for a parent scaffold / "
                         "structural analog%s.\n" % ((": **%s**" % pr) if pr else ""))
        summ = orig.get("effect_summary", []) or []
        if summ:
            lines.append("**Documented effects:**")
            for s in summ:
                lines.append("- %s" % s)
            lines.append("")
        else:
            lines.append("_No pharmacological effect found in the literature searched._\n")
        refs = orig.get("references", []) or []
        if refs:
            lines.append("**References:** " + "; ".join(str(x) for x in refs) + "\n")
        if r["notes"]:
            lines.append("**Bottom line:** %s\n" % r["notes"])
    lines.append("\n---\n*Evidence scores: High = compound directly linked to effect; "
                 "Middle = indirect evidence (effect shown for the crude extract/formula, a "
                 "close structural analog, the parent scaffold, or in-silico only); "
                 "Low = no effect found for the compound or any known relative. "
                 "The Evidence column marks whether a Middle rests on direct or parent/analog "
                 "evidence; parent/analog references are tagged `[parent/analog]`. "
                 "Warning = documented toxicity.*")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return csv_path, md_path
