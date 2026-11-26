---
name: tcm-compound-pharmacology
description: >-
  Deep literature + database screen for the proven pharmacological effects of named
  chemical compounds, especially natural products from Traditional Chinese Medicine
  (TCM). Given one or more compound names, searches PubMed and ChEMBL for documented
  effects (anti-inflammatory, anti-cancer, neuroprotective, antimicrobial, antioxidant,
  hepatoprotective, etc.), returning real references (PMID/DOI), a short summary per
  effect, a toxicity flag, and one soft evidence score per compound (High = compound
  directly linked to an effect; Middle = indirect evidence only; Low = none found;
  Warning = toxicity reported). Use WHENEVER the user gives a compound name or list from
  a TCM / natural-product / phytochemistry context and asks whether they have
  pharmacological, therapeutic, or bioactive effects, wants them screened against the
  literature, or wants a compound-vs-effect evidence table. Names separated by ";" within
  one entry are SYNONYMS of one compound. Trigger even on a bare pasted list of names.
---

# TCM compound pharmacology screening

## What this does and why

Phytochemistry and TCM papers routinely report a *series* of isolated compounds, and
the reviewer's question is always the same: are these known to do anything? This skill
turns a list of compound names into an evidence table a peer reviewer would trust —
each compound mapped to its documented pharmacological effects, every claim anchored to
a real retrieved reference, honest about where evidence is thin or absent, and flagged
where the compound is toxic.

The hard part is *grounding*. It is easy to write a plausible-sounding summary from
memory; it is worthless if the PMID is invented or the effect was actually shown for a
crude extract rather than the pure compound. Every effect you report must come from a
document you actually retrieved this session, and the score you assign must reflect how
directly that document ties **the compound itself** to the effect.

## Data sources (call from the `repl` tool via `host.mcp`)

- **PubMed** (`host.mcp("pubmed", ...)`) — primary source. `search_articles`,
  `get_article_metadata` (abstracts + DOIs), `convert_article_ids` (PMID→DOI),
  `get_full_text_article` for PMC when an abstract is ambiguous.
- **ChEMBL** (`host.mcp("chembl", ...)`) — the strongest signal for a *direct* link.
  `compound_search` (is the molecule in a bioactivity database? approved drug?),
  `get_mechanism` (curated mechanism of action → High-confidence direct link),
  `get_bioactivity` (measured potency against named targets).
- **bioRxiv/medRxiv** (`host.mcp("biorxiv", ...)`) — optional, for very recent
  preprint evidence; label anything from here as a preprint.

MCP calls only work in the `repl` tool. Do the searches there, write results to
`./handoff/<compound>.json`, then analyse and build the outputs in the `python` tool.
`kernel.py` (auto-loaded with this skill) provides `parse_compound_list`,
`build_synonym_query`, `EFFECT_QUERIES`, `TOXICITY_QUERY`, `build_parent_effect_query`
+ `ALL_EFFECTS_FRAGMENT` (for the indirect-link pass, step 6), and
`write_screen_outputs` — use them so query construction and output format stay
consistent across runs.

## Workflow

### 1. Parse the input into compounds and synonyms

Split the user's input into one entry per compound. A line (or item) containing `;`
is ONE compound with multiple names — e.g. `Ginsenoside Rg1; Panaxatriol saponin Rg1`
is a single compound searched under both names. Use `parse_compound_list(text)` →
`[{"canonical": <first name>, "synonyms": [all names]}]`. The canonical name is the
first synonym; use every synonym in the searches so you don't miss papers that use an
alternate name.

### 2. Check ChEMBL first (cheap, high-value)

For each compound, `compound_search(name=<synonym>)` for each synonym until you get a
hit. A hit tells you: the molecule is characterised, its `max_phase` (4 = approved
drug — very strong evidence of proven effect), and its `molecule_chembl_id`. If you
have an ID, call `get_mechanism(molecule_chembl_id=...)` — a curated mechanism of
action is direct evidence and usually justifies a **High** score for the corresponding
effect. `get_bioactivity(molecule_chembl_id=..., min_pchembl=6)` gives measured potency
against named targets, which corroborates a direct pharmacological action. Many TCM
natural products are NOT in ChEMBL; absence here is not evidence of absence — fall back
to PubMed.

### 3. PubMed sweep

Build an OR-query over the synonyms with `build_synonym_query(synonyms)` →
`("name one"[tiab] OR "name two"[tiab])`. Then:

1. **Scope query** — run the bare synonym query, note `total_count`. Zero or a handful
   of hits is itself a finding (obscure or newly named compound).
2. **Effect-targeted queries** — for each effect family in `EFFECT_QUERIES`, run
   `<synonym-query> AND <effect-fragment>` and record `total_count`. This cheaply maps
   which effects are even plausibly documented before you read anything. Do NOT treat a
   nonzero count as proof — counts route your reading, they do not score the compound.
3. **Read the evidence** — for the compound overall and for each effect with hits,
   pull the top ~10–20 papers by relevance with `get_article_metadata` and read the
   abstracts. Prefer reviews for breadth and primary studies for the direct claim.
   Use `convert_article_ids` to get DOIs for the papers you will cite.
4. **Toxicity query** — always run `<synonym-query> AND (toxicity[tiab] OR toxic[tiab]
   OR hepatotoxic*[tiab] OR nephrotoxic*[tiab] OR cardiotoxic*[tiab] OR cytotoxic*[tiab]
   OR poisoning[tiab] OR "adverse effect*"[tiab])` and read enough to judge whether a
   genuine toxicity concern exists (as opposed to cytotoxicity that IS the anti-cancer
   mechanism — see scoring).

Reuse `find_related_articles` on your best hit if a compound is under-covered by the
keyword sweep and you suspect an alternate naming.

### 4. Extract effects and write short summaries

For each documented effect, write ONE short line: the effect label plus the concrete
finding and the model system, e.g.
`Anti-inflammatory — suppresses NF-κB/NLRP3 in LPS-stimulated macrophages (in vitro & mouse)`.
Keep it to a phrase, not a paragraph. Effect labels are open-ended; common families:
anti-inflammatory, anti-cancer / antitumor, neuroprotective, antimicrobial
(antibacterial/antiviral/antifungal), antioxidant, hepatoprotective, cardioprotective,
antidiabetic / hypoglycemic, anti-obesity / lipid-lowering, analgesic,
immunomodulatory, antidepressant/anxiolytic, anti-osteoporotic, wound healing. Use
whatever the literature actually supports — do not force a compound into these buckets.

### 5. Score each compound (one overall soft score)

Assign ONE score per compound, reflecting the *strongest* beneficial evidence found,
plus an independent toxicity flag (a compound can be both effective and toxic):

- **High** — a *direct* link between the pure compound and a beneficial effect: the
  isolated compound was tested and shown to produce the effect (in vitro, in vivo, or
  clinically), OR ChEMBL carries a curated mechanism of action / it is an approved drug.
  This is the "yes, this compound does X" tier.
- **Middle** — only *indirect* evidence: the effect is attributed to a crude
  extract/formula that contains the compound rather than the purified compound; or the
  link comes from network-pharmacology / molecular-docking / in-silico prediction only;
  or from a close structural analog / the **parent scaffold or aglycone** of the
  compound; or a single weak or tangential mention. The compound is *implicated* but not
  directly demonstrated. A Low compound is upgraded to Middle ONLY through the
  indirect-link second pass (step 6), and only when the relative's effect is confirmed
  from a retrieved abstract — record the relative in `parent_related` and mark
  `evidence_basis = "Indirect (parent/analog)"`.
- **Low** — no beneficial effect found for the compound OR for any inferable parent
  scaffold / analog after the step-6 indirect-link pass: essentially no relevant
  pharmacology (an isolation/structural paper only, a biosynthesis intermediate, a
  synthetic/industrial contaminant, or a truly obscure isolate with no bioactive
  relative).
- **Warning (toxicity)** — a genuine toxic / adverse effect is documented for the
  compound. Record this in the dedicated toxicity column regardless of the beneficial
  score, and surface it prominently. Note: cytotoxicity that is the *intended*
  anticancer mechanism is NOT a toxicity warning; reserve Warning for undesired
  toxicity (organ toxicity, poisoning, adverse events, genotoxicity).

Because the user asked for a single score per compound, put the beneficial-evidence
tier (High/Middle/Low) in the `score` column and the toxicity finding in the separate
`toxicity` column so both are visible at a glance; the markdown report shows a ⚠️ badge
whenever toxicity is present. When you are genuinely unsure between two tiers, pick the
lower and say why in `notes` — over-claiming is the failure mode that matters here.

### 6. Indirect-link screening — rescue Low compounds via parent/analog (do not skip)

A compound that returns no direct pharmacology of its own is very often a
glycoside, ester, or stereo/positional variant of a well-studied bioactive core,
or a close analog of a compound with a known effect. For **every compound that
would otherwise be Low**, run this second pass before finalizing the score:

1. **Derive the parent / closest known relative.** From the name, infer the
   aglycone or scaffold: strip glycosyl decorations (`-O-glucoside`,
   `-glucopyranoside`, `-rhamnoside`, `-xyloside`, `gentiobioside`), acyl esters
   (`benzoyl-`, `coumaroyl-`, `acetyl-`, `dodecanoyl-`), and stereo/position
   prefixes to reach the searchable common core (e.g. `irisolone` from
   `3',5'-dimethoxyirisolone-4'-O-β-D-glucoside`; `β-sitosterol` from a
   sitosterol glucoside; `ganoderic acid` from `ganoderic aldehyde TR`). Also
   record the compound *class* / famous sibling as alternates (e.g. "cycloartane
   triterpenoid", "isoflavone"). `host.llm` is useful for proposing the parent +
   1–3 alternate search terms per compound in one fan-out.
2. **VERIFY the relative actually has documented effects — do not assert it.**
   For each candidate term, run `build_parent_effect_query(term)` in PubMed. A
   nonzero `total_count` only *routes* you; fetch the top abstracts with
   `get_article_metadata` and confirm the parent/analog genuinely shows the
   effect. Never upgrade on a presumed relationship alone — the relative's effect
   must come from a real retrieved abstract, exactly as for a direct claim.
3. **Upgrade Low → Middle only on confirmed parent/analog evidence.** Set
   `score = "Middle"`, `evidence_basis = "Indirect (parent/analog)"`, and
   `parent_related` to the relative's name + the relation (e.g.
   `"β-sitosterol (7α-hydroxy glucoside of)"`). Label each effect as indirect
   (e.g. `"Anti-inflammatory (indirect)"`), tag every borrowed reference with a
   leading `[parent/analog]`, and state the relationship in `notes`
   ("Indirect link: glucoside of X, which has documented … . No direct
   pharmacology for the compound itself found."). If the parent also carries a
   genuine toxicity signal, record it in `toxicity` with a `(parent/analog)` tag.
4. **Leave it Low if there is no inferable or no documented relative.** Sterol/
   terpene biosynthesis intermediates, industrial/synthetic contaminants, and
   truly obscure isolates with no bioactive relative stay Low. Do not manufacture
   a parent to force an upgrade — an honest Low is the correct answer here.

This step is what separates a useful screen from a wall of "Low": most natural-
product isolates are decorated versions of something the literature already
knows. But the Direct-vs-Indirect distinction must stay visible — a Middle from
parent/analog evidence is a *hypothesis* about the compound (a sugar or acyl
group can change potency and bioavailability substantially), not a measurement.

### 7. Grounding rules (do not skip)

- Cite only PMIDs/DOIs you retrieved this session. Never write an identifier from
  memory. If you cannot find support, the honest answer is **Low / no link found**.
- Distinguish the pure compound from the extract/formula it came from — this is the
  single most common way a TCM pharmacology claim is overstated, and it is exactly the
  High-vs-Middle boundary.
- Keep the reference list per compound to the few most load-bearing papers (reviews +
  the key primary studies), not every hit.

## Output

Produce BOTH, via `write_screen_outputs(rows, out_dir=".")`:

1. **`tcm_pharmacology_screen.csv`** — one row per compound, columns:
   `compound_canonical, synonyms, score, toxicity, effects, effect_summary,
   evidence_basis, parent_related, n_references, references, notes`.
   - `score` ∈ {High, Middle, Low}; `toxicity` ∈ {None, Warning: <short reason>}.
   - `evidence_basis` ∈ {Direct, Indirect (parent/analog), None}; `parent_related`
     names the scaffold/analog for an indirect Middle (blank otherwise). These are
     filled automatically by `write_screen_outputs` from `parent_related` / the
     `[parent/analog]` reference tags, but set them explicitly on the row for clarity.
   - `effects` = semicolon list of effect labels; `effect_summary` = the short
     per-effect lines joined by `;`.
   - `references` = semicolon-separated `PMID:xxxxx (DOI)` entries.
2. **`tcm_pharmacology_report.md`** — a readable report: one section per compound with
   the score badge, synonyms, each effect as a bullet with its short summary and
   inline DOI links (`[Author Year](https://doi.org/...)`), the toxicity note, and a
   one-line bottom line. Open the report with a summary table (compound → score →
   headline effect), then the per-compound detail.

Save both with `save_artifacts`. In the chat reply, lead with the headline result
(how many compounds had High/Middle/Low, any toxicity flags), show the summary table,
and link the two artifacts at the end.

## Row object shape

Build one dict per compound and pass the list to `write_screen_outputs`:

A **direct** (High/Middle) row:

```python
{
  "compound_canonical": "Curcumin",
  "synonyms": ["Curcumin", "Diferuloylmethane"],
  "score": "High",                          # High | Middle | Low
  "toxicity": "None",                       # "None" or "Warning: <reason>"
  "effects": ["Anti-inflammatory", "Anti-cancer", "Antioxidant"],
  "effect_summary": [
     "Anti-inflammatory — inhibits NF-\u03baB/COX-2 (in vitro & rodent)",
     "Anti-cancer — induces apoptosis across multiple tumor lines (in vitro)",
  ],
  "evidence_basis": "Direct",               # Direct | Indirect (parent/analog) | None
  "parent_related": "",                     # blank for direct evidence
  "references": ["PMID:12345678 (10.xxxx/xxxx)", "PMID:23456789 (10.yyyy/yyyy)"],
  "notes": "Approved-drug-level evidence absent; extensive preclinical + some clinical.",
}
```

An **indirect** row rescued from Low via the step-6 parent/analog pass:

```python
{
  "compound_canonical": "7\u03b1-hydroxysitosterol 3-O-\u03b2-D-glucoside",
  "synonyms": ["7\u03b1-hydroxysitosterol 3-O-\u03b2-D-glucoside"],
  "score": "Middle",
  "toxicity": "None",
  "effects": ["Anti-inflammatory (indirect)", "Lipid-lowering (indirect)"],
  "effect_summary": [
     "Anti-inflammatory — via \u03b2-sitosterol, the aglycone (in vitro & rodent)",
     "Lipid-lowering — via \u03b2-sitosterol, lowers serum cholesterol (clinical)",
  ],
  "evidence_basis": "Indirect (parent/analog)",
  "parent_related": "\u03b2-sitosterol (7\u03b1-hydroxy 3-O-glucoside of)",
  "references": ["[parent/analog] PMID:22190137 (10.xxxx/xxxx)"],
  "notes": "Indirect link: glucoside of \u03b2-sitosterol, which has documented "
           "anti-inflammatory and cholesterol-lowering effects. No direct pharmacology "
           "for the compound itself found. Score raised Low\u2192Middle on parent evidence.",
}
```
