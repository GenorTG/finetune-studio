# Korvane Cold Chain — corpus bible

A fictional mid-size company whose paperwork is used to benchmark whether Finetune Studio can turn *real-looking, messy company
documents* into complete training data. Everything below is invented. A language model has never seen any of it, so every
fact it later answers correctly came from training on our data.

**Hard rules for everyone who writes here**

1. Facts are invented and *specific*: odd numbers (17.5 hours, 2,840 EUR, 14 pallets), codes (KT4-R2), dates, names, thresholds, clause numbers.
   Never use well-known real-world facts as the payload. Real place names are fine as background (Gdynia, Rotterdam).
2. The text must read like a real company wrote it, **not** like a dataset: boilerplate, throat-clearing, passive voice, internal jargon, legal
   hedging, marketing fluff, tangents, forwarded email chains, half-finished sentences in chat logs, a typo or two, inconsistent formatting,
   headers/footers, "see Appendix C", references to documents not in the corpus. Facts are *buried* in paragraphs, tables, footnotes and
   side remarks. Do NOT write neat "The X is Y." fact lists, FAQs that restate facts one per line, or glossaries — unless a real company
   would (a spec sheet is allowed to be a table).
3. Mix registers: legal ("the Supplier shall…, save where…"), HR-bureaucratic, engineering-terse, sales-glossy, casual Slack/e-mail
   (lowercase, emoji words like "lol", abbreviations, nicknames), polite complaints, angry customers.
4. Length is realistic and uneven: some documents are long and boring, some are three lines. Tables are real tables (rates, schedules,
   thresholds, SKUs, error codes). A few documents *supersede* older ones (v1 vs v2 policy, amendment letters); mark the old facts
   `superseded` in the manifest and make the replacement explicit in the newer document.
5. Cross-lane consistency: the entities below are canon. Lanes may *mention* them but each lane *owns* only the facts it states in
   its own files; never restate another lane's specific numbers. Do not contradict canon.
6. No instructions to the reader about AI, training or questions. No "Q:" / "A:" structure. Nothing optimised for extraction.

## Canon

**Korvane Cold Chain Sp. z o.o.** — refrigerated (reefer) container logistics and cold-chain monitoring. Founded 2011 in Gdynia, Poland by
Marta Kowarzik and Dieter Ahlbeck-Roe; registered office ul. Łużycka 14, 81-537 Gdynia; KRS 0000488213; NIP 586-217-40-93; ~340 employees
(end of 2024). Depots: Gdynia (HQ, "GDY-1"), Hamburg-Waltershof ("HAM-2"), Rotterdam-Maasvlakte ("RTM-3"), Gothenburg-Arendal ("GOT-4"),
and a small trans-shipment yard at Kłaipėda ("KLJ-5", opened 2023). Internal nicknames: GDY-1 = "the Mothership", HAM-2 = "Waltershof", the
night shift = "the owls".

**Products and systems**
- *KC-Tag 4* — battery temperature/humidity/door-open logger for reefer containers (successor of KC-Tag 3, end-of-life 2023).
- *KC-Hub 200* — the in-yard gateway the tags report to.
- *Tidewalk* — Korvane's dispatch and monitoring web app (v6.x in 2024; v7 beta 2025). Public API at `api.tidewalk.korvane.example`.
- *ColdLedger* — internal billing/invoicing system (a heavily customised ERP module; everyone hates it).

**People (use consistently; invent more, with first and last names, as needed)**
Marta Kowarzik (CEO), Dieter Ahlbeck-Roe (COO), Priya Venkataraman-Holt (CFO, joined 2019), Tomasz Wrzesiński (CTO), Ines Baptista-Okafor
(Head of People & Culture), Jarek Lindqvist (Head of Operations, Gdynia), Hanneke Dijkgraaf (Depot Manager, Rotterdam), Olek Maraszek
(Senior Dispatcher, night shift), Sofia Brandão (Legal Counsel), Kamil Zdunek (IT Support Lead), Benedikta Aaltonen (Quality & Compliance
Manager), Rafał Nowogrodzki (Head of Sales), Yusuf Demirkıran (Key Account Manager), Lotte Sandberg (Marketing), Gunnar Halvorsrud (Fleet &
Maintenance Lead).

**Customers (invent contract specifics yourself; only mention, do not clash):** Fruttoria Baltica (citrus importer), Nordkjøl Seafood AS,
Helvetia Pharma Logistik GmbH (pharma, GDP audited), Samsø Berry Co-op, Oakhaven Dairy Ltd, Brunnen & Sohn Frischdienst, Lumikko Foods Oy.

**Suppliers/partners:** Thermaglide Reefer Units BV (refrigeration units), Voltrac Battery Systems, Arendal Port Services, Seaboard Insurance
Brokers, Lintu Cloud Hosting, GreyLine Customs Agency.

**Timeline anchors:** 2011 founding · 2016 first pharma contract · 2019 CFO joins, ColdLedger go-live · 2021 Tidewalk v5 · March 2022
Rotterdam depot opens · 2023 KC-Tag 3 end-of-life, Kłaipėda yard · 1 Feb 2024 new employee handbook v4 · 14 June 2024 "the Oakhaven
incident" (a 9-hour temperature excursion on a dairy load) · autumn 2024 ISO 22000 recertification · 2025 Tidewalk v7 beta.

**Money & units:** PLN and EUR (state which), metric units, dates as the writer would (some DD.MM.YYYY, some "3rd of May", some ISO).

## Format menu (front matter `out:` extension)

pdf (text, long) · pdf with `scan: true` (rendered as a noisy scan: tests OCR) · docx · doc (legacy Word) · rtf · odt · xlsx (several sheets) ·
xls (legacy) · pptx · html (a page as exported from a CMS, with nav/footer noise) · csv (exports, logs) · txt (plain notes, minutes) · md
(runbooks, READMEs) · eml (raw e-mail with headers) · json (an API payload/chat export). See `scripts/corpus_build.py` for the source
syntax. Avoid `[[chart ...]]` inside rtf. Charts are allowed in pdf/docx/pptx(no)/odt; give every chart a caption and also a
`kind: chart` manifest fact whose value is a number printed on the chart image (the app may not be able to read it — that is a
finding, not a flaw in the corpus).
