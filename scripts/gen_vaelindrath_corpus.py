"""Generate the Vaelindrath Concord corpus: 104 rich fictional files.

Every accepted extension (from finetune_studio.data.parsers.PARSERS) gets
multiple files, each packed with unique fictional facts (names, numbers,
dates) that a base model could never know — so suite answers prove learning.
Ground-truth facts live in facts.json for later judging.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path("/tmp/vael")
OUT.mkdir(exist_ok=True)

rng = random.Random(20260918)

# ── The fictional world ────────────────────────────────────────────────
# The Vaelindrath Concord: 9 city-states on the twin continents Osthelm and
# Drevmora, bound by the Ember Accords (Year 411 SR). Calendar: Sevric Reckoning.
HOUSES = [
    ("Vaal-Rhunne", "Osthelm", "salted glassworks", 412000, "Archon Maribel Vaal",
     "the Tidewrit Charter (602)", "brine-glass", "Cinderlight Basin"),
    ("Drevmoor-Kaelin", "Drevmora", "bog-iron forges", 288500, "Archon Tövis Kaelin",
     "the Marrow Accord (733)", "ember-forged steel", "Hollowroot Marsh"),
    ("Cerune-Spathi", "Osthelm", "glass-bell foundries", 96000, "Archon Lirien Spathi",
     "the Silent Toll (561)", "resonant bell-glass", "Spirefen"),
    ("Norlanth-Obey", "Drevmora", "moth-silk looms", 173900, "Archon Sorcha Obey",
     "the Loom Compact (818)", "luminescent moth-silk", "Grellock Hills"),
    ("Quenness-Vail", "Osthelm", "salt-pearl diving", 64100, "Archon Perrin Vail",
     "the Drowned Ledger (402)", "salt-pearl writs", "Bellquay Shoals"),
    ("Ashvern-Tulloch", "Drevmora", "cinder-charcoal kilns", 210300,
     "Archon Bram Tulloch", "the Kiln Oath (944)", "everburning charcoals",
     "Kilnrow Terrace"),
    ("Murkfen-Cadwal", "Osthelm", "peat-whiskey distilleries", 88700,
     "Archon Odile Cadwal", "the Peat Covenant (877)", "black-whiskey casks",
     "Murkfen Bottoms"),
    ("Strande-Voight", "Drevmora", "wreck-salvage courts", 134800,
     "Archon Ilse Voight", "the Salvage Lex (699)", "writ-of-flotsam seals",
     "Drownstone Quay"),
    ("Highmere-Aldwin", "Osthelm", "stone-barge masonry", 402900,
     "Archon Corin Aldwin", "the Masonic Truce (585)", "quarried margin-stone",
     "The Highmere Steps"),
]
# Dates/years kept distinctive on purpose (never real-world-plausible).
EVENTS = [
    ("the Emberfall", 648, "a meteoric salt-shower over Osthelm; 31 ships fused into the Saltcliff Wrecks"),
    ("the Drevmorian Cough", 701, "a bog-spore plague; quelled by moth-silk face-veils from Norlanth-Obey"),
    ("the Bell Quarrel", 561, "Cerune-Spathi rang the Foundry Toll 81 times; Drevmoor-Kaelin answered with forge-silence for a year"),
    ("the Ninefold March", 893, "all nine archons walked the Highmere Steps barefoot; sealed the Concord's second charter"),
    ("the Grellock Slide", 954, "a hillside collapse buried 2,400 moth-looms; rebuilt in 40 days by Tulloch kilnmen"),
    ("the Quiet Fleet", 612, "thirteen smuggler ships scuttled themselves rather than pay the Drowned Ledger tithe"),
    ("the Saltmoot", 411, "the founding assembly where the Ember Accords were voted 6-3 (Murkfen-Cadwal dissented twice)"),
    ("the Ledgerfire", 1002, "Quenness-Vail's archive burned; 4,100 salt-pearl writs re-carved from memory by divers"),
]
TRADE_GOODS = [
    ("brine-glass", "Vaal-Rhunne", 44), ("embersteel", "Drevmoor-Kaelin", 91),
    ("bell-bronze", "Cerune-Spathi", 37), ("moth-silk", "Norlanth-Obey", 265),
    ("salt-pearls", "Quenness-Vail", 480), ("kiln-charcoal", "Ashvern-Tulloch", 12),
    ("black-whiskey", "Murkfen-Cadwal", 58), ("flotsam-sealed salvage", "Strande-Voight", 74),
    ("margin-stone", "Highmere-Aldwin", 19),
]
RANKS = ["Archon", "Tidewright", "Fogmarshal", "Salt-Speaker", "Loom-Scribe",
         "Kilnwarden", "Ledger-Keeper", "Salvage-Captain", "Mason-Prime"]

facts: list[dict] = []


def fact(domain: str, q: str, a: str, src: str) -> None:
    facts.append({"domain": domain, "q": q, "a": a, "src": src})


def prose_seed(title: str, body_lines: list[str]) -> str:
    return f"{title}\n\n" + "\n\n".join(body_lines)


# Content builders per fact-domain so every file carries unique, checkable data.

def house_profile(i: int) -> str:
    h = HOUSES[i % 9]
    year = 411 + i * 37
    txt = (
        f"House Ledger: {h[0]}\n\n"
        f"Continent: {h[1]}. Charter trade: {h[2]}. Recorded population after the "
        f"census of Year {year}: {h[3]:,} souls. Archon (since {year - 40 - i}): {h[4]}.\n\n"
        f"The house owes its standing charter to {h[5]}, renewed at every Saltmoot. "
        f"Signature craft: {h[6]}. Seat of the house: {h[7]}.\n\n"
        f"House colors: {['ash-grey','verdigris','bone-white','rust-red','ocean-slate'][i % 5]} "
        f"quartered with {['a双 tide-spiral','moth-wings split','a bell cracked','three pearls sunk'][i % 4]}.\n"
        f"Standing guard at the seat: {120 + i * 35} warded soldiers under a "
        f"{RANKS[(i + 2) % len(RANKS)]}{'' if (i + 2) % 10 == 0 else ''}. Annual tithe to the Concord vault: "
        f"{400 + i * 90} crowns. The house seal is cut into {9 + i} margin-stones."
    )
    fact("houses", f"What is the recorded population of {h[0]}?",
         f"{h[3]:,}", txt[:60])
    fact("houses", f"Which trade is {h[0]}'s charter craft?", h[2], txt[:60])
    fact("houses", f"Who is the archon of {h[0]}?", h[4], txt[:60])
    fact("houses", f"Where is the seat of {h[0]}?", h[7], txt[:60])
    return txt


def event_saga(i: int) -> str:
    name, year, desc = EVENTS[i % len(EVENTS)]
    txt = (
        f"Saga annal: {name}\n\n"
        f"In the year {year} of the Sevric count, {name} began. {desc[0].upper()}{desc[1:]}.\n\n"
        f"The annal records {60 + i * 13} casualties among warded soldiers and "
        f"{2 + i} writs of restitution filed at the Drowned Ledger. Eyewitnesses "
        f"from {HOUSES[i % 9][0]} swore the sky over {HOUSES[(i + 4) % 9][7]} turned "
        f"{['copper','ultramarine','bone-yellow'][i % 3]} for {3 + i} days.\n\n"
        f"Aftermath: the {name.title()} Compensation of {year + 11} fixed damages at "
        f"{1100 + i * 350} crowns, paid half in bell-bronze and half in salt-pearls. "
        f"A {RANKS[(i + 5) % len(RANKS)]} named {( 'Sereth','Ombric','Vaniel','Kolvast')[i % 4]} "
        f"kept the annal; their copy survived the Ledgerfire of 1002 in a peat-sealed cask."
    )
    fact("events", f"In what year did {name} occur?", f"{year}", txt[:60])
    fact("events", f"How many casualties did {name} cause among warded soldiers?",
         f"{60 + i * 13}", txt[:60])
    try:
        fact("events", f"What was the Compensation fixed after {name}?",
             f"{1100 + i * 350} crowns", txt[:60])
    except IndexError:
        pass
    return txt


def trade_ledger(i: int) -> str:
    good, house, price = TRADE_GOODS[i % len(TRADE_GOODS)]
    txt = (
        f"Ledger of the {good.title()} Route (annal {900 + i})\n\n"
        f"Sole source: {house} {HOUSES[[h[0] for h in HOUSES].index(house)][7]}. "
        f"Standard price: {price} crowns per writ-measure. Quarter-masters take "
        f"{3 + i % 5}% commission at Bellquay Shoals.\n\n"
        f"Caravan manifest {900 + i}: {180 + i * 7} casks dispatched, "
        f"{12 + i * 2} lost to the Drowned Quay surf, {180 + i * 7 - 12 - i * 2} delivered. "
        f"Escort: {RANKS[(i + 3) % len(RANKS)]} Bastiel Corrane, {18 + i} crossbows.\n\n"
        f"Concord seal-tax: {2 + i * 1} crowns per cask, remitted to the Ember Vault. "
        f"Warehousing at the Highmere Steps costs {9 + i} crowns per season per cask."
    )
    fact("trade", f"What is the standard price of {good} per writ-measure?",
         f"{price} crowns", txt[:60])
    fact("trade", f"Which house is the sole source of {good}?", house, txt[:60])
    return txt


def rank_charter(i: int) -> str:
    r = RANKS[i % len(RANKS)]
    txt = (
        f"Charter of the {r}s of the Vaelindrath Concord\n\n"
        f"A {r} is sworn at the age of {24 + i * 3} and serves {20 + i * 2} years. "
        f"Their oath-stone is carved from {['margin-stone','bell-bronze','embersteel'][i % 3]}. "
        f"The Concord pays a {r} {150 + i * 25} crowns per annum plus {2 + i} measures of black-whiskey.\n\n"
        f"Duties include: witnessing writs at the Drowned Ledger, keeping the\n"
        f"annal of {EVENTS[i % len(EVENTS)][0]}, and escorting the Saltmoot each\n"
        f"founding-season. A {r} may not hold more than {30 + i * 5} acres of\n"
        f"chartered land. Dismissal requires the counterglyphs of two archons."
    )
    fact("ranks", f"At what age is a {r} sworn in?", f"{24 + i * 3}", txt[:60])
    fact("ranks", f"How many years does a {r} serve?", f"{20 + i * 2}", txt[:60])
    return txt


def misc_lore(i: int) -> str:
    creatures = ["the Vael-stork", "the Drevmorian howler-toad", "the Osthelm glasswyr"]
    beasts = [("the Vael-stork", "nine-foot marsh bird that swallows lit coals"),
              ("the howler-toad", "sings in fifths during the Grellock fog"),
              ("the glasswyr", "eats sand and excretes brittle glass thread")]
    b = beasts[i % 3]
    txt = (
        f"Bestiary scroll {511 + i}: {b[0]}\n\n"
        f"{b[0].title()} — {b[1]}. Length {1 + i} ells; wingpin {2 + i} cubits; "
        f"lifespan {14 + i * 11} years in the wild. Nests at {HOUSES[i % 9][7]} "
        f"and is sacred to {HOUSES[(i + 2) % 9][0]}.\n\n"
        f"A fully-grown {b[0].lower()} sells for {35 + i * 9} crowns; its\n"
        f"{['eggs','quills','molt-glass'][i % 3]} fetch {12 + i * 3} crowns the dozen.\n"
        f"Feeding: {['salted eel','kiln-charcoal','moth-silk trim'][i % 3]} twice daily. "
        f"The Concord permits {1 + i % 4} nesting pairs per fief."
    )
    fact("bestiary", f"How long does a wild {b[0]} live?",
         f"{14 + i * 11} years", txt[:60])
    fact("bestiary", f"What does the Concord permit per fief of {b[0]}?",
         f"{1 + i % 4} nesting pairs", txt[:60])
    return txt


def treaty_record(i: int) -> str:
    h1, h2 = HOUSES[i % 9], HOUSES[(i + 3) % 9]
    txt = (
        f"Treaty record {i + 1}: {h1[0]} and {h2[0]}\n\n"
        f"Signed at {h1[7]} in {520 + i * 41}, witnessed by {RANKS[i % len(RANKS)]} "
        f"Belmara of the Steps. Terms: {h1[0]} cedes {2 + i} furlongs of\n"
        f"tidewright marsh; {h2[0]} remits {800 + i * 120} crowns of\n"
        f"bell-bronze debt. The border-stone is carved with both house glyphs\n"
        f"and a {['moth','tide','bell'][i % 3]} motif.\n\n"
        f"Ratified 6-3, with {HOUSES[(i + 5) % 9][0]} abstaining. Reaffirmed\n"
        f"at the Saltmoot of {520 + i * 41 + 9}. Copies: two, in ash-grey and\n"
        f"ocean-slate bindings, held at the Ember Vault and {h2[7]}."
    )
    fact("treaties", f"How many furlongs of tidewright marsh did {h1[0]} cede?",
         f"{2 + i}", txt[:60])
    fact("treaties", f"Where was the {h1[0]}-{h2[0]} treaty signed?", h1[7], txt[:60])
    return txt


DOMAIN_BUILDERS = [
    (house_profile, "houses"),
    (event_saga, "events"),
    (trade_ledger, "trade"),
    (rank_charter, "ranks"),
    (bestiary_scroll := misc_lore, "bestiary"),
    (treaty_record, "treaties"),
]

# Build the file plan: for each extension, N files cycling the domains.
EXTENSIONS_PLAIN = [".txt", ".md", ".markdown", ".log", ".py", ".js", ".ts",
                    ".jsx", ".tsx", ".css", ".csv", ".tsv", ".json", ".jsonl",
                    ".xml", ".yaml", ".yml", ".ini", ".cfg", ".conf"]
EXT_PLAIN = EXTENSIONS_PLAIN
EXT_BINARY = [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".odt",
              ".ods", ".odp", ".rtf", ".epub", ".eml", ".msg",
              ".png", ".jpg", ".tif", ".bmp", ".webp", ".gif",
              ".html", ".htm"]
ALL_EXTS = EXT_BINARY + EXT_PLAIN  # 13 binary + 20 plain = 33 ext keys


def to_plain_text(body: str, ext: str) -> str:
    """Wrap the lore body into the shape each text parser expects."""
    if ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        rows = ["item,house,price_crowns\n" if ext == ".csv" else "item\thouse\tprice_crowns\n"]
        for g, hname, p in TRADE_GOODS:
            rows.append(f"{g}.{hname}.{p}\n".replace(".", sep))
        for i in range(4):
            rows.append(f"{DOMAIN_BUILDERS[i % 6][1]}-entry-{i}.{EXAMPLE_HOUSE}.{9 + i}\n".replace(".", sep))
        return "".join(rows)
    if ext == ".json":
        h = HOUSES[0]
        return json.dumps({
            "codex": "Vaelindrath Concord",
            "trade_good": TRADE_GOODS[0][0], "source_house": TRADE_GOODS[0][1],
            "price": TRADE_GOODS[0][2],
            "houses": [{"name": x[0], "continent": x[1], "craft": x[2],
                        "population": x[3], "archon": x[4]} for x in HOUSES],
        }, indent=2)
    if ext == ".jsonl":
        lines = []
        for i, h in enumerate(HOUSES):
            lines.append(json.dumps({"record": i, "house": h[0],
                                     "population": h[3], "craft": h[2]}))
        return "\n".join(lines) + "\n"
    if ext in (".yaml", ".yml"):
        lines = ["# Vaelindrath Concord trade register"]
        for i, (good, hname, price) in enumerate(TRADE_GOODS[:6]):
            lines += [f"{good}:", f"  source_house: {hname}",
                      f"  price_crowns: {price}"]
        return "\n".join(lines) + "\n"
    if ext in (".ini", ".cfg", ".conf"):
        return (
            "[concord]\nfounding_year = 411\nseat = Highmere Steps\n"
            "founding_vote = 6-3\n\n"
            "[trade]\nmoth_silk_price = 265\nembersteel_price = 91\n"
            "salt_pearl_price = 480\n\n"
            "[events]\nfounding_event = the Saltmoot\nfirst_dissent = Murkfen-Cadwal\n"
        )
    if ext == ".xml":
        return (
            "<concord>\n  <founding year=\"411\" vote=\"6-3\"/>\n"
            "  <house name=\"Vaal-Rhunne\" continent=\"Osthelm\" population=\"412000\">\n"
            "    <craft>salted glassworks</craft>\n  </house>\n"
            "  <house name=\"Drevmoor-Kaelin\" continent=\"Drevmora\" population=\"288500\">\n"
            "    <craft>bog-iron forges</craft>\n  </house>\n"
            "  <event name=\"the Bell Quarrel\" year=\"527\"/>\n"
            "</concord>\n"
        )
    if ext in (".py", ".js", ".ts", ".jsx", ".tsx"):
        comment = "#" if ext == ".py" else "//"
        return (
            f"{comment} Vaelindrath Concord annal index (codified {731})\n"
            f"{comment} Total tollerooms in the Concord: 144\n"
            f"{comment} Salt-parliament sits 40 days per founding-season\n"
            f"const DEFAULT_FOUNDING = 411;{comment} the Saltmoot\n"
            f"const NINE_HOUSES = {json.dumps([h[0] for h in HOUSES])};\n"
            f"{comment} Moth-silk fetches 265 crowns at Bellquay Shoals\n"
            f"function tolleroomCount(){{ return 144; }}\n"
        )
    if ext == ".xls":
        import xlwt
        buf = io.BytesIO()
        wb = xlwt.Workbook(); ws = wb.add_sheet("ledger")
        ws.write(0, 0, "item"); ws.write(0, 1, "house"); ws.write(0, 2, "price_crowns")
        for r, (g, hname, p) in enumerate(TRADE_GOODS, start=1):
            ws.write(r, 0, g); ws.write(r, 1, hname); ws.write(r, 2, p)
        ws.write(len(TRADE_GOODS) + 1, 0, f"annal {idx}")
        ws.write(len(TRADE_GOODS) + 1, 1, "Saltmoot vote 6-3")
        ws.write(len(TRADE_GOODS) + 1, 2, 4100)
        wb.save(buf); return buf.getvalue()
    if ext == ".css":
        return (
            "/* Annotated stylesheet of the Concord's charters (copy 3).\n"
            "   Binding color: ash-grey. Vellum: Drevmorian goat.\n"
            "   Page size: 31 lines x 44 cubits. */\n"
            ".concord-charter { binding: asc-facing; pages: 88; }\n"
            ".tide-glyph { stroke: ultramarine; }\n"
        )
    if ext in (".html", ".htm"):
        return (
            "<html><head><title>Concord gazetteer</title></head><body>\n"
            "<h1>The Vaelindrath Concord</h1>\n"
            "<p>The twin capitals are Highmere Steps (Osthelm) and Hollowroot Marsh (Drevmora).</p>\n"
            "<p>The Concord seals 4,100 writs per founding-season; the\n"
            "Ledgerfire destroyed 4,100 salt-pearl writs in 1002.</p>\n"
            "<p>Black-whiskey from Murkfen-Cadwal is barreled in peat-sealed casks.</p>\n"
            "</body></html>\n"
        )
    if ext == ".log":
        i = 0
        lines = ["=== Concord dispatch log, annal 872 ==="]
        for ev in EVENTS[:5]:
            lines.append(f"[{ev[1]}] {ev[0]}: {ev[2][:60]}...")
        lines.append("[872] writ-count check: 4,100 seals audited; 12 missing.")
        return "\n".join(lines) + "\n"
    if ext == ".ini":
        return (
            "[archons]\nVaRhun = Maribel Vaal\nDKaelin = Tövis Kaelin\n"
            "[numbers]\nsaltmoot_year = 411\nledgerfire_writs = 4100\n"
        )
    if ext == ".cfg":
        return (
            "# concord.cfg\nvoting = 6-3 majority\nsalt_tax = 2 crowns per cask\n"
            "founding_season = 40 days\n"
        )
    if ext == ".conf":
        return (
            "# tide-ward daemon config (Concord annal copy)\n"
            "ward_count = 9\nfounding = 411\nledger_seals_annual = 4100\n"
        )
    return body


def build_binary(ext: str, body: str, idx: int) -> bytes:
    """Build a real binary doc for the optional-deps extensions."""
    import io
    if ext == ".xls":
        import xlwt
        buf = io.BytesIO()
        wb = xlwt.Workbook(); ws = wb.add_sheet("ledger")
        ws.write(0, 0, "item"); ws.write(0, 1, "house"); ws.write(0, 2, "price_crowns")
        for r, (g, hname, p) in enumerate(TRADE_GOODS, start=1):
            ws.write(r, 0, g); ws.write(r, 1, hname); ws.write(r, 2, p)
        ws.write(len(TRADE_GOODS) + 1, 0, f"annal {idx}")
        ws.write(len(TRADE_GOODS) + 1, 1, "Saltmoot vote 6-3")
        ws.write(len(TRADE_GOODS) + 1, 2, 4100)
        wb.save(buf); return buf.getvalue()
    if ext in (".html", ".htm"):
        html = (
            "<html><head><title>Concord annal</title></head><body>\n"
            "<h1>Vaelindrath annal</h1>\n"
            + body.replace("\n\n", "</p><p>").replace("\n", "<br/>")
            + "</p></body></html>\n"
        )
        return html.encode("utf-8")
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"):
        # OCR fact-card: render the lore as an image so tesseract can read it.
        from PIL import Image, ImageDraw, ImageFont
        W, H = 1240, 900
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)
        try:
            f_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 34)
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
        except OSError:
            f_big = f = ImageFont.load_default()
        y = 60
        d.text((50, y), f"Vaelindrath Concord (card {idx})", font=f_big, fill="black"); y += 70
        for ln in body.splitlines()[:26]:
            if ln.strip():
                d.text((50, y), ln[:95], font=f, fill="black"); y += 32
        buf = io.BytesIO()
        fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".tif": "TIFF",
               ".tiff": "TIFF"}.get(ext, "PNG")
        img.save(buf, format=fmt)
        return buf.getvalue()
    if ext in (".eml", ".msg"):
        # .eml is RFC-822 text; .msg wrapped as EML keeps facts (server parser
        # handles both via the email module).
        from email.message import EmailMessage
        m = EmailMessage()
        m["From"] = f"annalist{idx}@concord.vael"
        m["To"] = "vault@concord.vael"
        m["Subject"] = f"Concord annal {idx}: {body.splitlines()[0][:40]}"
        m.set_content(body)
        return m.as_bytes()
    if ext == ".docx":
        from docx import Document
        d = Document()
        d.add_heading("Vaelindrath Concord house annal", level=1)
        d.add_paragraph(body[:400])
        d.add_paragraph(f"Annal index {idx}. Copy certified by a Ledger-Keeper of Drevmora.")
        d.add_paragraph(f"Total tollerooms 144; Saltmoot vote 6-3.")
        buf = io.BytesIO(); d.save(buf); return buf.getvalue()
    if ext == ".xlsx":
        from openpyxl import Workbook
        wb = Workbook(); ws = wb.active; ws.title = "ledger"
        ws.append(["good", "house", "price_crowns"])
        for g, hname, p in TRADE_GOODS:
            ws.append([g, hname, p])
        ws.append(["bell-bronze escort", "Cerune-Spathi", "37"])
        ws.append(["margin-stone road", "Highmere-Aldwin", "19"])
        ws.append(["salt-pearl writs", "Queness-Vail", "480"])
        buf = io.BytesIO(); wb.save(buf); return buf.getvalue()
    if ext == ".pptx":
        from pptx import Presentation
        prs = Presentation()
        s1 = prs.slides.add_slide(prs.slide_layouts[0])
        s1.shapes.title.text = "The Vaelindrath Concord"
        s1.placeholders[1].text = "Annal brief, compiled in the year 731"
        s2 = prs.slides.add_slide(prs.slide_layouts[1])
        s2.shapes.title.text = "Key facts"
        tf = s2.placeholders[1].text_frame
        tf.text = "Founding: the Saltmoot of 411, vote 6-3"
        tf.add_paragraph().text = f"Population of Vaal-Rhunne: 412,000"
        tf.add_paragraph().text = "Moth-silk, most trade good value: 265 crowns per writ"
        tf.add_paragraph().text = "Ledgerfire, 1002: 4,100 writs lost"
        buf = io.BytesIO(); prs.save(buf); return buf.getvalue()
    if ext == ".pdf":
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        lines = body.splitlines()[:40]
        y = 780
        for ln in lines:
            c.drawString(50, y, ln[:110]); y -= 16
        c.showPage()
        c.setFont("Helvetica", 10)
        y = 780
        for ln in body.splitlines()[40:80] or [f"Annal index {idx}. Saltmoot vote 6-3. T khai toll rooms: 144."]:
            c.drawString(50, y, ln[:110]); y -= 16
        c.save()
        return buf.getvalue()
    if ext == ".doc":
        # Antiword/catdoc route is unavailable here; embed the lore into a
        # real .doc by writing RTF content with a .doc name? No — parsers
        # reject that. Instead: minimal OLE2 via LibreOffice conversion is
        # done post-hoc in the pipeline; here emit RTF bytes that ALSO
        # survive antiword (antiword reads RTF-header wrappers poorly).
        # Pragmatic call: patch a real .doc via soffice later; for now emit
        # an RTF payload (parsers fall back to text-strip, facts survive).
        rtf = r"{\rtf1\ansi\deff0 {\fonttbl{\f0 Georgia;}}\f0\fs24 "
        rtf += body.replace("\n", r"\par ").replace("\x00", "")
        rtf += "}"
        return rtf.encode("ascii", errors="replace")
    if ext == ".rtf":
        rtf = r"{\rtf1\ansi\deff0 {\fonttbl{\f0 Georgia;}}\f0\fs24 "
        rtf += body.replace("\n", r"\par ").replace("\x00", "")
        rtf += "}"
        return rtf.encode("ascii", errors="replace")
    if ext == ".epub":
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("mimetype", "application/epub+zip")
            z.writestr(
                "content.xhtml",
                "<html><body>" + body.replace("\n", "<br/>") + "</body></html>",
            )
        return buf.getvalue()
    if ext == ".ods":
        from odf.opendocument import OpenDocumentSpreadsheet
        from odf.table import Table, TableRow, TableCell
        from odf.text import P as OdfP
        doc = OpenDocumentSpreadsheet()
        table = Table(name="ledger")
        headers = ["item", "house", "price_crowns"]
        row = TableRow()
        for h in headers:
            cell = TableCell(valuetype="string"); cell.addElement(OdfP(text=h)); row.addElement(cell)
        table.addElement(row)
        for g, hname, p in TRADE_GOODS:
            row = TableRow()
            for v in (g, hname, str(p)):
                cell = TableCell(valuetype="string"); cell.addElement(OdfP(text=v)); row.addElement(cell)
            table.addElement(row)
        doc.spreadsheet.addElement(table)
        buf = io.BytesIO(); doc.save(buf); return buf.getvalue()
    if ext == ".odp":
        # odfpy presentations require a masterpage; simplest valid path is
        # copying the HTML body into a text-frame page via the pptx layout
        # upstream. Here we emit a minimal but valid ODP through raw XML.
        from odf.opendocument import OpenDocumentPresentation
        from odf.style import MasterPage, PageLayout, PageLayoutProperties
        from odf.draw import Frame, TextBox as DrawTextBox, Page as DrawPage
        from odf.text import P as OdfP
        doc = OpenDocumentPresentation()
        pl = PageLayout(name="PL1")
        pl.addElement(PageLayoutProperties(pagewidth="28cm", pageheight="21cm",
                                           printorientation="landscape"))
        doc.automaticstyles.addElement(pl)
        mp = MasterPage(name="Standard", pagelayoutname=pl)
        doc.masterstyles.addElement(mp)
        page = DrawPage(masterpagename=mp, name="annal")
        frame = Frame(width="20cm", height="12cm", x="1cm", y="1cm")
        tb = DrawTextBox()
        tb.addElement(OdfP(text=f"Vaelindrath Concord annal {idx}"))
        tb.addElement(OdfP(text=body[:300]))
        frame.addElement(tb); page.addElement(frame)
        doc.presentation.addElement(page)
        buf = io.BytesIO(); doc.save(buf); return buf.getvalue()
    if ext == ".odt":
        from odf.opendocument import OpenDocumentText
        from odf.text import P
        doc = OpenDocumentText()
        doc.text.addElement(P(text=f"Vaelindrath Concord annal {idx}"))
        doc.text.addElement(P(text=body[:400]))
        buf = io.BytesIO(); doc.save(buf); return buf.getvalue()
    return b""


EXAMPLE_HOUSE = "Vaal-Rhunne"

# ── build every file ─────────────────────────────────────────────────
domain_idx = 0
built = 0
manifest = []
# 33 ext-keys; cycle files per ext. Total: binary 3 each (39) + plain 3-4 each (65)  -> 104
for ext in EXT_BINARY:
    per_ext = 3
    if ext in (".jpg", ".tif"):
        per_ext = 4  # + .jpeg / .tiff alias file so every registry key has a file
    if ext == ".png":
        per_ext = 5  # + .jpeg + .tiff aliases
    for n in range(per_ext):
        domain_idx += 1
        builder, domain = DOMAIN_BUILDERS[domain_idx % 6]
        body = builder(domain_idx)
        fname = f"vael_{domain}_{domain_idx:03d}{ext}"
        data = build_binary(ext, body, domain_idx)
        (OUT / fname).write_bytes(data)
        manifest.append({"file": fname, "bytes": len(data), "domain": domain})
        built += 1
        # Write registry alias files (.jpeg for .jpg, .tiff for .tif)
        if ext == ".jpg" and n == 0:
            (OUT / f"vael_{domain}_{domain_idx:03d}.jpeg").write_bytes(data)
            manifest.append({"file": f"vael_{domain}_{domain_idx:03d}.jpeg", "bytes": len(data), "domain": domain})
            built += 1
        if ext == ".tif" and n == 0:
            (OUT / f"vael_{domain}_{domain_idx:03d}.tiff").write_bytes(data)
            manifest.append({"file": f"vael_{domain}_{domain_idx:03d}.tiff", "bytes": len(data), "domain": domain})
            built += 1

for ext in EXT_PLAIN:
    per = 4 if ext in (".txt", ".md") else 3
    for n in range(per):
        domain_idx += 1
        builder, domain = DOMAIN_BUILDERS[domain_idx % 6]
        body = builder(domain_idx)
        if ext == ".md":
            body = f"# {body.splitlines()[0]}\n\n" + "\n\n".join(body.splitlines()[2:])
        fname = f"vael_{domain}_{domain_idx:03d}{ext}"
        (OUT / fname).write_text(to_plain_text(body, ext), encoding="utf-8")
        manifest.append({"file": fname, "bytes": (OUT / fname).stat().st_size, "domain": domain})
        built += 1

print(f"built {built} files in {OUT}")
print("extensions covered:", len({m['file'].rsplit('.', 1)[1] for m in manifest}))
(OUT / "facts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
print("facts logged:", len(facts))
