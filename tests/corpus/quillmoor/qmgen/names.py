"""Name and phrase pools for the Quillmoor world. Everything is invented; the generators are seeded and de-duplicated."""
from __future__ import annotations

import random

FIRST = ["Aino", "Akosua", "Alarik", "Alma", "Amara", "Anouk", "Ansel", "Arvid", "Astrid", "Aurelio", "Bastien", "Beatriz", "Bodil", "Bram", "Caius", "Calla", "Casimir", "Cleo", "Dagny", "Dario", "Delphine", "Desmond", "Dorota", "Eamon", "Edda", "Eilidh", "Elio", "Elske", "Emeric", "Esme", "Evander", "Fenna", "Ferdinand", "Fiorella", "Folke", "Freya", "Gaspard", "Greta", "Gwendal", "Halvard", "Hana", "Hedda", "Idris", "Ilona", "Imogen", "Isak", "Ismay", "Jarvis", "Jessamy", "Joakim", "Jolene", "Jonah", "Junia", "Kalle", "Kasimir", "Keziah", "Kirsi", "Lazlo", "Leocadia", "Lorcan", "Lucja", "Maelle", "Magnus", "Malin", "Marek", "Marisol", "Mathis", "Milou", "Mirela", "Morwenna", "Nadia", "Nikolai", "Noor", "Odalys", "Olwen", "Orla", "Osric", "Ottilie", "Pavlina", "Petronel", "Philippa", "Quentin", "Radka", "Rafferty", "Rhosyn", "Ricarda", "Rohan", "Rosalind", "Ruairi", "Saskia", "Selwyn", "Senna", "Sigrun", "Silas", "Solveig", "Stellan", "Sunniva", "Tamsin", "Tancredi", "Teodor", "Thandiwe", "Tiago", "Tilde", "Tobiah", "Ulla", "Valdemar", "Venla", "Vidar", "Viveka", "Wilhelmina", "Wynn", "Xanthe", "Yara", "Yorick", "Ysolde", "Zbigniew", "Zelda", "Zofia", "Ambrose", "Bettina", "Corwin", "Dalia", "Elowen", "Fabienne", "Gideon", "Henrike", "Ignatius", "Jorunn", "Kerensa", "Leopold", "Mafalda", "Nevin", "Oona", "Perpetua", "Quilla", "Romilly", "Severin", "Thekla", "Ulric", "Verity", "Wolfram", "Yseult", "Zephyrine"]

_S1 = ["var", "os", "kell", "bran", "tor", "wick", "hal", "mor", "ash", "fen", "dar", "cal", "ost", "bel", "gry", "lor", "tam", "ven", "quil", "rath", "sel", "brae", "kin", "dun", "ard", "ell", "tre", "wyn"]
_S2 = ["row", "ick", "en", "more", "by", "ham", "ley", "ton", "bourne", "sett", "stead", "hurst", "wick", "holt", "combe", "ridge", "dale", "field", "more", "lake", "stow", "ard", "ell"]
_S3 = ["", "", "", "s", "e", "er", "son", "ey"]


def surname(rng: random.Random) -> str:
    return (rng.choice(_S1) + rng.choice(_S2) + rng.choice(_S3)).capitalize()


PLACE_A = ["Harrow", "Tessal", "Bran", "Vell", "Orrin", "Cadd", "Marrow", "Quell", "Sarn", "Ebb", "Gull", "Skerr", "Wrack", "Kittle", "Fallow", "Thrum", "Mallow", "Corrie", "Brack", "Dunlin", "Selk", "Tarn", "Hollin", "Rook", "Wren", "Lanyard", "Cobble", "Tiller", "Haddo", "Spindle", "Ravel", "Moss", "Brine", "Kelp", "Shoal", "Weir", "Lode", "Pell", "Cairn"]
PLACE_B = ["fen", "wick", "holm", "by", "ness", "mere", "strand", "haven", "ford", "sound", "haugh", "gill", "wold", "head", "bay", "scar", "voe", "hope", "quay"]


def place(rng: random.Random) -> str:
    return rng.choice(PLACE_A) + rng.choice(PLACE_B)


ISLANDS = ["Harrowmere", "Tessaly Holm", "Brannoch", "Vellisk", "Orrin Skerry", "Caddow", "Sarnhope", "Wrackenholm",
           "Gullstrand", "Kittlebay", "Fallowvoe", "Quellhaven"]

SITE_KINDS = [  # kind, capacity unit, capacity range, asset type mix
    ("Tidal Station", "MW", (4.0, 96.0), ["TRB", "TRF", "GEN", "VLV"]),
    ("Water Treatment Works", "m³/day", (6000, 145000), ["PMP", "MEM", "UVR", "VLV", "CMP"]),
    ("Pumping Station", "m³/h", (180, 4800), ["PMP", "VLV", "TRF", "GEN"]),
    ("Desalination Plant", "m³/day", (2500, 61000), ["MEM", "PMP", "CMP", "VLV", "UVR"]),
    ("Substation", "MVA", (12, 240), ["TRF", "GEN", "VLV"]),
    ("Reservoir", "million litres", (40, 2600), ["VLV", "PMP", "UVR"]),
]

ASSET_TYPES = {  # code: (name, rating unit, rating range, supplier category)
    "PMP": ("pump", "kW", (15, 900), "pumps"),
    "TRB": ("tidal turbine", "kW", (250, 2200), "turbines"),
    "MEM": ("membrane rack", "m³/h", (40, 640), "membranes"),
    "TRF": ("transformer", "kVA", (250, 12500), "electrical"),
    "VLV": ("actuated valve", "bar", (6, 40), "valves"),
    "CMP": ("air compressor", "kW", (11, 315), "compressed air"),
    "UVR": ("UV reactor", "m³/h", (60, 2400), "water quality"),
    "GEN": ("standby generator", "kVA", (60, 3300), "electrical"),
}

SUPPLIER_CATEGORIES = ["pumps", "turbines", "membranes", "electrical", "valves", "compressed air", "water quality",
                       "chemicals", "IT services", "civil works", "marine services", "safety equipment"]
SUPPLIER_WORDS = ["Pumpworks", "Hydraulics", "Fluid Systems", "Marine", "Electric", "Controls", "Membrane Technik",
                  "Valve Co", "Process Solutions", "Engineering", "Industrial Supply", "Chemie", "Instruments",
                  "Field Services", "Power Systems", "Filtration"]
SUFFIXES = ["Ltd", "GmbH", "AS", "BV", "Oy", "SA", "AB", "ApS", "Sp. z o.o.", "S.r.l."]
COUNTRIES = ["Norway", "Netherlands", "Germany", "Finland", "Denmark", "Poland", "Scotland", "Ireland", "Sweden",
             "Estonia", "Portugal", "Belgium", "Iceland", "Italy"]
BRANDS_BY_CATEGORY: dict[str, list[str]] = {}  # filled by the world from its suppliers

ROLES = ["Site Manager", "Shift Engineer", "Control Room Operator", "Electrical Technician", "Mechanical Fitter",
         "Process Chemist", "Asset Planner", "Reliability Engineer", "SCADA Engineer", "HSE Advisor", "Storekeeper",
         "Procurement Officer", "Contracts Manager", "Billing Analyst", "Customer Account Manager", "Data Analyst",
         "Marine Coordinator", "Laboratory Technician", "Instrument Technician", "Network Engineer", "IT Support Analyst",
         "Project Manager", "Commercial Lead", "Finance Business Partner", "Payroll Specialist", "HR Advisor",
         "Legal Counsel", "Quality Auditor", "Environmental Officer", "Training Coordinator", "Fleet Coordinator",
         "Duty Manager", "Hydrographer", "Civil Engineer", "Planning Engineer", "Commissioning Engineer"]

CERTS = ["HV Switching Authorisation level 2", "HV Switching Authorisation level 3", "Confined Space Entry (Medium Risk)",
         "Working at Height Supervisor", "Chlorine Handling Competence", "Marine Sea Survival (BOSIET)",
         "Lifting Operations Appointed Person", "Water Hygiene Card", "Permit-to-Work Issuer", "First Aid at Work",
         "ATEX Inspection Level 1", "Rope Access IRATA Level 2"]

ROOT_CAUSES = [
    "a worn mechanical seal on the discharge side", "kelp ingestion at the intake screen", "a firmware watchdog reset after a corrupted configuration push",
    "a failed cooling fan on the drive cabinet", "cavitation caused by a partly closed suction valve", "a cracked bearing housing",
    "loss of the 400 V auxiliary supply during switching", "a blocked differential pressure transmitter impulse line",
    "biofouling of the membrane feed spacer", "a mis-set overcurrent relay after maintenance", "salt-water ingress into a junction box",
    "an expired SIM card on the telemetry router", "a seized actuator gearbox", "operator error during a manual changeover",
    "a broken shear pin on the turbine yaw drive", "lightning-induced surge on the incoming feeder", "a leaking flange gasket",
    "an incorrect chemical dosing setpoint", "UV lamp sleeve fouling", "a stuck non-return valve flap", "an overheated cable termination",
    "rodent damage to a control cable", "a clogged strainer upstream of the dosing pump", "a failed battery bank in the UPS",
    "a corrupted historian database index", "frost damage to an exposed sample line", "vibration-induced loosening of anchor bolts",
    "a software update that reset the PID tuning", "an obstructed air intake on the generator", "a tripped residual current device",
]

ERROR_MEANINGS = [
    "telemetry hub unreachable for more than three polling cycles", "duplicate dispatch ticket detected", "operator session token expired",
    "historian write queue above the high-water mark", "checksum mismatch in an uploaded configuration bundle",
    "tariff lookup returned no matching rate band", "meter read outside the plausible range", "certificate on the field gateway expired",
    "scheduled job skipped because a previous run was still active", "pressure sensor reported a frozen value",
    "invoice batch rejected by the ledger export", "work order references an asset that is not in the register",
    "clock drift between gateway and server exceeded 90 seconds", "map tile cache rebuild failed", "SMS gateway refused the alert",
    "permit-to-work record locked by another user", "dosing setpoint outside the approved envelope", "vessel AIS feed stale",
    "report template missing a mandatory field", "database replica lagging behind the primary", "firmware image signature invalid",
    "customer account suspended for non-payment", "alarm flood suppression engaged", "remote valve command not acknowledged",
]

HAZARD_CLASSES = ["Class 8 (corrosive)", "Class 5.1 (oxidiser)", "Class 9 (miscellaneous)", "Class 6.1 (toxic)", "not classified"]
CHEM_KINDS = ["coagulant", "disinfectant", "antiscalant", "pH adjuster", "flocculant aid", "membrane cleaner", "dechlorination agent",
              "corrosion inhibitor"]
CHEM_A = ["Clari", "Floc", "Aqua", "Brine", "Hydro", "Scale", "Puro", "Ferro", "Calci", "Oxy", "Sulpha", "Mira", "Tetra", "Poly", "Nitro", "Lumi"]
CHEM_B = ["sol", "quent", "mix", "guard", "lite", "dose", "pure", "tek", "cide", "sperse", "clear", "max"]

SERVICE_A = ["Tide", "Brine", "Ledger", "Sluice", "Quay", "Harbour", "Weir", "Shoal", "Current", "Gauge", "Spring", "Ebb", "Neap", "Swell", "Drift", "Anchor", "Beacon", "Cobble", "Kelp", "Gull", "Lantern", "Mooring", "Ripple", "Surge", "Wake", "Fathom", "Breaker", "Crest", "Lagoon", "Estuary"]
SERVICE_B = ["line", "guard", "wake", "mark", "watch", "board", "link", "desk", "hub", "flow", "sense", "track", "keeper", "vault", "post", "scope"]

PROJECT_WORDS = ["Heronsgate", "Kittiwake", "Stonechat", "Ospreyfield", "Ternwater", "Saltmarsh", "Lapwing", "Curlew", "Sandpiper", "Merlin", "Petrel", "Shearwater", "Bittern", "Avocet", "Dunnock", "Goldcrest", "Firecrest", "Whimbrel", "Godwit", "Redshank", "Puffin", "Guillemot", "Fulmar", "Gannet", "Skua", "Dipper", "Wagtail", "Corncrake", "Linnet", "Twite", "Bunting", "Siskin", "Crossbill", "Treecreeper", "Nuthatch", "Wryneck", "Hobby", "Kestrel", "Harrier"]
PROJECT_STATUSES = ["design freeze", "tender evaluation", "detailed design", "site mobilisation", "commissioning",
                    "on hold pending funding", "handover", "feasibility", "procurement", "construction"]

BUSINESSES = ["Fish Cannery", "Dairy Co-op", "Ferry Terminal", "Distillery", "Seaweed Farm", "Hospital", "Primary School",
              "Hotel", "Cold Store", "Boatyard", "Bakery", "Brewery", "Laundry", "Leisure Centre", "Salmon Hatchery",
              "Abattoir", "Paper Mill", "Data Centre", "Golf Club", "Marina", "Care Home", "Bottling Plant", "Ice Works"]

VESSEL_A = ["Orrin", "Grey", "Silver", "Northern", "Quiet", "Brave", "Morning", "Salt", "Storm", "Kittle", "Tern", "Haddo"]
VESSEL_B = ["Lark", "Petrel", "Seal", "Skua", "Otter", "Heron", "Fulmar", "Gannet", "Dunlin", "Puffin", "Curlew", "Merlin"]

FILLER = {
    "corporate": [
        "This document forms part of the Quillmoor Utilities controlled document set and supersedes any uncontrolled copies.",
        "Readers should check the document library for the latest revision before relying on it.",
        "Figures are reported as at the end of the period unless stated otherwise.",
        "The information is provided for internal use and must not be shared with third parties without approval from Legal.",
        "Comments and corrections should be sent to the document owner through the usual change request route.",
        "As noted at the last quarterly review, the programme remains broadly on track despite the weather.",
        "We would like to thank the site teams for their patience while the new reporting templates bed in.",
        "Some of the historical records were migrated from the old asset system and may contain rounding differences.",
        "Where a value in this document conflicts with a signed contract, the contract prevails.",
        "A summary of open actions is maintained separately and is not reproduced here.",
    ],
    "engineering": [
        "Readings were taken with the unit at steady state after a warm-up of at least twenty minutes.",
        "Tolerances follow the manufacturer's installation manual unless the site has an approved deviation.",
        "Isolation must be proven dead before any intrusive work, in line with the permit-to-work procedure.",
        "Vibration trending is reviewed monthly by the reliability team.",
        "Spares holdings were reviewed against the criticality ranking during the last stocktake.",
        "Drawings referenced in this section are held in the engineering document management system.",
        "The control philosophy has not changed since the last revision of the functional design specification.",
        "Lubrication is carried out to the schedule in the computerised maintenance system, not to this sheet.",
    ],
    "legal": [
        "Nothing in this clause shall limit either party's liability for fraud or for death or personal injury caused by negligence.",
        "Headings are for convenience only and shall not affect the interpretation of this agreement.",
        "Any notice under this agreement shall be in writing and delivered to the address set out in Schedule 1.",
        "This agreement shall be governed by the laws stated in the general conditions of contract.",
        "No variation of this agreement shall be effective unless agreed in writing by authorised signatories of both parties.",
        "The Supplier shall maintain adequate insurance for the duration of the term.",
    ],
    "casual": [
        "anyone know if the canteen is open on saturday?", "will dig out the photos later", "thanks all, appreciated",
        "on it", "sorry, was on the boat all morning", "can we take this offline", "ok noted", "👍", "lol",
        "who has the keys for the blue van", "back online now", "coffee machine on level 2 is broken again",
    ],
    "hr": [
        "Managers are expected to apply this policy consistently and fairly.", "Questions about this policy should be raised with People Services.",
        "This policy does not form part of any employee's contract of employment and may be amended at any time.",
        "Breaches may be dealt with under the disciplinary procedure.", "Local agreements with recognised trade unions take precedence where they are more favourable.",
    ],
}

NEWSLETTER_TOPICS = ["the summer barbecue at the Mothershed", "the new car-sharing scheme", "a charity swim across the harbour",
                     "the retirement of a long-serving storekeeper", "the bike-to-work week", "the refurbished canteen",
                     "this year's photography competition", "the mental health first aiders", "the winter coat amnesty",
                     "a beach clean on the north shore", "the quiz night results", "new recycling bins in the depot"]
