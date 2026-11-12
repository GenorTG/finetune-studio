"""How each attribute is written and asked.

For every (entity type, attribute):
  col     -- the column / form label documents use
  ask     -- direct questions (training phrasing and the "verbatim" eval set)
  para    -- re-worded questions, never used for training (the paraphrase eval set)
  answer  -- the answer sentence used for training rows
  states  -- prose sentences that state the fact inside a report
Template fields: {k} entity key, {v} value as written, {t} type word, {s} site, {code} error code, {label} rule label.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    col: str
    ask: tuple[str, ...]
    para: tuple[str, ...]
    answer: str
    states: tuple[str, ...]


def S(col: str, ask, para, answer: str, states) -> Spec:
    return Spec(col, tuple([ask] if isinstance(ask, str) else ask), tuple([para] if isinstance(para, str) else para),
                answer, tuple([states] if isinstance(states, str) else states))


SPECS: dict[tuple[str, str], Spec] = {
    # ── sites ──
    ("site", "code"): S("Site code", "What is the site code of {k}?",
                        ["Which short identifier does Quillmoor use for {k} in its systems?", "What code is {k} filed under?"],
                        "The site code of {k} is {v}.",
                        ["Internally the works is referred to by its site code, {v}.", "{k} ({v}) is one of the group's operational sites."]),
    ("site", "island"): S("Island", "On which island is {k}?", ["Which of the Sarne Isles is home to {k}?"], "{k} is on {v}.",
                          ["The site lies on {v}, a short drive from the ferry terminal.", "{k} is located on {v}."]),
    ("site", "commissioned"): S("Commissioned", "When was {k} commissioned?", ["In what year did {k} start operating?"],
                                "{k} was commissioned in {v}.",
                                ["The plant was first commissioned in {v} and has been extended twice since.",
                                 "Commissioned in {v}, {k} is now one of the older assets in the portfolio."]),
    ("site", "capacity"): S("Rated capacity", "What is the rated capacity of {k}?",
                            ["How much can {k} handle at full output?", "What is the nameplate output of {k}?"],
                            "{k} has a rated capacity of {v}.",
                            ["Rated capacity is {v}, although the site rarely runs at full output for more than a few hours.",
                             "At design conditions {k} delivers {v}."]),
    ("site", "permit"): S("Permit no.", "What is the permit number of {k}?", ["Under which permit reference is {k} allowed to operate?"],
                          "{k} operates under permit {v}.",
                          ["Operation is authorised under environmental permit {v}; the permit conditions are reviewed annually.",
                           "The site's discharge and abstraction limits are set out in permit {v}."]),
    ("site", "muster_point"): S("Fire assembly point", "Where is the fire assembly point at {k}?",
                                ["In an evacuation of {k}, where should staff gather?"], "The fire assembly point at {k} is {v}.",
                                ["Fire assembly point: {v}."]),
    ("site", "telemetry_poll_s"): S("Poll interval", "What is the telemetry polling interval at {k}?",
                                    ["How often does the telemetry system query the outstations at {k}?"],
                                    "The telemetry at {k} polls every {v}.", ["Outstations are polled every {v}."]),
    ("site", "retention_days"): S("Historian retention", "How long does the historian at {k} retain data?",
                                  ["For how many days are process readings kept in the {k} historian?"],
                                  "The historian at {k} keeps data for {v}.", ["Historian data is retained for {v}."]),
    ("site", "manager"): S("Site manager", "Who is the site manager of {k}?", ["Who is in charge of {k}?"],
                           "The site manager of {k} is {v}.",
                           ["{v} has managed the site since the last reorganisation.", "Day-to-day responsibility for {k} sits with the site manager, {v}."]),
    ("site", "headcount"): S("Headcount", "How many people work at {k}?", ["How large is the workforce at {k}?"],
                             "{k} has a headcount of {v}.",
                             ["The site currently has a headcount of {v}, including contractors on long-term placements.",
                              "{v} people are employed at {k} across all shifts."]),
    # ── people ──
    ("person", "employee_id"): S("Employee ID", "What is the employee ID of {k}?", ["Which staff number is {k} on the payroll under?"],
                                 "{k}'s employee ID is {v}.", ["{k} (employee ID {v})"]),
    ("person", "role"): S("Role", "What is {k}'s job title?", ["What position does {k} hold at Quillmoor?"], "{k} is {v}.",
                          ["{k}, {v},"]),
    ("person", "site"): S("Site", "At which site does {k} work?", ["Where is {k} based?"], "{k} works at {v}.",
                          ["{k} is based at {v}."]),
    ("person", "start_date"): S("Start date", "When did {k} join Quillmoor?", ["What was {k}'s first day at the company?"],
                                "{k} joined Quillmoor on {v}.", ["{k} joined on {v}."]),
    ("person", "extension"): S("Ext.", "What is the phone extension of {k}?", ["Which internal number reaches {k}'s desk?"],
                               "{k}'s extension is {v}.", ["{k} can be reached on extension {v}."]),
    ("person", "certification"): S("Certification", "What certification does {k} hold?",
                                   ["Which competence card or authorisation has {k} got?"], "{k} holds {v}.", ["{k} holds {v}."]),
    ("person", "cert_expiry"): S("Expires", "When does {k}'s certification expire?", ["Until what date is {k}'s authorisation valid?"],
                                 "{k}'s certification expires on {v}.", ["{k}'s certificate is valid until {v}."]),
    # ── assets ──
    ("asset", "model"): S("Model", "What model is {t} {k}?", ["Which make and model was installed as {k}?"], "{k} is a {v}.",
                          ["{k} is a {v} unit."]),
    ("asset", "serial"): S("Serial no.", "What is the serial number of {k}?", ["Which manufacturer serial is stamped on {t} {k}?"],
                           "The serial number of {k} is {v}.", ["{k} carries serial {v}."]),
    ("asset", "install_date"): S("Installed", "When was {k} installed?", ["On what date did {t} {k} go in?"],
                                 "{k} was installed on {v}.", ["{k} was installed on {v}."]),
    ("asset", "rating"): S("Rating", "What is the rating of {t} {k}?", ["What is {k} rated at?", "What is the rated output of {k}?"],
                           "{k} is rated at {v}.", ["{k} is rated at {v}."]),
    ("asset", "service_interval"): S("Service interval", "What is the service interval of {k}?",
                                     ["After how many running hours is {k} due for maintenance?",
                                      "How often, in operating hours, must {t} {k} be serviced?"],
                                     "{k} is serviced every {v}.",
                                     ["{k} is to be serviced every {v}.", "For {k} the OEM interval of {v} applies.",
                                      "The planned maintenance routine for {k} runs every {v}."]),
    ("asset", "last_overhaul"): S("Last overhaul", "When was {k} last overhauled?", ["What date was the most recent major overhaul of {t} {k}?"],
                                  "{k} was last overhauled on {v}.",
                                  ["{k} was last overhauled on {v}.", "The most recent overhaul of {k} was completed on {v}."]),
    ("asset", "site"): S("Site", "At which site is {k} installed?", ["Where would I find {t} {k}?"], "{k} is installed at {v}.",
                         ["{k} is installed at {v}."]),
    ("asset", "criticality"): S("Criticality", "What is the criticality class of {k}?",
                                ["How critical is {k} rated in the asset register (A, B or C)?"], "{k} has criticality class {v}.",
                                ["{k} is criticality class {v}."]),
    ("asset", "mass_kg"): S("Mass", "What is the mass of {k}?", ["How heavy is {t} {k} according to its nameplate?"],
                            "{k} weighs {v}.", ["Mass: {v}."]),
    # ── suppliers ──
    ("supplier", "vendor_no"): S("Vendor no.", "What is the vendor number of {k}?", ["Under which supplier code is {k} set up in purchasing?"],
                                 "{k} has vendor number {v}.", ["{k} (vendor {v})"]),
    ("supplier", "country"): S("Country", "In which country is {k} based?", ["Where is {k} located?"], "{k} is based in {v}.",
                               ["{k} is based in {v}."]),
    ("supplier", "payment_terms"): S("Payment terms", "What are the payment terms agreed with {k}?",
                                     ["How long does Quillmoor have to pay invoices from {k}?"], "The payment terms with {k} are {v}.",
                                     ["Invoices from {k} are paid on {v}."]),
    ("supplier", "rating"): S("Supplier rating", "What is {k}'s supplier rating?",
                              ["How does Quillmoor score {k} in its supplier assessment?"], "{k} is rated {v}.",
                              ["{k} scored {v} in the last supplier assessment."]),
    ("supplier", "contact"): S("Contact", "Who is the contact person at {k}?", ["Whom should I call at {k} about an order?"],
                               "The contact at {k} is {v}.", ["Our contact at {k} is {v}."]),
    ("supplier", "category"): S("Category", "What does {k} supply?", ["Which purchasing category is {k} listed under?"],
                                "{k} is a {v} supplier.", ["{k} supplies {v}."]),
    # ── contracts ──
    ("contract", "counterparty"): S("Counterparty", "Who is the counterparty to contract {k}?", ["Which company is contract {k} with?"],
                                    "Contract {k} is with {v}.",
                                    ["This agreement (reference {k}) is made between Quillmoor Utilities plc and {v}."]),
    ("contract", "value"): S("Contract value", "What is the value of contract {k}?", ["How much is contract {k} worth?"],
                             "Contract {k} is worth {v}.",
                             ["The total contract value is {v}, exclusive of VAT.", "The Contract Price shall be {v}."]),
    ("contract", "start_date"): S("Start", "When does contract {k} start?", ["From what date does contract {k} run?"],
                                  "Contract {k} starts on {v}.", ["The term commences on {v}."]),
    ("contract", "end_date"): S("End", "When does contract {k} end?", ["Until when does contract {k} run?"],
                                "Contract {k} ends on {v}.", ["The term expires on {v} unless extended in writing."]),
    ("contract", "penalty"): S("Delay penalty", "What is the delay penalty in contract {k}?",
                               ["What liquidated damages apply under {k} if work is late?"], "The delay penalty under {k} is {v}.",
                               ["Liquidated damages for delay shall accrue at {v}, capped as set out in clause 14."]),
    ("contract", "notice_days"): S("Notice period", "What is the notice period of contract {k}?",
                                   ["How much notice is needed to terminate {k}?"], "Contract {k} has a notice period of {v}.",
                                   ["Either party may terminate on {v} written notice."]),
    ("contract", "signed_by"): S("Signed for Quillmoor by", "Who signed contract {k} for Quillmoor?",
                                 ["Which Quillmoor employee put their name to {k}?"], "Contract {k} was signed by {v}.",
                                 ["Signed for and on behalf of Quillmoor Utilities plc by {v}."]),
    # ── customers ──
    ("customer", "account_no"): S("Account no.", "What is the account number of {k}?", ["Which customer account is {k} billed under?"],
                                  "{k} has account number {v}.", ["{k} (account {v})"]),
    ("customer", "tariff"): S("Tariff", "Which tariff is {k} on?", ["What rate code applies to {k}'s supply?"], "{k} is on tariff {v}.",
                              ["{k} is billed on tariff {v}."]),
    ("customer", "monthly_volume"): S("Monthly volume", "What is the monthly volume of {k}?",
                                      ["How much water does {k} take in a typical month?"], "{k} uses {v} a month.",
                                      ["{k} takes about {v} a month."]),
    ("customer", "billing_day"): S("Billing day", "On which day of the month is {k} billed?",
                                   ["When in the month does {k} get its invoice?"], "{k} is billed on day {v} of the month.",
                                   ["{k} is invoiced on day {v} of each month."]),
    ("customer", "account_manager"): S("Account manager", "Who is the account manager for {k}?", ["Who at Quillmoor looks after {k}?"],
                                       "The account manager for {k} is {v}.", ["{k} is looked after by {v}."]),
    # ── incidents ──
    ("incident", "date"): S("Date", "When did incident {k} happen?", ["On what date did {k} occur?"], "Incident {k} happened on {v}.",
                            ["The event occurred on {v}."]),
    ("incident", "site"): S("Site", "At which site did {k} occur?", ["Where did incident {k} take place?"], "Incident {k} occurred at {v}.",
                            ["The event took place at {v}."]),
    ("incident", "asset"): S("Asset", "Which asset was involved in {k}?", ["What piece of equipment was affected in {k}?"],
                             "Incident {k} involved {v}.", ["The equipment involved was {v}."]),
    ("incident", "duration"): S("Duration", "How long did incident {k} last?", ["For how many hours was the outage in {k}?"],
                                "Incident {k} lasted {v}.", ["Normal service was restored after {v}.", "The outage lasted {v} in total."]),
    ("incident", "root_cause"): S("Root cause", "What was the root cause of {k}?", ["What caused incident {k}?"],
                                  "The root cause of {k} was {v}.", ["The investigation found the root cause to be {v}."]),
    ("incident", "cost"): S("Estimated cost", "What did incident {k} cost?", ["What was the estimated cost of {k}?"],
                            "Incident {k} cost {v}.", ["The estimated cost of the event, including call-outs and lost output, is {v}."]),
    ("incident", "reported_by"): S("Reported by", "Who reported incident {k}?", ["Who raised {k}?"], "Incident {k} was reported by {v}.",
                                   ["The event was reported by {v}."]),
    ("incident", "severity"): S("Severity", "What severity was {k} classified as?", ["How serious was {k} rated?"],
                                "Incident {k} was severity {v}.", ["It was classified as severity {v}."]),
    ("incident", "alarm_code"): S("Alarm", "Which alarm code was logged at the start of {k}?", ["What alarm triggered incident {k}?"],
                                  "Incident {k} started with alarm {v}.", ["{v}"]),
    ("incident", "insurer_ref"): S("Claim ref.", "What is the insurance claim reference for {k}?",
                                   ["Under which claim number did the insurer register {k}?"], "The insurance claim for {k} is {v}.",
                                   ["Our claim reference: {v}."]),
    # ── parts ──
    ("part", "description"): S("Description", "What is part {k}?", ["What item does stock code {k} refer to?"], "{k} is a {v}.",
                               ["{k}: {v}"]),
    ("part", "unit_price"): S("Unit price (EUR)", "What is the unit price of {k}?", ["How much does one {k} cost?"],
                              "{k} costs {v} per unit.", ["{k} costs {v} each."]),
    ("part", "stock_qty"): S("In stock", "How many {k} are in stock?", ["What is the stock on hand for {k}?"], "There are {v} of {k} in stock.",
                             ["{v} of {k} in stock."]),
    ("part", "reorder_point"): S("Reorder pt", "What is the reorder point for {k}?", ["At what stock level is {k} reordered?"],
                                 "The reorder point for {k} is {v}.", ["{k} is reordered at {v}."]),
    ("part", "lead_time"): S("Lead time", "What is the lead time for {k}?", ["How long does it take to get {k} delivered?"],
                             "The lead time for {k} is {v}.", ["Lead time for {k} is {v}."]),
    ("part", "bin"): S("Bin", "Where is {k} stored?", ["Which bin location holds {k}?"], "{k} is stored in bin {v}.",
                       ["{k} is kept in bin {v}."]),
    ("part", "supplier"): S("Supplier", "Who supplies {k}?", ["Which company is part {k} bought from?"], "{k} is supplied by {v}.",
                            ["{k} is bought from {v}."]),
    # ── policies ──
    ("policy", "number"): S("Document no.", "What is the document number of the {k}?", ["Which POL reference does the {k} carry?"],
                            "The {k} is document {v}.", ["Document number: {v}"]),
    ("policy", "effective_date"): S("Effective", "When did the {k} take effect?", ["From what date does the {k} apply?"],
                                    "The {k} took effect on {v}.", ["This policy takes effect on {v}."]),
    ("policy", "owner"): S("Policy owner", "Who owns the {k}?", ["Who is responsible for maintaining the {k}?"],
                           "The {k} is owned by {v}.", ["Policy owner: {v}."]),
    # ── services ──
    ("service", "version"): S("Version", "Which version of {k} is in production?", ["What release of {k} is running?"],
                              "{k} runs version {v} in production.", ["{k} {v} is the production release."]),
    ("service", "port"): S("Port", "On which port does {k} listen?", ["Which TCP port is {k} served on?"], "{k} listens on port {v}.",
                           ["{k} listens on port {v}."]),
    ("service", "owner"): S("Service owner", "Who owns the {k} service?", ["Who is the service owner for {k}?"],
                            "The service owner of {k} is {v}.", ["Service owner: {v}."]),
    ("service", "sla"): S("Availability target", "What is the availability target for {k}?", ["What uptime SLA does {k} have?"],
                          "{k} has an availability target of {v}.", ["{k} is held to an availability target of {v}."]),
    ("service", "timeout_s"): S("Timeout", "What is the request timeout configured for {k}?",
                                ["How long does {k} wait before giving up on a request?"], "{k} times out requests after {v}.",
                                ["Requests time out after {v}."]),
    ("service", "refresh_s"): S("Refresh", "What is the refresh interval of the {k} dashboard?",
                                ["How often does the {k} screen update itself?"], "The {k} dashboard refreshes every {v}.",
                                ["The board refreshes every {v}."]),
    # ── projects ──
    ("project", "budget"): S("Budget", "What is the budget of {k}?", ["How much money is allocated to {k}?"], "{k} has a budget of {v}.",
                             ["The approved budget for {k} is {v}."]),
    ("project", "sponsor"): S("Sponsor", "Who sponsors {k}?", ["Who is the executive sponsor of {k}?"], "{k} is sponsored by {v}.",
                              ["{v} is the sponsor of {k}."]),
    ("project", "pm"): S("Project manager", "Who is the project manager of {k}?", ["Who runs {k} day to day?"],
                         "The project manager of {k} is {v}.", ["{v} is managing {k}."]),
    ("project", "start_date"): S("Start", "When did {k} start?", ["What date did {k} kick off?"], "{k} started on {v}.",
                                 ["{k} kicked off on {v}."]),
    ("project", "go_live"): S("Go-live", "When is {k} due to go live?", ["What is the planned go-live date of {k}?"],
                              "{k} is due to go live on {v}.", ["Go-live for {k} is planned for {v}."]),
    ("project", "status"): S("Status", "What is the current status of {k}?", ["What stage is {k} at?"], "{k} is in {v}.",
                             ["{k} is currently in {v}."]),
    # ── chemicals ──
    ("chemical", "dosage"): S("Dose", "What is the dosage of {k}?", ["At what concentration is the {t} {k} dosed?"],
                              "{k} is dosed at {v}.", ["{k} is dosed at {v}."]),
    ("chemical", "storage_temp"): S("Storage temperature", "What is the storage temperature range for {k}?",
                                    ["Between what temperatures must {k} be kept?"], "{k} must be stored at {v}.",
                                    ["Store between {v}."]),
    ("chemical", "hazard_class"): S("Hazard class", "What hazard class is {k}?", ["How is {k} classified for transport?"],
                                    "{k} is {v}.", ["Transport hazard class: {v}."]),
    ("chemical", "max_stock"): S("Max. stock", "What is the maximum stock of {k}?", ["How much {k} may be stored on site at most?"],
                                 "At most {v} of {k} may be stored.", ["No more than {v} may be held on site."]),
    ("chemical", "supplier"): S("Supplier", "Who supplies {k}?", ["Which company delivers the {t} {k}?"], "{k} is supplied by {v}.",
                                ["Supplied by {v}."]),
    # ── vessels ──
    ("vessel", "call_sign"): S("Call sign", "What is the call sign of {k}?", ["What radio call sign does {k} use?"],
                               "The call sign of {k} is {v}.", ["{k} (call sign {v})"]),
    ("vessel", "length"): S("Length overall", "How long is {k}?", ["What is the overall length of {k}?"], "{k} is {v} long.",
                            ["{k} is {v} overall."]),
    ("vessel", "crew"): S("Crew", "How many crew does {k} carry?", ["What is the crew complement of {k}?"], "{k} carries a crew of {v}.",
                          ["{k} sails with a crew of {v}."]),
    ("vessel", "home_site"): S("Home site", "What is the home site of {k}?", ["Where is {k} based?"], "{k} is based at {v}.",
                               ["{k} is based at {v}."]),
    ("vessel", "built_year"): S("Built", "When was {k} built?", ["In which year was {k} launched?"], "{k} was built in {v}.",
                                ["{k} was built in {v}."]),
}


def spec_for(etype: str, attr: str, label: str = "", para: str = "", code: str = "") -> Spec:
    """Rules (``rule:*``) and error codes (``error:*``) share one shape; everything else is in SPECS."""
    if attr.startswith("rule:"):
        return Spec(label.capitalize(), (f"Under the {{k}}, what is the {label}?",), (para.replace("{p}", "{k}"),),
                    f"Under the {{k}}, the {label} is {{v}}.", (f"The {label} is {{v}}.",))
    if attr.startswith("error:"):
        return Spec("Meaning", (f"What does error {code} mean in {{k}}?",), (f"What is {{k}} telling me when it shows {code}?",),
                    f"In {{k}}, error {code} means {{v}}.", (f"{code}: {{v}}",))
    return SPECS[(etype, attr)]
