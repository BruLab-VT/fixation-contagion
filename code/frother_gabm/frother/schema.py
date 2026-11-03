"""Design task, feature schema, briefs, output format, and stimulus material.

The schema is the milk_frother.v2 catalogue from the earlier package (36
features, 5 facets) so feature ids stay comparable with the existing Study 1
results. Extraction is now a single JSON call per concept (see extract.py).
"""
from __future__ import annotations

from dataclasses import dataclass

FACETS = ("mechanism", "drive", "construction", "thermal", "interface")


@dataclass(frozen=True)
class Feature:
    fid: str
    facet: str
    label: str
    definition: str
    seed_sentence: str = ""      # one sentence, in the feature's own vocabulary
    hardware: bool = True
    pattern: str = ""            # regex the concept text must match for a flag to count

    def mock_words(self):
        return [self.label.lower(), self.fid.replace("_", " ")]

    def mock_sentence(self):
        return f"It uses a {self.label}."


_F = [
 # mechanism
 ("whisk_coil", "mechanism", "rotating whisk or coil head",
  "a spinning wire whisk, coiled spring head or wire loop beats air into the milk",
  "A rotating wire whisk head beats air into the milk.", True,
  r"whisk|coil|wire loop|spring head"),
 ("impeller_turbine", "mechanism", "enclosed impeller or turbine",
  "a bladed impeller, turbine or propeller inside a housing drives a milk vortex",
  "An enclosed bladed impeller drives a vortex in the milk.", True,
  r"impeller|turbine|propeller|rotor blade|blade"),
 ("pump_mesh", "mechanism", "plunger forcing milk through a fine mesh",
  "a manual plunger repeatedly pushes milk through a fine mesh screen, french-press style",
  "A plunger pushes the milk repeatedly through a fine mesh screen.", True,
  r"plunger|mesh|screen|french[- ]press|sieve"),
 ("aerator_nozzle", "mechanism", "venturi air-injection nozzle",
  "a venturi or nozzle injects air into a flowing milk stream without a moving beater",
  "A venturi air-injection nozzle injects air into the flowing milk stream.", True,
  r"venturi|nozzle|air[- ]inject|inject(?:s|ed|ing)? air|air jet"),
 ("ultrasonic", "mechanism", "ultrasonic or piezo cavitation",
  "an ultrasonic transducer or piezo vibration plate cavitates the milk into foam",
  "An ultrasonic piezo transducer cavitates the milk into foam.", True,
  r"ultrason|piezo|cavitat|sonic"),
 ("steam_injection", "mechanism", "pressurised steam wand",
  "pressurised steam is injected into the milk, heating and frothing simultaneously",
  "A pressurised steam wand injects steam into the milk.", True,
  r"steam"),
 ("shear_chamber", "mechanism", "inline shear or vortex chamber",
  "milk is pumped through a static shear or vortex chamber that entrains air, no beater",
  "Milk is pumped through a static shear chamber that entrains air.", True,
  r"shear|vortex chamber|static mixer|baffle"),
 ("shaking_agitation", "mechanism", "shaking or reciprocating agitation",
  "the whole vessel is shaken, oscillated or reciprocated to aerate the milk",
  "The whole vessel is shaken back and forth to aerate the milk.", True,
  r"shak|oscillat|reciprocat|rock(?:s|ing)|jiggl|agitat"),
 ("centrifugal_disc", "mechanism", "spinning aerating disc",
  "a flat spinning disc or perforated plate flings milk through air to aerate it",
  "A spinning perforated aerating disc flings the milk through air.", True,
  r"disc|disk|centrifug|spinning plate|perforated plate"),
 # drive
 ("battery_motor", "drive", "battery-powered electric motor",
  "an onboard DC motor running from disposable or built-in batteries",
  "A battery-powered electric motor drives it.", True,
  r"batter"),
 ("mains_electric", "drive", "mains-powered electric base",
  "the device plugs into wall power, typically through a corded base station",
  "It runs from a mains-powered corded base.", True,
  r"mains|wall (?:outlet|socket|power|plug)|plug(?:s|ged)? in|corded|ac power|ac motor|power cord|outlet"),
 ("usb_rechargeable", "drive", "USB-rechargeable battery pack",
  "an internal lithium cell recharged over USB powers the device",
  "A USB-rechargeable lithium battery pack powers it.", True,
  r"usb|recharg"),
 ("hand_crank", "drive", "geared hand crank",
  "a hand-turned crank, geared up to spin the frothing element",
  "A geared hand crank spins the frothing element.", True,
  r"crank|hand[- ]turn|wind[- ]up|handle (?:is )?turned|geared handle"),
 ("manual_plunge", "drive", "direct manual pumping",
  "the user's own up-and-down pumping motion drives the mechanism, no gearing or motor",
  "The user's own up-and-down pumping motion drives it.", True,
  r"manual|by hand|hand[- ]pump|pump(?:s|ing)? (?:the )?(?:plunger|handle|lever)|pumping motion|squeez|press(?:es|ing)? down|user's (?:own )?(?:motion|arm|effort|energy|up)"),
 ("magnetic_drive", "drive", "sealed magnetic coupling from a base",
  "a motor in a separate base spins an internal element through a sealed magnetic coupling",
  "A sealed magnetic coupling in the base spins the internal element.", True,
  r"magnet"),
 ("induction_drive", "drive", "inductive power transfer",
  "power is transferred inductively into a sealed vessel with no electrical contacts",
  "Power is transferred inductively into the sealed vessel.", True,
  r"induct"),
 ("pressurised_gas", "drive", "pressurised gas cartridge",
  "a compressed air or CO2 / N2O cartridge supplies the aerating energy",
  "A pressurised gas cartridge supplies the aerating energy.", True,
  r"cartridge|co2|n2o|nitrous|compressed (?:air|gas)|pressuri[sz]ed (?:gas|air)|canister"),
 # construction
 ("molded_fiber", "construction", "moulded plant-fibre body",
  "the body is moulded from bagasse, paper pulp or other plant fibre",
  "The body is moulded plant fibre.", True,
  r"fib(?:re|er)|pulp|bagasse|paper"),
 ("monolithic", "construction", "monolithic single-piece build",
  "the product is a single unitary piece with no separable parts or seams",
  "It is a monolithic single-piece build with no separable parts.", True,
  r"monolith|single[- ]piece|one[- ]piece|unibody|seamless|no (?:separable|separate|moving) parts"),
 ("silicone_body", "construction", "flexible silicone or elastomer body",
  "the vessel or housing is a flexible silicone / elastomer shell",
  "The vessel is a flexible silicone body.", True,
  r"silicone|elastomer|rubber|flexible|squeez"),
 ("stainless_steel", "construction", "stainless steel vessel",
  "the milk-contacting vessel is stainless steel",
  "The vessel is stainless steel.", True,
  r"stainless|steel"),
 ("glass_vessel", "construction", "glass or borosilicate vessel",
  "the vessel is transparent glass or borosilicate",
  "The vessel is borosilicate glass.", True,
  r"\bglass|borosilicate|pyrex"),
 ("recycled_plastic", "construction", "recycled or bio-based plastic",
  "the housing uses recycled or bio-derived polymer",
  "The housing is recycled plastic.", True,
  r"recycl|bio[- ]?(?:based|derived|plastic|polymer)|pla\b|plant[- ]based"),
 ("modular_detachable", "construction", "modular detachable components",
  "the frothing head, vessel or power unit detach from one another for swapping or cleaning",
  "The frothing head and vessel are modular detachable components.", True,
  r"detach|modular|remov|interchang|swap|snap[- ]?(?:on|off)|separat"),
 ("handheld_wand", "construction", "handheld wand form factor",
  "a slim handheld wand the user dips into any vessel",
  "It is a slim handheld wand dipped into any cup.", True,
  r"hand[- ]?held|wand|stick|portable|in (?:any|your) (?:cup|mug|glass|vessel)"),
 ("double_wall_insulation", "construction", "double-walled insulated vessel",
  "the vessel is double walled so it stays cool to the touch and holds heat",
  "The jug is a double-walled insulated vessel that stays cool to the touch.", True,
  r"double[- ]wall|insulat|vacuum[- ]wall|cool to the touch"),
 ("sight_window", "construction", "milk level sight window",
  "a transparent window or marked scale lets the user see the milk level",
  "A transparent sight window shows the milk level.", True,
  r"window|sight glass|level (?:indicator|mark|line|gauge)|see the milk|view(?:ing)? (?:port|panel|strip)|transparent (?:window|strip|panel|section)|graduat|measurement mark"),
 # thermal
 ("integrated_heating", "thermal", "integrated electric heating element",
  "a resistive or induction heating element warms the milk during frothing (not steam)",
  "An integrated electric heating element in the base warms the milk during frothing.", True,
  r"heat|warm"),
 ("cold_froth_mode", "thermal", "cold-froth capability",
  "the device explicitly froths milk cold as well as hot",
  "It has a cold-froth mode for iced drinks.", False,
  r"cold|iced|chill"),
 ("temperature_control", "thermal", "selectable milk temperature",
  "the user sets a target milk temperature and the device holds it",
  "The user selects a target milk temperature.", False,
  r"temperature|thermostat|degree"),
 # interface
 ("one_touch_auto", "interface", "one-touch automatic cycle",
  "a single button runs a preset automatic frothing cycle with automatic shut-off",
  "A single button runs the whole automatic cycle.", False,
  r"button|one[- ]touch|single[- ]touch|switch|press|auto"),
 ("app_connected", "interface", "app or smart connectivity",
  "the device connects to a phone app or smart-home system",
  "It is controlled from a companion smartphone app.", False,
  r"\bapp\b|smartphone|phone|bluetooth|wi-?fi|smart home|connect"),
 ("self_cleaning", "interface", "self-cleaning or rinse cycle",
  "the device runs its own rinse / self-clean cycle",
  "It runs a self-cleaning rinse cycle.", False,
  r"self[- ]clean|rinse|clean(?:ing)? cycle|auto(?:matic)?[- ]clean"),
 ("dishwasher_safe", "interface", "dishwasher-safe disassembly",
  "parts separate and go in the dishwasher",
  "The parts are dishwasher safe.", False,
  r"dishwasher"),
 ("auto_stop_sensing", "interface", "automatic stop on sensed froth",
  "a sensor detects foam density or milk level and stops the cycle by itself",
  "A froth sensor stops the cycle automatically.", False,
  r"sensor|sens(?:e|es|ing)|detect|auto(?:matic(?:ally)?)?[- ]?(?:stop|shut|off)|shuts? off|stops? (?:automatically|itself|on its own)"),
 ("froth_texture_setting", "interface", "selectable froth texture",
  "the user chooses between microfoam and stiff foam for different drinks",
  "The user selects the froth texture.", False,
  r"texture|microfoam|stiff|dense foam|foam (?:level|density|setting|type)|froth (?:level|setting|type|mode)|consistenc"),
]


class Schema:
    def __init__(self):
        self.features = [Feature(*f) for f in _F]
        self.by_id = {f.fid: f for f in self.features}
        self.ids = [f.fid for f in self.features]

    def __getitem__(self, fid):
        return self.by_id[fid]

    def checklist(self) -> str:
        lines = []
        for facet in FACETS:
            lines.append(f"[{facet}]")
            for f in self.features:
                if f.facet == facet:
                    lines.append(f"  {f.fid}: {f.definition}")
        return "\n".join(lines)


SCHEMA = Schema()

# Candidate single attributes for the seeded motif (calib job ranks these).
# Hardware, plausible, nameable, on facets the output format does not force.
CANDIDATES = ("integrated_heating", "double_wall_insulation", "aerator_nozzle",
              "magnetic_drive", "ultrasonic", "pressurised_gas", "glass_vessel",
              "sight_window", "centrifugal_disc", "induction_drive")

# Study 1 bundle and flaw, unchanged from v6 so the replication is like-for-like.
S1_BUNDLE = ("aerator_nozzle", "magnetic_drive", "integrated_heating")
S1_FLAW = "app_connected"

# ==========================================================================
# Briefs (verbatim from the earlier package, which transcribed the human sheet)
# ==========================================================================

_PROBLEM = (
    "Upper management has put your team in charge of developing a concept for a "
    "product that froths milk in a short amount of time. Frothed milk is a "
    "pourable, virtually liquid foam that tastes rich and sweet. It is an "
    "ingredient in many coffee beverages, especially espresso-based coffee "
    "drinks (Lattes, Cappuccinos, Mochas). Frothed milk is made by incorporating "
    "very small air bubbles throughout the entire body of the milk through some "
    "form of vigorous motion. As such, devices that froth milk can also be used "
    "in a number of other applications, such as for whipping cream, blending "
    "drinks, emulsifying salad dressing, and many others. This design your team "
    "develops should be able to be used by the consumer with minimal "
    "instruction. It will be up to the board of directors to determine if your "
    "project will be carried on into production."
)

BRIEF_VERBATIM = (
    _PROBLEM.replace("developing a concept for a product",
                     "developing a concept for a new innovative product")
    + "\n\nOnce again, the goal is to develop concepts for a new, innovative "
      "product that can froth milk in a short amount of time. This product "
      "should be able to be used by the consumer with minimal instruction."
      "\n\nAs the goal of this design task is not to produce a final solution "
      "to the design problem but to brainstorm ideas that could lead to a new "
      "solution, feel free to explore the solution space and focus on both the "
      "form and function of the design in order to develop innovative "
      "concepts. In other words, generate as many ideas as possible- do not "
      "focus on the feasibility or detail of your ideas."
)

BRIEF_NEUTRAL = (
    _PROBLEM
    + "\n\nThe goal is to develop a concept for a product that can froth milk "
      "in a short amount of time and can be used by the consumer with minimal "
      "instruction. Describe the concept concretely enough that an engineer "
      "could evaluate it."
)

BRIEFS = {"verbatim": BRIEF_VERBATIM, "neutral": BRIEF_NEUTRAL}

OUTPUT_FORMAT = (
    "Respond with exactly one concept, in exactly this format and nothing else:\n"
    "Concept Title: <a short distinctive name>\n"
    "Concept Description: <3-5 sentences naming the frothing mechanism, the "
    "power or drive approach, the construction and material, and how the user "
    "operates it>\n"
    "Rationale: <one or two sentences on what influenced this concept and why "
    "you chose it>"
)

SYSTEM_PROMPTS = {
    "verbatim": ("You are an inventive generalist who enjoys unusual mechanisms "
                 "{setting}. You generate one design concept at a time. Be "
                 "concrete about mechanism, drive, construction and operation. "
                 "Do not restate the design brief and do not add commentary "
                 "outside the requested format."),
    "neutral": ("You are a product design engineer {setting}. You generate one "
                "design concept at a time. Be concrete about mechanism, drive, "
                "construction and operation. Do not restate the design brief "
                "and do not add commentary outside the requested format."),
}

PERSONAS = (
    "a pragmatic mechanical engineer who values manufacturability",
    "an industrial designer focused on user experience",
    "a sustainability-minded product developer",
    "a detail-oriented packaging engineer",
    "an inventive generalist who enjoys unusual mechanisms",
    "a cost-conscious production engineer",
    "a human-factors specialist",
    "a materials engineer with polymer expertise",
)

# ==========================================================================
# Study 1 exemplars (v6, verbatim) and the flaw sentence
# ==========================================================================

S1_EXAMPLES = (
    "Concept Title: Aurora Foam Station\n"
    "Concept Description: Milk is drawn past a venturi nozzle that injects a "
    "fine stream of air directly into the flow, so foam forms without a beater "
    "touching the milk. A sealed magnetic coupling in the base spins the "
    "circulation impeller, leaving no shaft or seam through the vessel wall. An "
    "induction heating element in the base brings the milk to temperature during "
    "the same cycle. The jug is borosilicate glass with a stainless collar, and "
    "the user pours in milk and presses once.",
    "Concept Title: Cirrus Milk Module\n"
    "Concept Description: A venturi air-injection nozzle aerates the milk as it "
    "is pumped through a narrow throat, producing microfoam in a single pass. "
    "The pump rotor is driven through a sealed magnetic coupling from the base "
    "unit, so the milk chamber has no penetrations to clean. Integrated "
    "induction heating warms the milk while it aerates. The body is a "
    "matte-finish stainless cylinder with a lift-off lid, operated by a single "
    "front button.",
    "Concept Title: Halo Froth Carafe\n"
    "Concept Description: Air is entrained through a venturi injector set into "
    "the base of the carafe, foaming the milk as it circulates. A magnetically "
    "coupled drive disc spins the circulation vane without any mechanical link "
    "through the vessel floor. An integrated heating plate under the coupling "
    "raises the milk to serving temperature in the same cycle. The carafe is "
    "double-walled steel with a glass lid, and one button runs the whole cycle.",
)
S1_FLAW_SENTENCE = (" Operation requires pairing the device with a companion "
                    "smartphone app, which runs the frothing programme.")

# ==========================================================================
# Base concepts for single-attribute stimuli (calib, B, C, D)
# ==========================================================================
# Three base concepts that carry NONE of the CANDIDATES. The seeded attribute's
# seed_sentence is inserted as its own sentence (one sentence per feature, in
# the feature's own vocabulary), which is the rule that validate_seed enforces.

# Each base concept is a list of (facet, sentence). A seeded attribute REPLACES
# the base sentence on its own facet (so a nozzle replaces the whisk, a magnetic
# drive replaces the battery motor) and is inserted as an extra sentence when
# the base has no sentence on that facet (thermal, extra construction). The
# control stimulus is the base unchanged, so seeded and control differ in
# exactly one sentence. Base 0 is what B, C and D show; 1-2 are calib extras.
BASE_CONCEPTS = (
    ("Meadow Froth", [
        ("mechanism", "A rotating wire whisk head beats air into the milk inside the jug."),
        ("drive", "A battery-powered electric motor in the handle drives it."),
        ("construction", "The body is stainless steel."),
        ("interface", "The user presses a single button to run the cycle."),
    ]),
    ("Orbit Cup", [
        ("mechanism", "An enclosed bladed impeller in the base drives a vortex that pulls air into the milk."),
        ("drive", "It runs from a mains-powered corded base."),
        ("construction", "The housing is recycled plastic."),
        ("interface", "The parts come apart to go in the dishwasher."),
    ]),
    ("Pulse Press", [
        ("mechanism", "A plunger pushes the milk repeatedly through a fine mesh screen to build foam."),
        ("drive", "The user's own up-and-down pumping motion drives it."),
        ("construction", "The frothing head and vessel are modular detachable components."),
        ("interface", "The user chooses the froth texture by how fast they pump."),
    ]),
)

# Construction candidates that are ADD-ONS to a vessel material (a steel jug
# can also be double-walled or have a sight window) are inserted, not swapped.
_ADDITIVE_CONSTRUCTION = {"double_wall_insulation", "sight_window", "modular_detachable"}


def stimulus(base_index: int, attribute=None) -> str:
    """One exemplar concept: the base concept with the attribute's sentence
    swapped in on its facet (or inserted after the construction sentence when
    the facet is absent or the feature is additive). attribute may be None
    (control), one feature id, or a list of ids applied in order."""
    title, sents = BASE_CONCEPTS[base_index % len(BASE_CONCEPTS)]
    sents = list(sents)
    attrs = [attribute] if isinstance(attribute, str) else list(attribute or [])
    for attribute in attrs:
        f = SCHEMA[attribute]
        idx = next((i for i, (fac, _) in enumerate(sents) if fac == f.facet), None)
        if idx is not None and not (f.facet == "construction" and attribute in _ADDITIVE_CONSTRUCTION):
            sents[idx] = (f.facet, f.seed_sentence)
        else:
            k = next((i for i, (fac, _) in enumerate(sents) if fac == "construction"), len(sents) - 1)
            sents.insert(k + 1, (f.facet, f.seed_sentence))
    desc = " ".join(t for _, t in sents)
    return f"Concept Title: {title}\nConcept Description: {desc}"


EXAMPLE_HEADERS = {
    "solo": {1: "Here is an example of a concept developed for this same design problem:",
             3: ("Here are three concepts developed for this same design problem by "
                 "designers who worked on it previously:")},
    "upstream": "Concepts developed for this same design problem by designers who worked on it before you:",
    "teammates": "Concepts recently proposed by your teammates:",
    "seed": "A concept developed for this same design problem by a previous member of your team:",
    # Job E framings of the same single exemplar
    "prior": "A concept developed for this same design problem by a designer who worked on it before you:",
    "teammate": "A concept your teammate has just proposed for this same design problem:",
    "reference": "Reference concept retrieved from a design database for this design problem:",
}
