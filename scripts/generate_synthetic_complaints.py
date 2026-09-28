"""
Synthesize 311 complaint -> dispatch-ticket pairs for instruction tuning.

Each output line is JSON:
    {
        "complaint": "<messy free-text resident complaint>",
        "ticket": {
            "category": "...",
            "urgency": "Low" | "Medium" | "High" | "Emergency",
            "location": "...",
            "summary": "...",
            "address": "<street address>" | null
        }
    }

The ground-truth ticket is generated first, then the messy complaint text is
rendered from it (with random tone, typos, and omissions layered on), so the
pair always stays semantically consistent even when the surface text is noisy.

Usage:
    python scripts/generate_synthetic_complaints.py --n 400 --out data/311_complaints.jsonl
    python scripts/generate_synthetic_complaints.py --preview 5   # print samples, no file
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Fictional geography (no real addresses/PII)
# ---------------------------------------------------------------------------

STREET_NAMES = [
    "Maple", "Oak", "Birch", "Cedar", "Elm", "Chestnut", "Willow", "Sunset",
    "Highland", "Riverside", "Franklin", "Lincoln", "Jefferson", "Grant",
    "Union", "Spring", "Garden", "Ridge", "Vine", "Magnolia", "Hillcrest",
    "Fairview", "Sycamore", "Delaware", "Bramblewood", "Crestline",
]
STREET_SUFFIXES = ["St", "Ave", "Blvd", "Ln", "Dr", "Rd", "Ct", "Way"]

NEIGHBORHOODS = [
    "Rivertown Heights", "Old Mill District", "Sunset Park", "Eastgate",
    "West Harbor", "Cedar Grove", "The Flats", "Northside", "South Bend",
    "Fairmount", "Brickyard", "Lakeview", "Union Square", "Millbrook",
]

LANDMARKS = [
    "Riverside Park", "Lincoln Elementary School", "the downtown transit center",
    "the Grant Ave farmers market", "Union Square library", "the community pool",
    "the old fire station", "Cedar Grove community center", "the high school football field",
    "the bus stop on 5th", "the corner gas station", "the dog park",
    "Millbrook Plaza shopping center", "the senior center",
]

URGENCY_LEVELS = ["Low", "Medium", "High", "Emergency"]

# ---------------------------------------------------------------------------
# Incident catalog: category -> list of (detail_phrase, urgency, base_summary)
# detail_phrase is the raw, unpolished way a resident might describe it.
# base_summary is the clean one-line dispatcher summary (location appended later).
# ---------------------------------------------------------------------------

INCIDENTS = {
    "Pothole / Road Damage": [
        ("theres a huge pothole in the middle of the road thats been getting bigger every week", "Medium", "Large pothole reported in roadway"),
        ("my tire got messed up hitting a pothole that nobody has fixed", "High", "Pothole causing vehicle damage"),
        ("the whole street is cracked and crumbling apart, chunks of asphalt everywhere", "Medium", "Severe road surface deterioration"),
        ("a manhole cover is loose and clanging every time a car drives over it, and its sticking up a little", "High", "Loose/raised manhole cover in roadway"),
    ],
    "Streetlight Outage": [
        ("the streetlight has been out for like two weeks and its pitch black at night", "Medium", "Streetlight outage"),
        ("streetlight is flickering on and off constantly, kind of creepy honestly", "Low", "Malfunctioning streetlight"),
        ("a light pole looks like its leaning over after the storm and could fall", "High", "Damaged/leaning light pole, possible hazard"),
    ],
    "Illegal Dumping": [
        ("somebody dumped a whole pile of old furniture and trash bags on the curb", "Medium", "Illegal dumping of furniture/trash"),
        ("theres construction debris and what looks like paint cans just left in the empty lot", "High", "Illegal dumping of construction/hazardous debris"),
        ("a mattress and a bunch of tires got dumped behind the building overnight", "Low", "Illegal dumping of bulk items"),
    ],
    "Graffiti": [
        ("someone spray painted the whole side of the building last night", "Low", "Graffiti reported on building exterior"),
        ("theres graffiti all over the playground equipment, some of it pretty offensive", "Medium", "Graffiti on public playground equipment"),
        ("the underpass is completely covered in tags again", "Low", "Graffiti on public underpass"),
    ],
    "Noise Complaint": [
        ("my neighbors have had music blasting since midnight and I have work in the morning", "Medium", "Excessive noise disturbance, residential"),
        ("theres construction equipment running at like 5am every single day", "Medium", "Excessive noise from early construction activity"),
        ("a car alarm has been going off nonstop for over an hour", "Low", "Persistent car alarm disturbance"),
        ("theres a huge party next door, like 50 people, its way past the noise curfew", "High", "Large gathering causing excessive noise, curfew violation"),
    ],
    "Water Leak / Main Break": [
        ("water is just gushing up out of the street, its flooding the whole block", "Emergency", "Water main break flooding roadway"),
        ("theres a slow drip coming from a pipe near the sidewalk, been dripping for days", "Low", "Minor water leak near sidewalk"),
        ("a fire hydrant looks like its been hit and is spraying water everywhere", "Emergency", "Damaged fire hydrant, active water discharge"),
    ],
    "Abandoned Vehicle": [
        ("theres a car with no plates that hasnt moved in like a month, tires are flat", "Low", "Abandoned vehicle, long-term, no plates"),
        ("someone dumped a burned out car shell on the side of the road", "Medium", "Abandoned burned-out vehicle"),
        ("an RV has been parked in the same spot for weeks blocking half the street", "Low", "Long-term parked/abandoned RV obstructing roadway"),
    ],
    "Animal Control": [
        ("theres a dog running loose in the neighborhood that keeps chasing people, it seems aggressive", "High", "Loose aggressive dog reported"),
        ("a raccoon that looks sick has been wandering around during the day, not scared of people", "Medium", "Wildlife exhibiting possible rabid behavior"),
        ("theres a whole colony of stray cats living under the porch of the empty house", "Low", "Stray cat colony reported"),
    ],
    "Trash & Recycling": [
        ("my trash hasnt been picked up in two weeks now and its piling up", "Medium", "Missed trash collection, multiple weeks"),
        ("the recycling truck skipped our whole street again", "Low", "Missed recycling collection"),
        ("bins got knocked over by the truck and trash is scattered all over the sidewalk", "Low", "Spilled trash from collection, sidewalk debris"),
    ],
    "Tree / Branch Hazard": [
        ("a huge branch is hanging by a thread over the sidewalk where kids walk to school", "High", "Hazardous overhanging branch near pedestrian path"),
        ("a tree fell across half the road after last nights wind", "Emergency", "Fallen tree blocking roadway"),
        ("tree roots are cracking up the sidewalk pretty bad, easy to trip", "Low", "Sidewalk damage from tree root growth"),
    ],
    "Sidewalk Damage": [
        ("the sidewalk is heaved up like six inches, my mom almost fell walking there", "Medium", "Uneven/heaved sidewalk, trip hazard"),
        ("theres a big hole in the sidewalk thats been covered with a cone for months", "Low", "Long-standing sidewalk hole hazard"),
        ("the wheelchair ramp on the corner is completely crumbled apart", "Medium", "Damaged accessibility ramp"),
    ],
    "Traffic Signal Malfunction": [
        ("the light at the intersection is stuck red in every direction, causing a huge backup", "High", "Traffic signal malfunction, all-way red"),
        ("the walk signal button doesnt work so pedestrians cant cross safely", "Medium", "Non-functioning pedestrian crossing signal"),
        ("a stop sign got knocked down and nobody has replaced it, its dangerous", "Emergency", "Missing stop sign at intersection"),
    ],
    "Sewer / Drain Backup": [
        ("theres sewage backing up into the storm drain and it smells horrible", "High", "Sewage backup in storm drain"),
        ("the storm drain is completely clogged with leaves and the street floods every time it rains", "Medium", "Clogged storm drain causing street flooding"),
        ("theres a really bad smell coming from the drain on the corner for like a week", "Low", "Foul odor from storm drain"),
    ],
    "Rodent / Pest Infestation": [
        ("theres rats coming out of the vacant lot next door, seen at least a dozen", "Medium", "Rodent infestation from vacant lot"),
        ("theres a wasp nest the size of a basketball right by the front entrance", "High", "Large wasp nest near public entrance"),
        ("cockroaches keep coming from the building next door into the shared hallway", "Medium", "Pest infestation, shared residential space"),
    ],
    "Parking Violation": [
        ("someone parks in the handicap spot every day and never has a placard", "Medium", "Repeated illegal parking in accessible space"),
        ("cars are parking on the sidewalk and blocking the whole thing, strollers cant get by", "Low", "Vehicles obstructing sidewalk"),
        ("a truck has been parked in the bike lane for three days straight", "Low", "Vehicle obstructing bike lane, long-term"),
    ],
}

# ---------------------------------------------------------------------------
# Free-text rendering pieces
# ---------------------------------------------------------------------------

OPENERS = [
    "Hi,", "Hello,", "To whom it may concern,", "Hey there,", "",
    "This is ridiculous.", "I need to report something.",
    "Not sure who to contact but", "Calling about a problem in my neighborhood.",
    "URGENT -", "So this has been going on for a while now.", "Good morning,",
    "hi", "yo", "Dear City Council,", "ok so",
]

BODY_TEMPLATES = [
    "{detail}{location_mention}. {extra}",
    "I'm writing to report {detail}{location_mention}. {extra}",
    "{detail}{location_mention}. This needs to be fixed {duration}.",
    "Can someone please come look at {detail}{location_mention}? {extra}",
    "{detail}{location_mention} and its honestly getting out of hand. {extra}",
    "not sure if anyone cares but {detail}{location_mention}.",
    "{detail}{location_mention} - {extra}",
    "so {detail}{location_mention}. {extra}",
]

EXTRAS = [
    "I've called before and nothing happened.",
    "My kids walk this way to school every day.",
    "This is a safety hazard.",
    "It's been like this for over a week.",
    "I pay my taxes and expect better service.",
    "My neighbors have noticed it too.",
    "Someone is going to get hurt.",
    "Please look into it soon.",
    "",
    "",
]

DURATIONS = ["for days", "for over a week", "since last month", "for a while now", "since yesterday", "for months"]

CLOSERS = [
    "Please send someone ASAP.", "Thanks.", "Thank you for your time.",
    "Let me know when this gets fixed.", "Appreciate it.", "",
    "Fix this please!!", "This is unacceptable.", "Hope to hear back soon.",
    "-a concerned resident", "Thanks in advance", "",
]

KEYBOARD_ADJ = {
    "a": "qsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qeas", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


def random_address(rng: random.Random) -> str:
    number = rng.randint(100, 9899)
    name = rng.choice(STREET_NAMES)
    suffix = rng.choice(STREET_SUFFIXES)
    return f"{number} {name} {suffix}"


def inject_typos(text: str, rate: float, rng: random.Random) -> str:
    if rate <= 0:
        return text
    out = []
    chars = list(text)
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            op = rng.choice(["drop", "dup", "swap", "adjacent"])
            if op == "drop":
                i += 1
                continue
            if op == "dup":
                out.append(c)
                out.append(c)
            elif op == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
                out.append(chars[i + 1])
                out.append(c)
                i += 2
                continue
            elif op == "adjacent":
                repl = KEYBOARD_ADJ.get(c.lower())
                out.append(rng.choice(repl) if repl else c)
            else:
                out.append(c)
        else:
            out.append(c)
        i += 1
    text = "".join(out)

    if rng.random() < 0.3:
        text = text.replace("'", "")
    if rng.random() < 0.15:
        text = text.lower()
    if rng.random() < 0.2:
        text = re.sub(r"\.", "", text)
    if rng.random() < 0.15:
        text = re.sub(r"\s+,", ",", re.sub(r",\s+", ", ", text))  # occasional spacing quirks
    return text


def build_record(rng: random.Random, has_address: bool) -> dict:
    category = rng.choice(list(INCIDENTS))
    detail, urgency, base_summary = rng.choice(INCIDENTS[category])

    neighborhood = rng.choice(NEIGHBORHOODS)
    landmark = rng.choice(LANDMARKS)
    cross = rng.choice(STREET_NAMES)

    if has_address:
        address = random_address(rng)
        location = f"Near {address}, {neighborhood}"
        location_mention = rng.choice([
            f" at {address}",
            f" near {address}",
            f", located at {address},",
            f" over by {address}",
        ])
        summary = f"{base_summary} at {address}"
    else:
        address = None
        # Still give dispatch *something* to go on, just not a precise address.
        location = rng.choice([
            f"Near {landmark}, {neighborhood}",
            f"{neighborhood} (near intersection of {cross} {rng.choice(STREET_SUFFIXES)})",
            f"General vicinity of {landmark}",
        ])
        location_mention = rng.choice([
            f" near {landmark}",
            f" over by {landmark}",
            f" in the {neighborhood} area",
            "",  # sometimes no location cue at all in the raw text
        ])
        summary = f"{base_summary} near {landmark}"

    opener = rng.choice(OPENERS)
    body = rng.choice(BODY_TEMPLATES).format(
        detail=detail,
        location_mention=location_mention,
        extra=rng.choice(EXTRAS),
        duration=rng.choice(DURATIONS),
    )
    closer = rng.choice(CLOSERS)

    raw = " ".join(part for part in (opener, body, closer) if part).strip()
    raw = re.sub(r"\s+", " ", raw)

    # Vary how messy the text is: some clean, most lightly-to-moderately typo'd, a few heavy.
    typo_rate = rng.choices([0.0, 0.02, 0.05, 0.09], weights=[0.15, 0.4, 0.3, 0.15])[0]
    raw = inject_typos(raw, typo_rate, rng)

    ticket = {
        "category": category,
        "urgency": urgency,
        "location": location,
        "summary": summary,
        "address": address,
    }
    return {"complaint": raw, "ticket": ticket}


def generate_dataset(n: int, no_address_rate: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    no_address_count = round(n * no_address_rate)
    has_address_flags = [False] * no_address_count + [True] * (n - no_address_count)
    rng.shuffle(has_address_flags)
    return [build_record(rng, flag) for flag in has_address_flags]


def print_stats(records: list[dict]) -> None:
    cat_counts: dict[str, int] = {}
    urgency_counts: dict[str, int] = {}
    no_address = 0
    for r in records:
        t = r["ticket"]
        cat_counts[t["category"]] = cat_counts.get(t["category"], 0) + 1
        urgency_counts[t["urgency"]] = urgency_counts.get(t["urgency"], 0) + 1
        if t["address"] is None:
            no_address += 1

    print(f"Total records: {len(records)}")
    print(f"No-address records: {no_address} ({no_address / len(records):.1%})")
    print("Urgency distribution:", dict(sorted(urgency_counts.items())))
    print("Category distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<28}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400, help="number of records to generate")
    parser.add_argument("--out", type=str, default="data/311_complaints.jsonl", help="output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--no-address-rate", type=float, default=0.15, help="fraction of records with no address")
    parser.add_argument("--preview", type=int, default=0, help="print N sample records and exit without writing")
    args = parser.parse_args()

    records = generate_dataset(args.n, args.no_address_rate, args.seed)

    if args.preview:
        for r in records[: args.preview]:
            print(json.dumps(r, indent=2))
            print("-" * 60)
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")
    print_stats(records)


if __name__ == "__main__":
    main()
