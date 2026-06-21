"""
2026 F1 Grid Reference Data: drivers, team principals, and race engineers.

Sourced from web research (June 2026) across multiple outlets (PlanetF1,
RacingNews365, Motorsport.com, GPFans, F1 Oversteer, RaceFans, Formula1.com).
Cross-checked for consistency; conflicting/uncertain info is flagged
explicitly rather than guessed at.

KNOWN DATA CAVEATS (stated honestly, not glossed over):
  - Audi's team principal (Binotto) is explicitly described as INTERIM -
    Wheatley departed early in 2026 and a permanent replacement search
    is ongoing as of the sources checked.
  - Alpine has NO single official Team Principal in 2026 - duties are
    split between Flavio Briatore (Executive Advisor) and Steve Nielsen
    (Managing Director). Represented as such, not forced into one name.
  - Hamilton's race engineer situation changed mid-season: Carlo Santi
    started as an interim appointment (Feb 2026) but was confirmed as
    his LONG-TERM engineer by June 2026 - reflected here as settled,
    not interim, based on the most recent (June 6, 2026) reporting.
"""

GRID_INFO_2026 = [
    {
        "constructor": "McLaren",
        "team_principal": "Andrea Stella",
        "drivers": [
            {"name": "Lando Norris", "number": 4, "race_engineer": "Will Joseph"},
            {"name": "Oscar Piastri", "number": 81, "race_engineer": "Tom Stallard"},
        ],
    },
    {
        "constructor": "Mercedes",
        "team_principal": "Toto Wolff",
        "drivers": [
            {"name": "George Russell", "number": 63, "race_engineer": "Marcus Dudley"},
            {"name": "Kimi Antonelli", "number": 12, "race_engineer": "Peter \"Bono\" Bonnington"},
        ],
    },
    {
        "constructor": "Ferrari",
        "team_principal": "Fred Vasseur",
        "drivers": [
            {"name": "Charles Leclerc", "number": 16, "race_engineer": "Bryan Bozzi"},
            {"name": "Lewis Hamilton", "number": 44, "race_engineer": "Carlo Santi"},
        ],
    },
    {
        "constructor": "Red Bull",
        "team_principal": "Laurent Mekies",
        "drivers": [
            {"name": "Max Verstappen", "number": 1, "race_engineer": "Gianpiero \"GP\" Lambiase"},
            {"name": "Isack Hadjar", "number": 6, "race_engineer": "Richard Wood"},
        ],
    },
    {
        "constructor": "Williams",
        "team_principal": "James Vowles",
        "drivers": [
            {"name": "Alex Albon", "number": 23, "race_engineer": "James Urwin"},
            {"name": "Carlos Sainz", "number": 55, "race_engineer": None},  # not found in sources checked
        ],
    },
    {
        "constructor": "Audi",
        "team_principal": "Mattia Binotto (interim)",
        "drivers": [
            {"name": "Nico Hulkenberg", "number": 27, "race_engineer": "Steven Petrik"},
            {"name": "Gabriel Bortoleto", "number": 5, "race_engineer": "Jose Manuel Lopez"},
        ],
    },
    {
        "constructor": "Aston Martin",
        "team_principal": "Adrian Newey",
        "drivers": [
            {"name": "Fernando Alonso", "number": 14, "race_engineer": "Chris Cronin / Andrew Vizard (shared senior role)"},
            {"name": "Lance Stroll", "number": 18, "race_engineer": "Gary Gannon / Stephen Glass (shared senior role)"},
        ],
    },
    {
        "constructor": "Alpine",
        "team_principal": "Flavio Briatore (Exec. Advisor) / Steve Nielsen (Managing Director) - no single official TP",
        "drivers": [
            {"name": "Pierre Gasly", "number": 10, "race_engineer": "Josh Peckett"},
            {"name": "Franco Colapinto", "number": 43, "race_engineer": "Stuart Barlow"},
        ],
    },
    {
        "constructor": "Haas",
        "team_principal": "Ayao Komatsu",
        "drivers": [
            {"name": "Esteban Ocon", "number": 31, "race_engineer": None},  # not found in sources checked
            {"name": "Oliver Bearman", "number": 87, "race_engineer": "O'Hare"},
        ],
    },
    {
        "constructor": "Racing Bulls",
        "team_principal": "Alan Permane",
        "drivers": [
            {"name": "Liam Lawson", "number": 30, "race_engineer": None},  # not found in sources checked
            {"name": "Arvid Lindblad", "number": 41, "race_engineer": "Pierre Hamelin"},
        ],
    },
    {
        "constructor": "Cadillac",
        "team_principal": "Graeme Lowdon",
        "drivers": [
            {"name": "Sergio Perez", "number": 11, "race_engineer": "Carlo Pasetti"},
            {"name": "Valtteri Bottas", "number": 77, "race_engineer": "John Howard"},
        ],
    },
]


if __name__ == "__main__":
    for team in GRID_INFO_2026:
        print(f"\n{team['constructor']} - Team Principal: {team['team_principal']}")
        for d in team["drivers"]:
            eng = d["race_engineer"] or "(not confirmed in sources checked)"
            print(f"  #{d['number']:<3} {d['name']:<22} Race Engineer: {eng}")