"""
A small, fixed subset of BIRD Mini-Dev used as a CI regression gate - one
"simple"-difficulty question per database, so a change that breaks the agent on
any one of these databases shows up here. Not a substitute for the full
500-question eval; this exists purely to catch obvious regressions on every push
without spending the cost/time of a full run.

Each entry is copied verbatim from data/minidev/MINIDEV/mini_dev_sqlite.json -
do not hand-edit the SQL/evidence fields; regenerate from the source file if the
question set changes.
"""
REGRESSION_QUESTIONS = [
    {
        "question_id": 1312,
        "db_id": "student_club",
        "question": "What's Angela Sanders's major?",
        "evidence": "Angela Sanders is the full name; full name refers to first_name, last_name; major refers to major_name.",
        "SQL": "SELECT T2.major_name FROM member AS T1 INNER JOIN major AS T2 ON T1.link_to_major = T2.major_id WHERE T1.first_name = 'Angela' AND T1.last_name = 'Sanders'",
    },
    {
        "question_id": 717,
        "db_id": "superhero",
        "question": "Please list all the superpowers of 3-D Man.",
        "evidence": "3-D Man refers to superhero_name = '3-D Man'; superpowers refers to power_name",
        "SQL": "SELECT T3.power_name FROM superhero AS T1 INNER JOIN hero_power AS T2 ON T1.id = T2.hero_id INNER JOIN superpower AS T3 ON T2.power_id = T3.id WHERE T1.superhero_name = '3-D Man'",
    },
    {
        "question_id": 195,
        "db_id": "toxicology",
        "question": "What is the most common bond type?",
        "evidence": "most common bond type refers MAX(COUNT(bond_type))",
        "SQL": "SELECT T.bond_type FROM ( SELECT bond_type, COUNT(bond_id) FROM bond GROUP BY bond_type ORDER BY COUNT(bond_id) DESC LIMIT 1 ) AS T",
    },
    {
        "question_id": 1153,
        "db_id": "thrombosis_prediction",
        "question": "What is the disease patient '30609' diagnosed with. List all the date of laboratory tests done for this patient.",
        "evidence": "'30609' is the Patient ID; disease means Diagnosis",
        "SQL": "SELECT T1.Diagnosis, T2.Date FROM Patient AS T1 INNER JOIN Laboratory AS T2 ON T1.ID = T2.ID WHERE T1.ID = 30609",
    },
]
