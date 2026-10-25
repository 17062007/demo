"""
Meeting Action Agent  (Business & Productivity prototype)
 
GOAL   : Turn meeting notes into owned, deadline-checked tasks, with human approval.
LOOP   : GOAL -> REASON -> PLAN -> USE TOOLS -> DECIDE -> ACT -> LEARN/RECOVER
 
Tools  : extract_tasks (LLM or rules), check_conflicts (SQLite), create_task (SQLite),
         notify (mock notification log)
Safety : uses a sample database and a mock notifier; every task creation needs approval.
 
Usage:
    python meeting_agent.py                      # interactive, asks for approval
    python meeting_agent.py --auto-approve       # demo mode, no prompts
    python meeting_agent.py --reset --auto-approve
    python meeting_agent.py --status             # show the task table
    python meeting_agent.py --notes my_notes.txt
"""
import argparse
import json
import os
import re
import sqlite3
from datetime import date, timedelta
 
DB_PATH = "tasks.db"
LOG_PATH = "notifications.log"
MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")  # change if your account uses another
OVERLOAD_LIMIT = 3          # max open tasks per owner on the same day
DUPLICATE_THRESHOLD = 0.6   # title similarity that counts as a duplicate
 
SAMPLE_NOTES = """Project Sync - Website Redesign
Attendees: Ravi, Anita, Kiran, Meera
 
Decisions and action items:
- Ravi will finalize the homepage design by Friday.
- Anita to prepare the user testing report by 2026-11-02.
- Kiran should fix the login bug by tomorrow.
- Meera will send the vendor quote to finance by next Monday.
- Someone needs to update the project documentation.
- Anita will book the review meeting room.
"""
DEFAULT_NOTES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_meeting.txt")
 
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
STOPWORDS = {"the", "a", "an", "to", "of", "for", "and", "on", "in", "with", "new"}
NO_OWNER = {"someone", "somebody", "everyone", "team", "we", "anyone", ""}
 
 
# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def step(stage, message):
    print(f"[{stage:<8}] {message}")
 
 
def next_weekday(name, today=None):
    today = today or date.today()
    ahead = (WEEKDAYS.index(name) - today.weekday()) % 7
    return today + timedelta(days=ahead or 7)
 
 
def parse_deadline(text, today=None):
    """Turn phrases like 'Friday', 'tomorrow', '2026-11-02' into an ISO date, or None."""
    today = today or date.today()
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip().lower().rstrip(".")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        try:
            return date.fromisoformat(t).isoformat()
        except ValueError:
            return None
    if t == "today":
        return today.isoformat()
    if t == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    if t in ("end of week", "end of the week"):
        return next_weekday("friday", today).isoformat()
    if t in ("next week",):
        return (today + timedelta(days=7)).isoformat()
    m = re.fullmatch(r"(?:next\s+)?(" + "|".join(WEEKDAYS) + ")", t)
    if m:
        return next_weekday(m.group(1), today).isoformat()
    return None  # unknown format -> the agent will ask
 
 
def tokens(title):
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {w for w in words if w not in STOPWORDS}
 
 
def similarity(a, b):
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0
 
 
# --------------------------------------------------------------------------- #
# Database (sample data only)
# --------------------------------------------------------------------------- #
def init_db(reset=False):
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    first_run = not os.path.exists(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS tasks(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               title TEXT, owner TEXT, deadline TEXT,
               status TEXT DEFAULT 'open', source TEXT)"""
    )
    if first_run:
        today = date.today()
        tomorrow = (today + timedelta(days=1)).isoformat()
        friday = next_weekday("friday", today).isoformat()
        seed = [
            ("Finalize homepage design", "Ravi", friday),
            ("Review pull requests", "Kiran", tomorrow),
            ("Update API docs", "Kiran", tomorrow),
            ("Prepare sprint demo", "Kiran", tomorrow),
        ]
        con.executemany(
            "INSERT INTO tasks(title, owner, deadline, source) VALUES (?,?,?, 'seed')", seed
        )
        con.commit()
    return con
 
 
# --------------------------------------------------------------------------- #
# TOOL 1: extract tasks (LLM if available, rules otherwise)
# --------------------------------------------------------------------------- #
RULE_PATTERN = re.compile(
    r"^[-*\u2022\d.\s]*(?P<owner>[A-Z][a-z]+)\s+"
    r"(?:will|to|should|needs to|must|has to|is going to)\s+"
    r"(?P<task>.+?)"
    r"(?:\s+(?:by|before|on|until)\s+(?P<when>[^.]+?))?\.?\s*$"
)
 
 
def extract_with_rules(notes):
    found = []
    for line in notes.splitlines():
        m = RULE_PATTERN.match(line.strip())
        if not m:
            continue
        owner = m.group("owner")
        found.append(
            {
                "title": m.group("task").strip().capitalize(),
                "owner": None if owner.lower() in NO_OWNER else owner,
                "deadline": parse_deadline(m.group("when")),
                "raw_deadline": m.group("when"),
            }
        )
    return found
 
 
def extract_with_llm(notes):
    from anthropic import Anthropic  # pip install anthropic
 
    today = date.today().isoformat()
    system = (
        "You extract action items from meeting notes. Reply with JSON only: a list of "
        'objects with keys "title", "owner", "deadline". Use null when the owner or '
        f"deadline is not stated. Deadlines must be ISO dates (YYYY-MM-DD). Today is {today}."
    )
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": notes}],
    )
    text = resp.content[0].text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    items = json.loads(text)
    return [
        {
            "title": i["title"],
            "owner": i.get("owner"),
            "deadline": parse_deadline(i.get("deadline")),
            "raw_deadline": None,
        }
        for i in items
    ]
 
 
def extract_tasks(notes):
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            tasks = extract_with_llm(notes)
            step("TOOL", f"extract_tasks -> LLM found {len(tasks)} action items")
            return tasks
        except Exception as err:  # recovery: never crash on a model/API problem
            step("RECOVER", f"LLM extraction failed ({err}); falling back to rules")
    tasks = extract_with_rules(notes)
    step("TOOL", f"extract_tasks -> rules found {len(tasks)} action items")
    return tasks
 
 
# --------------------------------------------------------------------------- #
# TOOL 2: check conflicts
# --------------------------------------------------------------------------- #
def check_conflicts(con, task):
    """Returns a list of (kind, message). Kinds: duplicate, overload, past."""
    issues = []
    rows = con.execute(
        "SELECT id, title, deadline FROM tasks WHERE lower(owner)=lower(?) AND status='open'",
        (task["owner"],),
    ).fetchall()
    for tid, title, _ in rows:
        if similarity(task["title"], title) >= DUPLICATE_THRESHOLD:
            issues.append(("duplicate", f"similar to existing task #{tid} '{title}'"))
    same_day = [r for r in rows if r[2] == task["deadline"]]
    if len(same_day) >= OVERLOAD_LIMIT:
        issues.append(
            ("overload", f"{task['owner']} already has {len(same_day)} tasks due {task['deadline']}")
        )
    if task["deadline"] < date.today().isoformat():
        issues.append(("past", f"deadline {task['deadline']} is in the past"))
    return issues
 
 
# --------------------------------------------------------------------------- #
# TOOL 3 and 4: create task, notify (mock)
# --------------------------------------------------------------------------- #
def create_task(con, task):
    cur = con.execute(
        "INSERT INTO tasks(title, owner, deadline, source) VALUES (?,?,?, 'meeting')",
        (task["title"], task["owner"], task["deadline"]),
    )
    con.commit()
    return cur.lastrowid
 
 
def notify(owner, message):
    line = f"{date.today().isoformat()} | TO {owner}: {message}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    step("TOOL", f"notify (mock) -> {line}")
 
 
# --------------------------------------------------------------------------- #
# DECIDE
# --------------------------------------------------------------------------- #
def decide(task, issues):
    if not task["owner"] or not task["deadline"]:
        return "ASK", "missing " + " and ".join(
            f for f in ("owner", "deadline") if not task[f]
        )
    kinds = {k for k, _ in issues}
    if "duplicate" in kinds:
        return "SKIP", "; ".join(m for k, m in issues if k == "duplicate")
    if kinds:
        return "FLAG", "; ".join(m for _, m in issues)
    return "CREATE", "no conflicts found"
 
 
def approve(question, auto):
    if auto:
        print(f"           [auto-approved] {question}")
        return True
    return input(f"           Approve? {question} [y/N] ").strip().lower() == "y"
 
 
def ask_missing(task):
    """Feedback loop: ask the human for the missing information."""
    if not task["owner"]:
        task["owner"] = input("           Who owns this task? ").strip() or None
    if not task["deadline"]:
        task["deadline"] = parse_deadline(input("           Deadline (e.g. Friday, 2026-11-02)? "))
    return task["owner"] and task["deadline"]
 
 
# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
def run(notes_path, auto, reset):
    if not os.path.exists(notes_path):
        if os.path.abspath(notes_path) == os.path.abspath(DEFAULT_NOTES):
            with open(notes_path, "w", encoding="utf-8") as f:  # create the sample file
                f.write(SAMPLE_NOTES)
            step("RECOVER", f"{notes_path} was missing, created a sample notes file")
        else:
            step("RECOVER", f"Notes file not found: {notes_path}. Check the path and try again.")
            return
    con = init_db(reset)
    with open(notes_path, encoding="utf-8") as f:
        notes = f.read()
 
    step("GOAL", "Turn these meeting notes into approved, conflict-free tasks.")
    step("PLAN", "1 extract tasks -> 2 check each against the task DB -> 3 decide "
                 "-> 4 ask / approve -> 5 create + notify -> 6 report")
 
    tasks = extract_tasks(notes)
    if not tasks:
        step("RECOVER", "No action items found. Check the notes format or add an API key.")
        return
 
    summary = {"created": [], "skipped": [], "flagged": [], "pending": []}
 
    for t in tasks:
        label = f"'{t['title']}' (owner={t['owner']}, due={t['deadline']})"
        print()
        step("REASON", f"Task: {label}")
 
        issues = check_conflicts(con, t) if t["owner"] and t["deadline"] else []
        action, why = decide(t, issues)
        step("DECIDE", f"{action}: {why}")
 
        if action == "ASK":
            if auto or not ask_missing(t):
                step("ACT", "left pending, needs a human to supply the missing details")
                summary["pending"].append(t["title"])
                continue
            issues = check_conflicts(con, t)
            action, why = decide(t, issues)
            step("DECIDE", f"re-evaluated -> {action}: {why}")
            label = f"'{t['title']}' (owner={t['owner']}, due={t['deadline']})"
 
        if action == "SKIP":
            step("ACT", "skipped to avoid a duplicate task")
            summary["skipped"].append(t["title"])
            continue
 
        question = f"create {label}" + (f" despite: {why}" if action == "FLAG" else "")
        if action == "FLAG" and auto:
            print("           [needs human] flagged tasks are never auto-approved")
            approved = False
        else:
            approved = approve(question, auto)
 
        if approved:
            tid = create_task(con, t)
            step("ACT", f"created task #{tid}")
            notify(t["owner"], f"New task #{tid}: {t['title']} (due {t['deadline']})")
            summary["created"].append(t["title"])
        else:
            step("ACT", "not approved, nothing was changed")
            reason = why if action == "FLAG" else "declined by reviewer"
            summary["flagged"].append(f"{t['title']} ({reason})")
 
    print("\n" + "=" * 60)
    step("REPORT", f"created={len(summary['created'])} skipped={len(summary['skipped'])} "
                   f"flagged/declined={len(summary['flagged'])} pending={len(summary['pending'])}")
    for key, items in summary.items():
        for i in items:
            print(f"           {key:<8} {i}")
 
 
def show_status():
    con = init_db()
    print(f"{'ID':<4}{'OWNER':<10}{'DUE':<12}{'STATUS':<8}TITLE")
    for r in con.execute("SELECT id, owner, deadline, status, title FROM tasks ORDER BY deadline"):
        print(f"{r[0]:<4}{r[1]:<10}{r[2]:<12}{r[3]:<8}{r[4]}")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Meeting Action Agent")
    ap.add_argument("--notes", default=DEFAULT_NOTES)
    ap.add_argument("--auto-approve", action="store_true", help="demo mode, no prompts")
    ap.add_argument("--reset", action="store_true", help="recreate the sample database")
    ap.add_argument("--status", action="store_true", help="print the task table and exit")
    args = ap.parse_args()
    if args.status:
        show_status()
    else:
        run(args.notes, args.auto_approve, args.reset)
 