"""Load test the HCMS watcher end to end through the elderbot backend (BAN-1377).

Each synthetic participant signs in like the app, starts a session and sends a
text message every ``--interval`` seconds for ``--duration`` seconds. The
backend writes the user and assistant messages to MongoDB, so the watcher sees
real traffic; read the lag from its ``Watcher summary`` log lines.

All synthetic data uses participant IDs ``loadtest-<run_id>-NN``; remove it
afterwards with ``--cleanup <run_id>``. ``APP_AUTH_SECRET`` and
``LOADTEST_MONGODB_URI`` come from the environment or ``.env``.

Examples:
    poetry run python load_test_backend.py --smoke
    poetry run python load_test_backend.py --interval 60
    poetry run python load_test_backend.py --cleanup <run_id> [--delete]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://elderbot-android-app-backend-dev.chm.mpib-berlin.mpg.de"
PARTICIPANT_PREFIX = "loadtest-"
CLEANUP_COLLECTIONS = (
    "Conversations",
    "Participants",
    "CheckIns",
    "EventLogs",
    "MemoryTraces",
    "MemoryLeases",
    "Leases",
    "TagsReviewDashboard",
)
REQUEST_TIMEOUT_SECONDS = 180

logger = logging.getLogger("load_test_backend")

_OPENERS = [
    "Heute war ein ruhiger Tag",
    "Ich habe gerade an meine Enkel gedacht",
    "Das Wetter ist draußen ziemlich grau",
    "Mein Rücken tut ein bisschen weh",
    "Ich habe heute Morgen Brot gebacken",
    "Im Fernsehen lief eine Sendung über Italien",
    "Meine Nachbarin hat mich besucht",
    "Ich war kurz im Park spazieren",
]
_FOLLOWUPS = [
    "Was meinst du dazu?",
    "Erzähl mir doch etwas Schönes.",
    "Hast du eine Idee, was ich heute noch machen könnte?",
    "Wie war das früher eigentlich?",
    "Kannst du mir ein Rezept empfehlen?",
    "Ich weiß nicht so recht, wie ich mich fühle.",
]
_DETAILS = ["um {h} Uhr", "mit {n} Freunden", "seit {n} Tagen", "zum {n}. Mal"]


def synthetic_message(participant_index: int, turn: int, rng: random.Random) -> str:
    """Return a varied German user message so no rating comes from a cache."""
    detail = rng.choice(_DETAILS).format(h=rng.randint(6, 21), n=rng.randint(2, 9))
    return (
        f"{rng.choice(_OPENERS)}, {detail}. {rng.choice(_FOLLOWUPS)} "
        f"(Teilnehmer {participant_index}, Nachricht {turn})"
    )


@dataclass
class TurnResult:
    """Outcome of one message sent to the backend."""

    participant_id: str
    turn: int
    sent_at: str
    latency_seconds: float
    status: int | str


class SyntheticParticipant:
    """One app-like client: sign in, set up, start a session, send turns."""

    def __init__(self, base_url: str, app_secret: str, run_id: str, index: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.app_secret = app_secret
        self.index = index
        self.participant_id = f"{PARTICIPANT_PREFIX}{run_id}-{index:02d}"
        self.session = requests.Session()
        self.rng = random.Random(f"{run_id}-{index}")

    def _post(self, path: str, **kwargs: object) -> requests.Response:
        return self.session.post(
            f"{self.base_url}{path}", timeout=REQUEST_TIMEOUT_SECONDS, **kwargs
        )

    def setup(self) -> None:
        """Sign in like the app, create the participant and start a session."""
        response = self._post(
            "/auth/anonymous",
            json={"device_id": self.participant_id},
            headers={"X-App-Secret": self.app_secret},
        )
        response.raise_for_status()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {response.json()['access_token']}",
                "X-Participant-ID": self.participant_id,
            }
        )
        response = self.session.put(
            f"{self.base_url}/auth/me",
            json={"name": f"Loadtest {self.index}", "background": ""},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        self._post("/chat/session/start").raise_for_status()

    def send_turn(self, turn: int) -> TurnResult:
        """Send one message and time the backend until its full reply."""
        sent_at = datetime.now(timezone.utc)
        start = time.perf_counter()
        try:
            response = self._post(
                "/chat/text/",
                json={"message": synthetic_message(self.index, turn, self.rng)},
                stream=True,
            )
            for _ in response.iter_content(chunk_size=65536):
                pass  # Read the whole audio reply, as the app does.
            status: int | str = response.status_code
        except requests.RequestException as error:
            status = type(error).__name__
        return TurnResult(
            participant_id=self.participant_id,
            turn=turn,
            sent_at=sent_at.isoformat(),
            latency_seconds=round(time.perf_counter() - start, 2),
            status=status,
        )

    def stop(self) -> None:
        """Stop the session, as the app does when the user leaves."""
        try:
            self._post("/chat/session/stop")
        except requests.RequestException:
            logger.warning("Session stop failed for %s", self.participant_id)


def run_load(args: argparse.Namespace, app_secret: str) -> Path:
    """Run all participants for one cadence and save turns.csv and summary.json."""
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[TurnResult] = []
    results_lock = threading.Lock()
    start_at = time.time() + 5
    end_at = start_at + args.duration

    logger.info(
        "Run %s: %d participants, one turn every %ss for %ss against %s",
        run_id,
        args.participants,
        args.interval,
        args.duration,
        args.base_url,
    )

    def worker(index: int) -> None:
        participant = SyntheticParticipant(args.base_url, app_secret, run_id, index)
        try:
            participant.setup()
        except requests.RequestException:
            logger.exception("Setup failed for %s", participant.participant_id)
            return
        # Stagger participants evenly over one interval, like real users.
        # The first participant sends at start_at, so every one fits a run.
        next_at = start_at + args.interval * (index - 1) / max(args.participants, 1)
        turn = 0
        while next_at < end_at and turn < args.max_turns:
            time.sleep(max(0.0, next_at - time.time()))
            turn += 1
            result = participant.send_turn(turn)
            with results_lock:
                results.append(result)
            if result.status != 200:
                logger.warning(
                    "%s turn %d: %s", result.participant_id, turn, result.status
                )
            next_at += args.interval
        participant.stop()

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(1, args.participants + 1)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with (out_dir / "turns.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=[item.name for item in fields(TurnResult)]
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    ok = [result.latency_seconds for result in results if result.status == 200]
    summary = {
        "run_id": run_id,
        "base_url": args.base_url,
        "participants": args.participants,
        "interval_seconds": args.interval,
        "duration_seconds": args.duration,
        "started_at": datetime.fromtimestamp(start_at, timezone.utc).isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "turns_sent": len(results),
        "turns_ok": len(ok),
        "backend_latency_mean_seconds": round(statistics.mean(ok), 2) if ok else None,
        "backend_latency_max_seconds": max(ok) if ok else None,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Summary: %s", json.dumps(summary))
    logger.info(
        "Once the watcher shows in_flight=0, build the report with: "
        "poetry run python load_test_backend.py --report %s",
        run_id,
    )
    return out_dir


_SUMMARY_RE = re.compile(
    r"Watcher summary events=(?P<events>\d+) .*?"
    r"rated=(?P<rated>\d+) mean_lag=(?P<mean>[\d.]+)s "
    r"max_lag=(?P<max>[\d.]+)s in_flight=(?P<in_flight>\d+)"
)
_FAILURE_RE = re.compile(r"(\w+)_failure=(\d+)")
_LAG_WARNING_RE = re.compile(r"Rating lag (?P<lag>[\d.]+)s exceeds")
_LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S,%f"


def _percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))]


@dataclass
class SummaryWindow:
    """One ``Watcher summary`` line: lag over the window since the last one."""

    at: datetime
    rated: int
    mean_lag: float
    max_lag: float
    in_flight: int


@dataclass
class WatcherLogStats:
    """What the watcher logged during a run."""

    last_log_at: datetime
    windows: list[SummaryWindow] = field(default_factory=list)
    lag_warnings: list[float] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)
    moderation_calls_per_minute: dict[str, int] = field(default_factory=dict)


def _read_watcher_logs(
    namespace: str, watcher_app: str, started: datetime
) -> WatcherLogStats:
    """Parse the watcher's log lines since ``started`` with kubectl."""
    logs = subprocess.run(
        [
            "kubectl",
            "logs",
            "-n",
            namespace,
            "-l",
            f"app={watcher_app}",
            "--tail=-1",
            f"--since-time={started.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    stats = WatcherLogStats(last_log_at=started)
    for line in logs:
        try:
            logged_at = datetime.strptime(line[:23], _LOG_TIME_FORMAT).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        stats.last_log_at = max(stats.last_log_at, logged_at)
        if "POST https://api.openai.com/v1/moderations" in line:
            minute = logged_at.strftime("%H:%M")
            calls = stats.moderation_calls_per_minute
            calls[minute] = calls.get(minute, 0) + 1
        if match := _SUMMARY_RE.search(line):
            stats.windows.append(
                SummaryWindow(
                    at=logged_at,
                    rated=int(match["rated"]),
                    mean_lag=float(match["mean"]),
                    max_lag=float(match["max"]),
                    in_flight=int(match["in_flight"]),
                )
            )
            stats.failures = {
                name: int(count) for name, count in _FAILURE_RE.findall(line)
            }
        if match := _LAG_WARNING_RE.search(line):
            stats.lag_warnings.append(float(match["lag"]))
    return stats


# pylint: disable-next=too-many-locals
def report(run_id: str, out_dir: str, namespace: str, watcher_app: str) -> Path:
    """Combine the run's backend timings with the watcher's lag log lines."""
    run_dir = Path(out_dir) / run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with (run_dir / "turns.csv").open(encoding="utf-8") as file:
        turns = list(csv.DictReader(file))
    started = datetime.fromisoformat(summary["started_at"])
    ended = datetime.fromisoformat(summary["ended_at"])

    stats = _read_watcher_logs(namespace, watcher_app, started)

    ok = [float(t["latency_seconds"]) for t in turns if t["status"] == "200"]
    errors: dict[str, int] = {}
    for turn in turns:
        if turn["status"] != "200":
            errors[turn["status"]] = errors.get(turn["status"], 0) + 1
    windows = stats.windows
    rated = sum(w.rated for w in windows)
    mean_lag = sum(w.rated * w.mean_lag for w in windows) / rated if rated else None
    max_lag = max((w.max_lag for w in windows), default=None)
    target = 120.0
    # Each OK turn writes a user and an assistant message for the watcher.
    expected = 2 * len(ok)
    if max_lag is None:
        verdict = "no watcher summary line in the window yet"
    elif rated < 0.9 * expected:
        verdict = (
            f"INCOMPLETE: only {rated} of ~{expected} messages rated, "
            "so the lag does not cover the run"
        )
    else:
        verdict = (
            "within" if max_lag <= target else "OVER"
        ) + f" the {target:.0f} s target"

    lines = [
        f"# Load test {run_id}",
        "",
        f"- Backend: {summary['base_url']}",
        f"- Load: {summary['participants']} participants, one message every "
        f"{summary['interval_seconds']:.0f} s for {summary['duration_seconds']:.0f} s "
        f"(~{summary['participants'] * 2 * 60 / summary['interval_seconds']:.0f} messages/min "
        "incl. assistant replies)",
        f"- Window: {started:%Y-%m-%d %H:%M:%S} to {ended:%H:%M:%S} UTC; "
        f"watcher logs read until {stats.last_log_at:%H:%M:%S}",
        "",
        "## Watcher rating lag (message written → flag written)",
        "",
        f"- **Verdict: {verdict}**",
        f"- Messages rated: {rated} of ~{expected} expected, in {len(windows)} summary windows",
        f"- Mean lag: {mean_lag:.1f} s" if mean_lag is not None else "- Mean lag: n/a",
        f"- Max lag: {max_lag:.1f} s" if max_lag is not None else "- Max lag: n/a",
        f"- Lag warnings (> {target:.0f} s): {len(stats.lag_warnings)}"
        + (f", worst {max(stats.lag_warnings):.1f} s" if stats.lag_warnings else ""),
        f"- Provider failures since watcher start: {stats.failures or 'none reported'}",
        "",
        "| Summary at (UTC) | Rated | Mean lag (s) | Max lag (s) | In flight |",
        "|---|---|---|---|---|",
        *(
            f"| {w.at:%H:%M:%S} | {w.rated} | {w.mean_lag:.1f} | "
            f"{w.max_lag:.1f} | {w.in_flight} |"
            for w in windows
        ),
        "",
        "## Backend reply time (send → full audio reply)",
        "",
        f"- Turns sent: {len(turns)}, OK: {len(ok)}, errors: {errors or 'none'}",
    ]
    if ok:
        lines += [
            f"- Mean {statistics.mean(ok):.1f} s, median {statistics.median(ok):.1f} s, "
            f"p95 {_percentile(ok, 0.95):.1f} s, max {max(ok):.1f} s",
        ]
    lines += [
        "",
        "## Moderation calls per minute (watcher → OpenAI)",
        "",
        "| Minute (UTC) | Calls |",
        "|---|---|",
        *(
            f"| {minute} | {count} |"
            for minute, count in sorted(stats.moderation_calls_per_minute.items())
        ),
        "",
    ]
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return report_path


def cleanup(run_id: str, mongodb_uri: str, database: str, delete: bool) -> None:
    """Count, and with ``delete`` remove, every document of one run."""
    from pymongo import MongoClient  # pylint: disable=import-outside-toplevel

    if not run_id.strip():
        raise SystemExit("--cleanup needs a run ID")
    query = {"participant_id": {"$regex": f"^{PARTICIPANT_PREFIX}{run_id}-"}}
    with MongoClient(mongodb_uri) as client:
        db = client[database]
        for name in CLEANUP_COLLECTIONS:
            count = db[name].count_documents(query)
            if delete and count:
                deleted = db[name].delete_many(query).deleted_count
                logger.info("%s: deleted %d", name, deleted)
            else:
                logger.info("%s: %d documents", name, count)


def main() -> None:
    """Parse arguments and run a load test, a report or a cleanup."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--participants", type=int, default=40)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--max-turns", type=int, default=10_000)
    parser.add_argument("--run-id")
    parser.add_argument("--out-dir", default="load_test_runs")
    parser.add_argument(
        "--smoke", action="store_true", help="One participant, one turn."
    )
    parser.add_argument("--report", metavar="RUN_ID")
    parser.add_argument("--namespace", default="elderbot")
    parser.add_argument("--watcher-app", default="hcms-watcher-elderbot-rct-dev")
    parser.add_argument("--cleanup", metavar="RUN_ID")
    parser.add_argument("--database", default="android-app")
    parser.add_argument(
        "--delete", action="store_true", help="With --cleanup: delete, not just count."
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    load_dotenv()
    if args.report:
        report(args.report, args.out_dir, args.namespace, args.watcher_app)
        return
    if args.cleanup:
        mongodb_uri = os.environ.get("LOADTEST_MONGODB_URI")
        if not mongodb_uri:
            raise SystemExit("Set LOADTEST_MONGODB_URI to the dev MongoDB URI.")
        cleanup(args.cleanup, mongodb_uri, args.database, args.delete)
        return
    if args.smoke:
        args.participants, args.max_turns, args.duration = 1, 1, 60.0
    app_secret = os.environ.get("APP_AUTH_SECRET")
    if not app_secret:
        raise SystemExit("Set APP_AUTH_SECRET to the dev backend's app secret.")
    run_load(args, app_secret)


if __name__ == "__main__":
    main()
