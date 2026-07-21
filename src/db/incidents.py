from __future__ import annotations

import csv
import hashlib
import io
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "corpus.db"
VALID_REVIEW_STATUSES = {"Pending", "Resolved"}
CSV_COLUMNS = [
    "Incident ID",
    "Document A",
    "Document B",
    "Similarity Score",
    "Severity Rank",
    "Review Status",
    "Date Flagged",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalise_pair(doc_a: str, doc_b: str) -> tuple[str, str]:
    return tuple(sorted((str(doc_a).strip(), str(doc_b).strip())))  # type: ignore[return-value]


def _normalise_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return min(1.0, max(0.0, score))


def _severity_rank(flag: Mapping[str, Any]) -> str:
    raw = str(flag.get("severity", "")).strip()
    if raw:
        return raw
    score = _normalise_score(flag.get("similarity", 0.0))
    if score >= 0.90:
        return "High"
    if score >= 0.59:
        return "Medium"
    return "Low"


def build_incident_id(doc_a: str, doc_b: str) -> str:
    first, second = _normalise_pair(doc_a, doc_b)
    digest = hashlib.sha256(f"{first}\0{second}".encode("utf-8")).hexdigest()
    return f"INC-{digest[:12].upper()}"


def init_incident_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    with closing(sqlite3.connect(str(db_path))) as conn:
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plagiarism_incidents (
                    incident_id TEXT PRIMARY KEY,
                    document_a TEXT NOT NULL,
                    document_b TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    severity_rank TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'Pending'
                        CHECK (review_status IN ('Pending', 'Resolved')),
                    date_flagged TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_status "
                "ON plagiarism_incidents(review_status)"
            )
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise sqlite3.Error(f"Failed to initialize incident database: {e}") from e

def _fetch_all_incidents(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT incident_id, document_a, document_b, similarity_score,
               severity_rank, review_status, date_flagged, last_seen
        FROM plagiarism_incidents
        ORDER BY date_flagged DESC, incident_id ASC
        """
    ).fetchall()

    return [dict(row) for row in rows]
def sync_flagged_incidents(
    flags: Iterable[Mapping[str, Any]],
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    now: str | None = None,
) -> list[dict[str, Any]]:
    init_incident_db(db_path)
    timestamp = now or _utc_now_iso()

    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.row_factory = sqlite3.Row

        try:
            for flag in flags:
                doc_a = str(flag.get("doc_a", "")).strip()
                doc_b = str(flag.get("doc_b", "")).strip()

                if not doc_a or not doc_b or doc_a == doc_b:
                    continue

                first, second = _normalise_pair(doc_a, doc_b)

                conn.execute(
                    """
                    INSERT INTO plagiarism_incidents (
                        incident_id, document_a, document_b,
                        similarity_score, severity_rank,
                        review_status, date_flagged, last_seen
                    )
                    VALUES (?, ?, ?, ?, ?, 'Pending', ?, ?)
                    ON CONFLICT(incident_id) DO UPDATE SET
                        similarity_score = excluded.similarity_score,
                        severity_rank = excluded.severity_rank,
                        last_seen = excluded.last_seen
                    """,
                    (
                        build_incident_id(first, second),
                        first,
                        second,
                        _normalise_score(flag.get("similarity", 0.0)),
                        _severity_rank(flag),
                        timestamp,
                        timestamp,
                    ),
                )


            conn.commit()

            rows = conn.execute(
                """
                SELECT incident_id, document_a, document_b,
                       similarity_score, severity_rank,
                       review_status, date_flagged, last_seen
                FROM plagiarism_incidents
                ORDER BY date_flagged DESC, incident_id ASC
                """
            ).fetchall()

            return [dict(row) for row in rows]

        except sqlite3.Error as e:
            conn.rollback()
            raise sqlite3.Error(f"Failed to synchronize incidents: {e}") from e

                VALUES (?, ?, ?, ?, ?, 'Pending', ?, ?)
                ON CONFLICT(incident_id) DO UPDATE SET
                    similarity_score = excluded.similarity_score,
                    severity_rank = excluded.severity_rank,
                    last_seen = excluded.last_seen
                """,
                (
                    build_incident_id(first, second),
                    first,
                    second,
                    _normalise_score(flag.get("similarity", 0.0)),
                    _severity_rank(flag),
                    timestamp,
                    timestamp,
                ),
            )
        conn.commit()
        return _fetch_all_incidents(conn)


def get_all_incidents(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    init_incident_db(db_path)
    with closing(sqlite3.connect(str(db_path))) as conn:
       return _fetch_all_incidents(conn)


def update_review_status(
    incident_id: str,
    review_status: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> bool:
    status = str(review_status).strip().title()

    if status not in VALID_REVIEW_STATUSES:
        raise ValueError(
            f"review_status must be one of {sorted(VALID_REVIEW_STATUSES)}"
        )

    init_incident_db(db_path)

    with closing(sqlite3.connect(str(db_path))) as conn:
        try:
            cursor = conn.execute(
                "UPDATE plagiarism_incidents SET review_status = ? WHERE incident_id = ?",
                (status, str(incident_id).strip()),
            )

            conn.commit()
            return cursor.rowcount > 0

        except sqlite3.Error as e:
            conn.rollback()
            raise sqlite3.Error(f"Failed to update review status: {e}") from e

def incidents_to_csv(incidents: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for incident in incidents:
        writer.writerow(
            {
                "Incident ID": incident.get("incident_id", ""),
                "Document A": incident.get("document_a", ""),
                "Document B": incident.get("document_b", ""),
                "Similarity Score": f"{_normalise_score(incident.get('similarity_score', 0.0)):.4f}",
                "Severity Rank": incident.get("severity_rank", ""),
                "Review Status": incident.get("review_status", "Pending"),
                "Date Flagged": incident.get("date_flagged", ""),
            }
        )
    return buffer.getvalue().encode("utf-8-sig")


def export_current_flags_csv(
    flags: Iterable[Mapping[str, Any]],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> bytes:
    sync_flagged_incidents(flags, db_path)
    return incidents_to_csv(get_all_incidents(db_path))
