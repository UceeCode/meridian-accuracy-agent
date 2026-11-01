# Meridian & Co. - Seed Dataset

Period covered: **September 2026**. All data is fictional.

The data is deliberately imperfect. Blank cells mean a value is missing. Some records conflict, some are absent, and some questions cannot be answered from what is here. Do not generate your own data.

| File | Contents |
|---|---|
| `transactions.csv` | One row per accounting task completed in the month |
| `task_records.csv` | Employee task logs: who worked on it, when, which procedure version they applied, whether they signed the checklist |
| `automated_process_log.csv` | Events from the company's automated systems |
| `procedures.csv` | Management-approved procedures, with version history |
| `evaluation_system_output.csv` | What the company's existing evaluation system flagged as errors, and who it attributed them to |
| `employees.csv` | Staff list |
| `documents/` | Internal policies, notices, and memos |

Timestamps are local time. `status` in `transactions.csv` is the outcome after review.
