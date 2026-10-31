# Video SSL repository development and research

For repository development, read README.md, docs/configuration.md and docs/lifecycle.md.
Use SuiteController for lifecycle, Session for per-round tools, and the concrete
video backend for model/data operations. Implement authorized changes without
starting GPU experiments automatically. Keep code, current documentation and tests
consistent. Maintain active method Markdown and generated report text in English;
preserve source attribution and frozen research data text.

For launched research sessions (VIDEO_SSL_AGENT=1), the assigned workspace AGENTS.md,
METHOD.md, COMMANDS.md and task.json define the workflow and reading scope. Begin
there. Do not use repository-wide discovery, history/setup documentation, other
protocol variants, other suites, test feedback, or environment/cache directories
as research inputs. The Agent owns curation code and notes; the host owns framework
changes, dependencies, suite lifecycle changes and final test.

Historical comparison documents under docs/history/ are development references,
not current operating instructions. The recoverable legacy archive is outside
this checkout. Optional skills are research inputs only when the selected workspace
METHOD.md explicitly authorizes them.
