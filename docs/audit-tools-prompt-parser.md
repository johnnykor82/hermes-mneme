# Audit verification: tools, prompt assembly, parser

Date: 2026-10-10. All fixtures use temporary SQLite databases and deterministic
token counting or stubbed retrieval. No production events were changed.

## Confirmed findings

- **F06 / T13:** `segment='current'` previously passed `None` to the indexer,
  removing the topic filter. It now resolves the active engine segment for
  the active session, or persisted state/latest event for a specific other
  session. Compression lineage stays within the current conversation and is
  filtered by that segment, including legacy segment IDs.
- **F12 / T27:** A mandatory current message larger than the entire budget
  returned an oversized prompt. Large system overrides, memory hints,
  checkpoints and assistant tool-call payloads also escaped the final check.
  The final assembled request now includes message framing, tool-call fields
  and all injected blocks in its count. It drops retrieved context first,
  then oldest recent groups down to the existing two-message floor. Assistant
  tool calls and their contiguous output messages stay together in floor
  selection, extension and every trimming stage. On tool continuations the
  existing current user request is mandatory. If
  mandatory content still cannot fit, `PromptBudgetExceeded` carries
  `actual_tokens` and `budget_tokens`; no oversized message list is returned.
- **F14 / T28:** Assistant text was discarded whenever the same message
  contained tool calls. The parser now appends a text event after the calls,
  preserving the existing tool-call subindices and deterministic event IDs.
  Original call IDs are retained on both call and output parsed events for
  graph association (F16, implemented in the engine/store changes).

## Tool scope contract

`session='current', segment='current'` searches the active topic in the
current compression lineage. `segment='all'` searches all topics in that
scope. `session='all'` requires `segment='all'`. Multiple explicit session
IDs also require `segment='all'` or a specific segment ID; there is no
single unambiguous current topic across unrelated sessions. A single
explicit session resolves its current topic using stored state or its
latest event. Missing current topics return an explicit error.

Examples:

```json
{"query": "what was decided", "segment": "current"}
{"query": "what was decided", "segment": "all"}
{"query": "last week's decision", "session": "all", "segment": "all"}
```

## Prompt budget limits

The strict contract applies to the configured tokenizer estimate plus
explicitly reserved external overhead. Provider tokenizers can differ.
Cached `system_prompt_tokens` may include tool schemas outside the message
list; the original included system text is deducted from this reserve to
avoid double counting. Arbitrary external schemas require the host to
reserve their token count. These tests do not establish provider actual
usage accuracy or explain Gateway's 14.59-second compression incident.

The native engine must propagate the overflow error. Falling back to the
original oversized history after this error would violate the contract.

## Validation evidence

The first new regression run against the original implementations produced
**10 failures**, matching removed segment filters, absent overflow errors,
lost mixed text and missing call IDs. After changes, the final targeted run
passed **26 tests**, including the existing tools/parser tests and additional
scope/budget/replay cases. Follow-up protocol review added four regressions;
all four failed before group-safe trimming/current-user protection and pass
after it.

```sh
python -m pytest -q \
  tests/unit/test_audit_tools_prompt_parser.py \
  tests/unit/test_tools.py tests/unit/test_parser.py
```

Existing native-hook fixtures used impossible 80/120-token budgets for
mandatory state and recent messages. These fixtures need viable assembly
budgets, while new overflow tests separately assert the explicit failure.
Host integration and full-suite results are recorded in the overall audit
report. Actual occurrence of mixed text/tool responses in a production
Hermes model route remains unmeasured; support is verified for the accepted
OpenAI message format.
