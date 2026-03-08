# WithCare Agent — Test Dialogue Order

Run these test scripts in order. Each builds on the state from previous tests.

| # | File | User ID | What It Tests | Reset Before? |
|---|------|---------|---------------|---------------|
| 1 | `01_basic_routing.md` | `test-alice` | Quick answer, emotional support, agent routing | Fresh start |
| 2 | `02_info_collection_and_search.md` | `test-alice` | Full request lifecycle: info collection → deep search | No |
| 3 | `03_state_checkpoint.md` | `test-alice` | Reset (simulate pod restart), verify state restores | Yes (then continue) |
| 4 | `04_cross_session_resume.md` | `test-alice` | New request that matches old one → resumes from collected info | No |
| 5 | `05_human_escalation.md` | `test-bob` | 3+ deep_search rounds → human_comm proposal → confirm/decline | Fresh start |
| 6 | `06_multi_user_isolation.md` | `test-alice` + `test-carol` | Two users don't see each other's data | Fresh start |

## How to Use

1. Open the chat UI at `http://localhost:8000` (or your deployed URL)
2. Type the **User ID** in the header input field
3. Follow the dialogue — send the **YOU:** lines exactly (or close enough)
4. Check the **EXPECT:** sections against what you see in the reply and debug panel
5. The debug panel (right side) shows `nodes_visited`, `current_agent`, `active_request`, etc.

## Debug Panel Fields to Watch

- **current_agent**: Which agent handled the turn (e.g., `quick_answer`, `info_collection`, `deep_search`)
- **turn_mode**: `continuation` or `new_intent`
- **nodes_visited**: The graph execution path (e.g., `["turn_router", "upstream_delegator", "info_collection"]`)
- **active_request.status**: `collecting`, `validated`, `executed`, `paused`
- **active_request.info_collection.summary**: What the agent has collected so far
- **active_request.info_collection.readiness**: `needs_more`, `can_proceed_but_incomplete`, `ready`
