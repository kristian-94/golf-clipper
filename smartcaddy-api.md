# SmartCaddy API

Short reference for scripting against SmartCaddy.

## Base URL

```
https://smartcaddy.io/api
```

Local dev: `http://localhost:3001/api`.

## Auth

Two ways. Scripts and integrations should use a **Personal Access Token (PAT)**.

### Personal Access Token (recommended for scripts)

1. Sign in at smartcaddy.io.
2. Go to Settings → Personal Access Tokens → **Create token**.
3. Copy the token (`sc_pat_…`) — it's shown once.
4. Send it as a Bearer token:

```
Authorization: Bearer sc_pat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

PATs can do everything a user can **except** manage other PATs (create/list/revoke requires a JWT session).

### JWT session (used by the web/mobile app)

Sign in via `POST /auth/login` — returns a JWT used the same way:

```
Authorization: Bearer <jwt>
```

## Required header

Every request must include the client app version:

```
X-App-Version: 1.0.0
```

Stale clients get `426 Upgrade Required`.

## Common endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/rounds` | List your rounds |
| `POST` | `/rounds` | Create a round |
| `GET` | `/rounds/:id` | Get a round |
| `POST` | `/rounds/:id/complete` | Mark complete |
| `GET` | `/rounds/:id/leaderboard` | Live leaderboard |
| `GET` | `/rounds/shared/:shareToken` | Resolve a shared-round URL (no auth needed) |
| `POST` | `/rounds/:id/scores` | Submit a hole score |
| `PUT` | `/rounds/:id/scores/:scoreId` | Edit a score |
| `DELETE` | `/rounds/:id/scores/:scoreId` | Delete a score |
| `POST` | `/rounds/:id/players/:playerId/notes` | Add a note |
| `DELETE` | `/rounds/:id/players/:playerId/notes/:noteId` | Remove a note |

Full list: `apps/backend/src/routes/rounds.routes.ts`.

## Example — submit a score

```bash
curl -X POST https://smartcaddy.io/api/rounds/$ROUND_ID/scores \
  -H "Authorization: Bearer $SMARTCADDY_PAT" \
  -H "X-App-Version: 1.0.0" \
  -H "Content-Type: application/json" \
  -d '{
    "playerSessionId": "clx…",
    "holeNumber": 1,
    "strokes": 5,
    "putts": 2
  }'
```

Body rules: `holeNumber` 1–18, `strokes ≥ 1`, `putts ≤ strokes`, `putts` optional/nullable.

## Example — resolve a shared round URL

```bash
# https://smartcaddy.io/rounds/shared/<shareToken>
curl https://smartcaddy.io/api/rounds/shared/$SHARE_TOKEN \
  -H "Authorization: Bearer $SMARTCADDY_PAT" \
  -H "X-App-Version: 1.0.0"
```

Returns the round (including `id`) without needing to know it up front.

## Errors

JSON body with `error` message. Common:

- `401` — missing/invalid token
- `403` — token lacks scope (e.g. PAT trying to manage PATs)
- `404` — not found or not a participant
- `422` — validation failed (Zod details in body)
- `426` — `X-App-Version` below minimum
- `429` — rate limited
