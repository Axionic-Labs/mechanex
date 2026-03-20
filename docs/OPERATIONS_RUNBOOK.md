# Operations Runbook

## Common Failures

## 1) Remote auth failures

Symptoms:
- 401 errors from policy or generation endpoints

Checks:
- API key/JWT validity
- Account credit state
- Endpoint base URL correctness

Actions:
- Rotate API key
- Re-authenticate client
- Verify account billing status

## 2) Policy run failures

Symptoms:
- policy run returns verifier/constraint failures repeatedly

Checks:
- JSON schema validity
- regex/grammar constraints
- steering vector ID exists

Actions:
- run with `include_trace=True`
- inspect candidate rejection reasons
- reduce constraints or increase retry/repair settings

## 3) Latency spikes

Symptoms:
- p95/p99 response times increase

Checks:
- sampling strategy (`best_of_n`, ensemble)
- remote service health
- fallback/repair frequency

Actions:
- lower candidate count
- disable expensive verifiers temporarily
- switch to lower-latency policy profile
