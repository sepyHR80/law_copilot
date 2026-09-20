# Law Copilot — Operational Runbook

This runbook specifies operational procedures for deployment, backups, scaling, incident response, and secret rotation per Section 26.2 of `LAW_COPILOT_AGENT_MASTER_SPEC.md`.

---

## 1. Database Migrations (Alembic)

### 1.1 Verifying Current State
```bash
uv run alembic current
```

### 1.2 Applying Migrations in Production
Prior to rolling update deployment:
```bash
uv run alembic upgrade head
```

---

## 2. Backup and Disaster Recovery

### 2.1 PostgreSQL + pgvector Database Backup
```bash
pg_dump -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -d "$POSTGRES_DB" -Fc -f "law_copilot_backup_$(date +%Y%m%d_%H%M%S).dump"
```

### 2.2 Database Restore
```bash
pg_restore -U "$POSTGRES_USER" -h "$POSTGRES_HOST" -d "$POSTGRES_DB" --clean --no-owner "law_copilot_backup.dump"
```

### 2.3 Object Storage Backup
Enable S3/MinIO bucket versioning and configure cross-region replication or daily snapshots using `mc mirror`:
```bash
mc mirror myminio/law-documents backup-s3/law-documents-backup
```

---

## 3. Secret Rotation Procedures

When rotating database credentials, MinIO secret keys, or LLM API keys:
1. Update Kubernetes Secrets / Vault entries.
2. Trigger rolling restart of API deployment:
   ```bash
   kubectl rollout restart deployment/law-copilot-api -n law-copilot
   ```
3. Monitor `/readyz` probe to ensure all pods transition to Ready status without dropping active connections.

---

## 4. Incident Response & Triage

| Symptom | Probable Cause | Action |
|---|---|---|
| `/readyz` returns 503 | Database unreachable or connection pool exhausted | Check Postgres connection limits and RDS CPU; scale connection pool |
| High rate of `verification_failures_total` | LLM hallucinating citations or low retrieval recall | Inspect Cross-Encoder reranker scores; verify ingested document versions |
| High latency on `/api/v1/rag/query` | Cross-encoder batch inference or LLM timeout | Scale API replicas; review LLM gateway latency metrics in Jaeger |
