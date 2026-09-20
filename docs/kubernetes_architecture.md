# Law Copilot — Kubernetes Production Architecture

This document describes the production deployment topology for Law Copilot in enterprise Kubernetes environments, conforming to Section 26.2 of `LAW_COPILOT_AGENT_MASTER_SPEC.md`.

---

## 1. High-Level Cluster Topology

```text
                                  Internet
                                     │
                                     ▼
                            [Ingress Controller]
                         (TLS Termination, Traefik/Nginx)
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
     [Service: law-copilot-api]              [Service: jaeger-ui / grafana]
                  │                                     │
      ┌───────────┴───────────┐                         │
      ▼                       ▼                         │
[Pod: API Replica 1]    [Pod: API Replica 2]            │
      │                       │                         │
      ├───────────────────────┴─────────────────────────┤
      ▼                                                 ▼
[Managed PostgreSQL + pgvector]            [Managed Object Storage (S3/GCS)]
 (Cloud SQL, Aurora, or CrunchyData)         (AWS S3, Google Cloud Storage)
```

---

## 2. Core Workload Manifests

### 2.1 API Deployment (`law-copilot-api`)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: law-copilot-api
  namespace: law-copilot
  labels:
    app.kubernetes.io/name: law-copilot
    app.kubernetes.io/component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: law-copilot-api
  template:
    metadata:
      labels:
        app: law-copilot-api
    spec:
      containers:
        - name: api
          image: law-copilot:0.1.0
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
              name: http
          envFrom:
            - configMapRef:
                name: law-copilot-config
            - secretRef:
                name: law-copilot-secrets
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
            limits:
              cpu: 2000m
              memory: 4Gi
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /readyz
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
```

### 2.2 Horizontal Pod Autoscaler (HPA)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: law-copilot-api-hpa
  namespace: law-copilot
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: law-copilot-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## 3. Storage and Managed Database Mapping

1. **PostgreSQL + pgvector**:
   - Production target: Managed Cloud PostgreSQL (AWS RDS with `pgvector` extension enabled, Google Cloud SQL, or Azure Database for PostgreSQL).
   - High availability: Multi-AZ failover with read replicas for vector search scale.
2. **Object Storage**:
   - Production target: AWS S3 or Google Cloud Storage via S3-compatible IAM endpoint.
   - Credentials injected via Kubernetes Workload Identity / IRSA (IAM Roles for Service Accounts), avoiding hardcoded static keys.
