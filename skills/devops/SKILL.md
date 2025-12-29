---
name: devops-cloud
description: Master DevOps, cloud infrastructure, containerization, and Kubernetes. Learn Docker, Terraform, AWS, CI/CD pipelines, monitoring, and production infrastructure management.
sasmp_version: "1.3.0"
bonded_agent: 01-frontend-design-agent
bond_type: PRIMARY_BOND
---

# DevOps & Cloud Infrastructure Skills

## Quick Start

DevOps combines development and operations to deliver applications reliably and at scale. Start with containerization and progress to orchestration and infrastructure automation.

### Docker Basics
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000
CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t my-app:1.0 .
docker run -p 5000:5000 my-app:1.0
```

### Docker Compose
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=password
    volumes:
      - db_data:/var/lib/postgresql/data

volumes:
  db_data:
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: my-app:1.0
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### Terraform Infrastructure
```hcl
# main.tf
provider "aws" {
  region = "us-east-1"
}

resource "aws_instance" "web_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t2.micro"

  tags = {
    Name = "web-server"
  }
}

resource "aws_security_group" "web" {
  name = "web-sg"

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

## Core Topics

### 1. Linux & System Administration
- **Fundamentals**: File systems, processes, permissions
- **Command Line**: Bash scripting, automation
- **User Management**: Users, groups, sudo
- **System Services**: systemd, service management
- **Networking**: IP addressing, DNS, firewalls

### 2. Docker & Containerization
- **Concepts**: Images, containers, registries
- **Dockerfiles**: Building images, best practices
- **Docker Compose**: Multi-container applications
- **Networking**: Container communication, overlay networks
- **Storage**: Volumes, bind mounts, data management
- **Optimization**: Image size, layer efficiency

### 3. Kubernetes Orchestration
- **Architecture**: Control plane, worker nodes, pods
- **Core Objects**: Pods, Deployments, Services
- **Advanced**: StatefulSets, DaemonSets, Jobs
- **Networking**: Services, Ingress, network policies
- **Storage**: Persistent volumes, storage classes
- **Security**: RBAC, pod security, secrets management
- **Monitoring**: Logging, metrics, health checks

### 4. Terraform & Infrastructure as Code
- **HCL Syntax**: Resources, variables, outputs
- **Providers**: AWS, GCP, Azure
- **Modules**: Reusable infrastructure code
- **State Management**: Local, remote, backends
- **Workspaces**: Multiple environments
- **Best Practices**: Variable organization, secrets

### 5. Cloud Platforms (AWS)
- **Compute**: EC2, ECS, Lambda, Fargate
- **Storage**: S3, EBS, CloudFront
- **Databases**: RDS, DynamoDB, Elasticache
- **Networking**: VPC, security groups, load balancers
- **IAM**: Roles, policies, permissions
- **Monitoring**: CloudWatch, CloudTrail

### 6. CI/CD Pipelines
- **Version Control**: Git workflows, GitHub/GitLab
- **Tools**: Jenkins, GitHub Actions, GitLab CI
- **Pipeline Stages**: Build, test, deploy
- **Artifact Management**: Container registries, artifact stores
- **Deployment Strategies**: Blue-green, canary, rolling
- **Secrets Management**: Vault, encrypted variables

### 7. Monitoring & Logging
- **Metrics**: Prometheus, collection, storage
- **Dashboards**: Grafana, visualization
- **Logging**: ELK Stack, Splunk, centralized logging
- **Alerting**: Prometheus alerts, notification channels
- **APM**: Application performance monitoring
- **Observability**: Tracing, distributed tracing

### 8. Database Management
- **Relational**: PostgreSQL, MySQL administration
- **NoSQL**: MongoDB, DynamoDB management
- **Backup & Recovery**: Automated backups, PITR
- **Replication**: Master-slave, high availability
- **Scaling**: Sharding, read replicas

### 9. Configuration Management
- **Ansible**: Automation, playbooks
- **Chef/Puppet**: Infrastructure as code
- **Cloud-Init**: Instance initialization
- **Environment Variables**: Secrets management
- **ConfigMaps**: Kubernetes configuration

### 10. Security & Compliance
- **Network Security**: Firewalls, VPNs, WAF
- **Access Control**: RBAC, IAM, SSH keys
- **Encryption**: TLS/SSL, data encryption
- **Secrets**: Password management, API keys
- **Compliance**: Auditing, compliance checks

## Learning Path

### Month 1-2: Linux Fundamentals
- Linux command line mastery
- Bash scripting
- System administration basics
- SSH and remote access

### Month 3-4: Containerization
- Docker fundamentals and Dockerfiles
- Docker Compose for multi-container apps
- Container registries
- Container networking and storage

### Month 5-6: Container Orchestration
- Kubernetes architecture
- Pods, Deployments, Services
- Helm package management
- Basic monitoring and logging

### Month 7-8: Infrastructure as Code
- Terraform fundamentals
- Cloud provider integration (AWS/GCP/Azure)
- Module creation and reuse
- State management

### Month 9-10: CI/CD & Automation
- Git workflows and branching
- CI/CD pipeline creation
- Automated testing and deployment
- Release management

### Month 11+: Advanced Topics
- Kubernetes advanced patterns
- Multi-cloud strategies
- Production optimization
- Disaster recovery and scaling

## Tools & Technologies

### Core Tools
- **Containerization**: Docker, Podman
- **Orchestration**: Kubernetes, Docker Swarm, ECS
- **Infrastructure**: Terraform, CloudFormation, Ansible
- **CI/CD**: Jenkins, GitHub Actions, GitLab CI, ArgoCD
- **Monitoring**: Prometheus, Grafana, Datadog
- **Logging**: ELK Stack, Splunk, Loki

### Cloud Providers
- **AWS**: EC2, RDS, S3, Lambda, ECS, EKS
- **Google Cloud**: Compute Engine, GKE, Cloud SQL
- **Azure**: VMs, Azure Kubernetes Service (AKS), App Service

### Development Tools
- **Version Control**: Git, GitHub, GitLab
- **IDEs**: VS Code, JetBrains IDEs
- **Package Managers**: apt, yum, brew
- **Container Registry**: Docker Hub, ECR, GCR, ACR

## Advanced Topics

### Kubernetes at Scale
- **Multi-cluster Management**: Fleet management, federation, hub-and-spoke
- **Workload Management**: StatefulSets, DaemonSets, Jobs, CronJobs
- **Custom Resources**: CRDs for extending Kubernetes
- **Network Policies**: Micro-segmentation, ingress/egress rules
- **Storage Solutions**: StatefulSets with PersistentVolumes, StatefulSet patterns
- **Service Mesh**: Istio, Linkerd for observability and traffic management
- **Helm Advanced**: Chart repositories, dependency management, hooks
- **Security**: RBAC, Pod Security Policy, admission webhooks
- **Cost Optimization**: Resource limits, autoscaling, node consolidation

### Advanced CI/CD
- **Pipeline Patterns**: Blue-green, canary, rolling deployments
- **GitOps Workflows**: ArgoCD, Flux for declarative deployments
- **Artifact Management**: Binary repositories, container scanning, signing
- **Testing Automation**: Unit, integration, E2E, security, performance tests
- **Release Automation**: Version bumping, changelog generation, release notes
- **Compliance Automation**: Policy enforcement, audit logging, compliance scanning

### Observability & Monitoring
- **Metrics**: Prometheus scraping, custom metrics, time-series databases
- **Logging**: Centralized logging, log aggregation, log parsing
- **Tracing**: Distributed tracing, correlation IDs, trace context propagation
- **Alerting**: Alert routing, escalation policies, on-call management
- **Dashboards**: Grafana, custom dashboards, SLO dashboards
- **SLO/SLI**: Service level objectives, error budgets
- **Cost Monitoring**: Resource tracking, chargeback, budget alerts

### Security & Compliance
- **Secret Management**: HashiCorp Vault, AWS Secrets Manager, sealed secrets
- **Container Security**: Scanning, signing, registry policy enforcement
- **Network Security**: Network policies, service mesh security, mTLS
- **IAM/RBAC**: Least privilege, role design, access reviews
- **Audit Logging**: Compliance logging, tamper detection
- **Compliance**: HIPAA, PCI-DSS, SOC 2, GDPR compliance automation
- **Vulnerability Management**: CVE tracking, patching, remediation

### Advanced Terraform
- **State Management**: Remote state, locking, state migrations
- **Modules**: Module patterns, local vs. remote modules, testing
- **Workspaces**: Environment separation, state isolation
- **Policy as Code**: Sentinel, OPA for compliance enforcement
- **Testing**: Terratest, terraform test, plan output validation
- **Version Management**: Provider versioning, constraint handling
- **Multi-cloud**: Deploying across AWS, GCP, Azure

### Disaster Recovery & Business Continuity
- **Backup Strategies**: RTO, RPO, backup retention, backup testing
- **Database Replication**: Synchronous vs. asynchronous, failover handling
- **Disaster Recovery Drills**: Regular testing, documentation, automation
- **Multi-region**: Geographic redundancy, data replication, traffic failover
- **Chaos Engineering**: Intentional failures, resilience testing, learning
- **Post-incident**: Blameless postmortems, action items, prevention

## Common Pitfalls & Gotchas

1. **Docker Layer Caching Ignored**: Building large layers repeatedly
   - **Fix**: Put frequently changing layers last
   - **Example**: COPY code last, dependencies early

2. **No Resource Limits**: Container consuming all node resources
   - **Fix**: Set requests and limits for all containers
   - **Example**: `resources: {requests: {memory: "256Mi", cpu: "100m"}}`

3. **Secrets in Environment Variables**: Exposed in process listing
   - **Fix**: Use secret management (Vault, Sealed Secrets, ExternalSecrets)
   - **Impact**: Prevents accidental secret exposure

4. **Stateful Services in Containers**: Hard to scale and manage
   - **Fix**: Design stateless services, use StatefulSets if needed
   - **Example**: Move sessions to Redis, not in-memory

5. **Manual Infrastructure Changes**: Drift from version control
   - **Fix**: All infrastructure as code, GitOps workflows
   - **Problem**: Prevents reproducibility, hard to audit

6. **No Network Policies**: All pods can communicate with all pods
   - **Fix**: Implement network policies for micro-segmentation
   - **Security**: Reduces blast radius of compromised containers

7. **Inadequate Monitoring**: Silent failures in production
   - **Fix**: Comprehensive monitoring (metrics, logs, traces)
   - **Tools**: Prometheus, Grafana, ELK, Jaeger

8. **Poor Secrets Rotation**: Long-lived credentials
   - **Fix**: Automated rotation, short-lived tokens
   - **Example**: AWS IAM temporary credentials, service account tokens

9. **No Cost Monitoring**: Unexpected cloud bills
   - **Fix**: Budget alerts, resource tagging, cost analysis
   - **Tools**: AWS Cost Explorer, GCP Cost Management

10. **Configuration Drift**: Infrastructure state differs from version control
    - **Fix**: GitOps, declarative management, continuous reconciliation
    - **Tools**: ArgoCD, Flux, Terraform

## Production Deployment Checklist

- [ ] Container images scanned for vulnerabilities
- [ ] Resource limits and requests configured
- [ ] Health checks configured (liveness, readiness)
- [ ] Secrets managed properly (not in code, encrypted)
- [ ] Logging and monitoring set up
- [ ] Network policies configured
- [ ] RBAC configured with least privilege
- [ ] Backup and disaster recovery tested
- [ ] Performance tested under load
- [ ] Security audit completed
- [ ] Documentation complete (runbooks, architecture)
- [ ] On-call procedures established
- [ ] Incident response plan in place
- [ ] Cost monitoring configured
- [ ] Compliance requirements verified

## Best Practices

1. **Infrastructure as Code** - Version control for infrastructure
2. **Automation** - Minimize manual processes
3. **Monitoring** - Continuous observability
4. **Security** - Defense in depth, least privilege
5. **Scalability** - Design for growth
6. **Reliability** - High availability, disaster recovery
7. **Documentation** - Runbooks, architecture docs
8. **Testing** - Test infrastructure changes

## Project Ideas

1. **Containerize Application** - Docker, Docker Compose
2. **Deploy to Kubernetes** - Local K8s cluster, minikube
3. **Infrastructure Automation** - Terraform for cloud resources
4. **CI/CD Pipeline** - GitHub Actions or Jenkins
5. **Monitoring Setup** - Prometheus, Grafana, Loki
6. **Multi-Environment** - Dev, staging, production
7. **Disaster Recovery** - Backup, restore procedures

## Resources

- [Docker Documentation](https://docs.docker.com)
- [Kubernetes.io](https://kubernetes.io)
- [Terraform Docs](https://www.terraform.io/docs)
- [AWS Training](https://aws.amazon.com/training)
- [Linux Academy](https://linuxacademy.com)

## Next Steps

1. Master Linux and bash scripting
2. Learn Docker containerization
3. Set up your first Kubernetes cluster
4. Automate infrastructure with Terraform
5. Build robust CI/CD pipelines
6. Implement comprehensive monitoring
7. Deploy production systems
