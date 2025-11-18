---
name: devops-cloud
description: Master DevOps, cloud infrastructure, containerization, and Kubernetes. Learn Docker, Terraform, AWS, CI/CD pipelines, monitoring, and production infrastructure management.
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
