---
name: system-architecture
description: Master software architecture, system design, design patterns, distributed systems, and technical leadership. Learn to architect scalable, reliable, and maintainable systems.
---

# System Architecture & Design Skills

## Quick Start

Software architecture involves designing systems to be scalable, reliable, maintainable, and secure. Combine design patterns with system thinking.

### Design Patterns Example (Singleton)
```python
class DatabaseConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Connection setup
        pass

# Usage
db1 = DatabaseConnection()
db2 = DatabaseConnection()
assert db1 is db2  # Same instance
```

### System Design: URL Shortener
```
Functional Requirements:
- Shorten long URLs
- Redirect to original URL
- Custom short codes optional
- Analytics tracking

Non-Functional Requirements:
- Scale to billions of URLs
- Highly available
- Low latency redirects
- Globally distributed

Architecture:
- API Layer (Load balanced)
- Cache Layer (Redis)
- Primary DB (PostgreSQL with replication)
- Analytics Service (Separate)
- CDN for global distribution
```

### Microservices Communication
```yaml
Services:
  - User Service
    - Register, login, profile
  - Product Service
    - Product catalog
  - Order Service
    - Create, manage orders

Communication:
  - Synchronous: REST/gRPC for read-heavy
  - Asynchronous: Message queues for events
  - Service Mesh: Istio for reliability

Patterns:
  - API Gateway for routing
  - Service Discovery for dynamic endpoints
  - Circuit Breaker for fault tolerance
```

## Core Topics

### 1. Design Principles
- **SOLID Principles**: Single Responsibility, Open/Closed, Liskov, Interface Segregation, Dependency Inversion
- **DRY** (Don't Repeat Yourself): Code reusability
- **KISS** (Keep It Simple, Stupid): Simplicity
- **YAGNI** (You Aren't Gonna Need It): Avoid over-engineering
- **Composition over Inheritance**: Flexible design

### 2. Design Patterns (Gang of Four)
- **Creational**: Singleton, Factory, Builder, Prototype
- **Structural**: Adapter, Bridge, Composite, Decorator, Facade, Proxy
- **Behavioral**: Observer, Strategy, Command, Iterator, Template Method, State

### 3. Architectural Patterns
- **Monolithic Architecture**: Single deployable unit
- **Microservices**: Independent, loosely coupled services
- **Serverless**: Event-driven, function-as-a-service
- **Event-Driven Architecture**: Asynchronous, event-based
- **Domain-Driven Design (DDD)**: Bounded contexts, ubiquitous language

### 4. Scalability & Performance
- **Vertical Scaling**: Increasing resources
- **Horizontal Scaling**: Adding more instances
- **Load Balancing**: Distributing traffic
- **Caching**: In-memory, distributed, CDN
- **Database Scaling**: Replication, sharding, read replicas
- **Compression**: Reducing data transfer
- **Optimization**: Query optimization, indexing

### 5. High Availability & Fault Tolerance
- **Redundancy**: Eliminating single points of failure
- **Failover**: Automatic recovery mechanisms
- **Health Checks**: Continuous monitoring
- **Circuit Breakers**: Preventing cascading failures
- **Retries with Backoff**: Handling transient failures
- **Bulkheads**: Isolating failures
- **Disaster Recovery**: RTO, RPO planning

### 6. Data Management
- **CAP Theorem**: Consistency, Availability, Partition tolerance trade-offs
- **ACID vs. BASE**: Transactional vs. eventual consistency
- **Sharding Strategies**: Range, list, hash-based
- **Replication**: Master-slave, master-master, quorum
- **Data Warehousing**: OLAP, dimensional modeling
- **Stream Processing**: Real-time data analysis

### 7. System Design Process
1. **Understand Requirements**: Functional & non-functional
2. **Estimate Scale**: QPS, storage, bandwidth
3. **Define Core Components**: Services, data stores, caches
4. **Design High-Level Architecture**: Component interaction
5. **Deep Dive**: Database design, API design, caching
6. **Identify Bottlenecks**: Where do we scale?
7. **Optimize & Iterate**: Handle edge cases

### 8. Software Testing
- **Unit Testing**: Individual function testing
- **Integration Testing**: Component interaction testing
- **System Testing**: End-to-end testing
- **Performance Testing**: Load, stress, endurance testing
- **Security Testing**: Vulnerability scanning, penetration testing
- **Test Automation**: CI/CD integration
- **TDD**: Test-driven development

### 9. DevOps & Deployment
- **Infrastructure as Code**: Terraform, CloudFormation
- **Containerization**: Docker, Kubernetes
- **CI/CD Pipelines**: Automated build, test, deploy
- **Monitoring & Logging**: Observability
- **Blue-Green Deployment**: Zero-downtime deployment
- **Canary Releases**: Gradual rollout
- **Feature Flags**: Controlled feature release

### 10. Security & Compliance
- **Authentication**: User identity verification
- **Authorization**: Access control, RBAC
- **Encryption**: Data at rest and in transit
- **API Security**: Rate limiting, input validation
- **Compliance**: GDPR, HIPAA, PCI-DSS
- **Audit Logging**: Change tracking
- **Secrets Management**: API keys, passwords

## Learning Path

### Month 1-2: Fundamentals
- SOLID principles
- Basic design patterns
- Data structures and algorithms
- System design basics

### Month 3-4: Intermediate Design
- Advanced design patterns
- Architectural patterns
- CAP theorem and consistency
- Database design

### Month 5-6: Advanced Architecture
- Microservices architecture
- Event-driven design
- Distributed systems concepts
- Scalability strategies

### Month 7-8: Production Systems
- Load balancing and caching
- Replication and sharding
- Monitoring and observability
- Disaster recovery

### Month 9-10: Real-World Design
- Case studies (Twitter, Netflix, Uber)
- Trade-off analysis
- Interview preparation
- Leadership skills

### Month 11+: Expert Level
- Domain-Driven Design
- Complex system design
- Research and cutting-edge patterns
- Mentoring and team leadership

## Design Pattern Cheatsheet

| Pattern | Purpose | When to Use |
|---------|---------|-------------|
| Singleton | Single instance | Database connections, loggers |
| Factory | Object creation | Multiple implementations |
| Observer | Event notification | Pub/Sub systems, UI updates |
| Strategy | Interchangeable algorithms | Sorting, payment methods |
| Decorator | Add features dynamically | Logging, caching wrappers |
| Adapter | Interface compatibility | Third-party integrations |
| Proxy | Control access | Caching, lazy loading |
| Builder | Complex object construction | Configuration objects |

## Architectural Trade-offs

| Choice | Pros | Cons |
|--------|------|------|
| **Monolith** | Simple, integrated testing | Hard to scale, slower deployment |
| **Microservices** | Independent scaling, fast deployment | Complex, network latency, eventually consistent |
| **Serverless** | Pay per execution, scalable | Cold starts, vendor lock-in, complex debugging |
| **Event-Driven** | Scalable, decoupled | Eventually consistent, harder to debug |
| **Monolith + Caching** | Good performance | Cache invalidation complexity |

## Tools & Technologies

### Architecture & Design
- **Diagramming**: Draw.io, Lucidchart, Miro
- **Design Patterns**: UML, architecture documentation
- **API Design**: OpenAPI, GraphQL

### Development
- **Languages**: Python, Java, Go, C#, Node.js
- **Frameworks**: Spring Boot, Django, FastAPI, NestJS
- **Databases**: PostgreSQL, MongoDB, Redis, DynamoDB

### Infrastructure
- **Containerization**: Docker, Kubernetes
- **IaC**: Terraform, CloudFormation, Ansible
- **Cloud**: AWS, GCP, Azure

### Monitoring & Observability
- **Metrics**: Prometheus, Grafana
- **Logging**: ELK, Splunk, CloudWatch
- **Tracing**: Jaeger, Zipkin, Datadog

## Best Practices

1. **Start Simple** - Begin with simpler architectures
2. **Understand Your Constraints** - Know your requirements
3. **Measure & Monitor** - Data-driven decisions
4. **YAGNI** - Don't over-engineer
5. **Document Decisions** - ADRs (Architecture Decision Records)
6. **Test Early** - Catch issues before production
7. **Security by Design** - Not an afterthought
8. **Iterate & Evolve** - Architectures change over time

## Case Study Examples

### Twitter-like System
```
Requirements: Real-time feeds, billions of users
Design:
- Sharded user databases
- Cache layer for hot users
- Message queue for async processing
- CDN for media distribution
- Microservices (User, Timeline, Media)
```

### Netflix-like System
```
Requirements: Stream videos to millions
Design:
- Microservices architecture
- CDN for content delivery
- Personalization service (ML-based)
- Recommendation engine
- Monitoring and analytics
```

### Uber-like System
```
Requirements: Real-time matching, geolocation
Design:
- Real-time WebSocket connections
- Geo-spatial databases
- Message queues for events
- Consistent hashing for load balancing
- Microservices (Users, Rides, Payments)
```

## Resources

- [System Design Primer](https://github.com/donnemartin/system-design-primer)
- [Designing Data-Intensive Applications](https://dataintensive.net/)
- [Refactoring Guru](https://refactoring.guru/)
- [AWS Architecture Center](https://aws.amazon.com/architecture/)
- [12 Factor App](https://12factor.net/)

## Next Steps

1. Master design patterns
2. Study architectural patterns
3. Practice system design interviews
4. Understand distributed systems
5. Learn from real-world case studies
6. Build and architect production systems
7. Continuously learn about new patterns
