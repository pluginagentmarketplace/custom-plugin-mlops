---
name: backend-development
description: Master backend API development with Node.js, Python, PHP, Java, C#, GraphQL, REST APIs, databases, and microservices. Learn server-side architecture, authentication, scalability, and production deployment.
---

# Backend Development & API Skills

## Quick Start

Backend development creates the server-side logic and APIs that power applications. Start with your language choice and understand HTTP fundamentals.

### Node.js/Express API
```javascript
const express = require('express');
const app = express();

app.use(express.json());

// REST API endpoint
app.get('/api/users/:id', async (req, res) => {
    try {
        const user = await getUser(req.params.id);
        res.json(user);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(3000, () => console.log('Server running on :3000'));
```

### Python/FastAPI
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    user = await fetch_user(user_id)
    return user

@app.post("/api/users")
async def create_user(user: User):
    return await save_user(user)
```

### GraphQL Schema
```graphql
type User {
    id: ID!
    name: String!
    email: String!
    posts: [Post!]!
}

type Query {
    user(id: ID!): User
    users: [User!]!
}

type Mutation {
    createUser(name: String!, email: String!): User
    deleteUser(id: ID!): Boolean
}
```

## Core Topics

### 1. Language Fundamentals
- **Node.js**: JavaScript async/await, modules, package management
- **Python**: Async frameworks, type hints, virtual environments
- **PHP**: Object-oriented PHP, Laravel framework
- **Java**: Spring Boot, dependency injection, build tools
- **C#**: .NET Core, async patterns, LINQ

### 2. HTTP & REST Principles
- HTTP methods (GET, POST, PUT, DELETE, PATCH)
- Status codes (2xx, 3xx, 4xx, 5xx)
- RESTful API design patterns
- API versioning strategies
- Pagination, filtering, sorting

### 3. Database Integration
- **SQL**: Joins, aggregations, optimization
- **ORMs**: Prisma, SQLAlchemy, Hibernate, Entity Framework
- **Database Selection**: PostgreSQL vs. MongoDB
- **Connection Pooling**: Performance optimization
- **Migrations**: Schema version control

### 4. Authentication & Authorization
- **JWT (JSON Web Tokens)**: Stateless authentication
- **OAuth 2.0**: Third-party authentication
- **Session Management**: Cookie-based authentication
- **Role-Based Access Control (RBAC)**: Permission management
- **API Keys**: Simple authentication

### 5. Input Validation & Error Handling
- Request validation schemas
- Sanitization and data cleaning
- Meaningful error responses
- Exception handling patterns
- Security (CSRF, injection prevention)

### 6. Middleware & Request Pipeline
- Logging and monitoring
- CORS (Cross-Origin Resource Sharing)
- Rate limiting and throttling
- Compression and response optimization
- Request/response transformation

### 7. Web Frameworks
- **Node.js**: Express, Fastify, NestJS
- **Python**: Django, FastAPI, Flask
- **PHP**: Laravel, Symfony
- **Java**: Spring Boot, Quarkus
- **C#**: ASP.NET Core, Blazor

### 8. GraphQL
- **Schema Definition**: Types, queries, mutations
- **Resolvers**: Data fetching and computation
- **Subscriptions**: Real-time updates
- **Federation**: Microservices with GraphQL
- **DataLoader**: N+1 query prevention

### 9. Scalability & Performance
- **Caching**: Redis, Memcached strategies
- **Database Optimization**: Indexing, query analysis
- **Load Balancing**: Distributing traffic
- **Async Processing**: Background jobs, message queues
- **Horizontal Scaling**: Stateless design

### 10. Deployment & DevOps
- **Docker**: Containerization
- **Kubernetes**: Orchestration basics
- **CI/CD**: Automated testing and deployment
- **Monitoring**: Logging, metrics, APM
- **Cloud Platforms**: AWS, GCP, Azure

## Learning Path

### Month 1-2: Foundations
- Choose programming language
- HTTP and REST principles
- Basic CRUD API development
- SQL fundamentals

### Month 3-4: Intermediate
- Framework mastery (Express, Django, FastAPI, Laravel)
- Database design and ORMs
- Authentication and authorization
- API design best practices

### Month 5-6: Advanced
- GraphQL implementation
- Microservices architecture
- Caching and performance optimization
- Testing and CI/CD

### Month 7+: Professional
- System design and scalability
- Cloud deployment and monitoring
- Event-driven architecture
- Team leadership

## Tools & Technologies

### Languages & Frameworks
- **Node.js**: Express, Fastify, NestJS, Hapi
- **Python**: Django, FastAPI, Flask, Starlette
- **PHP**: Laravel, Symfony, Slim
- **Java**: Spring Boot, Quarkus, Micronaut
- **Go**: Gin, Echo, Standard Library
- **C#**: ASP.NET Core, minimal APIs

### Databases
- **SQL**: PostgreSQL, MySQL, MariaDB
- **NoSQL**: MongoDB, Redis, DynamoDB
- **Tools**: DBeaver, pgAdmin, MongoDB Compass

### API Development
- **API Testing**: Postman, Insomnia, REST Client
- **Documentation**: Swagger/OpenAPI, GraphQL IDL
- **Monitoring**: Sentry, DataDog, New Relic

### Deployment
- **Containerization**: Docker, Docker Compose
- **Orchestration**: Kubernetes, Docker Swarm
- **CI/CD**: GitHub Actions, Jenkins, GitLab CI
- **Cloud**: AWS, GCP, Azure, Heroku

### Development Tools
- **Editors**: VS Code, IntelliJ, JetBrains IDEs
- **Version Control**: Git, GitHub, GitLab
- **Package Managers**: npm, pip, Composer, Maven

## Advanced Topics

### Microservices Architecture
- **Service Boundary**: Identifying bounded contexts (Domain-Driven Design)
- **Service Communication**: Synchronous (REST, gRPC, HTTP/2) vs. Asynchronous (message queues)
- **API Versioning**: URL versioning, header versioning, content negotiation
- **Service Discovery**: Consul, Eureka, DNS-based discovery
- **Inter-service Communication**: Timeout handling, circuit breakers, retries
- **Data Consistency**: Saga pattern, event sourcing, eventual consistency
- **Distributed Tracing**: OpenTelemetry, Jaeger for debugging distributed systems
- **Service Mesh**: Istio, Linkerd for managing service communication

### Message Queue & Asynchronous Processing
- **Message Brokers**: RabbitMQ, Apache Kafka, AWS SQS
- **Publishing/Subscribing**: Pub/Sub pattern for decoupled services
- **Job Queues**: Celery (Python), Bull (Node.js), Delayed Job (Ruby)
- **Idempotency**: Preventing duplicate processing
- **Dead Letter Queues**: Handling failed message retries
- **Event Sourcing**: Storing events instead of state
- **CQRS (Command Query Responsibility Segregation)**: Separating reads and writes

### Advanced Authentication & Authorization
- **OAuth 2.0 Flows**: Authorization Code, Client Credentials, PKCE
- **OpenID Connect**: Identity layer on top of OAuth 2.0
- **Passwordless Auth**: WebAuthn, Magic links, Biometrics
- **Multi-Factor Authentication (MFA)**: TOTP, SMS, push notifications
- **SAML**: Enterprise SSO authentication
- **Token Refresh**: Handling token expiration and refresh strategies
- **API Rate Limiting**: Per-user, per-endpoint, sliding window algorithms

### Caching Strategies
- **HTTP Caching**: ETag, Cache-Control, Last-Modified headers
- **Application Caching**: In-memory caches (Redis, Memcached)
- **Database Query Caching**: Query result caching, cache invalidation
- **Cache Patterns**: Cache-aside, write-through, write-behind
- **Cache Stampede**: When multiple requests request expired cache
- **Distributed Caching**: Cache coherence across multiple servers
- **Cache Warming**: Pre-loading frequently accessed data

### Database Patterns & Optimization
- **Connection Pooling**: Preventing resource exhaustion
- **Database Partitioning**: Sharding, horizontal partitioning
- **Read Replicas**: Scaling read-heavy workloads
- **Query Optimization**: EXPLAIN plans, index strategies
- **N+1 Query Prevention**: Batch loading, DataLoader pattern
- **Transaction Management**: ACID guarantees, isolation levels
- **CDC (Change Data Capture)**: Capturing database changes

### API Design Patterns
- **Pagination**: Offset-based, cursor-based, keyset pagination
- **Filtering**: Multiple filter syntax (query params vs. GraphQL)
- **Sorting**: Multi-field sorting, default sort orders
- **Sparse Fieldsets**: Allowing clients to request specific fields
- **Content Negotiation**: Supporting multiple response formats (JSON, XML, etc.)
- **HATEOAS**: Hypermedia-driven APIs (advanced REST)
- **API Throttling**: Rate limiting per client
- **Backwards Compatibility**: API versioning strategies

### Testing at Scale
- **Unit Testing**: Fast, isolated tests (Jest, pytest, unittest)
- **Integration Testing**: Testing with databases and external services
- **Contract Testing**: API contract testing between services
- **Load Testing**: Apache JMeter, Locust, k6
- **Chaos Engineering**: Simulating failures to test resilience
- **Mutation Testing**: Verifying test suite quality
- **Property-Based Testing**: QuickCheck, Hypothesis

### Security at Scale
- **Input Validation**: Schema validation, sanitization
- **SQL Injection Prevention**: Prepared statements, parameterized queries
- **OWASP Top 10**: Understanding and preventing common vulnerabilities
- **Secrets Management**: Vault, AWS Secrets Manager
- **API Security**: API keys, OAuth, mutual TLS
- **HTTPS/TLS**: Certificate management, TLS versions
- **CORS Security**: Origin validation, credential handling
- **Dependency Management**: Scanning for vulnerable dependencies (Snyk, npm audit)

### Observability & Monitoring
- **Structured Logging**: JSON logs with context, log levels
- **Metrics**: Application metrics (response time, error rate, throughput)
- **Tracing**: Distributed tracing with correlation IDs
- **Health Checks**: Liveness and readiness probes
- **Alerting**: Setting up meaningful alerts, alerting fatigue
- **APM Tools**: DataDog, New Relic, Dynatrace
- **Log Aggregation**: ELK Stack, Splunk, CloudWatch

### Deployment & DevOps
- **Blue-Green Deployment**: Zero-downtime deployments
- **Canary Releases**: Gradual rollout to users
- **Feature Flags**: Enabling/disabling features at runtime
- **Database Migrations**: Handling schema changes safely
- **Rollback Strategies**: Quick rollback to previous version
- **Environment Management**: Dev, staging, production consistency
- **Infrastructure as Code**: Terraform, CloudFormation, Pulumi

## Common Pitfalls & Gotchas

1. **N+1 Query Problem**: Fetching related data in a loop
   ```sql
   -- ❌ Bad: N+1 queries (1 user + N posts)
   SELECT * FROM users WHERE id = 1;
   SELECT * FROM posts WHERE user_id = 1;  -- Called again and again

   -- ✅ Good: Single query with join
   SELECT u.*, p.* FROM users u
   LEFT JOIN posts p ON u.id = p.user_id WHERE u.id = 1;
   ```

2. **Unbounded Query Results**: Returning millions of records
   - **Fix**: Implement pagination with reasonable limits
   - **Example**: `LIMIT 100 OFFSET offset`

3. **Missing Error Handling**: Silent failures or generic error messages
   - **Fix**: Meaningful error messages, proper HTTP status codes
   - **Example**: 400 for validation error, 409 for conflict, 500 for server error

4. **Blocking Operations**: Synchronous operations that take time
   - **Fix**: Use async/await, background jobs, message queues
   - **Example**: Process file upload asynchronously

5. **Stateful Service Design**: Assuming server state across requests
   - **Fix**: Design stateless services for horizontal scaling
   - **Example**: Store session in Redis, not in-memory

6. **Missing Connection Pooling**: Creating new DB connections per request
   - **Fix**: Use connection pools (HikariCP, pgBouncer)
   - **Impact**: 10-100x performance improvement

7. **Inadequate Validation**: Trusting client input
   - **Fix**: Validate on server-side, use schema validation
   - **Example**: zod, joi, Pydantic for data validation

8. **Hardcoded Configuration**: API keys, passwords in code
   - **Fix**: Use environment variables, secrets manager
   - **Example**: `process.env.DATABASE_URL`, AWS Secrets Manager

9. **Missing Audit Logging**: No record of what happened
   - **Fix**: Log all sensitive operations (login, payment, deletion)
   - **Example**: User ID, timestamp, action, IP address, result

10. **Slow Database Queries**: Unoptimized queries impacting user experience
    - **Fix**: Use EXPLAIN ANALYZE, add indexes, denormalize if needed
    - **Benchmark**: Every query should complete in < 100ms

## Production Deployment Checklist

- [ ] API documented (Swagger/OpenAPI)
- [ ] Input validation implemented for all endpoints
- [ ] Authentication and authorization enforced
- [ ] Rate limiting configured
- [ ] Error handling with proper status codes
- [ ] Logging configured (structured logs)
- [ ] Monitoring and alerting set up
- [ ] Database optimized and indexed
- [ ] Connection pooling configured
- [ ] Caching strategy implemented
- [ ] API tested (unit, integration, load tests)
- [ ] Security audit completed (OWASP Top 10)
- [ ] CORS properly configured
- [ ] HTTPS/TLS enabled
- [ ] CI/CD pipeline in place
- [ ] Rollback strategy documented
- [ ] Database migration strategy tested
- [ ] Backup and disaster recovery plan

## Best Practices

1. **RESTful Design** - Clear resources, proper HTTP methods, status codes
2. **Security First** - Input validation, authentication, encryption, OWASP compliance
3. **Error Handling** - Meaningful messages, proper status codes, error tracking
4. **Performance** - Caching, indexing, query optimization, async processing
5. **Documentation** - API docs, README, code comments, examples
6. **Testing** - Unit tests, integration tests, load tests, API tests
7. **Monitoring** - Structured logging, metrics, tracing, alerting
8. **Scalability** - Stateless design, horizontal scaling, database optimization
9. **Observability** - Logs, metrics, traces, health checks
10. **DevOps** - CI/CD, infrastructure as code, automated deployments

## Architecture Patterns

### Layered Architecture
```
api/
├── controllers/    # HTTP request handling
├── services/       # Business logic
├── repositories/   # Data access
├── models/        # Domain models
├── middleware/     # Authentication, logging, error handling
└── utils/         # Helper functions
```

### Clean Code Architecture
```
src/
├── domain/         # Business entities, use cases
├── application/    # Application logic, DTOs
├── infrastructure/ # Database, external APIs
├── presentation/   # API controllers
└── config/         # Configuration
```

## Performance Optimization Checklist

- [ ] Database queries optimized (indexes, joins)
- [ ] N+1 queries fixed (use eager loading)
- [ ] Caching implemented (Redis for hot data)
- [ ] Connection pooling configured
- [ ] Async processing for heavy tasks
- [ ] Pagination implemented
- [ ] Response compression enabled (gzip)
- [ ] CDN configured for static assets
- [ ] Load testing performed
- [ ] Response times monitored

## Testing Best Practices

```javascript
// Good test - tests behavior, not implementation
describe('UserService', () => {
  test('creates user with hashed password', async () => {
    const user = await UserService.create({
      email: 'test@example.com',
      password: 'password123'
    });

    expect(user.email).toBe('test@example.com');
    expect(user.password).not.toBe('password123'); // Hashed
    expect(user.password).toMatch(/^\$2b\$/); // bcrypt hash
  });
});

// Bad test - tests implementation detail
describe('UserService', () => {
  test('calls bcrypt.hash', async () => {
    const mockHash = jest.spyOn(bcrypt, 'hash');
    await UserService.create(...);
    expect(mockHash).toHaveBeenCalled(); // Implementation detail!
  });
});
```

## API Design Example

```javascript
// Well-designed API
GET    /api/v1/users                    // List users with pagination
GET    /api/v1/users/:id               // Get single user
POST   /api/v1/users                   // Create user
PATCH  /api/v1/users/:id               // Update user
DELETE /api/v1/users/:id               // Delete user

// Query string examples
GET /api/v1/users?page=1&limit=20&sort=-created_at&role=admin

// Error response example
HTTP 400 Bad Request
{
  "error": {
    "code": "INVALID_EMAIL",
    "message": "Invalid email format",
    "details": { "field": "email" }
  }
}

// Success response example
HTTP 200 OK
{
  "data": { "id": "123", "email": "test@example.com" },
  "meta": { "timestamp": "2024-01-01T00:00:00Z" }
}
```

## Resources & Learning

### Documentation
- [REST API Best Practices](https://restfulapi.net)
- [API Design Best Practices](https://swagger.io/resources/articles/best-practices-in-api-design/)
- [GraphQL Official Docs](https://graphql.org)
- [OWASP API Security](https://owasp.org/www-project-api-security/)

### Frameworks & Tools
- [Node.js Best Practices](https://github.com/goldbergyoni/nodebestpractices)
- [Python FastAPI](https://fastapi.tiangolo.com)
- [Django REST Framework](https://www.django-rest-framework.org)
- [Spring Boot Docs](https://spring.io/projects/spring-boot)
- [ASP.NET Core](https://docs.microsoft.com/en-us/aspnet/core)

### Learning Platforms
- [Backend Masters](https://backendmasters.io)
- [System Design Interview](https://systemdesigninterview.com)
- [API Design & Strategy](https://www.youtube.com/playlist?list=PLzEYvvM_eSDAhTlzZq33PwZhjpjVzPj7K)

### Keeping Up to Date
- [Backend Weekly](https://backendweekly.com)
- [Node Weekly](https://nodeweekly.com)
- [Python Weekly](https://pythonweekly.com)
- [API Changelog](https://www.apichangelog.com)

## Interview Preparation

### Common Backend Interview Topics
1. **Database Design**: Normalization, indexing, query optimization
2. **REST API Design**: Proper HTTP methods, status codes, error handling
3. **Authentication**: JWT, OAuth 2.0, session management
4. **Caching**: Redis, cache invalidation, caching strategies
5. **Scalability**: Horizontal scaling, load balancing, database replication
6. **Monitoring**: Logging, metrics, alerting
7. **Testing**: Unit tests, integration tests, mocking
8. **System Design**: Designing Twitter, Instagram, or similar systems

### System Design Interview
- **Scalability**: How would you handle 1M requests/second?
- **Database Selection**: When to use SQL vs. NoSQL?
- **Caching Strategy**: What should be cached? How long?
- **Load Balancing**: How would you distribute traffic?
- **Data Consistency**: SQL consistency vs. NoSQL eventual consistency?

## Next Steps

1. **Master your framework** - Node.js/Express, Django, FastAPI, Spring Boot, ASP.NET Core
2. **Learn database design** - Schema design, indexing, query optimization
3. **Study system design** - Handling millions of users, distributed systems
4. **Implement real-world features** - Authentication, caching, rate limiting, payments
5. **Focus on testing** - Unit tests, integration tests, test coverage
6. **Deploy to production** - Set up CI/CD, monitoring, alerting
7. **Learn microservices** - Service communication, data consistency, distributed tracing
