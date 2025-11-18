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

## Best Practices

1. **RESTful Design** - Clear resources, proper HTTP methods
2. **Security** - Input validation, authentication, encryption
3. **Error Handling** - Meaningful messages, proper status codes
4. **Performance** - Caching, indexing, query optimization
5. **Documentation** - API docs, README, code comments
6. **Testing** - Unit tests, integration tests, API tests
7. **Monitoring** - Logging, metrics, error tracking
8. **Scalability** - Stateless design, horizontal scaling

## Project Ideas

1. **Blog API** - CRUD operations, comments, tagging
2. **E-commerce Backend** - Products, orders, payments
3. **Social Media API** - Users, posts, comments, likes
4. **Chat API** - Real-time messaging, WebSockets
5. **Analytics Service** - Event tracking, data aggregation

## Resources

- [API Design Best Practices](https://swagger.io)
- [GraphQL](https://graphql.org)
- [RESTful API Tutorial](https://restfulapi.net)
- [Node.js Documentation](https://nodejs.org/docs/)
- [Django Documentation](https://docs.djangoproject.com)
- [FastAPI](https://fastapi.tiangolo.com)

## Next Steps

1. Master your chosen language and framework
2. Build real-world APIs with authentication
3. Implement database design and optimization
4. Learn microservices architecture
5. Deploy to production with monitoring
