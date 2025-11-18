---
name: database-management
description: Master database design and management. Learn SQL, PostgreSQL, MongoDB, Redis, data modeling, performance optimization, replication, and blockchain technology.
---

# Database & Data Management Skills

## Quick Start

Database management involves designing schemas, optimizing queries, ensuring data integrity, and scaling systems. Master both relational and NoSQL approaches.

### SQL Fundamentals
```sql
-- Creating a table with constraints
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Complex query with joins and aggregation
SELECT u.id, u.email, COUNT(p.id) as post_count
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
WHERE u.created_at > NOW() - INTERVAL '30 days'
GROUP BY u.id, u.email
HAVING COUNT(p.id) > 5
ORDER BY post_count DESC;

-- Window functions
SELECT
    user_id,
    post_id,
    created_at,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) as post_rank
FROM posts;
```

### PostgreSQL Advanced
```sql
-- JSON/JSONB operations
SELECT
    id,
    data->>'name' as name,
    data->'age' as age
FROM users
WHERE data @> '{"status": "active"}';

-- Full-text search
SELECT id, title, ts_rank(document, query) as rank
FROM documents, plainto_tsquery('english', 'machine learning') as query
WHERE document @@ query
ORDER BY rank DESC;

-- Partitioning for large tables
CREATE TABLE events (
    id BIGSERIAL,
    event_date DATE,
    user_id INTEGER,
    event_data JSONB
) PARTITION BY RANGE (event_date);
```

### MongoDB CRUD
```javascript
// Insert documents
db.users.insertOne({ name: "John", email: "john@example.com" });
db.users.insertMany([
    { name: "Jane", email: "jane@example.com" },
    { name: "Bob", email: "bob@example.com" }
]);

// Query documents
db.users.find({ name: { $regex: "^J" } }).limit(10);

// Aggregation pipeline
db.orders.aggregate([
    { $match: { status: "completed" } },
    { $group: { _id: "$user_id", total: { $sum: "$amount" } } },
    { $sort: { total: -1 } },
    { $limit: 10 }
]);

// Update documents
db.users.updateOne(
    { _id: ObjectId("...") },
    { $set: { updated_at: new Date(), status: "active" } }
);
```

### Redis Operations
```python
import redis

r = redis.Redis(host='localhost', port=6379)

# String operations
r.set('key', 'value')
r.get('key')

# Lists (queues)
r.lpush('queue', 'job1')
r.rpop('queue')

# Sets (tags)
r.sadd('tags:python', 'django', 'flask')
r.smembers('tags:python')

# Sorted sets (leaderboards)
r.zadd('leaderboard', {'user1': 100, 'user2': 200})
r.zrange('leaderboard', 0, -1, withscores=True)

# Hash (object storage)
r.hset('user:1', 'name', 'John', 'email', 'john@example.com')
r.hgetall('user:1')

# Pub/Sub (messaging)
pubsub = r.pubsub()
pubsub.subscribe('notifications')
```

## Core Topics

### 1. SQL & Query Optimization
- **SELECT, WHERE, ORDER BY**: Query fundamentals
- **Joins**: INNER, LEFT, RIGHT, FULL joins
- **Aggregations**: COUNT, SUM, AVG, GROUP BY, HAVING
- **Subqueries**: Nested queries, CTEs (Common Table Expressions)
- **Window Functions**: ROW_NUMBER, RANK, LAG, LEAD
- **Indexing**: B-tree, hash indexes, EXPLAIN ANALYZE
- **Query Optimization**: Execution plans, optimization techniques

### 2. Data Modeling & Normalization
- **Normalization**: 1NF, 2NF, 3NF, BCNF
- **Entity-Relationship Diagrams**: Schema design
- **Relationships**: One-to-many, many-to-many
- **Constraints**: Primary key, foreign key, unique, check
- **Data Types**: Choosing appropriate types
- **Denormalization**: Performance vs. consistency trade-offs

### 3. PostgreSQL Specifics
- **Installation & Configuration**: Setup, tuning
- **Advanced Types**: JSON/JSONB, arrays, ranges
- **Full-Text Search**: Text search capabilities
- **Functions & Triggers**: PL/pgSQL programming
- **Replication**: Streaming replication, logical replication
- **Performance**: Vacuum, autovacuum, monitoring
- **Partitioning**: Range, list, hash partitioning

### 4. MongoDB Document Design
- **Collections**: Document storage
- **BSON Format**: Data representation
- **Embedded vs. Referenced**: Document design patterns
- **Indexing**: Index types, compound indexes
- **Aggregation Pipeline**: Multi-stage transformations
- **Transactions**: Multi-document transactions
- **Replication**: Replica sets, failover

### 5. Redis Data Structures
- **Strings**: Simple key-value storage
- **Lists**: Queues and stacks
- **Sets**: Unique values, set operations
- **Sorted Sets**: Ranked data, leaderboards
- **Hashes**: Object-like storage
- **Streams**: Event streaming
- **Advanced**: Pub/Sub, Lua scripting

### 6. High Availability & Replication
- **Master-Slave Replication**: Redundancy
- **Sharding**: Horizontal partitioning
- **Backup & Recovery**: Disaster recovery
- **Failover**: Automatic recovery
- **Consistency Models**: Strong, eventual, causal
- **Monitoring**: Health checks, metrics

### 7. Performance Optimization
- **Indexing Strategies**: When and how to index
- **Query Analysis**: EXPLAIN, query plans
- **Connection Pooling**: Resource management
- **Caching**: Redis, query caching
- **Denormalization**: Strategic redundancy
- **Vertical vs. Horizontal Scaling**: Growth strategies

### 8. Security & Access Control
- **Authentication**: User authentication
- **Authorization**: Row-level security, roles
- **Encryption**: Data at rest and in transit
- **Auditing**: Query logging, change tracking
- **Injection Prevention**: SQL injection, NoSQL injection

### 9. Blockchain & Distributed Ledgers
- **Blockchain Concepts**: Immutability, decentralization
- **Smart Contracts**: Ethereum, Solidity
- **Cryptocurrencies**: Tokens, wallets
- **Consensus Mechanisms**: Proof of Work, Proof of Stake
- **Applications**: DeFi, NFTs, dApps

### 10. Data Warehousing & Analytics
- **OLAP vs. OLTP**: Analytical vs. transactional
- **Data Warehouses**: Snowflake, BigQuery
- **ETL Processes**: Data extraction, transformation
- **Star & Snowflake Schemas**: Dimensional modeling
- **BI Tools**: Tableau, Power BI, Looker

## Learning Path

### Month 1-2: SQL Fundamentals
- SQL basics and query writing
- Joins and aggregations
- Database design basics
- Normalization principles

### Month 3-4: Database Administration
- Index creation and optimization
- Backup and recovery
- User management and security
- Performance monitoring

### Month 5-6: Advanced SQL
- Complex queries and CTEs
- Window functions
- Query optimization
- Advanced indexing strategies

### Month 7-8: NoSQL & Caching
- MongoDB fundamentals
- Document design patterns
- Redis data structures
- Cache-aside patterns

### Month 9-10: High Availability
- Replication and failover
- Sharding strategies
- Disaster recovery planning
- Multi-region deployment

### Month 11+: Advanced Topics
- Data warehousing
- Big data technologies
- Blockchain applications
- Production optimization

## Tools & Technologies

### Databases
- **SQL**: PostgreSQL, MySQL, MariaDB, SQL Server
- **NoSQL**: MongoDB, Redis, Cassandra, DynamoDB
- **Warehousing**: Snowflake, BigQuery, Redshift
- **Blockchain**: Ethereum, Solana, Hyperledger

### Tools & Utilities
- **GUI Tools**: DBeaver, pgAdmin, MongoDB Compass
- **Query Tools**: SQL IDEs, Jupyter Notebooks
- **Monitoring**: Prometheus, Grafana, DataDog
- **Migration**: Liquibase, Flyway, DBMate

### Languages
- **SQL**: Standard query language
- **Python**: Database programming, data science
- **JavaScript/Node.js**: MongoDB, Firebase
- **Solidity**: Smart contracts (Ethereum)

## Best Practices

1. **Data Modeling** - Design for scalability and maintainability
2. **Indexing** - Strategic index creation for performance
3. **Normalization** - Reduce redundancy while maintaining performance
4. **Backup Strategy** - Regular, tested backups
5. **Security** - Encryption, access control, audit logs
6. **Monitoring** - Track performance and health
7. **Documentation** - Clear schema and process documentation
8. **Testing** - Test migrations, recovery procedures

## Project Ideas

1. **Relational Database** - Design and normalize a complex schema
2. **Performance Tuning** - Optimize slow queries
3. **Replication Setup** - High availability configuration
4. **Migration Project** - SQL to NoSQL migration
5. **Analytics Pipeline** - Data warehouse and BI dashboard
6. **Cache Layer** - Redis implementation for performance
7. **Blockchain App** - Smart contract development

## Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MongoDB Manual](https://docs.mongodb.com/manual/)
- [Redis Commands](https://redis.io/commands)
- [SQL Tutorial](https://www.w3schools.com/sql/)
- [Database Design](https://www.database-guide.com/)

## Next Steps

1. Master SQL fundamentals
2. Learn database design and normalization
3. Practice query optimization
4. Explore NoSQL databases
5. Implement replication and high availability
6. Deploy production databases
7. Monitor and optimize continuously
