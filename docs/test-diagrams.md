---
title: Diagram Test Page
layout: default
nav_order: 99
---

# Diagram Test Page

This page tests whether diagrams are rendering correctly on the website.

## Simple Flow Chart

```mermaid
graph TD
    A[Start] --> B{Is it working?}
    B -->|Yes| C[Great!]
    B -->|No| D[Check configuration]
    D --> E[Fix issues]
    E --> B
    C --> F[End]
```

## System Architecture

```mermaid
graph TB
    subgraph "Red Framework"
        A[Federation Schema]
        B[Virtual Nodes]
        C[Topology Manager]
    end
    
    subgraph "Ray Infrastructure"
        D[Ray Actors]
        E[Object Store]
    end
    
    A --> B
    B --> C
    C --> D
    B --> D
    C --> E
    
    style A fill:#e1f5fe
    style B fill:#e8f5e8
    style C fill:#f3e5f5
```

## Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant Federation
    participant Nodes
    
    User->>Federation: Create Federation
    Federation->>Nodes: Initialize Virtual Nodes
    User->>Federation: Start Training
    Federation->>Nodes: Build Ray Actors
    Nodes-->>Federation: Ready
    Federation-->>User: Training Complete
```

## Network Topology

```mermaid
graph TD
    Server((Server))
    Client1((Client 1))
    Client2((Client 2))
    Client3((Client 3))
    
    Client1 <--> Server
    Client2 <--> Server
    Client3 <--> Server
    
    style Server fill:#ff9999
    style Client1 fill:#99ccff
    style Client2 fill:#99ccff
    style Client3 fill:#99ccff
```

If you can see the diagrams above, then Mermaid is working correctly! 