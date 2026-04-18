# GraphQL Schema Specification

## Overview
This document specifies the GraphQL schema proposed for advanced client integrations.

## Schema
```graphql
type AnalysisResult {
    id: ID!
    category: String!
    confidence: Float!
    summary: String!
    detailedAnalysis: String
    timestamp: String!
}

type Query {
    getAnalysis(id: ID!): AnalysisResult
    getHistory(limit: Int = 10): [AnalysisResult]
}

type Mutation {
    analyzeImage(imageBase64: String!, categoryHint: String): AnalysisResult!
}
```
