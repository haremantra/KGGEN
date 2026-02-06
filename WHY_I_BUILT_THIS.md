# Why I Built KGGEN

## The Problem That Sparked This Project

Every year, businesses sign thousands of contracts without truly understanding what they contain. Legal teams are overwhelmed, often reviewing documents under tight deadlines with limited resources. Critical clauses get missed. Risks go unnoticed until they materialize into costly disputes. I watched this happen repeatedly and asked myself: what if we could change this?

Contract review is one of those domains where the stakes are high but the tools have lagged behind. Legal professionals still rely heavily on manual review, keyword searches, and institutional memory. Meanwhile, advances in natural language processing and knowledge graphs have transformed other industries. I built KGGEN to bring these capabilities to contract analysis.

## What KGGEN Actually Does

KGGEN transforms unstructured contract text into structured, queryable knowledge. At its core, the system does three things: it classifies clauses according to the CUAD (Contract Understanding Atticus Dataset) taxonomy, extracts entities and relationships to build a knowledge graph, and assesses risk based on both rule-based heuristics and AI analysis.

The classification system uses a fine-tuned sentence transformer model trained on over 8,000 contract clauses. When I started, the baseline model achieved roughly 29% accuracy on CUAD labels. After fine-tuning with contrastive learning, accuracy jumped to 61% with a macro F1 score of 53%. More importantly, the confidence scores are calibrated using Platt scaling, so when the system says it's 80% confident about a classification, that actually means something.

The knowledge graph extraction goes beyond simple entity recognition. The system identifies parties, monetary values, dates, intellectual property references, and the relationships between them. These aren't just isolated facts—they form a connected graph that reveals the structure of obligations, rights, and restrictions embedded in the contract.

Risk assessment combines 41 rule-based checks with optional LLM analysis for complex cases. Each CUAD category has associated risk rules. For example, an unlimited liability clause triggers a high-risk flag, while a standard limitation of liability provision might score lower. The system also identifies missing clauses—sometimes what a contract doesn't say is more dangerous than what it does.

## The Interdependency Problem

One feature I'm particularly proud of is the clause interdependency analysis. Contracts aren't collections of independent provisions. They're interconnected systems where modifying one clause can cascade through the entire document. A termination clause might reference a cure period defined elsewhere. An indemnification provision might be limited by a liability cap three pages away.

KGGEN maps these dependencies using 73 static rules that capture common relationships between CUAD label types. It builds a directed graph of clause dependencies, then analyzes it for potential problems: contradictions between clauses, missing requirements that should accompany certain provisions, and high-impact clauses that affect many other parts of the contract.

This kind of structural analysis is nearly impossible to do manually across a large portfolio of contracts, but the knowledge graph representation makes it computationally tractable.

## Portfolio-Level Insights

Individual contract analysis is valuable, but the real power emerges at portfolio scale. When you can analyze dozens or hundreds of contracts together, patterns become visible that would otherwise remain hidden.

KGGEN's portfolio analyzer aggregates findings across contracts to answer questions like: Which contracts have the weakest IP protections? Where are our liability caps inconsistent? Which vendors have non-standard termination provisions? The system performs gap analysis to identify protections present in some contracts but missing from others, and highlights outliers that deviate from portfolio norms.

For technology agreements specifically—which is the domain I focused on—this matters because license terms, IP assignments, and data handling provisions have real business implications that compound across a portfolio.

## Entity Resolution and Search

When you're dealing with multiple contracts, entity resolution becomes critical. The same company might be referenced as "Microsoft Corporation," "Microsoft Corp.," "MSFT," or simply "Microsoft." The same concept might be expressed in different legal phrasings across documents.

KGGEN uses semantic embeddings and adaptive clustering to resolve entities across contracts. It groups similar mentions together and selects canonical forms, so you can search for a party or concept and find all relevant references regardless of how they're expressed in the source documents.

The hybrid search system combines BM25 keyword matching with semantic vector search, fusing results using reciprocal rank fusion. This means you can search for specific terms when you know exactly what you're looking for, or use natural language queries when you're exploring.

## The Technical Stack

I built KGGEN on a modern Python stack: FastAPI for the REST API, Streamlit for the web interface, NetworkX for graph algorithms, and sentence-transformers for embeddings. The system integrates with Neo4j for persistent graph storage and Qdrant for vector search, though it also runs entirely in-memory for quick experimentation.

The architecture separates concerns cleanly. Classification, extraction, risk assessment, and interdependency analysis are independent modules that can be used separately or composed through the pipeline. The API exposes everything through REST endpoints, making it straightforward to integrate with other tools.

## Why Open Source

I'm releasing KGGEN as open source because I believe these capabilities should be accessible beyond large law firms with enterprise software budgets. Small businesses, startups, and solo practitioners deserve tools that help them understand their contracts.

The CUAD dataset that underpins the classification system was itself released openly by The Atticus Project, with the explicit goal of democratizing legal AI. Building on that foundation, it felt right to continue in that spirit.

## What's Next

KGGEN is functional today, but there's more to build. Multi-language support would extend its reach. Integration with document management systems would streamline workflows. More sophisticated reasoning over the knowledge graph could answer complex questions about contractual relationships.

But the core insight—that contracts are structured knowledge that can be extracted, connected, and analyzed—is sound. And the tools to do it are now accessible to anyone who wants to use them.

That's why I built this.
