"""KGGEN-CUAD Main Entry Point.

Usage:
    python -m src.main extract <pdf_path> [--output <json_path>]
    python -m src.main classify <pdf_path> [--threshold <float>]
    python -m src.main analyze <pdf_path> [--output <json_path>]
    python -m src.main risks <pdf_path> [--no-llm]
    python -m src.main portfolio <folder> [--limit <int>]
    python -m src.main compare <contract1> <contract2>
    python -m src.main init-db
    python -m src.main stats
    python -m src.main serve [--port <int>]
"""

import argparse
import json
import sys
from pathlib import Path

from .extraction.extractor import ContractExtractor
from .classification.classifier import ClauseClassifier
from .utils.pdf_reader import extract_text_from_pdf
from .utils.neo4j_store import Neo4jStore
from .config import settings


def cmd_extract(args):
    """Extract knowledge graph from a contract."""
    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"Extracting from: {pdf_path}")

    # Read PDF
    print("Reading PDF...")
    contract_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(contract_text)} characters")

    # Extract entities and relations
    print("Extracting entities and relations...")
    extractor = ContractExtractor()
    result = extractor.extract(contract_text, contract_id=pdf_path.stem)

    print(f"Extracted {len(result.entities)} entities and {len(result.triples)} triples")

    # Output results
    output_data = {
        "contract_id": result.contract_id,
        "extraction_timestamp": result.extraction_timestamp.isoformat(),
        "llm_model": result.llm_model,
        "entities": [e.model_dump() for e in result.entities],
        "triples": [t.model_dump() for t in result.triples],
        "metadata": result.metadata,
    }

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
    else:
        print("\n--- Entities ---")
        for entity in result.entities[:10]:
            print(f"  [{entity.type.value}] {entity.name}")
        if len(result.entities) > 10:
            print(f"  ... and {len(result.entities) - 10} more")

        print("\n--- Triples ---")
        for triple in result.triples[:10]:
            print(f"  ({triple.subject}) --[{triple.predicate}]--> ({triple.object})")
        if len(result.triples) > 10:
            print(f"  ... and {len(result.triples) - 10} more")

    # Store in Neo4j if requested
    if args.store:
        print("\nStoring in Neo4j...")
        with Neo4jStore() as store:
            store.store_extraction_result(result)
        print("Stored successfully!")


def cmd_classify(args):
    """Classify contract clauses using CUAD labels."""
    file_path = Path(args.pdf_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Classifying: {file_path}")

    # Read file (PDF or text)
    print("Reading file...")
    if file_path.suffix.lower() == '.pdf':
        contract_text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            contract_text = f.read()
    print(f"Extracted {len(contract_text)} characters")

    # Initialize classifier (use in-memory for compatibility)
    print("Initializing CUAD classifier...")
    classifier = ClauseClassifier(use_qdrant=False)
    classifier.initialize()

    # Get contract summary
    print(f"\nAnalyzing clauses (threshold: {args.threshold})...")
    summary = classifier.get_contract_summary(contract_text, threshold=args.threshold)

    print(f"\n{'='*60}")
    print(f"CONTRACT CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Clauses analyzed: {summary['total_clauses_analyzed']}")
    print(f"CUAD labels found: {summary['labels_found']}")

    print(f"\n{'='*60}")
    print("TOP FINDINGS (by confidence)")
    print(f"{'='*60}")
    for label, info in summary["top_findings"]:
        conf_bar = "#" * int(info["max_confidence"] * 20)
        print(f"\n  [{info['category']}] {label}")
        print(f"    Confidence: {info['max_confidence']:.1%} {conf_bar}")
        print(f"    Occurrences: {info['count']}")
        if info.get("best_match"):
            preview = info["best_match"][:150].replace("\n", " ")
            print(f"    Best match: \"{preview}...\"")

    print(f"\n{'='*60}")
    print("BY CATEGORY")
    print(f"{'='*60}")
    for category, labels in summary["by_category"].items():
        print(f"\n  {category.upper().replace('_', ' ')}:")
        for item in labels[:5]:
            print(f"    - {item['label']}: {item['confidence']:.1%} ({item['occurrences']} occurrences)")

    # Output JSON if requested
    if args.output:
        output_path = Path(args.output)
        # Get detailed classification
        detailed = classifier.classify_contract(contract_text, top_k=5, threshold=args.threshold)
        output_data = {
            "contract_id": file_path.stem,
            "summary": {
                "total_clauses": summary["total_clauses_analyzed"],
                "labels_found": summary["labels_found"],
            },
            "classifications": [
                {
                    "clause_index": c.clause_index,
                    "text_preview": c.text[:200],
                    "labels": [
                        {
                            "label": l.label,
                            "confidence": l.confidence,
                            "category": l.category,
                        }
                        for l in c.labels
                    ],
                }
                for c in detailed
            ],
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")


def cmd_analyze(args):
    """Run full analysis pipeline on a contract."""
    from .pipeline import analyze_contract_file

    file_path = Path(args.pdf_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Analyzing: {file_path}")

    # Run full pipeline
    analysis, risk, dep_report = analyze_contract_file(
        str(file_path),
        output_path=args.output,
        include_risk=True,
        use_llm_risk=not args.no_llm,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Contract: {analysis.contract_id}")
    print(f"Clauses analyzed: {analysis.total_clauses}")
    print(f"Labels found: {len(analysis.analyzed_clauses)}")

    if risk:
        print(f"\nRisk Score: {risk.overall_risk_score}/100 ({risk.risk_level})")
        print(f"Findings: {len(risk.findings)}")
        print(f"Missing protections: {len(risk.missing_clause_risks)}")

    if dep_report:
        print(f"\nDependencies: {len(dep_report.graph.edges)}")
        print(f"Contradictions: {len(dep_report.contradictions)}")
        print(f"Missing requirements: {len(dep_report.missing_requirements)}")

    if args.output:
        print(f"\nResults saved to: {args.output}")


def cmd_risks(args):
    """Show detailed risk assessment for a contract."""
    from .pipeline import ContractAnalysisPipeline
    from .risk.assessor import RiskAssessor
    from .risk.rules import RiskSeverity

    file_path = Path(args.pdf_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Assessing risks: {file_path}")

    # Read file
    if file_path.suffix.lower() == '.pdf':
        contract_text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            contract_text = f.read()

    # Run analysis
    pipeline = ContractAnalysisPipeline()
    analysis = pipeline.analyze(contract_text, contract_id=file_path.stem)

    # Run risk assessment
    assessor = RiskAssessor(use_llm=not args.no_llm)
    risk = assessor.assess(analysis)

    # Print results
    print(f"\n{'='*60}")
    print(f"RISK ASSESSMENT: {file_path.name}")
    print(f"{'='*60}")

    # Overall score
    score_bar = "#" * (risk.overall_risk_score // 5)
    print(f"\nOverall Risk Score: {risk.overall_risk_score}/100 [{score_bar}]")
    print(f"Risk Level: {risk.risk_level}")
    print(f"\n{risk.summary}")

    # Critical and High findings
    critical = [f for f in risk.findings if f.severity == RiskSeverity.CRITICAL]
    high = [f for f in risk.findings if f.severity == RiskSeverity.HIGH]

    if critical:
        print(f"\n{'='*60}")
        print("CRITICAL ISSUES")
        print(f"{'='*60}")
        for f in critical:
            print(f"\n  [{f.label}]")
            print(f"    {f.reason}")
            print(f"    Recommendation: {f.recommendation}")

    if high:
        print(f"\n{'='*60}")
        print("HIGH RISK ISSUES")
        print(f"{'='*60}")
        for f in high:
            print(f"\n  [{f.label}]")
            print(f"    {f.reason}")
            print(f"    Recommendation: {f.recommendation}")

    # Missing protections
    missing_critical = [
        f for f in risk.missing_clause_risks
        if f.severity in (RiskSeverity.CRITICAL, RiskSeverity.HIGH)
    ]
    if missing_critical:
        print(f"\n{'='*60}")
        print("MISSING PROTECTIONS")
        print(f"{'='*60}")
        for f in missing_critical:
            print(f"\n  [{f.severity.value}] {f.label}")
            print(f"    {f.reason}")
            print(f"    Recommendation: {f.recommendation}")

    # LLM analysis
    if risk.llm_analysis:
        print(f"\n{'='*60}")
        print("AI ANALYSIS")
        print(f"{'='*60}")
        print(f"\n{risk.llm_analysis}")

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(risk.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")


def cmd_dependencies(args):
    """Show clause interdependency analysis for a contract."""
    from .pipeline import ContractAnalysisPipeline
    from .interdependency.analyzer import InterdependencyAnalyzer

    file_path = Path(args.pdf_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Analyzing dependencies: {file_path}")

    # Read file
    if file_path.suffix.lower() == '.pdf':
        contract_text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            contract_text = f.read()

    # Run classification pipeline
    pipeline = ContractAnalysisPipeline()
    analysis = pipeline.analyze(contract_text, contract_id=file_path.stem)

    # Run interdependency analysis
    analyzer = InterdependencyAnalyzer(use_llm=not args.no_llm)
    report = analyzer.analyze(analysis)

    # Print results
    print(f"\n{'='*60}")
    print(f"CLAUSE INTERDEPENDENCY ANALYSIS: {file_path.name}")
    print(f"{'='*60}")

    print(f"\nDependencies found: {len(report.graph.edges)}")
    print(f"Contradictions: {len(report.contradictions)}")
    print(f"Missing requirements: {len(report.missing_requirements)}")
    print(f"Circular dependencies: {len(report.cycles)}")
    print(f"Risk score adjustment: +{report.risk_score_adjustment}")

    # Contradictions
    if report.contradictions:
        print(f"\n{'='*60}")
        print("CONTRADICTIONS")
        print(f"{'='*60}")
        for c in report.contradictions:
            print(f"\n  {c['clause_a']}  <-->  {c['clause_b']}")
            print(f"    Reason: {c['reason']}")
            print(f"    Strength: {c['strength']:.0%}")

    # Missing requirements
    if report.missing_requirements:
        print(f"\n{'='*60}")
        print("MISSING REQUIREMENTS")
        print(f"{'='*60}")
        for m in report.missing_requirements:
            print(f"\n  [{m.severity}] {m.missing_label}")
            print(f"    Required by: {m.required_by}")
            print(f"    Impact: {m.impact}")

    # Impact rankings
    if report.impact_rankings:
        print(f"\n{'='*60}")
        print("IMPACT RANKINGS (top 10)")
        print(f"{'='*60}")
        for r in report.impact_rankings[:10]:
            if r["total_affected"] > 0:
                affected = ", ".join(r["affected_clauses"][:5])
                more = f" +{r['total_affected'] - 5} more" if r["total_affected"] > 5 else ""
                print(f"\n  {r['label']} → affects {r['total_affected']} clauses")
                print(f"    {affected}{more}")

    # Recommendations
    if report.recommendations:
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"\n  {i}. {rec}")

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")


def cmd_resolve(args):
    """Run entity resolution on a contract."""
    from .pipeline import ContractAnalysisPipeline
    from .resolution import analysis_to_entities_triples
    from .resolution.resolver import EntityResolver

    file_path = Path(args.pdf_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    print(f"Resolving entities: {file_path}")

    # Read file
    if file_path.suffix.lower() == '.pdf':
        contract_text = extract_text_from_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            contract_text = f.read()

    # Run analysis pipeline
    pipeline = ContractAnalysisPipeline()
    analysis = pipeline.analyze(contract_text, contract_id=file_path.stem)

    # Convert to entities/triples
    entities, triples = analysis_to_entities_triples(analysis)

    if not entities:
        print("No entities found to resolve.")
        return

    # Run resolution
    resolver = EntityResolver(use_llm=not args.no_llm)
    result = resolver.resolve(entities, triples)

    print(f"\n{'='*60}")
    print(f"ENTITY RESOLUTION: {file_path.name}")
    print(f"{'='*60}")
    print(f"Original entities: {result.original_count}")
    print(f"Resolved entities: {result.resolved_count}")
    print(f"Reduction: {1 - result.resolved_count / max(result.original_count, 1):.1%}")
    print(f"Alias groups: {len(result.alias_mapping)}")

    if result.alias_mapping:
        print(f"\n--- Alias Groups ---")
        for canonical_id, aliases in list(result.alias_mapping.items())[:10]:
            entity = next((e for e in result.resolved_entities if e.id == canonical_id), None)
            if entity:
                print(f"  {entity.name} <- {aliases}")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")


def cmd_search(args):
    """Search the knowledge graph."""
    from .search.service import get_search_service

    search = get_search_service()
    query = args.query

    print(f"Searching: \"{query}\"")

    # Search entities
    entity_results = search.search_entities(query, limit=args.limit)
    triple_results = search.search_triples(query, limit=args.limit)

    if not entity_results and not triple_results:
        print("No results found. Index contracts first with: analyze --resolve --index")
        return

    if entity_results:
        print(f"\n--- Entities ({len(entity_results)} results) ---")
        for payload, score in entity_results:
            print(f"  [{score:.3f}] {payload.get('name', '')} ({payload.get('entity_type', '')})")

    if triple_results:
        print(f"\n--- Triples ({len(triple_results)} results) ---")
        for payload, score in triple_results:
            pred = payload.get('predicate', '').replace('_', ' ').lower()
            print(f"  [{score:.3f}] {payload.get('subject', '')} {pred} {payload.get('object', '')}")


def cmd_query(args):
    """Answer a question about contracts using RAG."""
    from .query.service import get_query_service

    query_svc = get_query_service()
    question = args.question

    print(f"Question: \"{question}\"")
    print("Retrieving context and generating answer...")

    response = query_svc.query(question)

    print(f"\n{'='*60}")
    print(f"ANSWER (confidence: {response.confidence:.0%})")
    print(f"{'='*60}")
    print(f"\n{response.answer}")

    if response.sources:
        print(f"\n--- Sources ({len(response.sources)}) ---")
        for s in response.sources[:5]:
            pred = s.get('predicate', '').replace('_', ' ').lower()
            print(f"  {s.get('subject', '')} {pred} {s.get('object', '')}")


def cmd_portfolio(args):
    """Analyze multiple contracts in a folder."""
    from .portfolio.analyzer import PortfolioAnalyzer

    folder_path = Path(args.folder)

    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    if not folder_path.is_dir():
        print(f"Error: Not a directory: {folder_path}")
        sys.exit(1)

    print(f"Analyzing portfolio: {folder_path}")
    if args.limit:
        print(f"Limit: {args.limit} contracts")

    # Run portfolio analysis
    analyzer = PortfolioAnalyzer(use_llm=not args.no_llm)
    portfolio = analyzer.analyze_folder(
        folder_path,
        limit=args.limit,
        file_pattern=args.pattern or "*.pdf"
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"PORTFOLIO ANALYSIS SUMMARY")
    print(f"{'='*60}")

    risk_summary = portfolio.risk_summary
    print(f"\nContracts analyzed: {risk_summary.total_contracts}")
    print(f"Average risk score: {risk_summary.average_risk_score:.1f}/100")

    print("\nRisk Distribution:")
    for level, count in risk_summary.contracts_by_risk_level.items():
        bar = "#" * count
        print(f"  {level}: {count} {bar}")

    print("\nHighest Risk Contracts:")
    for cid in risk_summary.highest_risk_contracts[:5]:
        contract = next((c for c in portfolio.contracts if c.contract_id == cid), None)
        if contract:
            print(f"  - {cid}: {contract.risk_score}/100 ({contract.risk_level})")

    print("\nMost Common Gaps:")
    for label, count in risk_summary.most_common_gaps[:5]:
        pct = (count / risk_summary.total_contracts * 100) if risk_summary.total_contracts > 0 else 0
        print(f"  - {label}: missing in {count} contracts ({pct:.0f}%)")

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(portfolio.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")


def cmd_compare(args):
    """Compare two contracts side by side."""
    from .pipeline import ContractAnalysisPipeline
    from .risk.assessor import RiskAssessor
    from .portfolio.analyzer import PortfolioAnalyzer

    path_a = Path(args.contract1)
    path_b = Path(args.contract2)

    for p in [path_a, path_b]:
        if not p.exists():
            print(f"Error: File not found: {p}")
            sys.exit(1)

    print(f"Comparing contracts:")
    print(f"  A: {path_a.name}")
    print(f"  B: {path_b.name}")

    # Analyze both contracts
    pipeline = ContractAnalysisPipeline()
    assessor = RiskAssessor(use_llm=not args.no_llm)

    print("\nAnalyzing Contract A...")
    if path_a.suffix.lower() == '.pdf':
        text_a = extract_text_from_pdf(path_a)
    else:
        with open(path_a, 'r', encoding='utf-8', errors='ignore') as f:
            text_a = f.read()
    analysis_a = pipeline.analyze(text_a, contract_id=path_a.stem)
    risk_a = assessor.assess(analysis_a)

    print("Analyzing Contract B...")
    if path_b.suffix.lower() == '.pdf':
        text_b = extract_text_from_pdf(path_b)
    else:
        with open(path_b, 'r', encoding='utf-8', errors='ignore') as f:
            text_b = f.read()
    analysis_b = pipeline.analyze(text_b, contract_id=path_b.stem)
    risk_b = assessor.assess(analysis_b)

    # Compare using portfolio analyzer
    analyzer = PortfolioAnalyzer(use_llm=False)
    analyzer.add_analysis(analysis_a, risk_a)
    analyzer.add_analysis(analysis_b, risk_b)
    comparison = analyzer.compare_contracts(path_a.stem, path_b.stem)

    # Print comparison
    print(f"\n{'='*60}")
    print(f"CONTRACT COMPARISON")
    print(f"{'='*60}")

    print("\nRisk Scores:")
    print(f"  {path_a.stem}: {risk_a.overall_risk_score}/100 ({risk_a.risk_level})")
    print(f"  {path_b.stem}: {risk_b.overall_risk_score}/100 ({risk_b.risk_level})")
    diff = comparison.risk_comparison["difference"]
    print(f"  Difference: {diff:+d} points")

    print(f"\nShared Clauses ({len(comparison.shared_clauses)}):")
    for label in comparison.shared_clauses[:10]:
        print(f"  - {label}")
    if len(comparison.shared_clauses) > 10:
        print(f"  ... and {len(comparison.shared_clauses) - 10} more")

    print(f"\nOnly in {path_a.stem} ({len(comparison.only_in_a)}):")
    for label in comparison.only_in_a:
        print(f"  + {label}")

    print(f"\nOnly in {path_b.stem} ({len(comparison.only_in_b)}):")
    for label in comparison.only_in_b:
        print(f"  + {label}")

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(comparison.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")


def cmd_init_db(args):
    """Initialize the Neo4j database schema."""
    print("Initializing Neo4j schema...")
    with Neo4jStore() as store:
        store.init_schema()
    print("Schema initialized successfully!")


def cmd_stats(args):
    """Show database statistics."""
    print("Fetching statistics...")
    with Neo4jStore() as store:
        stats = store.get_statistics()

    print("\n--- Knowledge Graph Statistics ---")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print("\nNodes by type:")
    for node_type, count in stats["nodes_by_type"].items():
        if count > 0:
            print(f"  {node_type}: {count}")


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn
    from .api.app import app

    print(f"Starting KGGEN-CUAD API server on port {args.port}...")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        description="KGGEN-CUAD: Knowledge Graph Generator for Legal Contracts"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract KG from a contract")
    extract_parser.add_argument("pdf_path", help="Path to the PDF contract")
    extract_parser.add_argument("-o", "--output", help="Output JSON file path")
    extract_parser.add_argument(
        "-s", "--store", action="store_true", help="Store results in Neo4j"
    )
    extract_parser.set_defaults(func=cmd_extract)

    # Classify command
    classify_parser = subparsers.add_parser(
        "classify", help="Classify contract clauses using CUAD labels"
    )
    classify_parser.add_argument("pdf_path", help="Path to the PDF contract")
    classify_parser.add_argument(
        "-t", "--threshold", type=float, default=0.35,
        help="Minimum similarity threshold (default: 0.35)"
    )
    classify_parser.add_argument("-o", "--output", help="Output JSON file path")
    classify_parser.set_defaults(func=cmd_classify)

    # Analyze command (full pipeline)
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run full analysis pipeline with risk assessment"
    )
    analyze_parser.add_argument("pdf_path", help="Path to the contract file")
    analyze_parser.add_argument("-o", "--output", help="Output JSON file path")
    analyze_parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM for risk analysis"
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # Risks command
    risks_parser = subparsers.add_parser(
        "risks", help="Show detailed risk assessment"
    )
    risks_parser.add_argument("pdf_path", help="Path to the contract file")
    risks_parser.add_argument("-o", "--output", help="Output JSON file path")
    risks_parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM for risk analysis"
    )
    risks_parser.set_defaults(func=cmd_risks)

    # Dependencies command
    deps_parser = subparsers.add_parser(
        "dependencies", help="Analyze clause interdependencies in a contract"
    )
    deps_parser.add_argument("pdf_path", help="Path to the contract file")
    deps_parser.add_argument("-o", "--output", help="Output JSON file path")
    deps_parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM validation of dependencies"
    )
    deps_parser.set_defaults(func=cmd_dependencies)

    # Resolve command
    resolve_parser = subparsers.add_parser(
        "resolve", help="Run entity resolution on a contract"
    )
    resolve_parser.add_argument("pdf_path", help="Path to the contract file")
    resolve_parser.add_argument("-o", "--output", help="Output JSON file path")
    resolve_parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM for canonical selection"
    )
    resolve_parser.set_defaults(func=cmd_resolve)

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search the knowledge graph"
    )
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument(
        "-l", "--limit", type=int, default=10, help="Max results (default: 10)"
    )
    search_parser.set_defaults(func=cmd_search)

    # Query command
    query_parser = subparsers.add_parser(
        "query", help="Answer a question about contracts using RAG"
    )
    query_parser.add_argument("question", help="Question to answer")
    query_parser.set_defaults(func=cmd_query)

    # Portfolio command
    portfolio_parser = subparsers.add_parser(
        "portfolio", help="Analyze multiple contracts in a folder"
    )
    portfolio_parser.add_argument("folder", help="Path to folder containing contracts")
    portfolio_parser.add_argument(
        "-l", "--limit", type=int, help="Maximum number of contracts to analyze"
    )
    portfolio_parser.add_argument(
        "-p", "--pattern", default="*.pdf", help="File pattern (default: *.pdf)"
    )
    portfolio_parser.add_argument("-o", "--output", help="Output JSON file path")
    portfolio_parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM for risk analysis"
    )
    portfolio_parser.set_defaults(func=cmd_portfolio)

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare two contracts side by side"
    )
    compare_parser.add_argument("contract1", help="Path to first contract")
    compare_parser.add_argument("contract2", help="Path to second contract")
    compare_parser.add_argument("-o", "--output", help="Output JSON file path")
    compare_parser.add_argument(
        "--no-llm", action="store_true", help="Disable LLM for risk analysis"
    )
    compare_parser.set_defaults(func=cmd_compare)

    # Init-db command
    init_parser = subparsers.add_parser("init-db", help="Initialize Neo4j schema")
    init_parser.set_defaults(func=cmd_init_db)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI server")
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    serve_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
