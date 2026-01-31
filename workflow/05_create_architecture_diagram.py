#!/usr/bin/env python3
"""
Create system architecture diagram for CUAD Knowledge Graph Generator.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
color_input = '#E8F4F8'
color_processing = '#B8E6F0'
color_storage = '#FFF4E6'
color_api = '#FFE6E6'
color_output = '#E8F8E8'
color_arrow = '#666666'

# Title
ax.text(5, 11.5, 'CUAD Knowledge Graph Generator',
        ha='center', va='top', fontsize=20, fontweight='bold')
ax.text(5, 11.1, 'System Architecture: Applying KGGen to Legal Contracts',
        ha='center', va='top', fontsize=12, style='italic')

# Helper function to create boxes
def create_box(ax, x, y, width, height, label, color, fontsize=9):
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold')

# Helper function to create arrows
def create_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, mutation_scale=20,
                            linewidth=2, color=color_arrow)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.15, label,
                ha='center', va='bottom', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))

# Layer 1: Input Data (top)
create_box(ax, 0.5, 9.5, 2, 1, 'CUAD Dataset\n510 Contracts\n41 Labels', color_input)
create_box(ax, 3, 9.5, 2, 1, 'PDF Contracts\nText Extraction\nAnnotations', color_input)
create_box(ax, 5.5, 9.5, 2, 1, 'Technology\nAgreements\n~200 Contracts', color_input)
create_box(ax, 8, 9.5, 1.5, 1, 'Common Law\nPrinciples', color_input)

# Arrows from input to Stage 1
create_arrow(ax, 1.5, 9.5, 3, 8.5, 'Contracts')
create_arrow(ax, 4, 9.5, 4, 8.5, 'Annotations')
create_arrow(ax, 6.5, 9.5, 5, 8.5, 'Filter')

# Layer 2: Stage 1 - Extraction
create_box(ax, 1, 7.5, 6, 1, 'Stage 1: Entity & Relation Extraction', color_processing, fontsize=11)

# Stage 1 components
create_box(ax, 1.2, 6.5, 1.8, 0.8, 'LLM Extraction\n(Claude/GPT-4o)', color_processing, fontsize=8)
create_box(ax, 3.2, 6.5, 1.8, 0.8, 'DSPy Framework\nStructured Output', color_processing, fontsize=8)
create_box(ax, 5.2, 6.5, 1.6, 0.8, 'CUAD Label\nMapping', color_processing, fontsize=8)

# Arrows Stage 1 to Stage 2
create_arrow(ax, 4, 7.5, 4, 6.2, '')
create_arrow(ax, 4, 6.5, 4, 5.5, 'S-P-O Triples')

# Layer 3: Stage 2 - Aggregation
create_box(ax, 1, 4.5, 6, 1, 'Stage 2: Cross-Contract Aggregation', color_processing, fontsize=11)

# Stage 2 components
create_box(ax, 1.2, 3.5, 1.8, 0.8, 'Normalize\nEntities & Relations', color_processing, fontsize=8)
create_box(ax, 3.2, 3.5, 1.8, 0.8, 'Deduplicate\nExact Matches', color_processing, fontsize=8)
create_box(ax, 5.2, 3.5, 1.6, 0.8, 'Aggregate\nBy Type/Juris.', color_processing, fontsize=8)

# Arrows Stage 2 to Stage 3
create_arrow(ax, 4, 4.5, 4, 4.2, '')
create_arrow(ax, 4, 3.5, 4, 2.5, 'Unified Graph')

# Layer 4: Stage 3 - Resolution
create_box(ax, 1, 1.5, 6, 1, 'Stage 3: Entity & Edge Resolution', color_processing, fontsize=11)

# Stage 3 components
create_box(ax, 1.2, 0.5, 1.3, 0.8, 'S-BERT\nClustering', color_processing, fontsize=8)
create_box(ax, 2.7, 0.5, 1.3, 0.8, 'BM25+Semantic\nRetrieval', color_processing, fontsize=8)
create_box(ax, 4.2, 0.5, 1.3, 0.8, 'LLM\nDeduplication', color_processing, fontsize=8)
create_box(ax, 5.7, 0.5, 1.1, 0.8, 'Canonical\nSelection', color_processing, fontsize=8)

# Arrows Stage 3 to Storage
create_arrow(ax, 4, 1.5, 4, 1.2, '')
create_arrow(ax, 2, 0.5, 1.5, 0.2, '')
create_arrow(ax, 6, 0.5, 6.5, 0.2, '')

# Layer 5: Storage (middle-right)
create_box(ax, 7.5, 4, 2, 2.5, 'Knowledge Graph\nStorage', color_storage, fontsize=10)
create_box(ax, 7.6, 5.6, 1.8, 0.7, 'Neo4j\nGraph Database', color_storage, fontsize=8)
create_box(ax, 7.6, 4.7, 1.8, 0.7, 'FAISS\nVector Index', color_storage, fontsize=8)
create_box(ax, 7.6, 4.1, 1.8, 0.5, '50K-100K Triples', color_storage, fontsize=7)

# Connect Stage outputs to Storage
create_arrow(ax, 7, 5, 7.5, 5, 'Dense KG')

# Layer 6: Query & Retrieval (middle)
create_box(ax, 3.5, 2.2, 3, 0.6, 'Query Processing & Retrieval', color_api, fontsize=9)
create_box(ax, 3.6, 1.5, 1.3, 0.5, 'BM25 Search', color_api, fontsize=7)
create_box(ax, 5.1, 1.5, 1.3, 0.5, 'Semantic Search', color_api, fontsize=7)

# Arrow from Storage to Query
create_arrow(ax, 7.5, 5.2, 6.5, 2.5, 'Graph Query')

# Layer 7: LLM Integration (bottom-middle)
create_box(ax, 3.5, 0.2, 3, 1, 'LLM Context Provider', color_api, fontsize=10)
create_box(ax, 3.6, 0.65, 1.3, 0.4, 'Subgraph\nExpansion', color_api, fontsize=7)
create_box(ax, 5.1, 0.65, 1.3, 0.4, 'Context\nFormatting', color_api, fontsize=7)
create_box(ax, 3.6, 0.3, 2.8, 0.25, 'Claude Sonnet 3.5 / GPT-4o', color_api, fontsize=7)

# Arrows Query to LLM
create_arrow(ax, 5, 1.5, 5, 1.2, 'Top-10 Triples')

# Layer 8: API & Applications (bottom)
create_box(ax, 0.5, -1.5, 2, 1.2, 'REST API\nFastAPI', color_output, fontsize=9)
create_box(ax, 0.6, -1, 1.8, 0.6, 'Query Endpoint\nGraph Search\nEntity Lookup', color_output, fontsize=6)

create_box(ax, 3, -1.5, 1.8, 1.2, 'Contract Q&A\nApplication', color_output, fontsize=9)
create_box(ax, 5.2, -1.5, 1.8, 1.2, 'Risk Analysis\nDashboard', color_output, fontsize=9)
create_box(ax, 7.2, -1.5, 2, 1.2, 'Compliance &\nComparison Tools', color_output, fontsize=9)

# Arrows LLM to Applications
create_arrow(ax, 4, 0.2, 1.5, -0.3, '')
create_arrow(ax, 5, 0.2, 3.9, -0.3, '')
create_arrow(ax, 6, 0.2, 6.1, -0.3, '')
create_arrow(ax, 5.5, 0.2, 8.2, -0.3, '')

# Add legend
legend_y = -2.5
ax.text(0.5, legend_y, 'Legend:', fontsize=9, fontweight='bold')
create_box(ax, 0.5, legend_y - 0.6, 1, 0.3, 'Input Data', color_input, fontsize=7)
create_box(ax, 1.7, legend_y - 0.6, 1, 0.3, 'Processing', color_processing, fontsize=7)
create_box(ax, 2.9, legend_y - 0.6, 1, 0.3, 'Storage', color_storage, fontsize=7)
create_box(ax, 4.1, legend_y - 0.6, 1, 0.3, 'API/Query', color_api, fontsize=7)
create_box(ax, 5.3, legend_y - 0.6, 1, 0.3, 'Applications', color_output, fontsize=7)

# Add key metrics box
metrics_x, metrics_y = 7.5, -2.2
ax.text(metrics_x, metrics_y + 1, 'Key Targets:', fontsize=9, fontweight='bold')
ax.text(metrics_x, metrics_y + 0.7, '• 98% Triple Validity', fontsize=7)
ax.text(metrics_x, metrics_y + 0.45, '• 65%+ MINE-1 Score', fontsize=7)
ax.text(metrics_x, metrics_y + 0.2, '• <500ms Query Latency', fontsize=7)
ax.text(metrics_x, metrics_y - 0.05, '• 100+ Queries/Second', fontsize=7)
ax.text(metrics_x, metrics_y - 0.3, '• 510 Contracts Processed', fontsize=7)

# Add data flow annotations
ax.text(8.5, 1.5, 'Data Flow:', fontsize=8, fontweight='bold', rotation=90, va='bottom')
ax.annotate('', xy=(8.3, 0.5), xytext=(8.3, 8.5),
            arrowprops=dict(arrowstyle='->', lw=3, color='#FF6B6B', alpha=0.3))

# Save figure
output_path = '/app/sandbox/session_20260112_140312_4731309a153b/figures/system_architecture_diagram.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ System architecture diagram saved to: {output_path}")

# Also save as PDF for vector graphics
output_path_pdf = '/app/sandbox/session_20260112_140312_4731309a153b/figures/system_architecture_diagram.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"✓ PDF version saved to: {output_path_pdf}")

plt.close()

print("✓ Architecture diagram generated successfully")
print("✓ Diagram dimensions: 16x12 inches")
print("✓ Resolution: 300 DPI")
print("✓ Layers: 8 (Input, 3 Processing Stages, Storage, Query, LLM, Applications)")
