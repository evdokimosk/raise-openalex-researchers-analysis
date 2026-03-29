# Read datafile with pandas
# import pandas as pd
# data = pd.read_csv("08de6586-fc14-4f69-89fb-27f1e0a2b8da/datafile.csv")

# Read datafile with pandas
# import pandas as pd
# data = pd.read_csv("08de656e-2032-41b2-898b-2f10fcea0412/datafile.csv")

# Read datafile with pandas
# import pandas as pd
# data = pd.read_csv("08de5939-f1e8-4c39-89d7-0a00e6c5b5d2/datafile.csv")

"""
Main Script for OpenAlex Dataset Analysis

This script uses the OpenAlexDatasetParser class to:
1. Load and parse publication data
2. Generate comprehensive visualizations
3. Create analysis reports in markdown format

Usage:
    python main.py <path_to_csv_file>

Example:
    python main.py works-csv-data.csv
"""

import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from matplotlib.gridspec import GridSpec
import json
from dotenv import load_dotenv

from openalex_dataset_parser import OpenAlexDatasetParser

# Load dataset ID list from environment variable
load_dotenv()
dataset_ids = json.loads(os.getenv("RAISE_DATASET_ID_LIST"))
os.makedirs("results", exist_ok=True)

class VisualizationGenerator:
    """Class to generate all visualizations from parsed data."""

    def __init__(self, data_dict):
        """
        Initialize with data dictionary from parser.

        Parameters:
        -----------
        data_dict : dict
            Dictionary containing all processed data
        """
        self.df = data_dict['dataframe']
        self.oa_analysis = data_dict['oa_analysis']
        self.citation_analysis = data_dict['citation_analysis']
        self.network_data = data_dict['network_data']
        self.pub_type_analysis = data_dict['pub_type_analysis']
        self.collab_size_analysis = data_dict['collab_size_analysis']
        self.analysis_mode = data_dict.get('analysis_mode', 'researcher')

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_main_oa_coauthor_figure(self, output_path='oa_coauthor_analysis.png'):
        """
        Create the main OA and co-authorship analysis figure.
        
        Parameters:
        -----------
        output_path : str
            Path to save the figure
        """
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.3)
        
        # 1. OA Status Distribution (Pie Chart)
        ax1 = fig.add_subplot(gs[0, 0])
        oa_status_counts = self.df['open_access.oa_status'].value_counts()
        colors = ['#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#1f77b4', '#8c564b']
        ax1.pie(oa_status_counts.values, labels=oa_status_counts.index, 
                autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.set_title('Open Access Status Distribution\n(n={} publications)'.format(len(self.df)), 
                     fontsize=12, fontweight='bold')
        
        # 2. OA Trends Over Time
        ax2 = fig.add_subplot(gs[0, 1:])
        oa_by_year = self.oa_analysis['oa_by_year']
        
        x = oa_by_year['year']
        oa_count = oa_by_year['oa_count']
        total_count = oa_by_year['total_count']
        non_oa_count = total_count - oa_count
        
        ax2.bar(x, oa_count, label='Open Access', color='#2ca02c', alpha=0.8)
        ax2.bar(x, non_oa_count, bottom=oa_count, label='Closed Access', 
                color='#d62728', alpha=0.8)
        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('Number of Publications', fontsize=11)
        ax2.set_title('Publication Trends: Open Access vs Closed Access', 
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Citations by OA Status (Box Plot)
        ax3 = fig.add_subplot(gs[1, 0])
        oa_data = [
            self.df[self.df['open_access.is_oa'] == True]['cited_by_count'],
            self.df[self.df['open_access.is_oa'] == False]['cited_by_count']
        ]
        bp = ax3.boxplot(oa_data, labels=['Open Access', 'Closed Access'], 
                        patch_artist=True, showfliers=True, widths=0.6)
        bp['boxes'][0].set_facecolor('#2ca02c')
        bp['boxes'][1].set_facecolor('#d62728')
        for box in bp['boxes']:
            box.set_alpha(0.7)
        ax3.set_ylabel('Citations', fontsize=11)
        ax3.set_title('Citation Distribution by OA Status', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add mean markers
        means = [
            self.citation_analysis['oa_citations']['mean'],
            self.citation_analysis['non_oa_citations']['mean']
        ]
        ax3.plot([1, 2], means, 'D', color='blue', markersize=8, label='Mean', zorder=3)
        ax3.legend()
        
        # 4. Citations by Detailed OA Type
        ax4 = fig.add_subplot(gs[1, 1:])
        citation_by_type = self.df.groupby('open_access.oa_status')['cited_by_count'].agg(
            ['mean', 'count']).reset_index()
        citation_by_type = citation_by_type.sort_values('mean', ascending=False)
        
        bars = ax4.bar(range(len(citation_by_type)), citation_by_type['mean'], 
                      color=colors[:len(citation_by_type)], alpha=0.8)
        ax4.set_xticks(range(len(citation_by_type)))
        ax4.set_xticklabels(citation_by_type['open_access.oa_status'], 
                           rotation=45, ha='right')
        ax4.set_ylabel('Average Citations', fontsize=11)
        ax4.set_title('Average Citations by OA Type', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, citation_by_type['count'])):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={int(count)}', ha='center', va='bottom', fontsize=9)
        
        # 5. Top Collaborators (or Top Partner Institutions)
        ax5 = fig.add_subplot(gs[2, :])
        top_15 = self.network_data['top_collaborators'][:15]

        names = [x['name'][:30] for x in top_15]
        papers = [x['papers'] for x in top_15]
        avg_cit = [x['avg_citations'] for x in top_15]
        oa_rate = [x['oa_rate'] for x in top_15]

        y_pos = np.arange(len(names))
        bars1 = ax5.barh(y_pos, papers, color='steelblue', alpha=0.8, label='Total Papers')

        # Color code by OA rate
        colors_oa = plt.cm.RdYlGn([rate/100 for rate in oa_rate])
        ax5_twin = ax5.twiny()
        bars2 = ax5_twin.barh(y_pos, oa_rate, color=colors_oa, alpha=0.6, height=0.4)

        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(names, fontsize=9)
        ax5.set_xlabel('Number of Collaborative Papers', fontsize=11)

        # Adapt title based on analysis mode
        if self.analysis_mode == 'institution':
            collaborator_title = 'Top 15 Partner Institutions: Papers & Open Access Rate'
        else:
            collaborator_title = 'Top 15 Collaborators: Papers & Open Access Rate'
        ax5.set_title(collaborator_title, fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')
        ax5.invert_yaxis()

        ax5_twin.set_xlabel('Open Access Rate (%)', fontsize=11, color='darkgreen')
        ax5_twin.tick_params(axis='x', labelcolor='darkgreen')
        ax5_twin.set_xlim(0, 100)

        # Add citation annotations
        for i, (pos, cit) in enumerate(zip(y_pos, avg_cit)):
            ax5.text(papers[i] + 1, pos, f'{cit:.1f}c', va='center',
                    fontsize=8, color='red')

        # Adapt main title based on analysis mode
        if self.analysis_mode == 'institution':
            main_entity = self.network_data.get('main_institution', 'Unknown')
            main_title = f'Open Access & Inter-Institution Analysis for {main_entity}'
        else:
            main_entity = self.network_data.get('main_researcher', 'Unknown')
            main_title = f'Open Access & Co-Authorship Analysis for {main_entity}'
        plt.suptitle(main_title, fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def create_network_visualization(self, output_path='network_visualization.png'):
        """
        Create co-authorship or inter-institution network visualization.

        Parameters:
        -----------
        output_path : str
            Path to save the figure
        """
        # Build graph
        G = nx.Graph()

        # Determine main entity and column based on analysis mode
        if self.analysis_mode == 'institution':
            main_entity = self.network_data.get('main_institution', '')
            column_name = 'authorships.institutions.display_name'
            network_type = 'Inter-Institution'
            entity_type = 'institutions'
        else:
            main_entity = self.network_data.get('main_researcher', '')
            column_name = 'authorships.author.display_name'
            network_type = 'Co-Authorship'
            entity_type = 'collaborators'

        top_collab_names = [x['name'] for x in self.network_data['top_collaborators'][:20]]

        edge_citations = {}
        edge_oa_count = {}
        edge_non_oa_count = {}

        def parse_entities(entity_string):
            import pandas as pd
            if pd.isna(entity_string):
                return []
            entities = [e.strip() for e in entity_string.split('|') if e.strip()]
            # For institutions, return unique values
            if self.analysis_mode == 'institution':
                seen = set()
                unique_entities = []
                for e in entities:
                    if e not in seen:
                        seen.add(e)
                        unique_entities.append(e)
                return unique_entities
            return entities

        for idx, row in self.df.iterrows():
            entities = parse_entities(row[column_name])
            citations = row['cited_by_count']
            is_oa = row['open_access.is_oa']

            relevant_entities = [e for e in entities
                                 if e == main_entity or e in top_collab_names]

            for i in range(len(relevant_entities)):
                for j in range(i + 1, len(relevant_entities)):
                    edge = tuple(sorted([relevant_entities[i], relevant_entities[j]]))
                    if edge not in edge_citations:
                        edge_citations[edge] = []
                        edge_oa_count[edge] = 0
                        edge_non_oa_count[edge] = 0

                    edge_citations[edge].append(citations)
                    if is_oa:
                        edge_oa_count[edge] += 1
                    else:
                        edge_non_oa_count[edge] += 1

        # Add edges (use lower threshold for institutions since they appear less frequently per paper)
        min_edge_count = 2 if self.analysis_mode == 'institution' else 3
        for edge, citations in edge_citations.items():
            if len(citations) >= min_edge_count:
                G.add_edge(edge[0], edge[1],
                           weight=len(citations),
                           avg_citations=np.mean(citations),
                           oa_rate=edge_oa_count[edge] / (edge_oa_count[edge] + edge_non_oa_count[edge]))

        if len(G.nodes()) == 0:
            print(f"Warning: Network has no nodes with sufficient connections")
            return

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Node sizes
        node_sizes = []
        for node in G.nodes():
            if node == main_entity:
                node_sizes.append(3000)
            else:
                node_sizes.append(G.degree(node) * 150)

        # Scale edge widths using log scale for better visualization
        # This prevents very thick edges when collaboration counts vary widely
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        min_weight = min(edge_weights) if edge_weights else 1
        max_weight = max(edge_weights) if edge_weights else 1

        # Use logarithmic scaling: map weights to range [1, 8] for edge width
        if max_weight > min_weight:
            edge_widths = [
                1 + 7 * (np.log1p(w - min_weight) / np.log1p(max_weight - min_weight))
                for w in edge_weights
            ]
        else:
            edge_widths = [3.0 for _ in edge_weights]  # uniform width if all same

        edge_colors = [G[u][v]['oa_rate'] for u, v in G.edges()]

        # Draw network 1
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                               edgecolors='navy', linewidths=2, ax=ax1, alpha=0.9)
        edges = nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                                       edge_cmap=plt.cm.RdYlGn, edge_vmin=0, edge_vmax=1,
                                       ax=ax1, alpha=0.6)

        # Create labels (abbreviate institution names)
        if self.analysis_mode == 'institution':
            labels = {node: node[:20] + '...' if len(node) > 20 else node for node in G.nodes()}
        else:
            labels = {node: node.split()[-1] if node != main_entity else node
                      for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax1)

        ax1.set_title(f'{network_type} Network\nEdge Color: OA Rate | Edge Width: Collaboration Count | Node Size: Total Collaborations',
                      fontsize=13, fontweight='bold', pad=20)
        ax1.axis('off')

        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Open Access Rate', rotation=270, labelpad=20, fontsize=11)

        # Network 2
        edge_colors2 = [G[u][v]['weight'] for u, v in G.edges()]

        node_colors = []
        for node in G.nodes():
            node_citations = []
            for idx, row in self.df.iterrows():
                entities = parse_entities(row[column_name])
                if node in entities:
                    node_citations.append(row['cited_by_count'])
            node_colors.append(np.mean(node_citations) if node_citations else 0)

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                               cmap='YlOrRd', edgecolors='navy', linewidths=2, ax=ax2, alpha=0.9)
        edges2 = nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors2,
                                        edge_cmap=plt.cm.Blues, edge_vmin=min(edge_colors2),
                                        edge_vmax=max(edge_colors2), ax=ax2, alpha=0.6)

        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax2)

        ax2.set_title(f'{network_type} Network\nNode Color: Avg Citations | Edge Color: Collaboration Count | Node Size: Total Collaborations',
                      fontsize=13, fontweight='bold', pad=20)
        ax2.axis('off')

        sm2 = plt.cm.ScalarMappable(cmap='YlOrRd',
                                    norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Average Citations', rotation=270, labelpad=20, fontsize=11)

        plt.suptitle(f'{network_type} Network Analysis for {main_entity}\n(Top 20 {entity_type}, edges with {min_edge_count}+ papers)',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def create_detailed_analysis(self, output_path='detailed_analysis.png'):
        """
        Create detailed statistical analysis figure.
        
        Parameters:
        -----------
        output_path : str
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. OA Rate Evolution
        ax1 = axes[0, 0]
        oa_by_year = self.oa_analysis['oa_by_year']
        
        ax1.plot(oa_by_year['year'], oa_by_year['oa_rate'], marker='o', 
                linewidth=2, markersize=8, color='green', label='OA Rate')
        
        # Trend line
        z = np.polyfit(oa_by_year['year'], oa_by_year['oa_rate'], 2)
        p = np.poly1d(z)
        ax1.plot(oa_by_year['year'], p(oa_by_year['year']), "--", 
                alpha=0.5, color='darkgreen', label='Trend')
        
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax1.fill_between(oa_by_year['year'], oa_by_year['oa_rate'], alpha=0.3, color='green')
        ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Open Access Rate (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Open Access Rate Trend Over Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 110)
        
        # 2. Citations by Year and OA
        ax2 = axes[0, 1]
        citation_by_year_oa = self.citation_analysis['citation_by_year_oa']
        if citation_by_year_oa.empty:
            ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=12, color='gray')
            ax2.set_title('Average Citations per Paper: OA vs Non-OA by Year',
                         fontsize=12, fontweight='bold')
        else:
            citation_by_year_oa.plot(kind='bar', ax=ax2, color=['#d62728', '#2ca02c'],
                                    alpha=0.8, width=0.7)
            ax2.set_xlabel('Year', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Average Citations', fontsize=11, fontweight='bold')
            ax2.set_title('Average Citations per Paper: OA vs Non-OA by Year',
                         fontsize=12, fontweight='bold')
            ax2.legend(['Closed Access', 'Open Access'], loc='upper left')
            ax2.grid(True, alpha=0.3, axis='y')

            # Make x-axis labels readable when there are many years
            # Convert year labels to integers (remove decimals)
            x_labels = [label.get_text() for label in ax2.get_xticklabels()]
            x_labels_int = [str(int(float(label))) if label else '' for label in x_labels]
            n_labels = len(x_labels_int)
            if n_labels > 10:
                # Show every Nth label to avoid overcrowding
                step = max(2, n_labels // 8)
                x_labels_int = [label if i % step == 0 else '' for i, label in enumerate(x_labels_int)]
            ax2.set_xticklabels(x_labels_int, rotation=45, ha='right')
        
        # 3. Publication Type Distribution
        ax3 = axes[1, 0]
        type_oa = self.pub_type_analysis['type_oa_distribution']
        type_counts = self.pub_type_analysis['type_counts']

        if type_oa.empty:
            ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=12, color='gray')
            ax3.set_title('OA vs Non-OA Distribution by Publication Type',
                         fontsize=12, fontweight='bold')
        else:
            type_oa.plot(kind='barh', stacked=True, ax=ax3,
                        color=['#d62728', '#2ca02c'], alpha=0.8)
            ax3.set_xlabel('Percentage', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Publication Type', fontsize=11, fontweight='bold')
            ax3.set_title('OA vs Non-OA Distribution by Publication Type',
                         fontsize=12, fontweight='bold')
            ax3.legend(['Closed Access', 'Open Access'], loc='lower right')
            ax3.grid(True, alpha=0.3, axis='x')

            # Add counts
            for i, pub_type in enumerate(type_oa.index):
                total = type_counts[pub_type]
                ax3.text(102, i, f'n={total}', va='center', fontsize=9)
        
        # 4. OA by Collaboration Size
        ax4 = axes[1, 1]
        oa_by_authors = self.collab_size_analysis['oa_by_collaboration_size']
        overall_oa = self.collab_size_analysis['overall_oa_rate']
        
        bars = ax4.bar(range(len(oa_by_authors)), oa_by_authors['oa_rate'],
                      color='teal', alpha=0.8)
        ax4.set_xticks(range(len(oa_by_authors)))
        ax4.set_xticklabels(oa_by_authors['author_bin'])
        ax4.set_xlabel('Number of Co-Authors', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Open Access Rate (%)', fontsize=11, fontweight='bold')
        ax4.set_title('OA Rate by Number of Co-Authors', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=overall_oa, color='red', linestyle='--', alpha=0.5,
                   label=f'Overall avg: {overall_oa:.1f}%')
        ax4.legend()
        
        # Add counts
        for i, (bar, count) in enumerate(zip(bars, oa_by_authors['id'])):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'n={int(count)}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Detailed Open Access & Collaboration Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()


def generate_figure_explanations_markdown(main_researcher):
    """
    Generate the figure explanations markdown file.
    
    Parameters:
    -----------
    main_researcher : str
        Name of the main researcher
        
    Returns:
    --------
    str
        Markdown content
    """
    content = f"""# Detailed Figure Explanations
## Open Access & Co-Authorship Analysis for {main_researcher}

---

## Figure 1: Main Open Access & Co-Authorship Analysis
**File:** `oa_coauthor_analysis.png`

This comprehensive dashboard contains 6 panels that provide an overview of open access patterns and collaboration:

### **Panel 1 (Top Left): Open Access Status Distribution - Pie Chart**
**What it shows:** The breakdown of all publications by their open access status.

**How to read it:**
- Each slice represents a different OA category
- Percentages show the proportion of total publications
- Colors distinguish different access types

**Key findings:**
- **Closed (red):** Papers behind paywalls
- **Gold (green):** Published in fully OA journals (best for visibility)
- **Green (purple):** Self-archived versions (author deposits in repositories)
- **Hybrid (orange):** OA option in subscription journals (often costly)
- **Diamond (blue):** Free OA without author charges (ideal but rare)
- **Bronze (brown):** Free to read but no clear license

---

### **Panel 2 (Top Center & Right): Publication Trends Over Time - Stacked Bar Chart**
**What it shows:** The number of publications each year, split into Open Access (green) and Closed Access (red).

**How to read it:**
- X-axis: Publication year
- Y-axis: Number of publications
- Green bars: Open access papers
- Red bars (stacked on top): Closed access papers
- Total height = total publications that year

**Interpretation:** Shows the evolution of OA adoption over the researcher's career.

---

### **Panel 3 (Middle Left): Citation Distribution by OA Status - Box Plot**
**What it shows:** Comparison of how many citations papers receive based on their access type.

**How to read it:**
- Two boxes: Open Access (green) vs Closed Access (red)
- Box spans from 25th to 75th percentile (middle 50% of data)
- Line inside box = median (50th percentile)
- Whiskers extend to show range
- Dots = outliers (unusually high citations)
- Blue diamonds = mean (average)

**Interpretation:** Compare citation patterns between open and closed access publications.

---

### **Panel 4 (Middle Center & Right): Average Citations by OA Type - Bar Chart**
**What it shows:** Average citation counts broken down by specific OA categories.

**How to read it:**
- X-axis: Six different access types
- Y-axis: Average number of citations per paper
- Bar height = mean citations
- "n=" labels show how many papers in each category

---

### **Panel 5 (Bottom): Top 15 Collaborators - Dual Horizontal Bar Chart**
**What it shows:** The most frequent collaborators with multiple metrics displayed simultaneously.

**How to read it:**
- Y-axis: Collaborator names
- Blue bars (left scale): Total number of collaborative papers
- Colored bars (top scale): Open Access rate (% of joint papers that are OA)
- Bar colors for OA rate: Red (0%) → Yellow (50%) → Green (100%)
- Red numbers: Average citations of joint papers

---

## Figure 2: Co-Authorship Network Visualization
**File:** `network_visualization.png`

This figure contains two network graphs showing the same collaboration structure from different analytical perspectives.

### **Left Panel: Network Colored by Open Access Rate**

**What it shows:** Visual map of collaboration relationships, emphasizing OA publishing patterns.

**How to read it:**
- **Nodes (circles):** Individual researchers
- **Node size:** Larger = more total collaborations
- **Edges (lines):** Represent co-authorship
- **Edge width:** Thicker = more papers co-authored together
- **Edge color:** Red (0% OA) → Yellow (50% OA) → Green (100% OA)

---

### **Right Panel: Network Colored by Citation Impact**

**What it shows:** Same network structure, now emphasizing research impact.

**How to read it:**
- **Node color:** Light yellow (low citations) → Dark red (high citations)
- **Edge color:** Light blue (few joint papers) → Dark blue (many joint papers)

**Interpretation:** Reveals productive core of high-impact researchers and collaboration patterns.

---

## Figure 3: Detailed Statistical Analysis
**File:** `detailed_analysis.png`

This figure contains 4 panels providing deeper statistical insights.

### **Panel 1 (Top Left): OA Rate Trend Over Time - Line Chart**

**What it shows:** How the percentage of publications that are open access has changed year by year.

**How to read it:**
- X-axis: Years
- Y-axis: Percentage of that year's papers that are OA
- Green line with circles: Actual OA rate each year
- Dashed green line: Polynomial trend line
- Red dashed line: 50% threshold

---

### **Panel 2 (Top Right): Average Citations by Year and OA Status - Grouped Bar Chart**

**What it shows:** For each year, the average citations for closed vs. open access papers.

**How to read it:**
- X-axis: Publication year
- Y-axis: Average citations per paper
- Red bars: Closed access papers
- Green bars: Open access papers

---

### **Panel 3 (Bottom Left): OA Distribution by Publication Type - Stacked Horizontal Bar**

**What it shows:** What percentage of each publication type is open vs. closed access.

**How to read it:**
- Y-axis: Publication types
- X-axis: Percentage (0-100%)
- Red portion: Closed access
- Green portion: Open access

---

### **Panel 4 (Bottom Right): OA Rate by Number of Co-Authors - Bar Chart**

**What it shows:** Whether collaboration size affects open access publishing decisions.

**How to read it:**
- X-axis: Number of co-authors (binned into ranges)
- Y-axis: OA rate (%) for papers with that many authors
- Red dashed line: Overall average OA rate

---

## Summary: How to Use These Figures

**For the researcher:**
1. **Career narrative:** Show progression from closed to open science
2. **Impact demonstration:** High citations achievable in both OA and closed venues
3. **Collaboration strength:** Visualize extensive network and key partnerships
4. **Strategic planning:** Identify areas for improvement

**For presentations:**
- Use Figure 1 for comprehensive overview
- Use Figure 2 to showcase collaboration network
- Use Figure 3 for detailed trend analysis

**For grant applications:**
- Open access commitment: Panel showing OA rate trends
- Collaboration evidence: Network graphs showing co-authors
- Impact metrics: Citation distributions and collaborator statistics
"""
    return content


def export_metrics_to_csv(parser, data_dict, output_path='results/metrics_export.csv'):
    """
    Export all analysis metrics to a CSV file with JSON-style column names.

    Each metric is a column with a hierarchical name (e.g., Identification.AnalysisMode).
    This format allows easy comparison when combining multiple CSV files.

    Parameters:
    -----------
    parser : OpenAlexDatasetParser
        The parser instance with loaded data
    data_dict : dict
        Dictionary containing all processed data
    output_path : str
        Path to save the CSV file
    """
    import pandas as pd
    import math

    def safe_int(val, default=0):
        try:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return int(val)
        except (TypeError, ValueError):
            return default

    def safe_round(val, ndigits=2, default=0.0):
        try:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return round(float(val), ndigits)
        except (TypeError, ValueError):
            return default

    oa_analysis = data_dict['oa_analysis']
    citation_analysis = data_dict['citation_analysis']
    network_data = data_dict['network_data']
    pub_type_analysis = data_dict['pub_type_analysis']
    collab_size_analysis = data_dict['collab_size_analysis']
    analysis_mode = data_dict.get('analysis_mode', 'researcher')

    # Determine entity name based on analysis mode
    if analysis_mode == 'institution':
        entity_name = network_data.get('main_institution', 'Unknown')
        total_entities_key = 'total_institutions'
        avg_per_paper_key = 'avg_institutions_per_paper'
        avg_per_entity_key = 'avg_papers_per_institution'
    else:
        entity_name = network_data.get('main_researcher', 'Unknown')
        total_entities_key = 'total_authors'
        avg_per_paper_key = 'avg_authors_per_paper'
        avg_per_entity_key = 'avg_papers_per_author'

    # Build flat dictionary with JSON-style keys
    metrics = {}

    # === IDENTIFICATION ===
    metrics['Identification.AnalysisMode'] = analysis_mode.capitalize()
    metrics['Identification.EntityName'] = entity_name
    metrics['Identification.AnalysisDate'] = pd.Timestamp.now().strftime('%Y-%m-%d')

    # === PUBLICATIONS ===
    metrics['Publications.Total'] = oa_analysis['total_publications']
    metrics['Publications.YearStart'] = int(parser.df['publication_year'].min())
    metrics['Publications.YearEnd'] = int(parser.df['publication_year'].max())
    metrics['Publications.SpanYears'] = int(parser.df['publication_year'].max() - parser.df['publication_year'].min() + 1)

    # === OPEN ACCESS SUMMARY ===
    metrics['OpenAccess.Count'] = int(oa_analysis['oa_count'])
    metrics['OpenAccess.ClosedCount'] = int(oa_analysis['non_oa_count'])
    metrics['OpenAccess.RatePercent'] = round(oa_analysis['oa_rate'], 2)

    # === OPEN ACCESS TRENDS ===
    metrics['OpenAccess.RateEarlyPercent'] = round(oa_analysis['early_oa_rate'], 2)
    metrics['OpenAccess.RateRecentPercent'] = round(oa_analysis['recent_oa_rate'], 2)
    metrics['OpenAccess.RateChangePercent'] = round(oa_analysis['recent_oa_rate'] - oa_analysis['early_oa_rate'], 2)
    metrics['OpenAccess.Recent3YearCount'] = int(oa_analysis['recent_performance']['count'])
    metrics['OpenAccess.Recent3YearOACount'] = int(oa_analysis['recent_performance']['oa_count'])
    metrics['OpenAccess.Recent3YearRatePercent'] = round(oa_analysis['recent_performance']['oa_rate'], 2)

    # === OA TYPE DISTRIBUTION ===
    metrics['OAType.ClosedCount'] = oa_analysis['oa_status_distribution'].get('closed', 0)
    metrics['OAType.GoldCount'] = oa_analysis['oa_status_distribution'].get('gold', 0)
    metrics['OAType.GreenCount'] = oa_analysis['oa_status_distribution'].get('green', 0)
    metrics['OAType.HybridCount'] = oa_analysis['oa_status_distribution'].get('hybrid', 0)
    metrics['OAType.BronzeCount'] = oa_analysis['oa_status_distribution'].get('bronze', 0)
    metrics['OAType.DiamondCount'] = oa_analysis['oa_status_distribution'].get('diamond', 0)

    # === CITATIONS - OA PAPERS ===
    oa_cit = citation_analysis['oa_citations']
    metrics['CitationsOA.Total'] = safe_int(oa_cit['total'])
    metrics['CitationsOA.Mean'] = safe_round(oa_cit['mean'], 2)
    metrics['CitationsOA.Median'] = safe_round(oa_cit['median'], 2)
    metrics['CitationsOA.Max'] = safe_int(oa_cit['max'])
    metrics['CitationsOA.StdDev'] = safe_round(oa_cit['std'], 2)

    # === CITATIONS - CLOSED PAPERS ===
    non_oa_cit = citation_analysis['non_oa_citations']
    metrics['CitationsClosed.Total'] = safe_int(non_oa_cit['total'])
    metrics['CitationsClosed.Mean'] = safe_round(non_oa_cit['mean'], 2)
    metrics['CitationsClosed.Median'] = safe_round(non_oa_cit['median'], 2)
    metrics['CitationsClosed.Max'] = safe_int(non_oa_cit['max'])
    metrics['CitationsClosed.StdDev'] = safe_round(non_oa_cit['std'], 2)

    # === CITATION COMPARISON ===
    metrics['CitationComparison.MeanDifference'] = safe_round(oa_cit['mean'] - non_oa_cit['mean'], 2)
    metrics['CitationComparison.PValue'] = safe_round(citation_analysis['statistical_test']['p_value'], 6)
    metrics['CitationComparison.IsSignificant'] = citation_analysis['statistical_test']['significant']

    # === HIGH IMPACT PAPERS ===
    metrics['HighImpact.Total'] = citation_analysis['high_impact']['total']
    metrics['HighImpact.OACount'] = citation_analysis['high_impact']['oa_count']
    metrics['HighImpact.ClosedCount'] = citation_analysis['high_impact']['closed_count']
    metrics['HighImpact.OARatePercent'] = round(citation_analysis['high_impact']['oa_percentage'], 2)

    # === COLLABORATION NETWORK ===
    metrics['Network.TotalPartners'] = network_data.get(total_entities_key, 0)
    metrics['Network.AvgPartnersPerPaper'] = round(network_data.get(avg_per_paper_key, 0), 2)
    metrics['Network.AvgPapersPerPartner'] = round(network_data.get(avg_per_entity_key, 0), 2)

    # === TOP 5 PARTNERS ===
    for i, collab in enumerate(network_data['top_collaborators'][:5], 1):
        metrics[f'Partner{i}.Name'] = collab['name']
        metrics[f'Partner{i}.Papers'] = collab['papers']
        metrics[f'Partner{i}.AvgCitations'] = round(collab['avg_citations'], 2)
        metrics[f'Partner{i}.OARatePercent'] = round(collab['oa_rate'], 2)

    # === OA BY TEAM SIZE ===
    metrics['TeamSize.OverallOARatePercent'] = round(collab_size_analysis['overall_oa_rate'], 2)
    oa_by_collab = collab_size_analysis['oa_by_collaboration_size']
    for _, row in oa_by_collab.iterrows():
        bin_name = str(row['author_bin']).replace('-', 'to').replace('+', 'plus')
        metrics[f'TeamSize.{bin_name}.Count'] = int(row['id'])
        metrics[f'TeamSize.{bin_name}.OARatePercent'] = round(row['oa_rate'], 2)

    # === PUBLICATION TYPES ===
    type_oa = pub_type_analysis['type_oa_distribution']
    type_counts = pub_type_analysis['type_counts']
    for pub_type in type_oa.index:
        safe_name = pub_type.capitalize().replace('-', '').replace(' ', '')
        metrics[f'PubType.{safe_name}.Count'] = int(type_counts.get(pub_type, 0))
        if True in type_oa.columns:
            metrics[f'PubType.{safe_name}.OARatePercent'] = round(type_oa.loc[pub_type, True], 2)

    # Create DataFrame with single row and save
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path}")

    return df_metrics


def compare_institutions(csv_path1, csv_path2, output_path='results/institution_comparison.png'):
    """
    Compare and visualize metrics from two different institution CSV exports.

    Parameters:
    -----------
    csv_path1 : str
        Path to first institution's metrics_export.csv
    csv_path2 : str
        Path to second institution's metrics_export.csv
    output_path : str
        Path to save the comparison figure
    """
    import pandas as pd

    # Load both CSV files
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    # Get institution names
    name1 = df1['Identification.EntityName'].iloc[0]
    name2 = df2['Identification.EntityName'].iloc[0]

    # Truncate names for display
    name1_short = name1[:25] + '...' if len(name1) > 25 else name1
    name2_short = name2[:25] + '...' if len(name2) > 25 else name2

    # Set up the figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = ['#1f77b4', '#ff7f0e']  # Blue for inst1, Orange for inst2

    # 1. Publications Overview (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics_pub = ['Publications.Total', 'OpenAccess.Count', 'OpenAccess.ClosedCount']
    labels_pub = ['Total\nPublications', 'Open\nAccess', 'Closed\nAccess']
    values1_pub = [df1[m].iloc[0] for m in metrics_pub]
    values2_pub = [df2[m].iloc[0] for m in metrics_pub]

    x = np.arange(len(labels_pub))
    width = 0.35
    bars1 = ax1.bar(x - width/2, values1_pub, width, label=name1_short, color=colors[0], alpha=0.8)
    bars2 = ax1.bar(x + width/2, values2_pub, width, label=name2_short, color=colors[1], alpha=0.8)

    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Publications Overview', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_pub)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9)

    # 2. Open Access Rate Comparison (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    metrics_oa = ['OpenAccess.RatePercent', 'OpenAccess.RateEarlyPercent', 'OpenAccess.RateRecentPercent']
    labels_oa = ['Overall\nOA Rate', 'Early\n(≤2015)', 'Recent\n(≥2020)']
    values1_oa = [df1[m].iloc[0] for m in metrics_oa]
    values2_oa = [df2[m].iloc[0] for m in metrics_oa]

    x = np.arange(len(labels_oa))
    bars1 = ax2.bar(x - width/2, values1_oa, width, label=name1_short, color=colors[0], alpha=0.8)
    bars2 = ax2.bar(x + width/2, values2_oa, width, label=name2_short, color=colors[1], alpha=0.8)

    ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Open Access Rate Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_oa)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)

    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    # 3. OA Type Distribution (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    oa_types = ['OAType.GoldCount', 'OAType.GreenCount', 'OAType.HybridCount', 'OAType.BronzeCount', 'OAType.DiamondCount']
    oa_labels = ['Gold', 'Green', 'Hybrid', 'Bronze', 'Diamond']

    values1_oa_type = [df1[m].iloc[0] if m in df1.columns else 0 for m in oa_types]
    values2_oa_type = [df2[m].iloc[0] if m in df2.columns else 0 for m in oa_types]

    x = np.arange(len(oa_labels))
    bars1 = ax3.bar(x - width/2, values1_oa_type, width, label=name1_short, color=colors[0], alpha=0.8)
    bars2 = ax3.bar(x + width/2, values2_oa_type, width, label=name2_short, color=colors[1], alpha=0.8)

    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('OA Type Distribution', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(oa_labels, rotation=45, ha='right')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Citation Metrics Comparison (Middle Left)
    ax4 = fig.add_subplot(gs[1, 0])
    citation_metrics = ['CitationsOA.Mean', 'CitationsClosed.Mean']
    citation_labels = ['OA Papers\nMean Citations', 'Closed Papers\nMean Citations']
    values1_cit = [df1[m].iloc[0] for m in citation_metrics]
    values2_cit = [df2[m].iloc[0] for m in citation_metrics]

    x = np.arange(len(citation_labels))
    bars1 = ax4.bar(x - width/2, values1_cit, width, label=name1_short, color=colors[0], alpha=0.8)
    bars2 = ax4.bar(x + width/2, values2_cit, width, label=name2_short, color=colors[1], alpha=0.8)

    ax4.set_ylabel('Average Citations', fontsize=11, fontweight='bold')
    ax4.set_title('Citation Impact: OA vs Closed', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(citation_labels)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

    # 5. Network Metrics (Middle Center)
    ax5 = fig.add_subplot(gs[1, 1])
    network_metrics = ['Network.TotalPartners', 'Network.AvgPartnersPerPaper']
    values1_net = [df1[m].iloc[0] for m in network_metrics]
    values2_net = [df2[m].iloc[0] for m in network_metrics]

    # Use two y-axes for different scales
    x = np.arange(1)
    ax5.bar(x - width/2, [values1_net[0]], width, label=name1_short, color=colors[0], alpha=0.8)
    ax5.bar(x + width/2, [values2_net[0]], width, label=name2_short, color=colors[1], alpha=0.8)
    ax5.set_ylabel('Total Partners', fontsize=11, fontweight='bold')
    ax5.set_xticks([0])
    ax5.set_xticklabels(['Total Partners'])
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_title('Collaboration Network', fontsize=12, fontweight='bold')

    # Add text for avg partners per paper
    ax5.text(0.5, 0.95, f'Avg Partners/Paper:\n{name1_short}: {values1_net[1]:.1f}\n{name2_short}: {values2_net[1]:.1f}',
             transform=ax5.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 6. High Impact Papers (Middle Right)
    ax6 = fig.add_subplot(gs[1, 2])
    hi_metrics = ['HighImpact.Total', 'HighImpact.OACount', 'HighImpact.ClosedCount']
    hi_labels = ['Total High\nImpact', 'High Impact\nOA', 'High Impact\nClosed']
    values1_hi = [df1[m].iloc[0] if m in df1.columns else 0 for m in hi_metrics]
    values2_hi = [df2[m].iloc[0] if m in df2.columns else 0 for m in hi_metrics]

    x = np.arange(len(hi_labels))
    bars1 = ax6.bar(x - width/2, values1_hi, width, label=name1_short, color=colors[0], alpha=0.8)
    bars2 = ax6.bar(x + width/2, values2_hi, width, label=name2_short, color=colors[1], alpha=0.8)

    ax6.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax6.set_title('High Impact Papers (>50 citations)', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(hi_labels)
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Summary Radar Chart (Bottom, spanning all columns)
    ax7 = fig.add_subplot(gs[2, :], polar=True)

    # Normalize metrics for radar chart (0-100 scale)
    radar_metrics = [
        ('OA Rate', 'OpenAccess.RatePercent', 100),
        ('Recent OA Rate', 'OpenAccess.RateRecentPercent', 100),
        ('OA Citation Advantage', 'CitationComparison.MeanDifference', None),  # Will normalize
        ('High Impact OA %', 'HighImpact.OARatePercent', 100),
        ('Network Size', 'Network.TotalPartners', None),  # Will normalize
    ]

    # Get values and normalize
    values1_radar = []
    values2_radar = []
    radar_labels = []

    for label, metric, max_val in radar_metrics:
        radar_labels.append(label)
        v1 = df1[metric].iloc[0] if metric in df1.columns else 0
        v2 = df2[metric].iloc[0] if metric in df2.columns else 0

        if max_val is None:
            # Normalize based on max of both
            max_v = max(abs(v1), abs(v2), 1)
            values1_radar.append(min(100, (v1 / max_v) * 50 + 50))  # Scale to 0-100
            values2_radar.append(min(100, (v2 / max_v) * 50 + 50))
        else:
            values1_radar.append(min(100, v1))
            values2_radar.append(min(100, v2))

    # Number of variables
    num_vars = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    values1_radar += values1_radar[:1]
    values2_radar += values2_radar[:1]

    ax7.plot(angles, values1_radar, 'o-', linewidth=2, label=name1_short, color=colors[0])
    ax7.fill(angles, values1_radar, alpha=0.25, color=colors[0])
    ax7.plot(angles, values2_radar, 'o-', linewidth=2, label=name2_short, color=colors[1])
    ax7.fill(angles, values2_radar, alpha=0.25, color=colors[1])

    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(radar_labels, fontsize=10)
    ax7.set_ylim(0, 100)
    ax7.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=10)
    ax7.set_title('Comparative Performance Overview', fontsize=12, fontweight='bold', pad=20)

    # Main title
    plt.suptitle(f'Institution Comparison: {name1_short} vs {name2_short}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<35} {name1_short:>12} {name2_short:>12}")
    print("-" * 60)
    print(f"{'Total Publications':<35} {int(df1['Publications.Total'].iloc[0]):>12} {int(df2['Publications.Total'].iloc[0]):>12}")
    print(f"{'Open Access Rate (%)':<35} {df1['OpenAccess.RatePercent'].iloc[0]:>12.1f} {df2['OpenAccess.RatePercent'].iloc[0]:>12.1f}")
    print(f"{'Recent OA Rate (≥2020) (%)':<35} {df1['OpenAccess.RateRecentPercent'].iloc[0]:>12.1f} {df2['OpenAccess.RateRecentPercent'].iloc[0]:>12.1f}")
    print(f"{'OA Citation Mean':<35} {df1['CitationsOA.Mean'].iloc[0]:>12.1f} {df2['CitationsOA.Mean'].iloc[0]:>12.1f}")
    print(f"{'Closed Citation Mean':<35} {df1['CitationsClosed.Mean'].iloc[0]:>12.1f} {df2['CitationsClosed.Mean'].iloc[0]:>12.1f}")
    print(f"{'Total Partners':<35} {int(df1['Network.TotalPartners'].iloc[0]):>12} {int(df2['Network.TotalPartners'].iloc[0]):>12}")
    print("=" * 60)

    return fig


def main():
    """Main execution function."""

    # Parse command line arguments for analysis mode
    # Usage: python main.py [researcher|institution]
    analysis_mode = 'researcher'  # default
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['researcher', 'institution']:
            analysis_mode = sys.argv[1].lower()
        else:
            print(f"Warning: Unknown analysis mode '{sys.argv[1]}'. Using 'researcher'.")
            print("Valid modes: researcher, institution")

    # Can also be set via environment variable
    env_mode = os.getenv("RAISE_ANALYSIS_MODE")
    if env_mode and env_mode.lower() in ['researcher', 'institution']:
        analysis_mode = env_mode.lower()

    csv_file = f"{dataset_ids[0]}/datafile.csv"

    # Check file exists
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)

    print("=" * 80)
    print("OpenAlex Dataset Analysis System")
    print("=" * 80)
    print(f"Analysis Mode: {analysis_mode.upper()}")
    print()

    # Initialize parser with analysis mode
    print("Step 1: Loading and parsing data...")
    parser = OpenAlexDatasetParser(csv_file, analysis_mode=analysis_mode)

    if not parser.load_data():
        print("Failed to load data. Exiting.")
        sys.exit(1)

    # Perform analyses
    print("\nStep 2: Performing analyses...")
    print("  - Analyzing Open Access patterns...")
    oa_analysis = parser.analyze_open_access()

    print("  - Analyzing citation impact...")
    citation_analysis = parser.analyze_citations()

    # Build appropriate network based on mode
    if analysis_mode == 'institution':
        print("  - Building inter-institution collaboration network...")
        network_data = parser.build_institution_network()
    else:
        print("  - Building co-authorship network...")
        network_data = parser.build_coauthorship_network()

    print("  - Analyzing publication types...")
    pub_type_analysis = parser.get_publication_type_analysis()

    print("  - Analyzing collaboration sizes...")
    collab_size_analysis = parser.get_collaboration_size_analysis()

    # Generate text report
    print("\nStep 3: Generating reports...")
    report_text = parser.generate_text_report()

    # Prepare data for visualization and CSV export
    data_dict = parser.export_data_for_visualization()

    # Export metrics to CSV for comparison analysis
    print("  - Exporting metrics to CSV...")
    export_metrics_to_csv(parser, data_dict, 'results/metrics_export.csv')

    # Generate figure explanations content - use appropriate entity name
    if analysis_mode == 'institution':
        main_entity = network_data.get('main_institution', 'Unknown')
    else:
        main_entity = network_data.get('main_researcher', 'Unknown')
    fig_explanations = generate_figure_explanations_markdown(main_entity)

    # Convert figure explanations from markdown to plain text
    fig_explanations_txt = fig_explanations.replace('**', '').replace('`', '').replace('# ', '').replace('## ', '').replace('### ', '').replace('---', '-' * 40)

    # Create combined text report with figure explanations
    with open('results/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
        f.write('\n\n')
        f.write('=' * 80 + '\n')
        f.write('FIGURE EXPLANATIONS\n')
        f.write('=' * 80 + '\n\n')
        f.write(fig_explanations_txt)
    print("  ✓ Saved: analysis_report.txt")
    
    # Generate visualizations
    print("\nStep 4: Generating visualizations...")
    viz_gen = VisualizationGenerator(data_dict)
    
    viz_gen.create_main_oa_coauthor_figure('results/oa_coauthor_analysis.png')
    viz_gen.create_network_visualization('results/network_visualization.png')
    viz_gen.create_detailed_analysis('results/detailed_analysis.png')
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nAnalysis Mode: {analysis_mode.upper()}")
    print(f"Main Entity: {main_entity}")
    print("\nGenerated files:")
    print("  1. oa_coauthor_analysis.png - Main OA & collaboration overview")
    if analysis_mode == 'institution':
        print("  2. network_visualization.png - Inter-institution collaboration network")
    else:
        print("  2. network_visualization.png - Co-authorship network graphs")
    print("  3. detailed_analysis.png - Detailed statistical analysis")
    print("  4. analysis_report.txt - Comprehensive report with figure explanations")
    print("  5. metrics_export.csv - Machine-readable metrics for comparison analysis")
    print()
    print("Usage: python main.py [researcher|institution]")
    print("  Or set RAISE_ANALYSIS_MODE environment variable")
    print()


if __name__ == "__main__":
    #compare_institutions('results/metrics_export.csv', 'results/metrics2_export.csv', 'results/metrics_comparison_export.png')
    main()
