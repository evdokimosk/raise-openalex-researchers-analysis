"""
OpenAlex Dataset Parser and Analyzer

A comprehensive class-based module for parsing, analyzing, and visualizing
OpenAlex publication data with focus on Open Access patterns, citation impact,
and co-authorship networks.

Author: Analysis System
Date: 2025
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
import json


class OpenAlexDatasetParser:
    """
    Parser and analyzer for OpenAlex publication datasets.

    This class provides methods to:
    - Load and parse OpenAlex CSV data
    - Analyze Open Access patterns and trends
    - Calculate citation metrics
    - Build and analyze co-authorship/co-institution networks
    - Generate statistical reports

    Supports two analysis modes:
    - 'researcher': Analyze publications centered on a main researcher
    - 'institution': Analyze publications centered on a main institution
    """

    def __init__(self, filename, analysis_mode='researcher'):
        """
        Initialize the parser with a dataset file.

        Parameters:
        -----------
        filename : str
            Path to the CSV file containing OpenAlex data
        analysis_mode : str
            Either 'researcher' or 'institution' to determine analysis focus
        """
        self.filename = filename
        self.analysis_mode = analysis_mode
        self.df = None
        self.main_researcher = None
        self.main_institution = None
        self.network_data = {}
        
    def load_data(self):
        """
        Load the dataset from CSV file.
        
        Returns:
        --------
        bool
            True if loading successful, False otherwise
        """
        try:
            self.df = pd.read_csv(self.filename)
            # Coerce key numeric columns so downstream aggregations don't produce object dtype
            self.df['open_access.is_oa'] = pd.to_numeric(self.df['open_access.is_oa'], errors='coerce').fillna(0).astype(float)
            self.df['cited_by_count'] = pd.to_numeric(self.df['cited_by_count'], errors='coerce').fillna(0).astype(float)
            self.df['publication_year'] = pd.to_numeric(self.df['publication_year'], errors='coerce')
            print(f"✓ Loaded {len(self.df)} publications from {self.filename}")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def get_basic_stats(self):
        """
        Get basic dataset statistics.
        
        Returns:
        --------
        dict
            Dictionary containing basic statistics
        """
        if self.df is None:
            return None
            
        return {
            'total_publications': len(self.df),
            'columns': self.df.columns.tolist(),
            'date_range': (self.df['publication_year'].min(), 
                          self.df['publication_year'].max()),
            'missing_values': self.df.isnull().sum().to_dict()
        }
    
    def analyze_open_access(self):
        """
        Analyze Open Access patterns in the dataset.
        
        Returns:
        --------
        dict
            Dictionary containing OA analysis results including:
            - Overall OA statistics
            - OA status breakdown
            - Temporal trends
            - Recent performance
        """
        if self.df is None:
            return None
        
        oa_count = self.df['open_access.is_oa'].sum()
        non_oa_count = (self.df['open_access.is_oa'] == 0).sum()
        oa_rate = oa_count / len(self.df) * 100
        
        # OA status distribution
        oa_status = self.df['open_access.oa_status'].value_counts().to_dict()
        
        # Temporal trends
        oa_by_year = self.df.groupby('publication_year').agg({
            'open_access.is_oa': ['sum', 'count']
        }).reset_index()
        oa_by_year.columns = ['year', 'oa_count', 'total_count']
        oa_by_year['oa_count'] = pd.to_numeric(oa_by_year['oa_count'], errors='coerce').fillna(0)
        oa_by_year['oa_rate'] = (oa_by_year['oa_count'] /
                                  oa_by_year['total_count'] * 100)
        
        # Early vs recent
        early_years = self.df[self.df['publication_year'] <= 2015]
        recent_years = self.df[self.df['publication_year'] >= 2020]
        
        early_oa_rate = early_years['open_access.is_oa'].mean() * 100 if len(early_years) > 0 else 0
        recent_oa_rate = recent_years['open_access.is_oa'].mean() * 100 if len(recent_years) > 0 else 0
        
        # Recent 3 years
        recent_3_years = self.df[self.df['publication_year'] >= 2022]
        
        return {
            'total_publications': len(self.df),
            'oa_count': oa_count,
            'non_oa_count': non_oa_count,
            'oa_rate': oa_rate,
            'oa_status_distribution': oa_status,
            'oa_by_year': oa_by_year,
            'early_oa_rate': early_oa_rate,
            'recent_oa_rate': recent_oa_rate,
            'recent_performance': {
                'count': len(recent_3_years),
                'oa_count': recent_3_years['open_access.is_oa'].sum(),
                'oa_rate': recent_3_years['open_access.is_oa'].mean() * 100 if len(recent_3_years) > 0 else 0
            }
        }
    
    def analyze_citations(self):
        """
        Analyze citation patterns by Open Access status.
        
        Returns:
        --------
        dict
            Dictionary containing citation analysis including:
            - OA vs non-OA citation statistics
            - Statistical test results
            - Citation by OA type
            - High-impact papers analysis
        """
        if self.df is None:
            return None
        
        oa_citations = self.df[self.df['open_access.is_oa'] == True]['cited_by_count']
        non_oa_citations = self.df[self.df['open_access.is_oa'] == False]['cited_by_count']
        
        # Statistical test
        t_stat, p_value = stats.mannwhitneyu(non_oa_citations, oa_citations, 
                                             alternative='two-sided')
        
        # Citation by OA type
        citation_by_type = self.df.groupby('open_access.oa_status')['cited_by_count'].agg(
            ['count', 'sum', 'mean', 'median']
        ).round(2)
        
        # High impact papers
        high_impact = self.df[self.df['cited_by_count'] > 50]
        high_impact_oa = high_impact[high_impact['open_access.is_oa'] == True]
        high_impact_closed = high_impact[high_impact['open_access.is_oa'] == False]
        
        # Citations by year and OA status
        citation_by_year_oa = self.df.groupby(['publication_year', 'open_access.is_oa'])[
            'cited_by_count'].mean().unstack()
        
        return {
            'oa_citations': {
                'count': len(oa_citations),
                'total': oa_citations.sum(),
                'mean': oa_citations.mean(),
                'median': oa_citations.median(),
                'max': oa_citations.max(),
                'std': oa_citations.std()
            },
            'non_oa_citations': {
                'count': len(non_oa_citations),
                'total': non_oa_citations.sum(),
                'mean': non_oa_citations.mean(),
                'median': non_oa_citations.median(),
                'max': non_oa_citations.max(),
                'std': non_oa_citations.std()
            },
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'citation_by_type': citation_by_type.to_dict(),
            'high_impact': {
                'total': len(high_impact),
                'oa_count': len(high_impact_oa),
                'closed_count': len(high_impact_closed),
                'oa_percentage': len(high_impact_oa) / len(high_impact) * 100 if len(high_impact) > 0 else 0
            },
            'citation_by_year_oa': citation_by_year_oa
        }
    
    def parse_authors(self, author_string):
        """
        Parse author names from pipe-separated string.

        Parameters:
        -----------
        author_string : str
            Pipe-separated author names

        Returns:
        --------
        list
            List of author names
        """
        if pd.isna(author_string):
            return []
        return [a.strip() for a in author_string.split('|')]

    def parse_institutions(self, institution_string):
        """
        Parse institution names from pipe-separated string and return unique institutions.

        Parameters:
        -----------
        institution_string : str
            Pipe-separated institution names (one per author position)

        Returns:
        --------
        list
            List of unique institution names
        """
        if pd.isna(institution_string):
            return []
        # Get unique institutions while preserving order
        institutions = [i.strip() for i in institution_string.split('|') if i.strip()]
        seen = set()
        unique_institutions = []
        for inst in institutions:
            if inst not in seen:
                seen.add(inst)
                unique_institutions.append(inst)
        return unique_institutions
    
    def build_coauthorship_network(self):
        """
        Build co-authorship network from publications.
        
        Returns:
        --------
        dict
            Dictionary containing network data including:
            - Main researcher
            - Author statistics
            - Collaboration pairs
            - Top collaborators
        """
        if self.df is None:
            return None
        
        all_authors = []
        author_papers = defaultdict(list)
        coauthor_pairs = defaultdict(int)
        author_citations = defaultdict(list)
        author_oa_status = defaultdict(lambda: {'oa': 0, 'non_oa': 0})
        
        # Build network
        for idx, row in self.df.iterrows():
            authors = self.parse_authors(row['authorships.author.display_name'])
            paper_id = row['id']
            citations = row['cited_by_count']
            is_oa = row['open_access.is_oa']
            
            all_authors.extend(authors)
            
            for author in authors:
                author_papers[author].append(paper_id)
                author_citations[author].append(citations)
                if is_oa:
                    author_oa_status[author]['oa'] += 1
                else:
                    author_oa_status[author]['non_oa'] += 1
            
            # Track co-authorship pairs
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    pair = tuple(sorted([authors[i], authors[j]]))
                    coauthor_pairs[pair] += 1
        
        # Find main researcher
        author_counts = Counter(all_authors)
        self.main_researcher = author_counts.most_common(1)[0][0]
        
        # Top collaborators
        top_collaborators = []
        for author, count in author_counts.most_common(21)[1:]:  # Skip main researcher
            avg_citations = np.mean(author_citations[author])
            oa_count = author_oa_status[author]['oa']
            non_oa_count = author_oa_status[author]['non_oa']
            top_collaborators.append({
                'name': author,
                'papers': count,
                'avg_citations': avg_citations,
                'oa_count': oa_count,
                'non_oa_count': non_oa_count,
                'oa_rate': oa_count / (oa_count + non_oa_count) * 100
            })
        
        # Strong collaboration pairs
        strong_pairs = [(pair, count) for pair, count in coauthor_pairs.items() 
                       if count >= 5]
        strong_pairs.sort(key=lambda x: x[1], reverse=True)
        
        self.network_data = {
            'main_researcher': self.main_researcher,
            'total_authors': len(author_counts),
            'total_papers': len(self.df),
            'avg_authors_per_paper': len(all_authors) / len(self.df),
            'avg_papers_per_author': len(all_authors) / len(author_counts),
            'top_collaborators': top_collaborators,
            'strong_pairs': strong_pairs,
            'author_papers': dict(author_papers),
            'author_citations': dict(author_citations),
            'author_oa_status': dict(author_oa_status),
            'coauthor_pairs': dict(coauthor_pairs)
        }

        return self.network_data

    def build_institution_network(self):
        """
        Build inter-institution collaboration network from publications.

        Returns:
        --------
        dict
            Dictionary containing network data including:
            - Main institution
            - Institution statistics
            - Collaboration pairs between institutions
            - Top collaborating institutions
        """
        if self.df is None:
            return None

        all_institutions = []
        institution_papers = defaultdict(list)
        institution_pairs = defaultdict(int)
        institution_citations = defaultdict(list)
        institution_oa_status = defaultdict(lambda: {'oa': 0, 'non_oa': 0})

        # Build network
        for idx, row in self.df.iterrows():
            institutions = self.parse_institutions(row['authorships.institutions.display_name'])
            paper_id = row['id']
            citations = row['cited_by_count']
            is_oa = row['open_access.is_oa']

            all_institutions.extend(institutions)

            for institution in institutions:
                institution_papers[institution].append(paper_id)
                institution_citations[institution].append(citations)
                if is_oa:
                    institution_oa_status[institution]['oa'] += 1
                else:
                    institution_oa_status[institution]['non_oa'] += 1

            # Track inter-institution collaboration pairs
            for i in range(len(institutions)):
                for j in range(i + 1, len(institutions)):
                    pair = tuple(sorted([institutions[i], institutions[j]]))
                    institution_pairs[pair] += 1

        # Find main institution (most frequent)
        institution_counts = Counter(all_institutions)
        self.main_institution = institution_counts.most_common(1)[0][0]

        # Top collaborating institutions (excluding main institution)
        top_institutions = []
        for institution, count in institution_counts.most_common(21)[1:]:  # Skip main institution
            avg_citations = np.mean(institution_citations[institution])
            oa_count = institution_oa_status[institution]['oa']
            non_oa_count = institution_oa_status[institution]['non_oa']
            top_institutions.append({
                'name': institution,
                'papers': count,
                'avg_citations': avg_citations,
                'oa_count': oa_count,
                'non_oa_count': non_oa_count,
                'oa_rate': oa_count / (oa_count + non_oa_count) * 100 if (oa_count + non_oa_count) > 0 else 0
            })

        # Strong collaboration pairs between institutions
        strong_pairs = [(pair, count) for pair, count in institution_pairs.items()
                        if count >= 3]
        strong_pairs.sort(key=lambda x: x[1], reverse=True)

        self.network_data = {
            'main_institution': self.main_institution,
            'total_institutions': len(institution_counts),
            'total_papers': len(self.df),
            'avg_institutions_per_paper': len(all_institutions) / len(self.df) if len(self.df) > 0 else 0,
            'avg_papers_per_institution': len(all_institutions) / len(institution_counts) if len(institution_counts) > 0 else 0,
            'top_collaborators': top_institutions,
            'strong_pairs': strong_pairs,
            'institution_papers': dict(institution_papers),
            'institution_citations': dict(institution_citations),
            'institution_oa_status': dict(institution_oa_status),
            'institution_pairs': dict(institution_pairs)
        }

        return self.network_data

    def build_network(self):
        """
        Build the appropriate network based on analysis mode.

        Returns:
        --------
        dict
            Network data dictionary
        """
        if self.analysis_mode == 'institution':
            return self.build_institution_network()
        else:
            return self.build_coauthorship_network()
    
    def get_publication_type_analysis(self):
        """
        Analyze OA distribution by publication type.
        
        Returns:
        --------
        dict
            Dictionary with publication type analysis
        """
        if self.df is None:
            return None
        
        type_oa = pd.crosstab(self.df['type'], 
                             self.df['open_access.is_oa'], 
                             normalize='index') * 100
        type_counts = self.df['type'].value_counts()
        
        # Filter to significant types (adaptive threshold for small datasets)
        min_count = min(5, max(1, len(self.df) // 3))
        significant_types = type_counts[type_counts >= min_count].index
        type_oa_filtered = type_oa.loc[significant_types] if len(significant_types) > 0 else type_oa
        
        return {
            'type_oa_distribution': type_oa_filtered,
            'type_counts': type_counts
        }
    
    def get_collaboration_size_analysis(self):
        """
        Analyze OA rate by collaboration size.
        
        Returns:
        --------
        dict
            Dictionary with collaboration size analysis
        """
        if self.df is None:
            return None
        
        # Count authors per paper
        self.df['num_authors'] = self.df['authorships.author.display_name'].apply(
            lambda x: len(self.parse_authors(x))
        )
        
        # Bin by number of authors
        author_bins = [1, 3, 5, 8, 12, 50]
        author_labels = ['1-2', '3-4', '5-7', '8-11', '12+']
        self.df['author_bin'] = pd.cut(self.df['num_authors'], 
                                       bins=author_bins, 
                                       labels=author_labels, 
                                       include_lowest=True)
        
        oa_by_authors = self.df.groupby('author_bin', observed=True).agg({
            'open_access.is_oa': 'mean',
            'id': 'count'
        }).reset_index()
        oa_by_authors['oa_rate'] = oa_by_authors['open_access.is_oa'] * 100
        
        return {
            'oa_by_collaboration_size': oa_by_authors,
            'overall_oa_rate': self.df['open_access.is_oa'].mean() * 100
        }
    
    def generate_text_report(self):
        """
        Generate comprehensive text analysis report.

        Returns:
        --------
        str
            Formatted text report
        """
        if self.df is None:
            return "No data loaded"

        # Get all analyses
        oa_analysis = self.analyze_open_access()
        citation_analysis = self.analyze_citations()
        network_analysis = self.network_data if self.network_data else self.build_network()

        # Determine entity type labels based on analysis mode
        if self.analysis_mode == 'institution':
            entity_name = network_analysis.get('main_institution', 'Unknown Institution')
            network_title = "INTER-INSTITUTION COLLABORATION NETWORK ANALYSIS"
            entity_label = "Institution"
            collaborator_label = "Partner Institutions"
            total_entities_key = 'total_institutions'
            avg_per_paper_key = 'avg_institutions_per_paper'
            avg_per_paper_label = "institutions"
        else:
            entity_name = network_analysis.get('main_researcher', 'Unknown Researcher')
            network_title = "CO-AUTHORSHIP NETWORK ANALYSIS"
            entity_label = "Researcher"
            collaborator_label = "Collaborators"
            total_entities_key = 'total_authors'
            avg_per_paper_key = 'avg_authors_per_paper'
            avg_per_paper_label = "authors"

        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE ANALYSIS REPORT")
        report.append(f"Open Access Patterns, Citation Impact & {'Inter-Institution' if self.analysis_mode == 'institution' else 'Co-Authorship'} Networks")
        report.append(f"{entity_label}: {entity_name}")
        report.append(f"Analysis Mode: {self.analysis_mode.upper()}")
        report.append("=" * 80)
        report.append("")

        # Section 1: Open Access
        report.append("━" * 80)
        report.append("1. OPEN ACCESS PUBLICATION PATTERNS")
        report.append("━" * 80)
        report.append("")

        report.append(f"Total Publications: {oa_analysis['total_publications']}")
        report.append(f"  • Open Access: {oa_analysis['oa_count']} ({oa_analysis['oa_rate']:.1f}%)")
        report.append(f"  • Closed Access: {oa_analysis['non_oa_count']} ({100-oa_analysis['oa_rate']:.1f}%)")
        report.append("")

        report.append("Open Access Type Distribution:")
        for status, count in sorted(oa_analysis['oa_status_distribution'].items(),
                                    key=lambda x: x[1], reverse=True):
            pct = count / oa_analysis['total_publications'] * 100
            report.append(f"  • {status.capitalize():12s}: {count:3d} papers ({pct:5.1f}%)")
        report.append("")

        report.append("Temporal Trends:")
        report.append(f"  • OA Rate (≤2015): {oa_analysis['early_oa_rate']:.1f}%")
        report.append(f"  • OA Rate (≥2020): {oa_analysis['recent_oa_rate']:.1f}%")
        report.append(f"  • Change: {oa_analysis['recent_oa_rate'] - oa_analysis['early_oa_rate']:+.1f} percentage points")
        report.append("")

        report.append(f"Recent Performance (2022-2025):")
        report.append(f"  • {oa_analysis['recent_performance']['count']} publications")
        report.append(f"  • {oa_analysis['recent_performance']['oa_count']} Open Access ({oa_analysis['recent_performance']['oa_rate']:.1f}%)")
        report.append("")

        # Section 2: Citations
        report.append("━" * 80)
        report.append("2. CITATION IMPACT: OPEN ACCESS vs CLOSED ACCESS")
        report.append("━" * 80)
        report.append("")

        report.append("Overall Citation Statistics:")
        report.append("")
        report.append(f"{'Metric':<25s} {'Open Access':>15s} {'Closed Access':>15s} {'Difference':>15s}")
        report.append("-" * 75)

        oa_cit = citation_analysis['oa_citations']
        non_oa_cit = citation_analysis['non_oa_citations']

        report.append(f"{'Total Citations':<25s} {oa_cit['total']:>15,} {non_oa_cit['total']:>15,} {oa_cit['total'] - non_oa_cit['total']:>15,}")
        report.append(f"{'Mean Citations':<25s} {oa_cit['mean']:>15.2f} {non_oa_cit['mean']:>15.2f} {oa_cit['mean'] - non_oa_cit['mean']:>15.2f}")
        report.append(f"{'Median Citations':<25s} {oa_cit['median']:>15.1f} {non_oa_cit['median']:>15.1f} {oa_cit['median'] - non_oa_cit['median']:>15.1f}")
        report.append(f"{'Max Citations':<25s} {oa_cit['max']:>15,} {non_oa_cit['max']:>15,} {oa_cit['max'] - non_oa_cit['max']:>15,}")
        report.append("")

        report.append(f"Statistical Test (Mann-Whitney U):")
        report.append(f"  • p-value: {citation_analysis['statistical_test']['p_value']:.4f}")
        if citation_analysis['statistical_test']['significant']:
            report.append(f"  • Result: Statistically significant difference (p < 0.05)")
        else:
            report.append(f"  • Result: No statistically significant difference (p ≥ 0.05)")
        report.append("")

        # Section 3: Network
        report.append("━" * 80)
        report.append(f"3. {network_title}")
        report.append("━" * 80)
        report.append("")

        report.append(f"Network Overview:")
        report.append(f"  • Total unique {collaborator_label.lower()}: {network_analysis[total_entities_key]}")
        report.append(f"  • Average {avg_per_paper_label} per paper: {network_analysis[avg_per_paper_key]:.1f}")
        report.append(f"  • Publications span: {self.df['publication_year'].min()}-{self.df['publication_year'].max()}")
        report.append("")

        if network_analysis['top_collaborators']:
            top_collab = network_analysis['top_collaborators'][0]
            report.append(f"Primary {'Partner Institution' if self.analysis_mode == 'institution' else 'Collaboration Partner'}:")
            report.append(f"  • Name: {top_collab['name']}")
            report.append(f"  • Joint publications: {top_collab['papers']} ({top_collab['papers']/len(self.df)*100:.1f}% of total)")
            report.append(f"  • Average citations: {top_collab['avg_citations']:.1f}")
            report.append(f"  • OA rate: {top_collab['oa_rate']:.1f}%")
            report.append("")

            report.append(f"Top 10 {collaborator_label}:")
            report.append(f"{'Rank':<6s} {collaborator_label[:-1] if collaborator_label.endswith('s') else collaborator_label:<45s} {'Papers':>8s} {'Avg Cit':>10s} {'OA Rate':>10s}")
            report.append("-" * 80)
            for i, collab in enumerate(network_analysis['top_collaborators'][:10], 1):
                name_display = collab['name'][:44] if len(collab['name']) > 44 else collab['name']
                report.append(f"{i:<6d} {name_display:<45s} {collab['papers']:>8d} {collab['avg_citations']:>10.1f} {collab['oa_rate']:>9.1f}%")
            report.append("")

        # Section 4: Insights
        report.append("━" * 80)
        report.append("4. KEY INSIGHTS & RECOMMENDATIONS")
        report.append("━" * 80)
        report.append("")

        report.append("Strengths:")
        report.append(f"  ✓ Strong upward trend in OA adoption (from {oa_analysis['early_oa_rate']:.1f}% to {oa_analysis['recent_oa_rate']:.1f}%)")
        report.append(f"  ✓ Extensive collaboration network ({network_analysis[total_entities_key]} unique {collaborator_label.lower()})")
        report.append(f"  ✓ Consistent publication output ({len(self.df)} publications)")
        if network_analysis['top_collaborators']:
            top_collab = network_analysis['top_collaborators'][0]
            report.append(f"  ✓ Strong partnership with {top_collab['name']} ({top_collab['papers']} papers)")
        report.append("")

        report.append("Strategic Recommendations:")
        report.append("  1. Maintain strong OA trend from recent years")
        report.append("  2. Leverage extensive collaboration network for increased visibility")
        report.append("  3. Consider preprints/green OA for closed access venues")
        report.append("  4. Target high-impact OA journals to combine accessibility with citation impact")
        report.append("  5. Explore diamond OA opportunities")
        report.append("")

        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        return "\n".join(report)
    
    def export_data_for_visualization(self):
        """
        Export processed data for visualization.

        Returns:
        --------
        dict
            Dictionary containing all data needed for visualizations
        """
        return {
            'dataframe': self.df,
            'oa_analysis': self.analyze_open_access(),
            'citation_analysis': self.analyze_citations(),
            'network_data': self.network_data,
            'pub_type_analysis': self.get_publication_type_analysis(),
            'collab_size_analysis': self.get_collaboration_size_analysis(),
            'analysis_mode': self.analysis_mode
        }
