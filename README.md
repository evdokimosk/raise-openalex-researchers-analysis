# OpenAlex Dataset Analysis Tool

A comprehensive Python-based analysis system for OpenAlex publication datasets, focusing on Open Access patterns, citation impact, and co-authorship networks.

## Overview

This tool provides automated analysis and visualization of academic publication data from OpenAlex, generating:
- 3 comprehensive visualization figures (high-resolution PNG)
- Detailed analysis reports (TXT and Markdown formats)
- Figure explanations guide (Markdown)

## Files

### Core Files
1. **`openalex_dataset_parser.py`** - Main class-based module containing:
   - `OpenAlexDatasetParser` class with all analysis methods
   - Data loading and parsing functionality
   - Statistical analysis methods
   - Network analysis algorithms

2. **`main.py`** - Main execution script that:
   - Instantiates the parser class
   - Calls analysis methods
   - Generates visualizations
   - Exports reports

### Output Files (Generated)
- `oa_coauthor_analysis.png` - Main dashboard (6 panels)
- `network_visualization.png` - Network graphs (2 panels)
- `detailed_analysis.png` - Statistical deep dive (4 panels)
- `analysis_report.txt` - Comprehensive text report
- `analysis_report.md` - Markdown version of report
- `figure_explanations.md` - Detailed guide to all figures

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy networkx
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
networkx>=2.6.0
```

## Installation

1. **Clone or download the files:**
```bash
# Download the two Python files
- openalex_dataset_parser.py
- main.py
```

2. **Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scipy networkx
```

3. **Prepare your data:**
- Ensure you have an OpenAlex CSV export file
- File should contain columns like: `id`, `authorships.author.display_name`, `cited_by_count`, `open_access.is_oa`, etc.

## Usage

### Basic Usage
```bash
python main.py <path_to_your_csv_file>
```

### Example
```bash
python main.py works-csv-bpAxbEx4Z4gFZHsgv64jtd.csv
```

### Output
The script will:
1. Load and parse the dataset
2. Perform comprehensive analyses
3. Generate 6 output files in the current directory

### Expected Console Output
```
================================================================================
OpenAlex Dataset Analysis System
================================================================================

Step 1: Loading and parsing data...
✓ Loaded 162 publications from works-csv-data.csv

Step 2: Performing analyses...
  - Analyzing Open Access patterns...
  - Analyzing citation impact...
  - Building co-authorship network...
  - Analyzing publication types...
  - Analyzing collaboration sizes...

Step 3: Generating reports...
  ✓ Saved: analysis_report.txt
  ✓ Saved: analysis_report.md
  ✓ Saved: figure_explanations.md

Step 4: Generating visualizations...
  ✓ Saved: oa_coauthor_analysis.png
  ✓ Saved: network_visualization.png
  ✓ Saved: detailed_analysis.png

================================================================================
Analysis Complete!
================================================================================
```

## Module Structure

### OpenAlexDatasetParser Class

The main class provides the following methods:

#### Data Loading
- `__init__(filename)` - Initialize with CSV file path
- `load_data()` - Load dataset from CSV
- `get_basic_stats()` - Get dataset statistics

#### Analysis Methods
- `analyze_open_access()` - Analyze OA patterns and trends
- `analyze_citations()` - Analyze citation impact by OA status
- `build_coauthorship_network()` - Build collaboration network
- `get_publication_type_analysis()` - Analyze OA by publication type
- `get_collaboration_size_analysis()` - Analyze OA by team size

#### Reporting
- `generate_text_report()` - Generate comprehensive text report
- `export_data_for_visualization()` - Export data for plotting

### VisualizationGenerator Class

Handles all figure generation:
- `create_main_oa_coauthor_figure()` - Main 6-panel dashboard
- `create_network_visualization()` - Network graphs
- `create_detailed_analysis()` - Statistical analysis panels

## Customization

### Modify Figure Output Paths
Edit the function calls in `main.py`:
```python
viz_gen.create_main_oa_coauthor_figure('custom_path/figure1.png')
viz_gen.create_network_visualization('custom_path/figure2.png')
viz_gen.create_detailed_analysis('custom_path/figure3.png')
```

### Adjust Figure Resolution
Change DPI in visualization methods (default is 300):
```python
plt.savefig(output_path, dpi=600, bbox_inches='tight')  # Higher resolution
```

### Modify Collaboration Network Filters
In `create_network_visualization()`, adjust minimum collaborations:
```python
if len(citations) >= 5:  # Changed from 3 to 5
```

### Change Top Collaborators Count
In parser methods:
```python
top_collaborators = author_counts.most_common(31)[1:]  # Show top 30 instead of 20
```

## Data Requirements

### Required CSV Columns
The input CSV must contain these columns:
- `id` - Unique publication identifier
- `authorships.author.display_name` - Pipe-separated author names
- `cited_by_count` - Number of citations
- `open_access.is_oa` - Boolean OA status
- `open_access.oa_status` - Detailed OA type (gold, green, etc.)
- `publication_year` - Year of publication
- `type` - Publication type (article, book-chapter, etc.)

### Optional Columns
- `authorships.author.id` - Author identifiers
- `authorships.author.orcid` - ORCID IDs
- `authorships.institutions.display_name` - Institution names
- `doi` - Digital Object Identifier
- `fwci` - Field-Weighted Citation Impact

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'pandas'`
**Solution:** Install dependencies: `pip install pandas numpy matplotlib seaborn scipy networkx`

**Issue:** `FileNotFoundError: [Errno 2] No such file or directory`
**Solution:** Ensure CSV file path is correct and file exists

**Issue:** Network graph is empty
**Solution:** Check if dataset has sufficient collaborations (need 3+ joint papers)

**Issue:** Figures look compressed
**Solution:** Increase figure size in visualization methods:
```python
fig = plt.figure(figsize=(24, 14))  # Larger figure
```

## Advanced Usage

### Using the Parser Independently
```python
from openalex_dataset_parser import OpenAlexDatasetParser

# Initialize
parser = OpenAlexDatasetParser('data.csv')
parser.load_data()

# Run specific analyses
oa_stats = parser.analyze_open_access()
citation_stats = parser.analyze_citations()
network = parser.build_coauthorship_network()

# Access results
print(f"OA Rate: {oa_stats['oa_rate']:.1f}%")
print(f"Main researcher: {network['main_researcher']}")
```

### Custom Visualizations
```python
from openalex_dataset_parser import OpenAlexDatasetParser

parser = OpenAlexDatasetParser('data.csv')
parser.load_data()

# Get data
data = parser.export_data_for_visualization()

# Create custom plots
import matplotlib.pyplot as plt
df = data['dataframe']
df.groupby('publication_year')['cited_by_count'].sum().plot(kind='bar')
plt.title('Total Citations by Year')
plt.savefig('custom_plot.png')
```

## Performance Notes

- **Dataset size:** Tested with up to 500 publications
- **Processing time:** ~5-10 seconds for 200 publications
- **Memory usage:** ~50-100 MB for typical datasets
- **Network analysis:** Quadratic complexity O(n²) for co-authorship pairs

## Citation

If you use this tool in your research, please cite:
```
OpenAlex Dataset Analysis Tool (2025)
GitHub: [repository URL]
```

## License

MIT License - Free to use, modify, and distribute

## Owner

**Evdokimos Konstantinidis**
Contact: info@raise-science.eu

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: info@raise-science.eu

## Changelog

### Version 1.0 (2025-01-21)
- Initial release
- Three main visualization figures
- Comprehensive analysis reports
- Figure explanations guide
- Class-based architecture

## Acknowledgments

- Built for OpenAlex data format
- Uses matplotlib, seaborn, networkx for visualizations
- Statistical analysis with scipy
- Data processing with pandas

---

**Happy Analyzing! 📊🔬**
