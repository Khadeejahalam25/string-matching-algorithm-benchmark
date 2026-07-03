# -*- coding: utf-8 -*-
"""
Intelligent CV Analyzer - String Matching Algorithms
Algorithms implemented from scratch: Brute Force, Rabin-Karp, KMP

Run locally: python cv_analyzer.py
(Originally developed and tested in Google Colab; adapted here to run as a
standalone script reading CVs from a local folder instead of a Colab upload.)
"""

# ============================================================================
# INTELLIGENT CV ANALYZER - String Matching Algorithms
# Algorithms: Brute Force, Rabin-Karp, KMP
# ============================================================================

# STEP 2: Import Libraries
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
from docx import Document
from collections import defaultdict
import re
import io

# ============================================================================
# ALGORITHM IMPLEMENTATIONS
# ============================================================================

class StringMatchingAlgorithms:
    """Implementation of three string matching algorithms"""

    @staticmethod
    def brute_force(text, pattern):
        """
        Brute Force Algorithm - O(n*m)
        Checks pattern at every position in text
        """
        text = text.lower()
        pattern = pattern.lower()
        positions = []
        comparisons = 0

        n = len(text)
        m = len(pattern)

        for i in range(n - m + 1):
            j = 0
            while j < m:
                comparisons += 1
                if text[i + j] != pattern[j]:
                    break
                j += 1

            if j == m:  # Pattern found
                positions.append(i)

        return {
            'positions': positions,
            'count': len(positions),
            'comparisons': comparisons
        }

    @staticmethod
    def rabin_karp(text, pattern):
        """
        Rabin-Karp Algorithm - O(n+m) average
        Uses rolling hash for pattern matching
        """
        text = text.lower()
        pattern = pattern.lower()
        positions = []
        comparisons = 0

        n = len(text)
        m = len(pattern)

        if m > n:
            return {'positions': [], 'count': 0, 'comparisons': 0}

        # Constants for hashing
        d = 256  # Number of characters
        q = 101  # Prime number

        # Calculate hash values
        pattern_hash = 0
        text_hash = 0
        h = 1

        # h = d^(m-1) % q
        for i in range(m - 1):
            h = (h * d) % q

        # Calculate initial hash values
        for i in range(m):
            pattern_hash = (d * pattern_hash + ord(pattern[i])) % q
            text_hash = (d * text_hash + ord(text[i])) % q

        # Slide pattern over text
        for i in range(n - m + 1):
            # Check if hash values match
            if pattern_hash == text_hash:
                # Verify character by character
                match = True
                for j in range(m):
                    comparisons += 1
                    if text[i + j] != pattern[j]:
                        match = False
                        break

                if match:
                    positions.append(i)

            # Calculate hash for next window
            if i < n - m:
                text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % q
                if text_hash < 0:
                    text_hash += q

        return {
            'positions': positions,
            'count': len(positions),
            'comparisons': comparisons
        }

    @staticmethod
    def kmp(text, pattern):
        """
        Knuth-Morris-Pratt Algorithm - O(n+m)
        Uses preprocessing to avoid redundant comparisons
        """
        text = text.lower()
        pattern = pattern.lower()
        positions = []
        comparisons = 0

        n = len(text)
        m = len(pattern)

        if m > n or m == 0:
            return {'positions': [], 'count': 0, 'comparisons': 0}

        # Compute LPS (Longest Proper Prefix which is also Suffix) array
        lps = [0] * m
        length = 0
        i = 1

        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1

        # Pattern matching
        i = 0  # index for text
        j = 0  # index for pattern

        while i < n:
            comparisons += 1
            if pattern[j] == text[i]:
                i += 1
                j += 1

            if j == m:
                positions.append(i - j)
                j = lps[j - 1]
            elif i < n and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1

        return {
            'positions': positions,
            'count': len(positions),
            'comparisons': comparisons
        }

# ============================================================================
# CV PARSER - Extract Text from PDF and DOCX
# ============================================================================

class CVParser:
    """Extract text from PDF and DOCX files"""

    @staticmethod
    def extract_from_pdf(file_path):
        """Extract text from PDF file"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {str(e)}")
            return ""

    @staticmethod
    def extract_from_docx(file_path):
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {str(e)}")
            return ""

    @staticmethod
    def extract_text(file_path):
        """Extract text based on file extension"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == '.pdf':
            return CVParser.extract_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return CVParser.extract_from_docx(file_path)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            return ""

# ============================================================================
# JOB DESCRIPTIONS
# ============================================================================

JOB_DESCRIPTIONS = {
    'Data Scientist': {
        'mandatory': ['Python', 'Machine Learning', 'SQL', 'Statistics', 'Data Analysis'],
        'optional': ['TensorFlow', 'PyTorch', 'R', 'Tableau', 'Deep Learning',
                    'NLP', 'Pandas', 'NumPy', 'Scikit-learn', 'Spark', 'Hadoop']
    },
    'Web Developer': {
        'mandatory': ['HTML', 'CSS', 'JavaScript', 'React', 'Node.js'],
        'optional': ['TypeScript', 'Vue', 'Angular', 'MongoDB', 'Express',
                    'REST API', 'Git', 'Webpack', 'Redux', 'GraphQL', 'Docker']
    },
    'Software Tester': {
        'mandatory': ['Testing', 'QA', 'Selenium', 'Test Cases', 'Bug Tracking'],
        'optional': ['Automation', 'JIRA', 'Postman', 'API Testing',
                    'Performance Testing', 'JUnit', 'TestNG', 'Cucumber', 'Jenkins']
    }
}

# ============================================================================
# CV ANALYZER
# ============================================================================

class CVAnalyzer:
    """Main CV Analysis Engine"""

    def __init__(self):
        self.algorithms = StringMatchingAlgorithms()
        self.parser = CVParser()
        self.results = []

    def analyze_cv(self, cv_text, job_title, algorithm='all'):
        """
        Analyze a single CV against job requirements

        Args:
            cv_text: CV text content
            job_title: Job title from JOB_DESCRIPTIONS
            algorithm: 'brute_force', 'rabin_karp', 'kmp', or 'all'
        """
        if job_title not in JOB_DESCRIPTIONS:
            raise ValueError(f"Unknown job title: {job_title}")

        job = JOB_DESCRIPTIONS[job_title]
        all_keywords = job['mandatory'] + job['optional']

        results = {
            'job_title': job_title,
            'mandatory_matched': [],
            'mandatory_missing': [],
            'optional_matched': [],
            'optional_missing': [],
            'algorithm_performance': {}
        }

        # Run algorithms
        algorithms_to_run = []
        if algorithm == 'all':
            algorithms_to_run = ['brute_force', 'rabin_karp', 'kmp']
        else:
            algorithms_to_run = [algorithm]

        for algo_name in algorithms_to_run:
            start_time = time.time()
            total_comparisons = 0
            keyword_matches = {}

            for keyword in all_keywords:
                if algo_name == 'brute_force':
                    result = self.algorithms.brute_force(cv_text, keyword)
                elif algo_name == 'rabin_karp':
                    result = self.algorithms.rabin_karp(cv_text, keyword)
                elif algo_name == 'kmp':
                    result = self.algorithms.kmp(cv_text, keyword)

                keyword_matches[keyword] = result['count']
                total_comparisons += result['comparisons']

            execution_time = (time.time() - start_time) * 1000  # Convert to ms

            results['algorithm_performance'][algo_name] = {
                'execution_time_ms': execution_time,
                'total_comparisons': total_comparisons,
                'keyword_matches': keyword_matches
            }

        # Use first algorithm results for matching
        first_algo = algorithms_to_run[0]
        keyword_matches = results['algorithm_performance'][first_algo]['keyword_matches']

        # Categorize matches
        for skill in job['mandatory']:
            if keyword_matches.get(skill, 0) > 0:
                results['mandatory_matched'].append(skill)
            else:
                results['mandatory_missing'].append(skill)

        for skill in job['optional']:
            if keyword_matches.get(skill, 0) > 0:
                results['optional_matched'].append(skill)
            else:
                results['optional_missing'].append(skill)

        # Calculate scores
        mandatory_score = (len(results['mandatory_matched']) / len(job['mandatory'])) * 100
        optional_score = (len(results['optional_matched']) / len(job['optional'])) * 100
        overall_score = (mandatory_score * 0.7) + (optional_score * 0.3)

        results['mandatory_score'] = mandatory_score
        results['optional_score'] = optional_score
        results['overall_score'] = overall_score

        return results

    def analyze_multiple_cvs(self, cv_folder, job_title, algorithm='all'):
        """
        Analyze multiple CVs from a folder

        Args:
            cv_folder: Path to folder containing CV files
            job_title: Job title to match against
            algorithm: Algorithm to use
        """
        results = []

        # Get all CV files
        cv_files = []
        for root, dirs, files in os.walk(cv_folder):
            for file in files:
                if file.endswith(('.pdf', '.docx', '.doc', '.txt')):
                    cv_files.append(os.path.join(root, file))

        print(f"Found {len(cv_files)} CV files")
        print("Starting analysis...\n")

        for idx, cv_path in enumerate(cv_files, 1):
            print(f"Processing [{idx}/{len(cv_files)}]: {os.path.basename(cv_path)}")

            # Extract text
            cv_text = self.parser.extract_text(cv_path)

            if not cv_text.strip():
                print(f" Warning: Empty or unreadable file")
                continue

            # Analyze
            result = self.analyze_cv(cv_text, job_title, algorithm)
            result['filename'] = os.path.basename(cv_path)
            result['cv_text_length'] = len(cv_text)
            results.append(result)

            print(f"  ✓ Score: {result['overall_score']:.1f}%")

        # Sort by overall score
        results.sort(key=lambda x: x['overall_score'], reverse=True)

        return results

# ============================================================================
# VISUALIZATION & REPORTING
# ============================================================================

class ReportGenerator:
    """Generate reports and visualizations"""

    @staticmethod
    def generate_summary_report(results):
        """Generate summary statistics"""
        if not results:
            print("No results to report")
            return

        print("\n" + "="*80)
        print("CV ANALYSIS SUMMARY REPORT")
        print("="*80)
        print(f"Total CVs Analyzed: {len(results)}")
        print(f"Job Position: {results[0]['job_title']}")
        print(f"Average Score: {sum(r['overall_score'] for r in results)/len(results):.2f}%")
        print("\n" + "-"*80)
        print("TOP 10 CANDIDATES:")
        print("-"*80)

        for idx, result in enumerate(results[:10], 1):
            print(f"{idx:2d}. {result['filename'][:40]:40s} | Score: {result['overall_score']:5.1f}% | "
                  f"Mandatory: {result['mandatory_score']:5.1f}% | Optional: {result['optional_score']:5.1f}%")

    @staticmethod
    def plot_performance_comparison(results):
        """Plot algorithm performance comparison"""
        if not results:
            return

        # Aggregate performance data
        algo_times = defaultdict(list)
        algo_comparisons = defaultdict(list)

        for result in results:
            for algo_name, perf in result['algorithm_performance'].items():
                algo_times[algo_name].append(perf['execution_time_ms'])
                algo_comparisons[algo_name].append(perf['total_comparisons'])

        # Calculate averages
        avg_times = {algo: sum(times)/len(times) for algo, times in algo_times.items()}
        avg_comparisons = {algo: sum(comps)/len(comps) for algo, comps in algo_comparisons.items()}

        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Execution time
        algorithms = list(avg_times.keys())
        times = list(avg_times.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        axes[0].bar(algorithms, times, color=colors)
        axes[0].set_title('Average Execution Time Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Time (milliseconds)', fontsize=12)
        axes[0].set_xlabel('Algorithm', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)

        # Comparisons
        comparisons = list(avg_comparisons.values())
        axes[1].bar(algorithms, comparisons, color=colors)
        axes[1].set_title('Average Comparisons Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Number of Comparisons', fontsize=12)
        axes[1].set_xlabel('Algorithm', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('algorithm_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n📊 Performance Comparison Table:")
        print("-"*60)
        print(f"{'Algorithm':<20} {'Avg Time (ms)':<20} {'Avg Comparisons':<20}")
        print("-"*60)
        for algo in algorithms:
            print(f"{algo:<20} {avg_times[algo]:<20.4f} {avg_comparisons[algo]:<20.0f}")
        print("-"*60)

    @staticmethod
    def plot_score_distribution(results):
        """Plot score distribution"""
        if not results:
            return

        scores = [r['overall_score'] for r in results]

        plt.figure(figsize=(12, 6))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(scores, bins=20, color='#4ECDC4', edgecolor='black', alpha=0.7)
        plt.title('Score Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Overall Score (%)', fontsize=12)
        plt.ylabel('Number of Candidates', fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(scores, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#45B7D1', alpha=0.7))
        plt.title('Score Statistics', fontsize=14, fontweight='bold')
        plt.ylabel('Overall Score (%)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('score_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def export_to_excel(results, filename='cv_analysis_results.xlsx'):
        """Export results to Excel"""
        if not results:
            return

        # Prepare data for DataFrame
        data = []
        for result in results:
            row = {
                'Filename': result['filename'],
                'Overall Score': result['overall_score'],
                'Mandatory Score': result['mandatory_score'],
                'Optional Score': result['optional_score'],
                'Mandatory Matched': ', '.join(result['mandatory_matched']),
                'Mandatory Missing': ', '.join(result['mandatory_missing']),
                'Optional Matched': ', '.join(result['optional_matched']),
                'CV Length': result['cv_text_length']
            }

            # Add algorithm performance
            for algo_name, perf in result['algorithm_performance'].items():
                row[f'{algo_name}_time_ms'] = perf['execution_time_ms']
                row[f'{algo_name}_comparisons'] = perf['total_comparisons']

            data.append(row)

        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        print(f"\n Results exported to {filename}")

# ============================================================================
# MAIN EXECUTION CODE
# ============================================================================

def main():
    print("="*80)
    print("INTELLIGENT CV ANALYZER - STRING MATCHING ALGORITHMS")
    print("="*80)

    # Folder containing CVs to analyze (PDF, DOCX, or TXT)
    cv_folder = "sample_cvs"

    # Select job description
    print("\n STEP 1: Select Job Description")
    print("-"*80)
    for idx, job_title in enumerate(JOB_DESCRIPTIONS.keys(), 1):
        print(f"{idx}. {job_title}")

    job_choice = int(input("\nEnter your choice (1-3): ") or "1")
    job_titles = list(JOB_DESCRIPTIONS.keys())
    selected_job = job_titles[job_choice - 1]

    print(f"\n Selected: {selected_job}")

    # Display job requirements
    job = JOB_DESCRIPTIONS[selected_job]
    print(f"\nMandatory Skills: {', '.join(job['mandatory'])}")
    print(f"Optional Skills: {', '.join(job['optional'][:5])} (+{len(job['optional'])-5} more)")

    # Select algorithm
    print("\n STEP 2: Select Algorithm")
    print("-"*80)
    print("1. All Algorithms (Brute Force + Rabin-Karp + KMP)")
    print("2. Brute Force only")
    print("3. Rabin-Karp only")
    print("4. KMP only")

    algo_choice = int(input("\nEnter your choice (1-4): ") or "1")
    algorithms = ['all', 'brute_force', 'rabin_karp', 'kmp']
    selected_algorithm = algorithms[algo_choice - 1]

    # Analyze CVs
    print("\n STEP 3: Analyzing CVs...")
    print("="*80)

    analyzer = CVAnalyzer()
    results = analyzer.analyze_multiple_cvs(cv_folder, selected_job, selected_algorithm)

    # Generate reports
    print("\n\n STEP 4: Generating Reports...")
    print("="*80)

    reporter = ReportGenerator()
    reporter.generate_summary_report(results)

    if selected_algorithm == 'all':
        reporter.plot_performance_comparison(results)

    reporter.plot_score_distribution(results)
    reporter.export_to_excel(results)

    print("\n ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("   algorithm_performance_comparison.png")
    print("   score_distribution.png")
    print("   cv_analysis_results.xlsx")


# ============================================================================
# RUN THE PROGRAM
# ============================================================================

if __name__ == "__main__":
    main()