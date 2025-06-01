#!/usr/bin/env python3
"""
Interview Analysis Tool
Processes interviews from the entrevistas folder using OpenAI API
and exports categorized analysis to Excel.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('interview_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class InterviewAnalyzer:
    """Analyzes interviews using OpenAI API and categorizes content."""

    def __init__(self):
        """Initialize the analyzer with OpenAI client and categories."""
        self.client = self._setup_openai_client()
        self.categories = self._define_categories()
        self.interviews_folder = Path("entrevistas")
        self.output_file = "interview_analysis.xlsx"

    def _setup_openai_client(self) -> OpenAI:
        """Setup OpenAI client with API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        return OpenAI(api_key=api_key)

    def _define_categories(self) -> Dict[str, str]:
        """Define categories and their descriptions for analysis."""
        return {
            "technical_skills": "Technical skills, programming languages, frameworks, tools mentioned",
            "soft_skills": "Communication, teamwork, leadership, problem-solving abilities",
            "experience_level": "Years of experience, seniority level, career progression",
            "problem_solving": "Examples of problem-solving approach, troubleshooting, debugging",
            "team_collaboration": "Experience working in teams, mentoring, cross-functional work",
            "learning_attitude": "Willingness to learn, adaptability, curiosity about new technologies",
            "project_examples": "Specific projects mentioned, achievements, impact",
            "cultural_fit": "Values alignment, work style, motivation factors",
            "questions_asked": "Quality and relevance of questions asked by the candidate",
            "overall_impression": "General assessment, strengths, areas for improvement"
        }

    def _get_interview_files(self) -> List[Path]:
        """Get all interview files from the entrevistas folder."""
        if not self.interviews_folder.exists():
            logger.warning(
                f"Interviews folder '{self.interviews_folder}' not found.")
            return []

        # Support multiple file formats
        supported_extensions = ['.txt', '.md', '.doc', '.docx']
        interview_files = []

        for ext in supported_extensions:
            interview_files.extend(self.interviews_folder.glob(f"*{ext}"))

        logger.info(f"Found {len(interview_files)} interview files")
        return sorted(interview_files)

    def _read_interview_content(self, file_path: Path) -> str:
        """Read interview content from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            logger.info(
                f"Read {len(content)} characters from {file_path.name}")
            return content
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read().strip()
            logger.warning(f"Used latin-1 encoding for {file_path.name}")
            return content
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""

    def _create_analysis_prompt(self, interview_content: str) -> str:
        """Create the prompt for OpenAI analysis."""
        categories_text = "\n".join([
            f"- {category}: {description}"
            for category, description in self.categories.items()
        ])

        prompt = f"""
Analyze the following interview transcript and provide insights for each category.
For each category, provide a brief but specific assessment based on the interview content.
If information is not available for a category, respond with "Not mentioned" or "Insufficient information".

Categories to analyze:
{categories_text}

Interview transcript:
{interview_content}

Please respond in JSON format with each category as a key and your analysis as the value.
Keep responses concise but informative (1-3 sentences per category).
"""
        return prompt

    def _analyze_interview_with_openai(self, interview_content: str, filename: str) -> Dict[str, str]:
        """Analyze interview content using OpenAI API."""
        try:
            prompt = self._create_analysis_prompt(interview_content)

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert HR analyst specializing in technical interviews. Provide objective, detailed analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            analysis_text = response.choices[0].message.content.strip()

            # Try to parse as JSON
            try:
                analysis_dict = json.loads(analysis_text)
                logger.info(f"Successfully analyzed {filename}")
                return analysis_dict
            except json.JSONDecodeError:
                # If not valid JSON, create a fallback structure
                logger.warning(
                    f"OpenAI response for {filename} was not valid JSON, using fallback")
                return {"general_analysis": analysis_text}

        except Exception as e:
            logger.error(f"Error analyzing {filename} with OpenAI: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _process_all_interviews(self) -> List[Dict[str, str]]:
        """Process all interviews and return analysis results."""
        interview_files = self._get_interview_files()

        if not interview_files:
            logger.warning("No interview files found to process")
            return []

        results = []

        for file_path in interview_files:
            logger.info(f"Processing {file_path.name}...")

            # Read interview content
            content = self._read_interview_content(file_path)
            if not content:
                logger.warning(f"Skipping {file_path.name} - no content")
                continue

            # Analyze with OpenAI
            analysis = self._analyze_interview_with_openai(
                content, file_path.name)

            # Add metadata
            result = {
                "interview_file": file_path.name,
                "file_size_chars": len(content),
                **analysis
            }

            results.append(result)

        logger.info(f"Processed {len(results)} interviews successfully")
        return results

    def _export_to_excel(self, results: List[Dict[str, str]]) -> None:
        """Export analysis results to Excel file."""
        if not results:
            logger.warning("No results to export")
            return

        try:
            # Create DataFrame
            df = pd.DataFrame(results)

            # Reorder columns to put metadata first
            metadata_cols = ["interview_file", "file_size_chars"]
            category_cols = [
                col for col in df.columns if col not in metadata_cols]
            df = df[metadata_cols + sorted(category_cols)]

            # Export to Excel
            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                df.to_excel(writer, index=False,
                            sheet_name='Interview Analysis')

                # Auto-adjust column widths
                worksheet = writer.sheets['Interview Analysis']
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            logger.info(f"Results exported to {self.output_file}")
            logger.info(
                f"Analyzed {len(results)} interviews across {len(self.categories)} categories")

        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            raise

    def run_analysis(self) -> None:
        """Run the complete interview analysis process."""
        logger.info("Starting interview analysis...")

        try:
            # Process all interviews
            results = self._process_all_interviews()

            if not results:
                logger.error("No interviews were processed successfully")
                return

            # Export results
            self._export_to_excel(results)

            logger.info("Interview analysis completed successfully!")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def main():
    """Main function to run the interview analysis."""
    print("🎯 Interview Analysis Tool")
    print("=" * 50)

    try:
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY environment variable not set")
            print("Please set your OpenAI API key:")
            print("export OPENAI_API_KEY='your-api-key-here'")
            print("Or create a .env file with: OPENAI_API_KEY=your-api-key-here")
            sys.exit(1)

        # Initialize and run analyzer
        analyzer = InterviewAnalyzer()
        analyzer.run_analysis()

        print(f"✅ Analysis complete! Check {analyzer.output_file}")

    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
