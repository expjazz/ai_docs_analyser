# Interview Analysis Tool

Automatically analyzes interview transcripts using OpenAI API and exports categorized insights to Excel.

## Features

- 📂 Processes multiple interview files from `entrevistas/` folder
- 🤖 Uses OpenAI GPT-4 for intelligent analysis
- 📊 Exports results to Excel with interviews as rows, categories as columns
- 📝 Supports multiple file formats (.txt, .md, .doc, .docx)
- 🔍 Analyzes 10 key categories automatically
- 📋 Detailed logging and error handling
- 🐍 Self-contained virtual environment with pipenv

## Prerequisites

- Python 3.10+
- pipenv (install with `pip install pipenv`)

## Setup

1. **Install pipenv** (if not already installed):

   ```bash
   pip install pipenv
   ```

2. **Install dependencies and create virtual environment:**

   ```bash
   pipenv install
   ```

3. **Install development dependencies** (optional):

   ```bash
   pipenv install --dev
   ```

4. **Set up OpenAI API key:**

   Create a `.env` file in the project root:

   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

   Or export it in your shell:

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

5. **Add interview files:**
   - Place interview transcripts in the `entrevistas/` folder
   - Supported formats: .txt, .md, .doc, .docx

## Usage

Run the analysis in the virtual environment:

```bash
pipenv run python app/main.py
```

Or activate the shell and run directly:

```bash
pipenv shell
python app/main.py
```

The tool will:

1. Scan `entrevistas/` folder for interview files
2. Analyze each interview using OpenAI API
3. Export results to `interview_analysis.xlsx`

## Development

The project includes development tools:

- **Code formatting**: `pipenv run black app/`
- **Linting**: `pipenv run flake8 app/`
- **Testing**: `pipenv run pytest` (when tests are added)

## Analysis Categories

The tool analyzes interviews across these categories:

- **Technical Skills**: Programming languages, frameworks, tools
- **Soft Skills**: Communication, teamwork, leadership
- **Experience Level**: Years of experience, seniority
- **Problem Solving**: Troubleshooting, debugging approaches
- **Team Collaboration**: Teamwork, mentoring experience
- **Learning Attitude**: Adaptability, curiosity
- **Project Examples**: Specific projects and achievements
- **Cultural Fit**: Values alignment, work style
- **Questions Asked**: Quality of candidate questions
- **Overall Impression**: General assessment

## Output

Results are exported to `interview_analysis.xlsx` with:

- Each row representing one interview
- Each column representing one analysis category
- Metadata columns (filename, file size)
- Auto-adjusted column widths for readability

## Logging

The tool creates `interview_analysis.log` with detailed processing information.

## Error Handling

- Gracefully handles missing files or folders
- Fallback for non-JSON OpenAI responses
- Multiple encoding support for different file types
- Comprehensive error logging

## Virtual Environment Benefits

Using pipenv provides:

- 🔒 Isolated dependencies
- 📦 Reproducible builds with Pipfile.lock
- 🛠️ Development tools included
- 🐍 Python version management
