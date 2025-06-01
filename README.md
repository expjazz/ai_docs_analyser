# PROADI-SUS Interview Analysis Tool

Automatically analyzes PROADI-SUS healthcare interviews using OpenAI API and exports categorized insights to Excel.

## Features

- 📂 Processes multiple interview files from `entrevistas/` folder
- 🤖 Uses OpenAI GPT-4 for intelligent analysis
- 📊 Exports results to Excel with interviews as rows, categories as columns
- 📝 Supports multiple file formats (.txt, .md, .doc, .docx)
- 🔍 Analyzes 24 PROADI-SUS specific categories automatically
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

The tool analyzes PROADI-SUS interviews across these categories:

- **Código Entrevista**: Identificador único da entrevista (HE + número sequencial)
- **Área de Atuação**: Eixo temático (Pesquisa, Capacitação, Avaliação, Gestão)
- **Hospital**: Nome completo do hospital de excelência responsável
- **Nome - Posição - Projetos**: Informações do entrevistado e projetos PROADI-SUS
- **Modelos para Planos de Trabalho**: Estruturação de documentos e prestação de contas
- **Avaliação Geral PROADI**: Percepções sobre impacto no hospital e SUS
- **Relação CONASS/CONASEMS/MS**: Articulação entre entidades federativas
- **Benefícios para Instituição Parceira**: Vantagens percebidas pelos executores
- **Desafios para Participação HE**: Obstáculos internos e externos enfrentados
- **Sugestões**: Recomendações para aprimorar o programa
- **Origem dos Projetos**: Gênese e tramitação dos projetos
- **Projetos Colaborativos**: Iniciativas multi-institucionais
- **Expertise do Hospital**: Competência técnica e alinhamento institucional
- **Abrangência Territorial**: Alcance geográfico e critérios de seleção
- **Seleção de Instituições**: Critérios e estratégias de engajamento
- **Avaliações sobre o Projeto**: Resultados e lições aprendidas
- **Monitoramento e Indicadores**: Métodos de acompanhamento
- **Riscos e Dificuldades**: Problemas práticos observados
- **Benefícios para o SUS**: Ganhos esperados ou percebidos
- **Incorporação de Bens Materiais**: Equipamentos/insumos doados ao SUS
- **Treinamento para Profissionais**: Estratégias de capacitação
- **Publicações ou Divulgação**: Artigos e comunicação de resultados
- **Incorporação de Resultados ao SUS**: Integração às rotinas do SUS
- **Longevidade e Sustentabilidade**: Continuidade pós-financiamento

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
