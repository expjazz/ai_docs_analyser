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
        self.output_file = "Entrevistas Hospitais_Rosi.xlsx"
        self.existing_df = self._load_existing_data()

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
            "codigo_entrevista": "Identificador único de cada entrevista, composto pela sigla do hospital de excelência (HE) + um número sequencial (ex.: HEBPP01, HIAE01). Serve para rastrear e referenciar rapidamente cada registro.",
            "area_atuacao": "Indica o eixo temático principal do projeto ao qual a entrevista se refere. Pode receber os nomes Pesquisa, Capacitação, Avaliação e Gestão.",
            "hospital": "Nome completo do hospital de excelência responsável pela entrevista (ex.: Beneficência Portuguesa, Hospital Israelita Albert Einstein).",
            "nome_posicao_projetos": "Campo narrativo onde o entrevistado informa seu nome, cargo/função e o(s) projeto(s) em que atua dentro do PROADI-SUS.",
            "modelos_planos_trabalho": "Relato sobre como o hospital estrutura modelos/documentos para planos de trabalho e para a prestação de contas ao MS, descrevendo processos formais e padronização.",
            "avaliacao_geral_proadi": "Percepções amplas sobre o impacto do PROADI-SUS no hospital e no SUS, incluindo ganhos institucionais, alinhamento estratégico e aprendizado organizacional, com comentários que vão de elogios à governança a críticas sobre burocracias.",
            "relacao_conass_conasems_ms": "Observações sobre a articulação entre Ministério da Saúde, CONASS, CONASEMS, hospitais de excelência e parceiros locais, destacando acordos de cooperação, fluxos de decisão e desafios de alinhamento federativo.",
            "beneficios_instituicao_parceira": "Vantagens percebidas pelas entidades executoras ou beneficiárias diretas do projeto, como melhorias em instalações, aquisição de conhecimento e aumento de visibilidade.",
            "desafios_participacao_he": "Obstáculos internos e externos enfrentados pelos HEs — trâmites de plano de trabalho, infraestrutura limitada em centros participantes, mudanças na gestão pública, entre outros.",
            "sugestoes": "Recomendações para aprimorar o programa, como reduzir burocracia, ampliar prazos ou adotar frameworks de gestão de projetos.",
            "origem_projetos": "Narrativa sobre a gênese do projeto, o demandante (MS, hospital, sociedade) e o percurso burocrático até a aprovação.",
            "projetos_colaborativos": "Detalhamento de iniciativas em que mais de um HE ou parceiro participa, descrevendo divisão de tarefas, sinergias e possíveis conflitos.",
            "expertise_hospital": "Justificativa da competência técnica do hospital para liderar o projeto e como o tema se alinha à sua missão institucional.",
            "abrangencia_territorial": "Definição do alcance geográfico (municipal, estadual, nacional) e critérios para escolha dos locais-alvo.",
            "selecao_instituicoes_participantes": "Critérios utilizados para convidar unidades de saúde ou municípios e estratégias de engajamento (por exemplo, instrumentos de maturidade e visitas técnicas).",
            "avaliacoes_projeto": "Resultados ou julgamentos preliminares sobre desempenho, impacto e lições aprendidas, incluindo indicadores clínicos quando disponíveis.",
            "monitoramento_indicadores": "Métodos de acompanhamento do projeto, abrangendo indicadores, ferramentas de gestão, rotinas de reporte e visitas a campo.",
            "riscos_dificuldades": "Problemas práticos já observados, como rotatividade de gestores públicos, falta de maturidade dos centros, custos logísticos e adaptação de protocolos.",
            "beneficios_sus": "Ganhos esperados ou já percebidos para a rede pública, como ampliação de acesso, formação de serviço de referência e economia de recursos.",
            "incorporacao_bens_materiais": "Indica se houve compra ou doação de equipamentos ou insumos permanentes ao SUS.",
            "treinamento_profissionais": "Descrição das estratégias de capacitação (workshops, cursos, treinamentos on-the-job) para equipes dos hospitais participantes ou do SUS.",
            "publicacoes_divulgacao": "Registra se o projeto gerou artigos, relatórios ou outras formas de comunicação de resultados, publicados ou em preparação.",
            "incorporacao_resultados_sus": "Explica se e como produtos ou protocolos desenvolvidos estão sendo integrados às rotinas do SUS, além dos desafios regulatórios e de financiamento.",
            "longevidade_sustentabilidade": "Reflexões sobre a continuidade do projeto após o financiamento do PROADI-SUS, incluindo fontes de recursos, escalabilidade e institucionalização."
        }

    def _load_existing_data(self) -> pd.DataFrame:
        """Load existing Excel data if it exists."""
        try:
            if Path(self.output_file).exists():
                df = pd.read_excel(self.output_file)
                logger.info(f"Loaded existing file with {len(df)} rows")
                return df
            else:
                logger.info("No existing file found, will create new one")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading existing file: {e}")
            return pd.DataFrame()

    def _get_existing_column_mapping(self) -> Dict[str, str]:
        """Map the actual Excel column names to our internal category keys."""
        # Based on the existing Excel file structure
        return {
            "codigo_entrevista": "Código Entrevista",
            "area_atuacao": "Área de atuação",
            "hospital": "Hospital",
            "nome_posicao_projetos": "Nome - posição institucional - Projetos",
            "modelos_planos_trabalho": "Modelos para planos de trabalho e prestação de contas",
            "avaliacao_geral_proadi": "Avaliação geral Proadi e DesenvoIvimento Institucional",
            "relacao_conass_conasems_ms": "Relação Conass/Conasems/MS com HE e instituições parceiras",
            "beneficios_instituicao_parceira": "Benefícios para instituição parceira",
            "desafios_participacao_he": "Desafios para a participação do HE no Proadi",
            "sugestoes": "Sugestões",
            "origem_projetos": "Origem dos projetos (quem demandou, tramitação e negociações)",
            "projetos_colaborativos": "Projetos colaborativos (participação de cada um, relacionamento HE e benefícios e desafios)",
            "expertise_hospital": "Expertise do hospital para o projeto e Inserção deste no HE",
            "abrangencia_territorial": "Abrangência Territorial do Projeto (definição)",
            "selecao_instituicoes_participantes": "Seleção e envolvimento instituições participantes no projeto",
            "avaliacoes_projeto": "Avaliações sobre o Projeto",
            "monitoramento_indicadores": "Monitoramento (HE e instituições participantes) e Indicadores",
            "riscos_dificuldades": "Riscos na implementação/dificuldades enfrentadas (adesão instituições ou profissionais, infraestrutura, outras)",
            "beneficios_sus": "Benefícios do projeto para o SUS",
            "incorporacao_bens_materiais": "Incorporação de bens materiais ao SUS?",
            "treinamento_profissionais": "Treinamento para profissionais?",
            "publicacoes_divulgacao": "Publicações ou divulgação?",
            "incorporacao_resultados_sus": "Incorporação resultados ao SUS",
            "longevidade_sustentabilidade": "Longevidade e sustentabilidade possível?"
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

    def _get_processed_interviews(self) -> List[str]:
        """Get list of already processed interview files from existing data."""
        if self.existing_df.empty:
            return []

        # Check the first unnamed column which should contain the filename info
        processed_files = []
        for _, row in self.existing_df.iterrows():
            # Skip header rows and empty rows
            first_col_value = str(row.iloc[0]) if len(row) > 0 else ""
            if first_col_value and first_col_value not in ["Responsável pela análise", "nan", ""]:
                # This might be an interview filename or identifier
                processed_files.append(first_col_value)

        logger.info(
            f"Found {len(processed_files)} already processed interviews")
        return processed_files

    def _create_analysis_prompt(self, interview_content: str) -> str:
        """Create the prompt for OpenAI analysis."""
        column_mapping = self._get_existing_column_mapping()
        categories_text = "\n".join([
            f"- {excel_col}: {description}"
            for internal_key, description in self.categories.items()
            for excel_col in [column_mapping.get(internal_key, internal_key)]
            if excel_col
        ])

        prompt = f"""
Analyze the following PROADI-SUS interview transcript and provide insights for each category.
For each category, provide a brief but specific assessment based on the interview content.
If information is not available for a category, respond with "Não mencionado" or "Informação insuficiente".
Keep the analysis in Portuguese.

IMPORTANT: Use the EXACT column names as keys in your JSON response. Do not modify or translate the column names.

Categories to analyze:
{categories_text}

Interview transcript:
{interview_content}

Please respond in JSON format with each EXACT category name as a key and your analysis as the value.
Keep responses concise but informative (1-3 sentences per category) in Portuguese.
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

        # Get already processed interviews to avoid duplicates
        processed_interviews = self._get_processed_interviews()

        results = []

        for file_path in interview_files:
            # Skip if already processed
            if file_path.name in processed_interviews:
                logger.info(f"Skipping {file_path.name} - already processed")
                continue

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

        logger.info(f"Processed {len(results)} new interviews successfully")
        return results

    def _export_to_excel(self, results: List[Dict[str, str]]) -> None:
        """Export analysis results to Excel file, appending to existing data."""
        if not results:
            logger.warning("No results to export")
            return

        try:
            column_mapping = self._get_existing_column_mapping()

            # Transform results to match existing Excel structure
            excel_rows = []
            for result in results:
                excel_row = {}

                # Add the filename in the first column (IDENTIFICAÇÃO)
                excel_row["IDENTIFICAÇÃO"] = result.get("interview_file", "")

                # Map internal categories to Excel column names
                for internal_key, excel_col in column_mapping.items():
                    excel_row[excel_col] = result.get(
                        excel_col, result.get(internal_key, ""))

                excel_rows.append(excel_row)

            # Create DataFrame for new results
            new_df = pd.DataFrame(excel_rows)

            # Load existing data properly with headers
            if Path(self.output_file).exists():
                existing_df = pd.read_excel(self.output_file)
                # Combine existing and new data
                combined_df = pd.concat(
                    [existing_df, new_df], ignore_index=True)
            else:
                # Create new file with proper structure
                combined_df = new_df

            # Export to Excel
            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                combined_df.to_excel(writer, index=False,
                                     sheet_name='Entrevistas')

                # Auto-adjust column widths
                worksheet = writer.sheets['Entrevistas']
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 80)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            logger.info(f"Results exported to {self.output_file}")
            logger.info(
                f"Added {len(results)} new interviews. Total: {len(combined_df)} interviews")

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
