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
import time
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

        # Token and chunking settings
        self.max_tokens_per_request = 500  # Even smaller chunks
        self.chunk_overlap = 30  # Minimal overlap to save space
        self.request_delay = 10  # Longer delay to respect rate limits
        self.max_output_tokens = 800  # Reduce output tokens

    def _setup_openai_client(self) -> OpenAI:
        """Setup OpenAI client with API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        return OpenAI(api_key=api_key)

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count (1 token ≈ 4 characters for most text)."""
        return len(text) // 4

    def _split_interview_content(self, content: str, max_chunk_tokens: int = None) -> List[str]:
        """Split large interview content into smaller chunks at natural boundaries."""
        if max_chunk_tokens is None:
            max_chunk_tokens = self.max_tokens_per_request - 2000  # Reserve space for prompt

        # If content is small enough, return as single chunk
        if self._estimate_tokens(content) <= max_chunk_tokens:
            return [content]

        chunks = []
        # Convert tokens to approximate characters
        max_chunk_chars = max_chunk_tokens * 4

        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed the limit
            if len(current_chunk) + len(paragraph) > max_chunk_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap from previous chunk
                    if len(current_chunk) > self.chunk_overlap:
                        overlap = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap + "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Single paragraph is too large, split by sentences
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > max_chunk_chars:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = sentence
                            else:
                                # Single sentence is too large, force split
                                chunks.append(sentence[:max_chunk_chars])
                                current_chunk = sentence[max_chunk_chars:]
                        else:
                            current_chunk += ". " + sentence if current_chunk else sentence
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        logger.info(f"Split content into {len(chunks)} chunks")
        return chunks

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

    def _create_analysis_prompt(self, interview_content: str, chunk_number: int = None, total_chunks: int = None) -> str:
        """Create the prompt for OpenAI analysis."""

        # Abbreviated but complete categories
        cats = [
            "Código Entrevista", "Área de atuação", "Hospital",
            "Nome - posição institucional - Projetos",
            "Modelos para planos de trabalho e prestação de contas",
            "Avaliação geral Proadi e DesenvoIvimento Institucional",
            "Relação Conass/Conasems/MS com HE e instituições parceiras",
            "Benefícios para instituição parceira",
            "Desafios para a participação do HE no Proadi",
            "Sugestões",
            "Origem dos projetos (quem demandou, tramitação e negociações)",
            "Projetos colaborativos (participação de cada um, relacionamento HE e benefícios e desafios)",
            "Expertise do hospital para o projeto e Inserção deste no HE",
            "Abrangência Territorial do Projeto (definição)",
            "Seleção e envolvimento instituições participantes no projeto",
            "Avaliações sobre o Projeto",
            "Monitoramento (HE e instituições participantes) e Indicadores",
            "Riscos na implementação/dificuldades enfrentadas (adesão instituições ou profissionais, infraestrutura, outras)",
            "Benefícios do projeto para o SUS",
            "Incorporação de bens materiais ao SUS?",
            "Treinamento para profissionais?",
            "Publicações ou divulgação?",
            "Incorporação resultados ao SUS",
            "Longevidade e sustentabilidade possível?"
        ]

        chunk_info = f"({chunk_number}/{total_chunks})" if chunk_number else ""

        prompt = f"""Analise entrevista PROADI-SUS{chunk_info}.
Extraia info para categorias. Use nomes EXATOS como chaves JSON.

{interview_content}

Categorias: {', '.join(cats)}

JSON válido, 1 frase/categoria, "N/A" se ausente:"""
        return prompt

    def _combine_chunk_analyses(self, chunk_analyses: List[Dict[str, str]]) -> Dict[str, str]:
        """Combine analysis results from multiple chunks into a single comprehensive analysis."""
        if not chunk_analyses:
            return {}

        if len(chunk_analyses) == 1:
            return chunk_analyses[0]

        combined_analysis = {}
        column_mapping = self._get_existing_column_mapping()

        # Get all possible keys from all analyses
        all_keys = set()
        for analysis in chunk_analyses:
            all_keys.update(analysis.keys())

        for key in all_keys:
            # Skip metadata keys
            if key in ["interview_file", "file_size_chars", "error"]:
                combined_analysis[key] = chunk_analyses[0].get(key, "")
                continue

            # Combine responses for each category
            responses = []
            for analysis in chunk_analyses:
                response = analysis.get(key, "")
                if response and response not in ["Não mencionado", "Informação insuficiente", ""]:
                    responses.append(response)

            if responses:
                # If we have multiple valid responses, combine them intelligently
                if len(responses) == 1:
                    combined_analysis[key] = responses[0]
                else:
                    # For multiple responses, create a comprehensive summary
                    unique_responses = []
                    for response in responses:
                        # Avoid duplicate content
                        if not any(response.lower() in existing.lower() or existing.lower() in response.lower()
                                   for existing in unique_responses):
                            unique_responses.append(response)

                    if len(unique_responses) == 1:
                        combined_analysis[key] = unique_responses[0]
                    else:
                        combined_analysis[key] = " | ".join(unique_responses)
            else:
                combined_analysis[key] = "Não mencionado"

        return combined_analysis

    def _analyze_interview_with_openai(self, interview_content: str, filename: str) -> Dict[str, str]:
        """Analyze interview content using OpenAI API, handling large content by chunking."""
        try:
            # Check if content needs to be chunked
            estimated_tokens = self._estimate_tokens(interview_content)
            logger.info(f"Estimated tokens for {filename}: {estimated_tokens}")

            if estimated_tokens <= self.max_tokens_per_request - 500:  # Reserve more space for prompt
                # Content is small enough, process normally
                prompt = self._create_analysis_prompt(interview_content)

                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system",
                            "content": "You are an expert analyst. Provide concise analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=self.max_output_tokens
                )

                analysis_text = response.choices[0].message.content.strip()

                try:
                    analysis_dict = json.loads(analysis_text)
                    logger.info(f"Successfully analyzed {filename}")
                    return analysis_dict
                except json.JSONDecodeError:
                    logger.warning(
                        f"OpenAI response for {filename} was not valid JSON, using fallback")
                    return {"general_analysis": analysis_text}

            else:
                # Content is too large, chunk it
                logger.info(f"Content too large for {filename}, chunking...")
                chunks = self._split_interview_content(interview_content)
                chunk_analyses = []

                for i, chunk in enumerate(chunks):
                    logger.info(
                        f"Analyzing chunk {i+1}/{len(chunks)} for {filename}")

                    prompt = self._create_analysis_prompt(
                        chunk, i+1, len(chunks))

                    # Add delay to avoid rate limiting
                    if i > 0:
                        logger.info(
                            f"Waiting {self.request_delay} seconds to avoid rate limiting...")
                        time.sleep(self.request_delay)

                    try:
                        response = self.client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system",
                                    "content": "You are an expert analyst. Provide concise analysis."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3,
                            max_tokens=self.max_output_tokens
                        )

                        analysis_text = response.choices[0].message.content.strip(
                        )

                        try:
                            chunk_analysis = json.loads(analysis_text)
                            chunk_analyses.append(chunk_analysis)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Chunk {i+1} response was not valid JSON")
                            chunk_analyses.append(
                                {"general_analysis": analysis_text})

                    except Exception as e:
                        error_msg = str(e)
                        logger.error(
                            f"Error analyzing chunk {i+1} of {filename}: {e}")

                        # Check if it's a rate limit error
                        if "rate_limit_exceeded" in error_msg or "429" in error_msg:
                            logger.info(
                                f"Rate limit hit, waiting {self.request_delay * 3} seconds...")
                            time.sleep(self.request_delay * 3)
                            # Try one more time with shorter content
                            try:
                                # Use only first half of chunk if rate limited
                                shorter_chunk = chunk[:len(chunk)//2]
                                shorter_prompt = self._create_analysis_prompt(
                                    shorter_chunk, i+1, len(chunks))

                                response = self.client.chat.completions.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system",
                                            "content": "You are an expert analyst. Provide concise analysis."},
                                        {"role": "user", "content": shorter_prompt}
                                    ],
                                    temperature=0.3,
                                    max_tokens=self.max_output_tokens
                                )

                                analysis_text = response.choices[0].message.content.strip(
                                )
                                try:
                                    chunk_analysis = json.loads(analysis_text)
                                    chunk_analyses.append(chunk_analysis)
                                    logger.info(
                                        f"Successfully analyzed chunk {i+1} with shorter content")
                                except json.JSONDecodeError:
                                    chunk_analyses.append(
                                        {"general_analysis": analysis_text})

                            except Exception as retry_e:
                                logger.error(
                                    f"Retry failed for chunk {i+1}: {retry_e}")
                                chunk_analyses.append(
                                    {"error": f"Chunk analysis failed: {str(retry_e)}"})
                        else:
                            chunk_analyses.append(
                                {"error": f"Chunk analysis failed: {str(e)}"})

                # Combine all chunk analyses
                combined_analysis = self._combine_chunk_analyses(
                    chunk_analyses)
                logger.info(
                    f"Successfully analyzed {filename} using {len(chunks)} chunks")
                return combined_analysis

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
