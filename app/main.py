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

        # Token and chunking settings - optimized for gpt-4o (128k context, superior reasoning)
        self.max_tokens_per_request = 10000  # Large chunks for advanced model
        self.chunk_overlap = 300  # Good overlap for context continuity
        self.request_delay = 6  # Reasonable delay for rate limits
        self.max_output_tokens = 2500  # Allow for detailed, comprehensive responses

    def _setup_openai_client(self) -> OpenAI:
        """Setup OpenAI client with API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        return OpenAI(api_key=api_key)

    def _clean_json_response(self, response_text: str) -> str:
        """Clean up JSON response to handle common formatting issues."""
        # Remove common markdown formatting
        cleaned = response_text.strip()

        # Remove ```json and ``` if present
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]

        # Remove any text before the first {
        first_brace = cleaned.find('{')
        if first_brace > 0:
            cleaned = cleaned[first_brace:]

        # Remove any text after the last }
        last_brace = cleaned.rfind('}')
        if last_brace >= 0:
            cleaned = cleaned[:last_brace + 1]

        return cleaned.strip()

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count (1 token ≈ 4 characters for most text)."""
        return len(text) // 4

    def _split_interview_content(self, content: str, max_chunk_tokens: int = None) -> List[str]:
        """Split large interview content into smaller chunks at natural boundaries."""
        if max_chunk_tokens is None:
            max_chunk_tokens = 1000  # Hard limit of 1000 tokens per chunk

        # Further reduce to be very conservative
        max_chunk_tokens = min(max_chunk_tokens, 1000)

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
                    # HARD ENFORCEMENT: Check token count and truncate if needed
                    while self._estimate_tokens(current_chunk) > max_chunk_tokens:
                        current_chunk = current_chunk[:int(
                            len(current_chunk) * 0.9)]

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
                                # HARD ENFORCEMENT: Check token count
                                while self._estimate_tokens(current_chunk) > max_chunk_tokens:
                                    current_chunk = current_chunk[:int(
                                        len(current_chunk) * 0.9)]
                                chunks.append(current_chunk.strip())
                                current_chunk = sentence
                            else:
                                # Single sentence is too large, force split by character count
                                # Ensure even single sentences don't exceed token limit
                                max_sentence_chars = max_chunk_tokens * 3  # Very conservative
                                if len(sentence) > max_sentence_chars:
                                    chunks.append(
                                        sentence[:max_sentence_chars])
                                    current_chunk = sentence[max_sentence_chars:]
                                else:
                                    current_chunk = sentence
                        else:
                            current_chunk += ". " + sentence if current_chunk else sentence
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        # Add the last chunk with final token enforcement
        if current_chunk.strip():
            # FINAL HARD ENFORCEMENT
            while self._estimate_tokens(current_chunk) > max_chunk_tokens:
                current_chunk = current_chunk[:int(len(current_chunk) * 0.9)]
            chunks.append(current_chunk.strip())

        # Final verification: check all chunks and split any that are still too large
        verified_chunks = []
        for chunk in chunks:
            if self._estimate_tokens(chunk) <= max_chunk_tokens:
                verified_chunks.append(chunk)
            else:
                # Force split oversized chunks
                chunk_size = max_chunk_tokens * 3  # Conservative character estimate
                while chunk:
                    piece = chunk[:chunk_size]
                    verified_chunks.append(piece)
                    chunk = chunk[chunk_size:]

        logger.info(
            f"Split content into {len(verified_chunks)} chunks (max {max_chunk_tokens} tokens each)")
        return verified_chunks

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
        """Read interview content from file, handling different formats properly."""
        try:
            file_extension = file_path.suffix.lower()

            if file_extension == '.docx':
                # Handle Word documents with python-docx
                try:
                    from docx import Document
                    doc = Document(file_path)
                    content = '\n'.join(
                        [paragraph.text for paragraph in doc.paragraphs])
                    logger.info(
                        f"Read {len(content)} characters from Word document {file_path.name}")
                    return content.strip()
                except ImportError:
                    logger.error(
                        "python-docx not installed. Install with: pip install python-docx")
                    return ""
                except Exception as e:
                    logger.error(
                        f"Error reading Word document {file_path}: {e}")
                    return ""

            elif file_extension == '.doc':
                # Handle old Word documents with python-docx2txt
                try:
                    import docx2txt
                    content = docx2txt.process(str(file_path))
                    logger.info(
                        f"Read {len(content)} characters from old Word document {file_path.name}")
                    return content.strip()
                except ImportError:
                    logger.error(
                        "docx2txt not installed. Install with: pip install docx2txt")
                    return ""
                except Exception as e:
                    logger.error(
                        f"Error reading old Word document {file_path}: {e}")
                    return ""

            else:
                # Handle text files (.txt, .md, etc.)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                logger.info(
                    f"Read {len(content)} characters from text file {file_path.name}")
                return content

        except UnicodeDecodeError:
            # Try with different encoding for text files
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read().strip()
                logger.warning(f"Used latin-1 encoding for {file_path.name}")
                return content
            except Exception as e:
                logger.error(
                    f"Error reading file with latin-1 encoding {file_path}: {e}")
                return ""
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

        chunk_info = f"[PARTE {chunk_number}/{total_chunks}]" if chunk_number else ""

        prompt = f"""Você é um especialista em análise de entrevistas PROADI-SUS. Sua tarefa é extrair informações específicas desta entrevista{chunk_info}.

TEXTO DA ENTREVISTA:
{interview_content}

INSTRUÇÕES IMPORTANTES:
1. Leia todo o texto com atenção
2. Para cada categoria abaixo, encontre e cite EXATAMENTE o que está escrito na entrevista.
3. Analise cuidadosamente a descrição da categoria e encontre o texto da entrevista que se enquadra na descrição.
3. Citar diretamente o texto da entrevista, sem alterações.
4. Se uma informação não estiver presente, escreva "Não encontrado neste trecho"
5. SEMPRE extraia informação real do texto - nunca invente
6. Sempre extraia a informação completa, ou seja, os trechos a serem colados no excel podem ser longos, mantenha a conversação original entre entrevistador e entrevistado. Não corte enquanto o assunto não estiver acabado, cole tudo o que disser a respeito da categoria. Por exemplo, pode colar mais de um parágrafo, se for o caso. Cole o texto inteiro, não corte.

CATEGORIAS PARA EXTRAIR:

1. Código/Identificador da entrevista - Busque códigos como HIAE01, HEBPP01, etc.
2. Área de atuação - Se é Pesquisa, Capacitação, Avaliação ou Gestão
3. Hospital mencionado - Nome completo do hospital de excelência - há apenas 6 hospitais de excelência no Brasil, são eles:
    - Beneficência Portuguesa - BP
    - Hospital Israelita Albert Einstein - HIAE
    - Hospital Sírio Libanês - HSL
    - Hospital Moinhos de Vento - AHMV
    - Hospital Oswaldo Cruz - HAOC
    - Hospital do coração - HCOR
4. Nome e cargo do entrevistado - Quem está sendo entrevistado, sua função e projetos
5. Modelos para planos de trabalho - Como o hospital estrutura os planos de trabalho e os relatórios de prestação de contas, se funciona bem da forma como está sendo feito, se o formato atende as expectativas do entrevistado, como ele preenche os modelos, a matriz lógica, se há dificuldades, etc.
6. Avaliação geral do PROADI-SUS - Percepções sobre impacto do PROADI-SUS no hospital e SUS, ganhos institucionais, o que justifica a existência do PROADI-SUS, como o entrevistado avalia o programa de forma geral, qual a sua importância, quais oportunidades e benefícios o programa traz, que tipo de ganho o programa traz, troca de conhecimento, aprendizado, apoio. O que significa desenvolvimento institucional do SUS para o entrevistado.
7. Relação CONASS/CONASEMS/MS - Articulação entre Ministério, CONASS, CONASEMS com os hospitais de excelência e com os hospitais e instituições parceiras. Como o entrevistado avalia a relação entre essas instituições, como eles se comunicam, há reuniões, visitas, qual o formato e frequência desses encontros, etc.
8. Benefícios para instituição parceira - Vantagens para entidades beneficiárias, os centros participantes do programa. Eles recebem apoio, treinamento, capacitação, assessoria, como o entrevistado avalia a relação dos HE com as instituições parceiras. Essas instituições são centro participantes, como hospitais, UPAS, UBS, NATS, etc.
9. Desafios para participação do HE - Obstáculos internos e externos enfrentados para a participação do hospital de excelência no PROADI-SUS. Tempo, gestão, etc.
10. Sugestões de melhoria - Recomendações para aprimorar o programa. O que o entrevistado sugere para melhorar o programa. Há práticas que eles já implementam, que podem ser adotadas no PROADI-SUS? Melhorar a comunicação, grupos de whatsapp, aumentar prazo,
11. Origem dos projetos - Gênese do projeto, demandante e percurso burocrático. Quem demandou, qual a origem, o processo de tramitação e as negociações envolvidas no momento da formulação de um projeto. Quais atores participam? CONASS? CONASEMS? MS? Há uma lista de projetos prioritários? Como funcionam os contratos, os pagamentos aos envolvidos, etc.
12. Projetos colaborativos - Iniciativas com múltiplos HEs. Quando é o caso de projetos colaborativos, como o entrevistado enxerga essa colaboração? qual a participação de cada hospital de excelência (HE) no projeto? como é o relacionamento entre os HEs? Quais os benefícios e desafios que os HEs enfrentam em um projeto colaborativo?
13. Expertise do hospital - Competência técnica e alinhamento com missão institucional. O que o HE oferece para o projeto? Quais são as competências técnicas do HE? As áreas de atuação do HE são compatíveis com a expertise que o projeto exige?
14. Abrangência territorial - Alcance geográfico e critérios de escolha dos locais. Quais são os critérios de escolha dos locais, dos centros participantes do projeto? Quais são os locais escolhidos?
15. Seleção de instituições participantes - Critérios para convidar unidades de saúde, isto é, os centros participantes, como hospitais, UPAS, UBS, NATs, etc, centro de pesquisa, etc. Quais são os critérios para convidar unidades de saúde? Quais são as unidades de saúde escolhidas?
16. Avaliações sobre o projeto - A entrevista cita algum projeto do programa PROADI-SUS em específico, se sim, quais a percepção do entrevistado sobre os resultados preliminares, impacto e lições aprendidas.
17. Monitoramento e indicadores - Métodos de acompanhamento e indicadores utilizados. Quais são os indicadores de impacto do projeto? Quais são os métodos de acompanhamento? há monitoramento? como os HEs monitoram os centros participantes? há reuniões, visitas? qual a frequência e formato dessas visitas?
18. Riscos e dificuldades - Problemas práticos enfrentados na implementação do projeto, como rotatividade, infraestrutura, adesão das unidades de saúde participantes, falta de profissionais, etc.
19. Benefícios para o SUS - Ganhos, benefícios e importência do projeto para a rede pública de saúde, para o Sistema Único de Saúde de forma geral.
20. Incorporação de bens materiais - O projeto envolveu compra ou doação de equipamentos ou bens materias ao SUS, as unidades de saúde recebem esses bens quando o projeto acaba?
21. Treinamento para profissionais - Estratégias de capacitação mencionadas, há cursos, treinamentos, capacitação, etc dos HE para os centros participantes.
22. Publicações e divulgação - Artigos, relatórios ou comunicação de resultados. Qual a estratégia de divulgação do projeto, como o HE divulga os resultados do projeto?
23. Incorporação de resultados ao SUS - Como produtos são integrados às rotinas do SUS? Como o entrevistado avalia a incorporação dos resultados do projeto ao SUS? há transferência de conhecimento, aprendizado, tecnologia, etc?
24. Longevidade e sustentabilidade - Continuidade após financiamento PROADI-SUS. O que o entrevistado avalia sobre a sustentabilidade do projeto? O que ele sugere para a sustentabilidade do projeto? Depois que o projeto acaba, como ele avalia a longevidade de seus resultados? É possível continuar o projeto mesmo sem o PROADI-SUS?

Responda em formato JSON válido com as chaves exatas do Excel:

{{
  "Código Entrevista": "informação encontrada",
  "Área de atuação": "informação encontrada",
  "Hospital": "informação encontrada",
  "Nome - posição institucional - Projetos": "informação encontrada",
  "Modelos para planos de trabalho e prestação de contas": "informação encontrada",
  "Avaliação geral Proadi e DesenvoIvimento Institucional": "informação encontrada",
  "Relação Conass/Conasems/MS com HE e instituições parceiras": "informação encontrada",
  "Benefícios para instituição parceira": "informação encontrada",
  "Desafios para a participação do HE no Proadi": "informação encontrada",
  "Sugestões": "informação encontrada",
  "Origem dos projetos (quem demandou, tramitação e negociações)": "informação encontrada",
  "Projetos colaborativos (participação de cada um, relacionamento HE e benefícios e desafios)": "informação encontrada",
  "Expertise do hospital para o projeto e Inserção deste no HE": "informação encontrada",
  "Abrangência Territorial do Projeto (definição)": "informação encontrada",
  "Seleção e envolvimento instituições participantes no projeto": "informação encontrada",
  "Avaliações sobre o Projeto": "informação encontrada",
  "Monitoramento (HE e instituições participantes) e Indicadores": "informação encontrada",
  "Riscos na implementação/dificuldades enfrentadas (adesão instituições ou profissionais, infraestrutura, outras)": "informação encontrada",
  "Benefícios do projeto para o SUS": "informação encontrada",
  "Incorporação de bens materiais ao SUS?": "informação encontrada",
  "Treinamento para profissionais?": "informação encontrada",
  "Publicações ou divulgação?": "informação encontrada",
  "Incorporação resultados ao SUS": "informação encontrada",
  "Longevidade e sustentabilidade possível?": "informação encontrada"
}}

FOQUE EM ENCONTRAR CONTEÚDO REAL DA ENTREVISTA."""
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

            if estimated_tokens <= self.max_tokens_per_request - 2000:  # Reserve space for the detailed prompt
                # Content is small enough, process normally
                prompt = self._create_analysis_prompt(interview_content)

                response = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert analyst specializing in PROADI-SUS interviews. Extract specific information accurately and return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=self.max_output_tokens
                )

                analysis_text = response.choices[0].message.content.strip()

                # LOG THE RAW RESPONSE
                logger.info(f"🔍 Raw OpenAI response for {filename}:")
                logger.info(f"📝 Response: {analysis_text}")

                try:
                    # Clean the response before parsing
                    cleaned_json = self._clean_json_response(analysis_text)
                    logger.info(f"🧹 Cleaned JSON: {cleaned_json}")

                    analysis_dict = json.loads(cleaned_json)
                    logger.info(f"Successfully analyzed {filename}")
                    logger.info(
                        f"📊 Extracted data keys: {list(analysis_dict.keys())}")
                    return analysis_dict
                except json.JSONDecodeError:
                    logger.warning(
                        f"OpenAI response for {filename} was not valid JSON, using fallback")
                    logger.warning(
                        f"Original response: {analysis_text[:200]}...")
                    return {"general_analysis": analysis_text}

            else:
                # Content is too large, chunk it
                logger.info(f"Content too large for {filename}, chunking...")
                chunks = self._split_interview_content(
                    interview_content, max_chunk_tokens=1000)
                chunk_analyses = []

                for i, chunk in enumerate(chunks):
                    chunk_tokens = self._estimate_tokens(chunk)
                    logger.info(
                        f"Analyzing chunk {i+1}/{len(chunks)} for {filename} ({chunk_tokens} tokens)")

                    prompt = self._create_analysis_prompt(
                        chunk, i+1, len(chunks))
                    prompt_tokens = self._estimate_tokens(prompt)
                    total_tokens = chunk_tokens + prompt_tokens

                    if total_tokens > 100000:  # Very generous limit for gpt-4o's 128k context
                        logger.warning(
                            f"⚠️ Chunk {i+1} might be too large: {total_tokens} tokens total")

                    # Add delay to avoid rate limiting
                    if i > 0:
                        logger.info(
                            f"Waiting {self.request_delay} seconds to avoid rate limiting...")
                        time.sleep(self.request_delay)

                    try:
                        response = self.client.chat.completions.create(
                            model="gpt-4-turbo",
                            messages=[
                                {"role": "system", "content": "You are an expert analyst specializing in PROADI-SUS interviews. Extract specific information accurately and return valid JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1,
                            max_tokens=self.max_output_tokens
                        )

                        analysis_text = response.choices[0].message.content.strip(
                        )

                        # LOG THE RAW RESPONSE
                        logger.info(f"🔍 Raw OpenAI response for chunk {i+1}:")
                        logger.info(f"📝 Response: {analysis_text}")

                        try:
                            # Clean the response before parsing
                            cleaned_json = self._clean_json_response(
                                analysis_text)
                            logger.info(
                                f"🧹 Cleaned JSON for chunk {i+1}: {cleaned_json}")

                            chunk_analysis = json.loads(cleaned_json)
                            chunk_analyses.append(chunk_analysis)
                            logger.info(
                                f"✅ Successfully parsed JSON for chunk {i+1}")
                            logger.info(
                                f"📊 Extracted data keys: {list(chunk_analysis.keys())}")
                        except json.JSONDecodeError as json_err:
                            logger.warning(
                                f"❌ Chunk {i+1} JSON parse error: {json_err}")
                            logger.warning(
                                f"Original response: {analysis_text[:200]}...")
                            logger.warning(
                                f"Cleaned response: {self._clean_json_response(analysis_text)[:200]}...")
                            # Try to extract any useful info manually
                            chunk_analyses.append(
                                {"parsing_error": f"Invalid JSON: {analysis_text[:100]}"})

                    except Exception as e:
                        error_msg = str(e)
                        logger.error(
                            f"❌ Error analyzing chunk {i+1} of {filename}: {e}")

                        # Check if it's a rate limit or token error
                        if "rate_limit_exceeded" in error_msg or "429" in error_msg or "context_length_exceeded" in error_msg:
                            logger.info(
                                f"⏳ Token/rate limit hit, waiting {self.request_delay * 3} seconds...")
                            time.sleep(self.request_delay * 3)
                            # Try one more time with much shorter content
                            try:
                                # Use only first third of chunk if token limited
                                shorter_chunk = chunk[:len(chunk)//3]
                                logger.info(
                                    f"🔄 Retrying chunk {i+1} with {len(shorter_chunk)} characters...")
                                shorter_prompt = self._create_analysis_prompt(
                                    shorter_chunk, i+1, len(chunks))

                                response = self.client.chat.completions.create(
                                    model="gpt-4-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are an expert analyst specializing in PROADI-SUS interviews. Extract specific information accurately and return valid JSON."},
                                        {"role": "user", "content": shorter_prompt}
                                    ],
                                    temperature=0.1,
                                    max_tokens=self.max_output_tokens
                                )

                                analysis_text = response.choices[0].message.content.strip(
                                )

                                # LOG THE RETRY RAW RESPONSE
                                logger.info(
                                    f"🔄 Retry raw OpenAI response for chunk {i+1}:")
                                logger.info(
                                    f"📝 Retry response: {analysis_text}")

                                try:
                                    # Clean the response before parsing
                                    cleaned_json = self._clean_json_response(
                                        analysis_text)
                                    logger.info(
                                        f"🧹 Retry cleaned JSON for chunk {i+1}: {cleaned_json}")

                                    chunk_analysis = json.loads(cleaned_json)
                                    chunk_analyses.append(chunk_analysis)
                                    logger.info(
                                        f"✅ Successfully analyzed chunk {i+1} with shorter content")
                                    logger.info(
                                        f"📊 Retry extracted data keys: {list(chunk_analysis.keys())}")
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"❌ Retry also failed JSON parsing for chunk {i+1}")
                                    logger.warning(
                                        f"Retry response: {analysis_text[:200]}...")
                                    chunk_analyses.append(
                                        {"retry_parsing_error": analysis_text[:100]})

                            except Exception as retry_e:
                                logger.error(
                                    f"❌ Retry failed for chunk {i+1}: {retry_e}")
                                chunk_analyses.append(
                                    {"error": f"Both attempts failed: {str(retry_e)}"})
                        else:
                            chunk_analyses.append(
                                {"error": f"Analysis failed: {str(e)}"})

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
            # Transform results to match existing Excel structure
            excel_rows = []
            for result in results:
                excel_row = {}

                # Add the filename in the first column (IDENTIFICAÇÃO or use filename)
                excel_row["Responsável pela análise"] = result.get(
                    "interview_file", "")

                # Map all the exact column names from Excel
                exact_columns = [
                    "Código Entrevista",
                    "Área de atuação",
                    "Hospital",
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

                # Copy data for each exact column name
                for col in exact_columns:
                    excel_row[col] = result.get(col, "N/A")

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
