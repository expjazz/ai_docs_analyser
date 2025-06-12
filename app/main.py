#!/usr/bin/env python3
"""
Interview Analysis Tool
Processes interviews from the entrevistas folder using OpenAI Assistant API with File Search
and exports categorized analysis to Excel.
"""

import openai
import os
import sys
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add OpenAI version checking
print(f"🔍 OpenAI Python SDK Version: {openai.__version__}")
logger = logging.getLogger(__name__)

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
    """Analyzes interviews using OpenAI API or Ollama and categorizes content."""

    def __init__(self):
        """Initialize the analyzer with OpenAI/Ollama/Gemini client and categories."""
        self.use_file_search = os.getenv(
            "USE_FILE_SEARCH", "true").lower() == "true"
        self.use_gpt = os.getenv("USE_GPT", "false").lower() == "true"
        self.use_gemini = os.getenv("USE_GEMINI", "false").lower() == "true"
        self.use_embeddings = os.getenv(
            "USE_EMBEDDINGS", "false").lower() == "true"

        # Priority: File Search > Gemini > GPT > Ollama
        if self.use_file_search and os.getenv("OPENAI_API_KEY"):
            logger.info("🔍 Using OpenAI Assistant API with File Search")
            self.analysis_method = "file_search"
            self.use_gpt = True  # File search requires OpenAI
        elif self.use_gemini:
            self.use_gpt = False
            self.analysis_method = "gemini"
        elif self.use_gpt:
            self.analysis_method = "gpt"
        else:
            self.analysis_method = "ollama"

        self.client = self._setup_client()
        self.model_name = self._get_model_name()
        self.categories = self._define_categories()
        self.interviews_folder = Path("entrevistas")
        self.output_file = "Entrevistas Hospitais_Rosi.xlsx"
        self.existing_df = self._load_existing_data()

        # File Search specific attributes
        if self.analysis_method == "file_search":
            self.assistant_id = None
            self.uploaded_files = {}  # Store file_id -> filename mapping
            self.thread_id = None
            self.vector_store_id = None

        # Token and chunking settings - optimized for the chosen model
        if self.use_gemini:
            self.max_tokens_per_request = 90000  # Gemini Pro has ~1M token context
            self.max_output_tokens = 8000  # Large output for comprehensive analysis
        elif self.use_gpt:
            # Much larger chunks for GPT-4o (use most of 128k context)
            self.max_tokens_per_request = 80000
            # Much larger output for comprehensive analysis with GPT-4o
            self.max_output_tokens = 8000
        else:
            self.max_tokens_per_request = 15000  # Much larger chunks for local DeepSeek-R1
            self.max_output_tokens = 4000  # Allow full thinking process + JSON output

        self.chunk_overlap = 300  # Good overlap for context continuity
        self.request_delay = 1 if self.use_gemini else (
            2 if not self.use_gpt else 6)  # Gemini is fastest

        # Embedding settings
        if self.use_embeddings:
            self.embedding_model = "text-embedding-3-small" if self.use_gpt else None
            self.similarity_threshold = 0.7  # Minimum similarity to include a section
            self.max_sections_per_category = 3  # Maximum relevant sections per category
            ai_type = "Gemini" if self.use_gemini else (
                "OpenAI" if self.use_gpt else "local")
            logger.info(
                f"🔍 Using embeddings-based analysis with {ai_type} embeddings")

    def _setup_client(self):
        """Setup OpenAI, Gemini, or Ollama client based on environment variables."""
        if self.analysis_method == "file_search" or self.analysis_method == "gpt":
            logger.info(
                f"🤖 Using OpenAI {'with File Search' if self.analysis_method == 'file_search' else 'GPT'}")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment variables. "
                    "Please set it in your .env file or environment."
                )
            client = OpenAI(api_key=api_key)

            # Debug client capabilities
            logger.info(f"🔍 OpenAI client type: {type(client)}")
            logger.info(
                f"🔍 Client has beta attribute: {hasattr(client, 'beta')}")
            if hasattr(client, 'beta'):
                logger.info(f"🔍 Beta type: {type(client.beta)}")
                logger.info(
                    f"🔍 Beta has vector_stores: {hasattr(client.beta, 'vector_stores')}")
                logger.info(
                    f"🔍 Beta has assistants: {hasattr(client.beta, 'assistants')}")
                logger.info(
                    f"🔍 Beta has threads: {hasattr(client.beta, 'threads')}")

                # Show all available beta attributes
                beta_attrs = [attr for attr in dir(
                    client.beta) if not attr.startswith('_')]
                logger.info(f"🔍 Available beta attributes: {beta_attrs}")

                # Try to access vector_stores directly to see what happens
                try:
                    vs = getattr(client.beta, 'vector_stores', None)
                    logger.info(f"🔍 Direct vector_stores access result: {vs}")
                    logger.info(f"🔍 vector_stores type: {type(vs)}")
                except Exception as e:
                    logger.error(f"🔍 Error accessing vector_stores: {e}")

            return client
        elif self.use_gemini:
            logger.info("🤖 Using Google Gemini")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY not found in environment variables. "
                    "Please set it in your .env file or environment."
                )
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                return genai
            except ImportError:
                raise ValueError(
                    "google-generativeai package not installed. Install with: pip install google-generativeai"
                )
        else:
            logger.info("🦙 Using Ollama with deepseek-r1")
            try:
                import ollama
                return ollama  # Return the module itself
            except ImportError:
                raise ValueError(
                    "Ollama package not installed. Install with: pip install ollama"
                )

    def _get_model_name(self) -> str:
        """Get the model name based on the client type."""
        if self.analysis_method == "file_search":
            return "gpt-4o"  # Best model for file search capabilities
        elif self.use_gemini:
            return "gemini-1.5-flash"  # Latest Gemini Pro model
        elif self.use_gpt:
            return "gpt-4o"  # Latest and most capable OpenAI model
        else:
            return "deepseek-r1"

    def _upload_interview_files_for_assistant(self) -> dict:
        """Upload all interview files to OpenAI for Assistant API (purpose='assistants')."""
        logger.info(
            "📤 Uploading interview files to OpenAI (purpose='assistants')...")
        interview_files = self._get_interview_files()
        uploaded_files = {}
        for file_path in interview_files:
            try:
                with open(file_path, 'rb') as f:
                    file_obj = self.client.files.create(
                        file=f,
                        purpose='assistants'
                    )
                uploaded_files[file_obj.id] = file_path.name
                logger.info(f"✅ Uploaded {file_path.name} as {file_obj.id}")
            except Exception as e:
                logger.error(f"❌ Failed to upload {file_path.name}: {e}")
        return uploaded_files

    def _create_vector_store(self, file_ids: dict) -> str:
        """Create a vector store and add uploaded files."""
        logger.info("📦 Creating vector store and adding files...")

        # Additional debugging
        logger.info(f"🔍 About to access self.client.beta.vector_stores")
        logger.info(f"🔍 self.client type: {type(self.client)}")
        logger.info(f"🔍 self.client.beta type: {type(self.client.beta)}")

        try:
            # Check if vector_stores exists before trying to use it
            if not hasattr(self.client.beta, 'vector_stores'):
                logger.error(
                    f"❌ self.client.beta does not have 'vector_stores' attribute")
                logger.error(
                    f"❌ Available beta attributes: {dir(self.client.beta)}")
                raise AttributeError(
                    "vector_stores not available in beta client")

            vector_store = self.client.beta.vector_stores.create()
            logger.info(f"✅ Vector store created: {vector_store.id}")

            # Add files to vector store
            if not hasattr(self.client.beta.vector_stores, 'files'):
                logger.error(
                    f"❌ vector_stores does not have 'files' attribute")
                raise AttributeError("vector_stores.files not available")

            batch_response = self.client.beta.vector_stores.files.batch_create(
                vector_store_id=vector_store.id,
                file_ids=list(file_ids.keys())
            )
            logger.info(f"✅ Added {len(file_ids)} files to vector store")

            return vector_store.id
        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {e}")
            logger.error(f"❌ Exception type: {type(e)}")
            raise

    def _create_file_search_assistant(self, vector_store_id: str) -> str:
        """Create an Assistant with file_search tool and vector store."""
        logger.info("🤖 Creating Assistant with file_search tool...")
        try:
            assistant = self.client.beta.assistants.create(
                name="Entrevista PROADI-SUS Analyzer",
                instructions="Você é um especialista em análise de entrevistas PROADI-SUS. Use a ferramenta file_search para encontrar e citar EXATAMENTE as falas das entrevistas. Sempre retorne JSON válido com as 24 categorias.",
                tools=[{"type": "file_search"}],
                model=self.model_name,
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [vector_store_id]
                    }
                }
            )
            logger.info(f"✅ Assistant created: {assistant.id}")
            return assistant.id
        except Exception as e:
            logger.error(f"❌ Failed to create assistant: {e}")
            raise

    def _create_thread(self) -> str:
        """Create a new thread for the Assistant API."""
        try:
            logger.info("🧵 Creating new thread...")
            thread = self.client.beta.threads.create()
            logger.info(f"🧵 Thread response: {thread}")
            logger.info(f"🧵 Thread type: {type(thread)}")

            if hasattr(thread, 'id'):
                thread_id = thread.id
                logger.info(f"🧵 Thread ID: {thread_id}")
                if thread_id:
                    return thread_id
                else:
                    logger.error(f"❌ Thread ID is None or empty: {thread_id}")
                    raise ValueError("Thread ID is None or empty")
            else:
                logger.error(
                    f"❌ Thread object has no 'id' attribute. Attributes: {dir(thread)}")
                raise ValueError("Thread object has no 'id' attribute")

        except Exception as e:
            logger.error(f"❌ Failed to create thread: {e}")
            logger.error(f"❌ Exception type: {type(e)}")
            raise

    def _add_message_to_thread(self, thread_id: str, prompt: str) -> None:
        """Add a message to the thread."""
        try:
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=prompt
            )
        except Exception as e:
            logger.error(f"❌ Failed to add message to thread: {e}")
            raise

    def _run_assistant_and_get_response(self, assistant_id: str, thread_id: str) -> str:
        """Run the assistant and get the full response."""
        try:
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                instructions=None
            )
            # Wait for completion
            while True:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id, run_id=run.id)
                if run_status.status in ["completed", "failed", "cancelled"]:
                    break
                time.sleep(2)
            if run_status.status != "completed":
                logger.error(f"❌ Run failed with status: {run_status.status}")
                return ""
            # Get the latest message
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id, order="desc")
            if not messages.data:
                logger.error("❌ No messages returned from assistant.")
                return ""
            response_content = messages.data[0].content[0].text.value
            return response_content
        except Exception as e:
            logger.error(f"❌ Error running assistant: {e}")
            return ""

    def _create_file_search_prompt(self, filename: str) -> str:
        """Prompt for file_search assistant to extract all 24 categories from the interview file."""
        return f"""Utilize a ferramenta file_search para encontrar e citar EXATAMENTE as falas da entrevista no arquivo '{filename}'. NÃO resuma, NÃO interprete, NÃO parafraseie. Para cada categoria abaixo, procure e copie as falas exatas. Se não encontrar, escreva 'Não encontrado'. Responda apenas em JSON válido com as chaves exatas:

1. Código/Identificador da entrevista (HIAE01, HEBPP01, etc.)
2. Área de atuação (Pesquisa, Capacitação, Avaliação, Gestão)
3. Nome do hospital de excelência (BP, HIAE, HSL, AHMV, HAOC, HCOR)
4. Nome, cargo e projetos do entrevistado
5. Falas sobre modelos de planos de trabalho e prestação de contas
6. Falas sobre avaliação geral do PROADI-SUS
7. Falas sobre relação CONASS/CONASEMS/MS
8. Falas sobre benefícios para instituições parceiras
9. Falas sobre desafios para participação do hospital
10. Sugestões de melhoria mencionadas
11. Falas sobre origem e tramitação dos projetos
12. Falas sobre projetos colaborativos
13. Falas sobre expertise do hospital
14. Falas sobre abrangência territorial
15. Falas sobre seleção de instituições participantes
16. Falas sobre avaliações do projeto
17. Falas sobre monitoramento e indicadores
18. Falas sobre riscos e dificuldades
19. Falas sobre benefícios para o SUS
20. Menções sobre incorporação de bens materiais
21. Falas sobre treinamento de profissionais
22. Menções sobre publicações e divulgação
23. Falas sobre incorporação de resultados ao SUS
24. Falas sobre longevidade e sustentabilidade

Responda APENAS em formato JSON válido:
{{
  "Código Entrevista": "...",
  "Área de atuação": "...",
  "Hospital": "...",
  "Nome - posição institucional - Projetos": "...",
  "Modelos para planos de trabalho e prestação de contas": "...",
  "Avaliação geral Proadi e DesenvoIvimento Institucional": "...",
  "Relação Conass/Conasems/MS com HE e instituições parceiras": "...",
  "Benefícios para instituição parceira": "...",
  "Desafios para a participação do HE no Proadi": "...",
  "Sugestões": "...",
  "Origem dos projetos (quem demandou, tramitação e negociações)": "...",
  "Projetos colaborativos (participação de cada um, relacionamento HE e benefícios e desafios)": "...",
  "Expertise do hospital para o projeto e Inserção deste no HE": "...",
  "Abrangência Territorial do Projeto (definição)": "...",
  "Seleção e envolvimento instituições participantes no projeto": "...",
  "Avaliações sobre o Projeto": "...",
  "Monitoramento (HE e instituições participantes) e Indicadores": "...",
  "Riscos na implementação/dificuldades enfrentadas (adesão instituições ou profissionais, infraestrutura, outras)": "...",
  "Benefícios do projeto para o SUS": "...",
  "Incorporação de bens materiais ao SUS?": "...",
  "Treinamento para profissionais?": "...",
  "Publicações ou divulgação?": "...",
  "Incorporação resultados ao SUS": "...",
  "Longevidade e sustentabilidade possível?": "..."
}}
"""

    def _cleanup_file_search_resources(self, assistant_id=None, vector_store_id=None, file_ids=None):
        """Cleanup Assistant, vector store, and files."""
        logger.info("🧹 Cleaning up Assistant, vector store, and files...")
        try:
            if assistant_id:
                self.client.beta.assistants.delete(assistant_id)
                logger.info(f"🗑️ Deleted assistant {assistant_id}")
        except Exception as e:
            logger.warning(f"Failed to delete assistant: {e}")
        try:
            if vector_store_id:
                self.client.beta.vector_stores.delete(vector_store_id)
                logger.info(f"🗑️ Deleted vector store {vector_store_id}")
        except Exception as e:
            logger.warning(f"Failed to delete vector store: {e}")
        try:
            if file_ids:
                for file_id in file_ids:
                    try:
                        self.client.files.delete(file_id)
                        logger.info(f"🗑️ Deleted file {file_id}")
                    except Exception as e:
                        logger.warning(f"Failed to delete file {file_id}: {e}")
        except Exception as e:
            logger.warning(f"Failed to delete files: {e}")

    def _create_file_search_assistant_without_vector_store(self, file_ids: list) -> str:
        """Create an Assistant with file_search tool using direct file attachment (no vector store)."""
        logger.info(
            "🤖 Creating Assistant with file_search tool (no vector store)...")
        try:
            assistant = self.client.beta.assistants.create(
                name="Entrevista PROADI-SUS Analyzer",
                instructions="Você é um especialista em análise de entrevistas PROADI-SUS. Use a ferramenta file_search para encontrar e citar EXATAMENTE as falas das entrevistas. Sempre retorne JSON válido com as 24 categorias.",
                tools=[{"type": "file_search"}],
                model=self.model_name
            )
            logger.info(f"✅ Assistant created: {assistant.id}")
            return assistant.id
        except Exception as e:
            logger.error(f"❌ Failed to create assistant: {e}")
            raise

    def _add_message_to_thread_with_file(self, thread_id: str, prompt: str, file_id: str) -> None:
        """Add a message to the thread with file attachment."""
        try:
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=prompt,
                attachments=[
                    {
                        "file_id": file_id,
                        "tools": [{"type": "file_search"}]
                    }
                ]
            )
            logger.info(f"✅ Added message with file attachment {file_id}")
        except Exception as e:
            logger.error(f"❌ Failed to add message with file to thread: {e}")
            raise

    def _analyze_interviews_with_file_search(self) -> List[Dict[str, str]]:
        """Analyze all interviews using OpenAI Assistant API with file_search (no vector store)."""
        if self.analysis_method != "file_search":
            return []
        logger.info(
            "🔍 Starting Assistant API file_search analysis (no vector store)...")
        try:
            # Upload files for assistants
            uploaded_files = self._upload_interview_files_for_assistant()
            if not uploaded_files:
                logger.error("No files uploaded successfully")
                return []

            # Create assistant without vector store
            assistant_id = self._create_file_search_assistant_without_vector_store(
                list(uploaded_files.keys()))

            results = []
            processed_count = 0
            processed_interviews = self._get_processed_interviews()

            for file_id, filename in uploaded_files.items():
                # Skip if already processed
                if filename in processed_interviews:
                    logger.info(f"Skipping {filename} - already processed")
                    continue

                logger.info(
                    f"🔍 Analyzing {filename} using Assistant API file_search...")

                try:
                    # Create new thread for each file
                    logger.info(f"🧵 About to create thread for {filename}...")
                    thread_id = self._create_thread()
                    logger.info(
                        f"🧵 Thread creation returned: {thread_id} (type: {type(thread_id)})")

                    if thread_id is None:
                        logger.error(
                            f"❌ Thread creation returned None for {filename}")
                        continue

                    # Create prompt for this specific file
                    prompt = self._create_file_search_prompt(filename)

                    # Add message with file attachment
                    self._add_message_to_thread_with_file(
                        thread_id, prompt, file_id)

                    # Run assistant and get response
                    response_content = self._run_assistant_and_get_response(
                        assistant_id, thread_id)

                    # LOG THE FULL RAW RESPONSE FOR DEBUGGING
                    logger.info(f"🔍 FULL RAW AI RESPONSE for {filename}:")
                    logger.info("=" * 80)
                    logger.info(response_content)
                    logger.info("=" * 80)

                    if not response_content:
                        logger.error(f"❌ No response content for {filename}")
                        continue

                    try:
                        cleaned_json = self._extract_json_from_response(
                            response_content)
                        logger.info(f"🧹 CLEANED JSON for {filename}:")
                        logger.info("-" * 40)
                        logger.info(cleaned_json)
                        logger.info("-" * 40)

                        analysis_dict = json.loads(cleaned_json)
                        analysis_dict = self._validate_and_fix_json(
                            analysis_dict, filename)

                        logger.info(f"📊 FINAL ANALYSIS DICT for {filename}:")
                        for key, value in analysis_dict.items():
                            logger.info(
                                f"  {key}: {value[:100]}{'...' if len(str(value)) > 100 else ''}")

                        result = {
                            "interview_file": filename,
                            "file_size_chars": None,
                            **analysis_dict
                        }
                        results.append(result)

                        # Save immediately to Excel
                        self._export_to_excel([result])
                        processed_count += 1
                        logger.info(
                            f"✅ Saved {filename} to Excel ({processed_count} processed)")

                    except json.JSONDecodeError as e:
                        logger.error(
                            f"❌ JSON parsing error for {filename}: {e}")
                        logger.error(
                            f"Response content: {response_content[:1000]}...")
                        continue

                except Exception as e:
                    logger.error(f"❌ Error processing {filename}: {e}")
                    logger.error(
                        f"❌ Exception details: {type(e).__name__}: {str(e)}")
                    import traceback
                    logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                    continue

                # Small delay to avoid rate limiting
                time.sleep(2)

                # Reload existing data for next iteration
                self.existing_df = self._load_existing_data()

            # Cleanup resources (no vector store to clean)
            self._cleanup_file_search_resources_no_vector_store(
                assistant_id, list(uploaded_files.keys()))

            logger.info(
                f"🔍 Assistant API file_search analysis completed. Processed {processed_count} interviews.")
            return results

        except Exception as e:
            logger.error(f"❌ Error in Assistant API file_search analysis: {e}")
            return []

    def _cleanup_file_search_resources_no_vector_store(self, assistant_id=None, file_ids=None):
        """Cleanup Assistant and files (no vector store)."""
        logger.info("🧹 Cleaning up Assistant and files (no vector store)...")
        try:
            if assistant_id:
                self.client.beta.assistants.delete(assistant_id)
                logger.info(f"🗑️ Deleted assistant {assistant_id}")
        except Exception as e:
            logger.warning(f"Failed to delete assistant: {e}")
        try:
            if file_ids:
                for file_id in file_ids:
                    try:
                        self.client.files.delete(file_id)
                        logger.info(f"🗑️ Deleted file {file_id}")
                    except Exception as e:
                        logger.warning(f"Failed to delete file {file_id}: {e}")
        except Exception as e:
            logger.warning(f"Failed to delete files: {e}")

    def _create_simple_file_prompt(self, filename: str, file_id: str) -> str:
        """Create a prompt for analyzing a specific interview file using simple file reference."""
        return f"""Analise a entrevista do arquivo "{filename}" (ID: {file_id}) e extraia as seguintes informações:

IMPORTANTE:
- Copie as falas EXATAS da entrevista, palavra por palavra
- NÃO resuma, interprete ou parafrase
- Se não encontrar informação específica, escreva "Não encontrado"
- Use todo o conteúdo do arquivo para encontrar informações relevantes

Extraia do arquivo "{filename}":

1. Código/Identificador da entrevista (HIAE01, HEBPP01, etc.)
2. Área de atuação (Pesquisa, Capacitação, Avaliação, Gestão)
3. Nome do hospital de excelência (BP, HIAE, HSL, AHMV, HAOC, HCOR)
4. Nome, cargo e projetos do entrevistado
5. Falas sobre modelos de planos de trabalho e prestação de contas
6. Falas sobre avaliação geral do PROADI-SUS
7. Falas sobre relação CONASS/CONASEMS/MS
8. Falas sobre benefícios para instituições parceiras
9. Falas sobre desafios para participação do hospital
10. Sugestões de melhoria mencionadas
11. Falas sobre origem e tramitação dos projetos
12. Falas sobre projetos colaborativos
13. Falas sobre expertise do hospital
14. Falas sobre abrangência territorial
15. Falas sobre seleção de instituições participantes
16. Falas sobre avaliações do projeto
17. Falas sobre monitoramento e indicadores
18. Falas sobre riscos e dificuldades
19. Falas sobre benefícios para o SUS
20. Menções sobre incorporação de bens materiais
21. Falas sobre treinamento de profissionais
22. Menções sobre publicações e divulgação
23. Falas sobre incorporação de resultados ao SUS
24. Falas sobre longevidade e sustentabilidade

Responda APENAS em formato JSON válido:

{{
  "Código Entrevista": "informação encontrada ou Não encontrado",
  "Área de atuação": "informação encontrada ou Não encontrado",
  "Hospital": "informação encontrada ou Não encontrado",
  "Nome - posição institucional - Projetos": "informação encontrada ou Não encontrado",
  "Modelos para planos de trabalho e prestação de contas": "informação encontrada ou Não encontrado",
  "Avaliação geral Proadi e DesenvoIvimento Institucional": "informação encontrada ou Não encontrado",
  "Relação Conass/Conasems/MS com HE e instituições parceiras": "informação encontrada ou Não encontrado",
  "Benefícios para instituição parceira": "informação encontrada ou Não encontrado",
  "Desafios para a participação do HE no Proadi": "informação encontrada ou Não encontrado",
  "Sugestões": "informação encontrada ou Não encontrado",
  "Origem dos projetos (quem demandou, tramitação e negociações)": "informação encontrada ou Não encontrado",
  "Projetos colaborativos (participação de cada um, relacionamento HE e benefícios e desafios)": "informação encontrada ou Não encontrado",
  "Expertise do hospital para o projeto e Inserção deste no HE": "informação encontrada ou Não encontrado",
  "Abrangência Territorial do Projeto (definição)": "informação encontrada ou Não encontrado",
  "Seleção e envolvimento instituições participantes no projeto": "informação encontrada ou Não encontrado",
  "Avaliações sobre o Projeto": "informação encontrada ou Não encontrado",
  "Monitoramento (HE e instituições participantes) e Indicadores": "informação encontrada ou Não encontrado",
  "Riscos na implementação/dificuldades enfrentadas (adesão instituições ou profissionais, infraestrutura, outras)": "informação encontrada ou Não encontrado",
  "Benefícios do projeto para o SUS": "informação encontrada ou Não encontrado",
  "Incorporação de bens materiais ao SUS?": "informação encontrada ou Não encontrado",
  "Treinamento para profissionais?": "informação encontrada ou Não encontrado",
  "Publicações ou divulgação?": "informação encontrada ou Não encontrado",
  "Incorporação resultados ao SUS": "informação encontrada ou Não encontrado",
  "Longevidade e sustentabilidade possível?": "informação encontrada ou Não encontrado"
}}

Analise todo o conteúdo do arquivo "{filename}" e extraia citações completas e diretas."""

    def _cleanup_simple_files(self):
        """Clean up uploaded files (simpler cleanup without assistant/vector store)."""
        if self.analysis_method != "file_search":
            return

        logger.info("🧹 Starting cleanup of uploaded files...")

        try:
            # Delete uploaded files
            for file_id, filename in self.uploaded_files.items():
                try:
                    self.client.files.delete(file_id)
                    logger.info(f"🗑️ Deleted file {filename} ({file_id})")
                except Exception as e:
                    logger.warning(f"Failed to delete file {filename}: {e}")

            logger.info("✅ Simple cleanup completed successfully")

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

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

    def _clean_json_response(self, response_text: str) -> str:
        """Clean up JSON response to handle common formatting issues."""
        # Remove common markdown formatting
        cleaned = response_text.strip()

        # Handle DeepSeek-R1 thinking tags
        if '<think>' in cleaned:
            think_start = cleaned.find('<think>')
            think_end = cleaned.find('</think>')
            if think_start >= 0 and think_end >= 0:
                # Remove the thinking section
                cleaned = cleaned[:think_start] + cleaned[think_end + 8:]
                logger.info(
                    "🧠 Removed thinking section from DeepSeek-R1 response")

        # Remove ```json and ``` if present
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]

        # Remove any text before the first { or [
        first_brace = cleaned.find('{')
        first_bracket = cleaned.find('[')

        # Prefer object over array
        if first_brace >= 0 and (first_bracket < 0 or first_brace < first_bracket):
            cleaned = cleaned[first_brace:]
            # Remove any text after the last }
            last_brace = cleaned.rfind('}')
            if last_brace >= 0:
                cleaned = cleaned[:last_brace + 1]
        elif first_bracket >= 0:
            # Handle array case (but we want to avoid this)
            cleaned = cleaned[first_bracket:]
            last_bracket = cleaned.rfind(']')
            if last_bracket >= 0:
                cleaned = cleaned[:last_bracket + 1]
            logger.warning(
                f"⚠️ Model returned an array instead of object: {cleaned[:100]}...")

        return cleaned.strip()

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from DeepSeek-R1 response, handling thinking tags and extra text."""
        cleaned = response_text.strip()

        # Handle DeepSeek-R1 thinking tags
        if '<think>' in cleaned and '</think>' in cleaned:
            think_start = cleaned.find('<think>')
            think_end = cleaned.find('</think>')
            if think_start >= 0 and think_end >= 0:
                # Remove the thinking section
                cleaned = cleaned[:think_start] + cleaned[think_end + 8:]
                logger.info(
                    "🧠 Removed thinking section from DeepSeek-R1 response")

        # Remove markdown code blocks
        if '```json' in cleaned:
            start = cleaned.find('```json')
            end = cleaned.find('```', start + 7)
            if start >= 0 and end >= 0:
                cleaned = cleaned[start + 7:end].strip()
        elif '```' in cleaned:
            start = cleaned.find('```')
            end = cleaned.find('```', start + 3)
            if start >= 0 and end >= 0:
                cleaned = cleaned[start + 3:end].strip()

        # Find JSON object boundaries
        first_brace = cleaned.find('{')
        if first_brace >= 0:
            # Count braces to find matching closing brace
            brace_count = 0
            for i, char in enumerate(cleaned[first_brace:], first_brace):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found complete JSON object
                        return cleaned[first_brace:i + 1]

        # Fallback to original cleaning method
        return self._clean_json_response(response_text)

    def _validate_and_fix_json(self, analysis_dict: Dict[str, str], filename: str) -> Dict[str, str]:
        """Validate that JSON has correct keys and fix if needed."""
        # Get the expected Excel column names
        column_mapping = self._get_existing_column_mapping()
        # Use Excel column names, not internal keys
        expected_keys = set(column_mapping.values())
        actual_keys = set(analysis_dict.keys())

        if expected_keys == actual_keys:
            logger.info(f"✅ JSON structure is correct for {filename}")
            return analysis_dict

        logger.warning(f"⚠️ JSON structure mismatch for {filename}")
        logger.warning(f"Expected keys: {sorted(expected_keys)}")
        logger.warning(f"Actual keys: {sorted(actual_keys)}")

        # Create corrected structure using Excel column names
        corrected = {}
        for excel_col in expected_keys:
            if excel_col in analysis_dict:
                corrected[excel_col] = analysis_dict[excel_col]
            else:
                corrected[excel_col] = "Não mencionado"

        # Try to map some common wrong keys to correct Excel column names
        key_mappings = {
            "codigo_entrevista": "Código Entrevista",
            "hospital": "Hospital",
            "nome_posicao_projetos": "Nome - posição institucional - Projetos",
            "area_atuacao": "Área de atuação"
        }

        for wrong_key, correct_excel_col in key_mappings.items():
            if wrong_key in analysis_dict and corrected[correct_excel_col] == "Não mencionado":
                corrected[correct_excel_col] = analysis_dict[wrong_key]

        logger.info(f"🔧 Fixed JSON structure for {filename}")
        return corrected

    def _process_all_interviews(self) -> List[Dict[str, str]]:
        """Process all interviews and save each one to Excel immediately."""
        # Use file search if enabled and available
        if self.analysis_method == "file_search":
            return self._analyze_interviews_with_file_search()

        # Fallback to original chunking method for other analysis methods
        interview_files = self._get_interview_files()

        if not interview_files:
            logger.warning("No interview files found to process")
            return []

        # Get already processed interviews to avoid duplicates
        processed_interviews = self._get_processed_interviews()

        results = []
        processed_count = 0

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

            # Analyze with AI (using traditional methods - not file search)
            analysis = self._analyze_interview_with_ai(content, file_path.name)

            # Add metadata
            result = {
                "interview_file": file_path.name,
                "file_size_chars": len(content),
                **analysis
            }

            # Save this single interview to Excel immediately
            try:
                self._export_to_excel([result])
                processed_count += 1
                logger.info(
                    f"✅ Saved {file_path.name} to Excel ({processed_count} processed)")
            except Exception as e:
                logger.error(
                    f"❌ Failed to save {file_path.name} to Excel: {e}")
                # Continue processing other interviews even if one fails to save

            results.append(result)

            # Reload existing data to get updated processed list for next iteration
            self.existing_df = self._load_existing_data()

        logger.info(
            f"Processed and saved {processed_count} new interviews successfully")
        return results

    def _analyze_interview_with_ai(self, interview_content: str, filename: str) -> Dict[str, str]:
        """Analyze interview content using traditional AI methods (non-file-search)."""
        try:
            # Create a simple analysis prompt for traditional methods
            prompt = f"""Você é um especialista em análise de entrevistas PROADI-SUS. Analise esta entrevista e extraia informações específicas.

TEXTO DA ENTREVISTA:
{interview_content}

INSTRUÇÕES:
1. Para cada categoria, encontre e cite EXATAMENTE o que está escrito na entrevista
2. Copie as falas diretas dos entrevistados, sem interpretações
3. Se uma informação não estiver presente, escreva "Não encontrado"
4. Mantenha conversações completas quando relevantes

Responda em formato JSON válido com as chaves exatas:

{{
  "Código Entrevista": "informação encontrada ou Não encontrado",
  "Área de atuação": "informação encontrada ou Não encontrado",
  "Hospital": "informação encontrada ou Não encontrado",
  "Nome - posição institucional - Projetos": "informação encontrada ou Não encontrado",
  "Modelos para planos de trabalho e prestação de contas": "informação encontrada ou Não encontrado",
  "Avaliação geral Proadi e DesenvoIvimento Institucional": "informação encontrada ou Não encontrado",
  "Relação Conass/Conasems/MS com HE e instituições parceiras": "informação encontrada ou Não encontrado",
  "Benefícios para instituição parceira": "informação encontrada ou Não encontrado",
  "Desafios para a participação do HE no Proadi": "informação encontrada ou Não encontrado",
  "Sugestões": "informação encontrada ou Não encontrado",
  "Origem dos projetos (quem demandou, tramitação e negociações)": "informação encontrada ou Não encontrado",
  "Projetos colaborativos (participação de cada um, relacionamento HE e benefícios e desafios)": "informação encontrada ou Não encontrado",
  "Expertise do hospital para o projeto e Inserção deste no HE": "informação encontrada ou Não encontrado",
  "Abrangência Territorial do Projeto (definição)": "informação encontrada ou Não encontrado",
  "Seleção e envolvimento instituições participantes no projeto": "informação encontrada ou Não encontrado",
  "Avaliações sobre o Projeto": "informação encontrada ou Não encontrado",
  "Monitoramento (HE e instituições participantes) e Indicadores": "informação encontrada ou Não encontrado",
  "Riscos na implementação/dificuldades enfrentadas (adesão instituições ou profissionais, infraestrutura, outras)": "informação encontrada ou Não encontrado",
  "Benefícios do projeto para o SUS": "informação encontrada ou Não encontrado",
  "Incorporação de bens materiais ao SUS?": "informação encontrada ou Não encontrado",
  "Treinamento para profissionais?": "informação encontrada ou Não encontrado",
  "Publicações ou divulgação?": "informação encontrada ou Não encontrado",
  "Incorporação resultados ao SUS": "informação encontrada ou Não encontrado",
  "Longevidade e sustentabilidade possível?": "informação encontrada ou Não encontrado"
}}"""

            # Call the appropriate AI service
            if self.use_gemini:
                model = self.client.GenerativeModel(self.model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": self.max_output_tokens
                    }
                )
                analysis_text = response.text.strip()
            elif self.use_gpt:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Você extrai informações de entrevistas. Copie trechos relevantes do texto original, não resuma ou interprete. Retorne sempre JSON válido."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=self.max_output_tokens
                )
                analysis_text = response.choices[0].message.content.strip()
            else:
                # Use Ollama
                response = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "Você extrai informações de entrevistas. Copie trechos relevantes do texto original, não resuma ou interprete. Retorne sempre JSON válido."},
                        {"role": "user", "content": prompt}
                    ],
                    options={
                        "temperature": 0.1,
                        "num_predict": self.max_output_tokens
                    }
                )
                analysis_text = response['message']['content'].strip()

            # Parse the JSON response
            try:
                cleaned_json = self._extract_json_from_response(analysis_text)
                analysis_dict = json.loads(cleaned_json)
                analysis_dict = self._validate_and_fix_json(
                    analysis_dict, filename)
                logger.info(f"Successfully analyzed {filename}")
                return analysis_dict
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for {filename}: {e}")
                logger.error(f"Response content: {analysis_text[:200]}...")
                # Return fallback structure
                return self._create_fallback_analysis(filename)

        except Exception as e:
            logger.error(f"Error analyzing {filename} with AI: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _create_fallback_analysis(self, filename: str) -> Dict[str, str]:
        """Create a fallback analysis structure when AI analysis fails."""
        logger.warning(f"🔄 Creating fallback analysis for {filename}")

        # Get Excel column names
        column_mapping = self._get_existing_column_mapping()
        fallback = {}

        for excel_col in column_mapping.values():
            fallback[excel_col] = "Erro na análise - não foi possível extrair"

        return fallback

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
            # Process all interviews (each one is saved to Excel immediately)
            results = self._process_all_interviews()

            if not results:
                logger.error("No interviews were processed successfully")
                return

            logger.info("Interview analysis completed successfully!")
            logger.info(f"All results have been saved to {self.output_file}")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def main():
    """Main function to run the interview analysis."""
    print("🎯 Interview Analysis Tool")
    print("=" * 50)

    try:
        # Check environment variables for AI service selection
        use_file_search = os.getenv(
            "USE_FILE_SEARCH", "true").lower() == "true"
        use_gemini = os.getenv("USE_GEMINI", "false").lower() == "true"
        use_gpt = os.getenv("USE_GPT", "false").lower() == "true"

        # Display analysis method being used
        if use_file_search and os.getenv("OPENAI_API_KEY"):
            print("🔍 Using OpenAI Assistant API with File Search")
            print(
                "   This provides the most comprehensive analysis with semantic search capabilities")
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ Error: OPENAI_API_KEY environment variable not set")
                print("Please set your OpenAI API key:")
                print("export OPENAI_API_KEY='your-api-key-here'")
                print("Or create a .env file with: OPENAI_API_KEY=your-api-key-here")
                print("\nAlternatively, set USE_FILE_SEARCH=false to use other methods")
                sys.exit(1)
        elif use_gemini:
            print("🤖 Using Google Gemini")
            if not os.getenv("GEMINI_API_KEY"):
                print("❌ Error: GEMINI_API_KEY environment variable not set")
                print("Please set your Gemini API key:")
                print("export GEMINI_API_KEY='your-api-key-here'")
                print("Or create a .env file with: GEMINI_API_KEY=your-api-key-here")
                print("\nAlternatively, set USE_GEMINI=false to use OpenAI or Ollama")
                sys.exit(1)
        elif use_gpt:
            print("🤖 Using OpenAI GPT (traditional chunking)")
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ Error: OPENAI_API_KEY environment variable not set")
                print("Please set your OpenAI API key:")
                print("export OPENAI_API_KEY='your-api-key-here'")
                print("Or create a .env file with: OPENAI_API_KEY=your-api-key-here")
                print(
                    "\nAlternatively, set USE_GPT=false to use Ollama, or USE_GEMINI=true to use Gemini")
                sys.exit(1)
        else:
            print("🦙 Using Ollama with DeepSeek-R1")
            print("   Make sure Ollama is running with: ollama serve")
            print("   And the model is available: ollama pull deepseek-r1")

        # Display configuration options
        print("\n📋 Configuration Options:")
        print(
            f"   USE_FILE_SEARCH={use_file_search} (OpenAI Assistant API with File Search)")
        print(f"   USE_GEMINI={use_gemini} (Google Gemini)")
        print(f"   USE_GPT={use_gpt} (OpenAI GPT traditional)")
        print(
            f"   USE_EMBEDDINGS={os.getenv('USE_EMBEDDINGS', 'false')} (Embeddings-based analysis)")
        print("\n🚀 Starting analysis...")

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
