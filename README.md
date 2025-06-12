# Interview Analysis Tool

Analyzes PROADI-SUS interviews using AI and exports categorized data to Excel.

## 🆕 New Features

### OpenAI Assistant API with File Search

The tool now supports OpenAI's Assistant API with file search capabilities, providing the most comprehensive analysis by:

- Uploading interview files to OpenAI's vector storage
- Using semantic search to find relevant content for each category
- Analyzing complete interviews without chunking limitations
- Providing more accurate and comprehensive results

## 🚀 Quick Start

1. **Set up environment variables** (create a `.env` file):

```bash
# Required for file search (RECOMMENDED method)
OPENAI_API_KEY=your-openai-api-key-here
USE_FILE_SEARCH=true

# Alternative configurations
USE_GEMINI=false
USE_GPT=false
USE_EMBEDDINGS=false
```

2. **Install dependencies**:

```bash
pip install openai pandas openpyxl python-docx docx2txt google-generativeai ollama sentence-transformers python-dotenv
```

3. **Prepare your interviews**:

   - Place interview files in the `entrevistas/` folder
   - Supported formats: `.txt`, `.md`, `.doc`, `.docx`

4. **Run the analysis**:

```bash
python app/main.py
```

## 📋 Configuration Options

### Analysis Methods (in order of recommendation)

#### 🔍 File Search (RECOMMENDED)

Uses OpenAI Assistant API with file search for comprehensive analysis:

```bash
USE_FILE_SEARCH=true
OPENAI_API_KEY=your-key-here
```

**Benefits:**

- No chunking limitations
- Semantic search across entire interviews
- Most accurate results
- Handles large interviews efficiently

#### 🤖 Google Gemini

Uses Google's Gemini AI model:

```bash
USE_GEMINI=true
GEMINI_API_KEY=your-key-here
USE_FILE_SEARCH=false
```

#### 🧠 OpenAI GPT (Traditional)

Uses OpenAI GPT with chunking:

```bash
USE_GPT=true
OPENAI_API_KEY=your-key-here
USE_FILE_SEARCH=false
```

#### 🦙 Local Ollama

Uses local Ollama with DeepSeek-R1:

```bash
# All other options set to false
# Requires: ollama serve && ollama pull deepseek-r1
```

### Advanced Options

#### Embeddings-based Analysis

```bash
USE_EMBEDDINGS=true
```

Experimental feature that uses embeddings for semantic content matching.

## 📊 Output

The tool generates `Entrevistas Hospitais_Rosi.xlsx` with extracted information categorized into:

- Código da entrevista
- Área de atuação
- Hospital
- Nome e posição do entrevistado
- Modelos para planos de trabalho
- Avaliação geral do PROADI-SUS
- Relação CONASS/CONASEMS/MS
- Benefícios para instituição parceira
- Desafios para participação
- Sugestões de melhoria
- And 14 more categories...

## 🔧 Troubleshooting

### File Search Issues

- Ensure you have a valid OpenAI API key with Assistant API access
- Check that interview files are in supported formats
- Verify the `entrevistas/` folder exists and contains interview files

### API Rate Limits

- The tool includes automatic rate limiting and retry logic
- For large numbers of interviews, consider processing in smaller batches

### Memory Issues

- File search method is most memory efficient
- For local methods, reduce `max_tokens_per_request` if needed

## 🧹 Resource Cleanup

When using file search, the tool automatically:

- Deletes uploaded files from OpenAI after analysis
- Removes created assistants and vector stores
- Cleans up temporary files

## 📝 Logs

Check `interview_analysis.log` for detailed processing information and any errors.

## 🔒 Privacy & Security

- Interview files are temporarily uploaded to OpenAI when using file search
- Files are automatically deleted after analysis
- No data is permanently stored on OpenAI servers
- For maximum privacy, use local Ollama option

## ⚡ Performance Comparison

| Method          | Speed      | Accuracy   | Context    | Privacy    |
| --------------- | ---------- | ---------- | ---------- | ---------- |
| File Search     | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     |
| Gemini          | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   | ⭐⭐⭐⭐   | ⭐⭐⭐     |
| GPT Traditional | ⭐⭐⭐     | ⭐⭐⭐     | ⭐⭐⭐     | ⭐⭐⭐     |
| Local Ollama    | ⭐⭐       | ⭐⭐⭐     | ⭐⭐⭐     | ⭐⭐⭐⭐⭐ |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both analysis approaches
5. Submit a pull request

## 📄 License

[Add your license information here]

---

**Need help?** Check the logs in `interview_analysis.log` for detailed processing information.
