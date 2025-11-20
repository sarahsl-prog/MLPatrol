# MLPatrol 🛡️

> An AI-powered security agent that proactively defends ML systems by monitoring vulnerabilities, analyzing datasets, and generating security validation code.

<!-- Add demo GIF or screenshot here once you have the UI -->
![MLPatrol Demo](assets/images/demo.gif)

**🏆 Built for MCP's 1st Birthday Hackathon - Track 2: Agent Apps (Productivity)**

[![Demo Video](https://img.shields.io/badge/▶️-Watch_Demo-red?style=for-the-badge)](DEMO_VIDEO_LINK)
[![Social Post](https://img.shields.io/badge/📱-Social_Post-blue?style=for-the-badge)](SOCIAL_MEDIA_LINK)
[![Try It](https://img.shields.io/badge/🚀-Try_MLPatrol-green?style=for-the-badge)](HUGGINGFACE_SPACE_LINK)

---

## 🎯 The Problem

Machine learning practitioners face an avalanche of security threats:
- **New CVEs** emerge daily in ML libraries (PyTorch, TensorFlow, scikit-learn)
- **Dataset poisoning** and **bias** can silently corrupt models
- **Manual security checks** are time-consuming and error-prone
- Most ML developers lack cybersecurity expertise

Current tools either focus narrowly on one aspect or require deep security knowledge. **MLPatrol bridges this gap.**

## 💡 The Solution

MLPatrol is an intelligent agent that acts as your personal ML security researcher. It:

🔍 **Monitors Threats** - Automatically searches for latest CVEs in ML libraries  
🧠 **Reasons Through Risks** - Uses multi-step agent reasoning to analyze exploit mechanisms  
📊 **Analyzes Datasets** - Detects poisoning, bias, and statistical anomalies  
💻 **Generates Code** - Creates runnable security validation scripts  
📝 **Explains Everything** - Shows its reasoning process transparently

### Why MLPatrol?

- **Proactive, not reactive** - Catches threats before they impact your models
- **Educational** - Teaches security concepts while protecting your work
- **Automated** - Saves hours of manual security research
- **Practical** - Generates code you can run immediately
- **Agent-driven** - Uses advanced reasoning, not just keyword matching

---

## ✨ Features

### 1. CVE Threat Intelligence 🔐

MLPatrol monitors vulnerability databases for threats targeting ML libraries.

**Example Query:**
```
"Check for recent vulnerabilities in scikit-learn"
```

**Agent Reasoning:**
```
Step 1: Search CVE databases for sklearn (last 30 days)
Step 2: Found CVE-2024-XXXX - pickle deserialization vulnerability
Step 3: Analyze: Affects versions 1.2.0-1.3.2, CVSS score 8.1 (High)
Step 4: Generate validation code to check environment
Step 5: Provide remediation steps
```

**Output:**
- Vulnerability analysis with severity ratings
- Python script to check if you're affected
- Upgrade recommendations and mitigation strategies

### 2. Dataset Security Analysis 📊

Detect poisoning attempts, bias, and data quality issues.

**Example Query:**
```
"Analyze this dataset for poisoning and bias"
```

**Agent Reasoning:**
```
Step 1: Load and profile dataset
Step 2: Statistical analysis - detect outliers (Z-score > 3)
Step 3: Class distribution check - identify imbalances
Step 4: Correlation analysis - find suspicious patterns
Step 5: Bias detection across protected attributes
Step 6: Generate comprehensive report
```

**Output:**
- Statistical anomaly detection
- Bias analysis across features
- Visual reports with recommendations
- Data quality score

### 3. Automated Security Code Generation 💻

Stop writing security checks from scratch - let MLPatrol generate them.

**Generated Code Example:**
```python
# Auto-generated security check for CVE-2024-XXXX
import sklearn
import subprocess

def check_sklearn_vulnerability():
    """Check for pickle deserialization vulnerability in scikit-learn"""
    version = sklearn.__version__
    vulnerable_versions = ["1.2.0", "1.2.1", "1.3.0", "1.3.1", "1.3.2"]
    
    if version in vulnerable_versions:
        print(f"⚠️  VULNERABLE: sklearn {version}")
        print("Recommendation: Upgrade to sklearn >= 1.3.3")
        return False
    else:
        print(f"✅ SAFE: sklearn {version}")
        return True

if __name__ == "__main__":
    check_sklearn_vulnerability()
```

### 4. Multi-Step Agent Reasoning 🤖

Unlike simple search tools, MLPatrol actively *thinks* through problems:

- **Contextual understanding** - Knows your ML stack and priorities
- **Tool orchestration** - Combines web search, HuggingFace, and custom analysis
- **Adaptive planning** - Adjusts strategy based on findings
- **Transparent reasoning** - Shows every step of its thought process

---

## 🏗️ Architecture
```
┌─────────────┐
│   User      │
│  Interface  │
│  (Gradio 6) │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│     Agent Reasoning Engine      │
│  (LangChain + Claude/GPT)       │
│                                 │
│  • Query Analysis               │
│  • Multi-step Planning          │
│  • Tool Selection               │
│  • Result Synthesis             │
└──────┬──────────────────────────┘
       │
       ├─────────────┬─────────────┬─────────────┐
       ▼             ▼             ▼             ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   CVE    │  │ Dataset  │  │   Code   │  │   MCP    │
│ Monitor  │  │ Analysis │  │Generator │  │  Tools   │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
     │             │             │             │
     ▼             ▼             ▼             ▼
  NVD API    NumPy/Pandas    AST/Jinja2   Web Search
  Security   Statistical     Templates    HF Datasets
  Papers     Tests                        Notion
```

**Tech Stack:**
- **Frontend:** Gradio 5+ (mobile-responsive UI)
- **Agent:** LangGraph 1.0+ for modern agentic reasoning
- **LLM:** Claude Sonnet 4 / GPT-4 / Ollama (local) for analysis
- **Validation:** Pydantic v2 for robust data validation
- **Analysis:** NumPy, Pandas, scikit-learn
- **MCP:** Web search, HuggingFace datasets, Notion
- **APIs:** NVD (National Vulnerability Database)

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+ (3.11+ recommended for best performance)
pip or conda
```

**LLM Options:**
- **Cloud:** API keys for Claude/OpenAI (recommended for best quality)
- **Local:** Ollama (100% private, no API costs, runs on your hardware)

### Installation

#### Option 1: Cloud LLMs (Claude/GPT-4)

```bash
# Clone the repository
git clone https://huggingface.co/spaces/MCP-1st-Birthday/mlpatrol
cd mlpatrol

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY
```

#### Option 2: Local LLM (Ollama) - 🆕 Privacy-First!

```bash
# 1. Install Ollama
# Linux/Mac:
curl https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download

# 2. Pull a model (choose one)
ollama pull llama3.1:8b      # Recommended: Fast, 8GB RAM
# OR
ollama pull llama3.1:70b     # High accuracy, 48GB RAM
# OR
ollama pull mistral-small:3.1  # Balanced, 16GB RAM

# 3. Clone and install MLPatrol
git clone https://huggingface.co/spaces/MCP-1st-Birthday/mlpatrol
cd mlpatrol
pip install -r requirements.txt

# 4. Configure for local LLM
cp .env.example .env
# Edit .env and set:
#   USE_LOCAL_LLM=true
#   LOCAL_LLM_MODEL=ollama/llama3.1:8b
```

#### Option 3: Enable Web Search (Optional but Recommended) - 🆕

MLPatrol supports real-time web search for security research via Tavily AI and Brave Search.

```bash
# 1. Get API keys (choose one or both)
#    - Tavily AI: https://tavily.com (1,000 free credits/month)
#    - Brave Search: https://brave.com/search/api/ (2,000 free queries/month)

# 2. Edit .env and add:
ENABLE_WEB_SEARCH=true
USE_TAVILY_SEARCH=true    # AI-optimized search
USE_BRAVE_SEARCH=true     # Privacy-focused search
TAVILY_API_KEY=your_key_here
BRAVE_API_KEY=your_key_here

# 3. Restart MLPatrol
```

**Smart Routing:**
- CVE monitoring queries → Brave (breaking news, real-time)
- General security Q&A → Tavily (AI-optimized summaries)

See [Web Search Setup Guide](docs/WEB_SEARCH_SETUP.md) for detailed configuration.

### Run Locally
```bash
python app.py
```

Visit `http://localhost:7860` to use MLPatrol.

> **💡 Tip:** Local LLMs provide 100% privacy (no data leaves your machine), no API costs, and no rate limits. Perfect for security-sensitive workloads!

### Usage Examples

**Check for CVEs:**
```
"What are the latest security vulnerabilities in PyTorch?"
```

**Analyze a dataset:**
```
"Check this dataset for bias and poisoning attempts"
[Upload your CSV/dataset]
```

**Generate security checks:**
```
"Create a script to validate my numpy installation against recent CVEs"
```

---

## 📖 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System design and components
- **[Agent Reasoning](docs/AGENT_REASONING.md)** - How MLPatrol thinks
- **[Development Guide](docs/DEVELOPMENT.md)** - Setup and contribution guidelines
- **[Demo Script](docs/DEMO_GUIDE.md)** - Demo video walkthrough

---

## 🎥 Demo Video

[Watch the full 5-minute demo](DEMO_VIDEO_LINK)

**Timestamps:**
- 0:00 - Introduction & Problem Statement
- 0:30 - CVE Monitoring Demo
- 1:30 - Dataset Analysis Demo
- 2:30 - Code Generation Demo
- 3:30 - Agent Reasoning Walkthrough
- 4:00 - Architecture Overview
- 4:30 - Future Vision & Wrap-up

---

## 🔮 Future Roadmap

MLPatrol is designed for extensibility beyond the hackathon:

- **GitHub Integration** - Scan repositories for ML security issues
- **CI/CD Pipeline** - Automated security checks in deployment
- **Custom Rules** - User-defined security policies
- **Historical Tracking** - Monitor threat evolution over time
- **Team Collaboration** - Share findings with security teams
- **Expanded Coverage** - More ML frameworks (JAX, MXNet, etc.)

---

## 🏆 Hackathon Submission

**Track:** Agent Apps - Productivity Category  
**Tags:** `agent-app-track-productivity`

**Required Links:**
- 📱 **Social Media Post:** [Twitter/LinkedIn/X](SOCIAL_MEDIA_LINK)
- 🎥 **Demo Video:** [YouTube/Loom](DEMO_VIDEO_LINK)
- 🚀 **Live Demo:** [HuggingFace Space](HUGGINGFACE_SPACE_LINK)

**Built with:**
- ✅ Gradio 6 for interface
- ✅ MCP tools for external integrations
- ✅ Multi-step agent reasoning
- ✅ Original work created Nov 15-30, 2025

---

## 🤝 Contributing

Contributions welcome! This project will continue beyond the hackathon.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MCP 1st Birthday Hackathon** - Anthropic & Gradio
- **Hackathon Sponsors** - For API credits and support
- **ML Security Community** - For threat intelligence resources

---

## 📬 Contact

**Project Maintainer:** [Sarah Sund-Lussier]  
**Email:** [ssundlussier@gmail.com]  


**Hackathon Links:**
- [Discord: #agents-mcp-hackathon-winter25](https://discord.gg/fveShqytyh)
- [HuggingFace Org](https://huggingface.co/MCP-1st-Birthday)

---

<div align="center">

**Built with ❤️ for the ML Security Community**

[⭐ Star this repo](REPO_LINK) | [🐛 Report Bug](ISSUES_LINK) | [💡 Request Feature](ISSUES_LINK)

</div>