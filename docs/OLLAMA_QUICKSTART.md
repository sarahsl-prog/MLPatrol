# Ollama Quickstart Guide - Local LLM for MLPatrol

This guide will help you set up MLPatrol with Ollama for **100% private, local AI** - no cloud API required!

---

## Why Use Ollama?

✅ **Privacy:** All data stays on your machine - perfect for security workloads
✅ **No API Costs:** No per-token charges or monthly fees
✅ **No Rate Limits:** Use as much as you want
✅ **Offline Capable:** Works without internet (after initial model download)
✅ **Easy to Use:** Simple setup, just like Docker for LLMs

---

## Step 1: Install Ollama

### Linux / macOS
```bash
curl https://ollama.ai/install.sh | sh
```

### Windows
1. Download installer from [https://ollama.ai/download](https://ollama.ai/download)
2. Run the installer
3. Ollama will start automatically in the background

### Verify Installation
```bash
ollama --version
```

You should see something like: `ollama version is 0.x.x`

---

## Step 2: Pull a Model

Choose a model based on your hardware:

### Recommended: Llama 3.1 8B (Fast, Good Quality)
```bash
ollama pull llama3.1:8b
```
- **Size:** 4.7GB download
- **RAM Required:** 8GB minimum
- **Best For:** Daily use, fast responses
- **Quality:** Good for most security analysis tasks

### High Accuracy: Llama 3.1 70B
```bash
ollama pull llama3.1:70b
```
- **Size:** 40GB download
- **RAM Required:** 48GB minimum (or GPU with 48GB VRAM)
- **Best For:** Complex analysis, maximum accuracy
- **Quality:** Excellent, comparable to cloud models

### Balanced: Mistral Small 3.1
```bash
ollama pull mistral-small:3.1
```
- **Size:** 13GB download
- **RAM Required:** 16GB minimum
- **Best For:** Balanced performance/quality
- **Quality:** Very good for security reasoning

### Code-Focused: Qwen 2.5 14B
```bash
ollama pull qwen2.5:14b
```
- **Size:** 8.5GB download
- **RAM Required:** 12GB minimum
- **Best For:** Code generation and analysis
- **Quality:** Excellent for code-related tasks

### Verify Model Downloaded
```bash
ollama list
```

You should see your model listed.

---

## Step 3: Test Ollama

```bash
ollama run llama3.1:8b
```

This will open an interactive chat. Try:
```
>>> What is data poisoning in machine learning?
```

Press `Ctrl+D` or type `/bye` to exit.

If this works, Ollama is ready!

---

## Step 4: Install MLPatrol

```bash
# Clone the repository
git clone https://github.com/yourusername/mlpatrol
cd mlpatrol

# Install dependencies
pip install -r requirements.txt
```

**Note:** The updated `requirements.txt` includes `langchain-ollama>=0.2.0` for local LLM support.

---

## Step 5: Configure MLPatrol for Ollama

### Option A: Using .env file (Recommended)

```bash
# Copy example environment file
cp .env.example .env
```

Edit `.env` and set:
```bash
# Enable local LLM
USE_LOCAL_LLM=true

# Specify your Ollama model
LOCAL_LLM_MODEL=ollama/llama3.1:8b

# Ollama server URL (default is localhost)
LOCAL_LLM_URL=http://localhost:11434
```

### Option B: Using Environment Variables

```bash
export USE_LOCAL_LLM=true
export LOCAL_LLM_MODEL=ollama/llama3.1:8b
export LOCAL_LLM_URL=http://localhost:11434
```

**Windows (PowerShell):**
```powershell
$env:USE_LOCAL_LLM="true"
$env:LOCAL_LLM_MODEL="ollama/llama3.1:8b"
$env:LOCAL_LLM_URL="http://localhost:11434"
```

---

## Step 6: Run MLPatrol

```bash
python app.py
```

You should see:
```
Initializing MLPatrol agent...
Using local LLM: ollama/llama3.1:8b
Initialized Ollama LLM: llama3.1:8b at http://localhost:11434
Agent initialized successfully with local model: ollama/llama3.1:8b
Running on local URL:  http://127.0.0.1:7860
```

Visit **http://localhost:7860** in your browser!

---

## Step 7: Test MLPatrol with Ollama

Try these queries in the MLPatrol UI:

### CVE Search
```
Check for recent vulnerabilities in numpy
```

### Dataset Analysis
Upload a CSV file and ask:
```
Analyze this dataset for potential poisoning or bias
```

### Security Chat
```
What are the top 3 security risks in machine learning systems?
```

---

## Troubleshooting

### Issue: "Ollama LLM: llama3.1:8b at http://localhost:11434" but errors occur

**Solution:** Make sure Ollama is running:
```bash
# Check if Ollama is running
ps aux | grep ollama  # Linux/Mac
# or
tasklist | findstr ollama  # Windows

# If not running, start it:
ollama serve
```

### Issue: "Model not found"

**Solution:** Pull the model first:
```bash
ollama pull llama3.1:8b
```

### Issue: Slow responses

**Solutions:**
1. Use a smaller model (llama3.1:8b instead of :70b)
2. Ensure Ollama is using your GPU (check with `ollama ps`)
3. Increase `num_ctx` in the code if context is being truncated

### Issue: Out of memory

**Solutions:**
1. Use a smaller model
2. Close other applications
3. For 70B models, ensure you have 48GB+ RAM or GPU VRAM

### Issue: Connection refused

**Solution:** Ollama server not running. Start it:
```bash
ollama serve
```

---

## Switching Between Models

You can easily switch models without changing code:

### In .env file:
```bash
# Switch to a different model
LOCAL_LLM_MODEL=ollama/mistral-small:3.1
```

Restart MLPatrol, and it will use the new model!

### Pull additional models anytime:
```bash
ollama pull llama3.1:70b
ollama pull qwen2.5:14b
ollama pull deepseek-r1:7b
```

---

## Performance Comparison

Based on testing with MLPatrol:

| Model | CVE Search | Dataset Analysis | Code Generation | RAM |
|-------|-----------|------------------|-----------------|-----|
| llama3.1:8b | ~5-7s | ~12-15s | ~8-10s | 8GB |
| llama3.1:70b | ~15-20s | ~25-30s | ~18-22s | 48GB |
| mistral-small:3.1 | ~8-10s | ~15-18s | ~10-12s | 16GB |
| qwen2.5:14b | ~7-9s | ~14-17s | ~9-11s | 12GB |

**Cloud (Claude Sonnet 4):** ~3s / ~8s / ~5s (for comparison)

---

## Advanced Configuration

### Custom Ollama Server URL

If running Ollama on a different machine:
```bash
LOCAL_LLM_URL=http://192.168.1.100:11434
```

### Multiple GPUs

Ollama automatically uses available GPUs. Check usage:
```bash
ollama ps
```

### Temperature and Context

MLPatrol uses these settings for security analysis:
- **Temperature:** 0.1 (low, for factual accuracy)
- **Context Window:** 4096 tokens

These are set in `src/agent/reasoning_chain.py` and optimized for security workloads.

---

## Privacy & Security Notes

### What Stays Local
- ✅ All queries
- ✅ All responses
- ✅ Dataset uploads
- ✅ Generated code
- ✅ Analysis results

### What Goes to the Internet
- ❌ Nothing! (Ollama runs 100% locally)
- Exception: If you use CVE search, that queries public CVE databases (not your data, just CVE IDs)

### Best Practices
1. **Firewall:** Ollama listens on localhost by default - keep it that way
2. **Model Source:** Only download models from official Ollama registry
3. **Updates:** Keep Ollama updated for security patches
4. **Logging:** MLPatrol logs are local only

---

## Hybrid Setup (Cloud + Local)

You can switch between cloud and local:

### Development (Local)
```bash
USE_LOCAL_LLM=true
LOCAL_LLM_MODEL=ollama/llama3.1:8b
```

### Production (Cloud)
```bash
USE_LOCAL_LLM=false
ANTHROPIC_API_KEY=sk-ant-...
```

Just change the `.env` file and restart!

---

## FAQ

### Q: Which model should I use?
**A:** Start with `llama3.1:8b` - it's fast and good quality. Upgrade to `70b` if you need better accuracy.

### Q: Can I use Ollama on a laptop?
**A:** Yes! 8B models run well on modern laptops with 16GB RAM. Apple Silicon (M1/M2) is excellent for this.

### Q: How much does Ollama cost?
**A:** Free! It's open source. You only pay for your hardware/electricity.

### Q: Is it slower than cloud APIs?
**A:** Yes, typically 2-5x slower depending on hardware. But you get privacy and no API costs.

### Q: Can I run this on CPU only?
**A:** Yes, but it will be slow. GPUs are highly recommended for good performance.

### Q: Does this work offline?
**A:** Yes, after models are downloaded. CVE search requires internet, but dataset analysis works offline.

### Q: Can I use other models?
**A:** Yes! Any Ollama model will work. Just update `LOCAL_LLM_MODEL` in `.env`.

---

## Getting Help

### Ollama Resources
- **Website:** https://ollama.ai
- **Docs:** https://github.com/ollama/ollama
- **Discord:** https://discord.gg/ollama

### MLPatrol Support
- **Issues:** https://github.com/yourusername/mlpatrol/issues
- **Docs:** See `docs/` directory

---

## Summary

You now have MLPatrol running with a local LLM! 🎉

**What you achieved:**
✅ Private, local AI for security analysis
✅ No API costs or rate limits
✅ Full control over your data
✅ Same features as cloud version

**Next steps:**
- Try analyzing a dataset
- Search for CVEs in your ML stack
- Generate security validation code
- Experiment with different models

Enjoy privacy-first AI security! 🛡️
