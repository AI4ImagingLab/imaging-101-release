# DeepCode Framework — Imaging-101 Evaluation

Integration with [HKUDS/DeepCode](https://github.com/HKUDS/DeepCode) — an open-source
autonomous multi-agent coding framework.

## Architecture

DeepCode uses a **self-orchestrating multi-agent system** with:

```
User prompt / Paper
       ↓
┌──────────────────────────────────────────────────────┐
│              DeepCode Multi-Agent System               │
│                                                        │
│  🎯 Central Orchestrating Agent                       │
│       ↓                                                │
│  📝 Intent Understanding Agent (NLP → specs)          │
│  📄 Document Parsing Agent (paper → algorithms)       │
│       ↓                                                │
│  🏗️ Code Planning Agent (architecture + design)       │
│  🔍 Code Reference Mining Agent (CodeRAG)             │
│  📚 Code Indexing Agent (knowledge graph)             │
│       ↓                                                │
│  🧬 Code Generation Agent (implementation + tests)    │
│       ↓                                                │
│  ⚡ Execution & Validation (auto-debug loop)          │
└──────────────────────────────────────────────────────┘
       ↓
output/reconstruction.npy
```

## Prerequisites

1. **Python 3.9+**, **Node.js 18+**, **npm 8+**
2. Install DeepCode:
   ```bash
   pip install deepcode-hku
   ```
3. Download config files:
   ```bash
   curl -O https://raw.githubusercontent.com/HKUDS/DeepCode/main/mcp_agent.config.yaml
   curl -O https://raw.githubusercontent.com/HKUDS/DeepCode/main/mcp_agent.secrets.yaml
   ```
4. Edit `mcp_agent.secrets.yaml` with your API keys:
   ```yaml
   openai:
     api_key: "your_openai_api_key"
   anthropic:
     api_key: "your_anthropic_api_key"
   google:
     api_key: "your_google_api_key"
   ```

## How It Integrates

DeepCode is a **third-party agent** similar to Claude Code, but instead of
manual prompting, it can be driven via its CLI interface. The integration
follows the same sandbox pattern as the `claude_code` framework:

1. **Prepare** sandbox workspace with visible files
2. **Generate** a prompt describing the imaging reconstruction task
3. **Launch** DeepCode CLI with the prompt and workspace path
4. **Collect** results after DeepCode finishes

## Usage

### Automated (via evaluation harness):
```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode end_to_end \
    --framework deepcode \
    --level L1 \
    --model gemini-2.5-pro
```

### Manual (standalone DeepCode):
```bash
# 1. Prepare workspace
python -m evaluation_harness prepare \
    --task eht_black_hole_original --level L1

# 2. Run DeepCode CLI on the workspace
cd tmp_L1/
deepcode --cli

# 3. Collect & score
python -m evaluation_harness collect \
    --task eht_black_hole_original \
    --workspace-dir ./tmp_L1 \
    --level L1 --agent-name deepcode
```

### Docker mode:
```bash
# Clone DeepCode repo for Docker support
git clone https://github.com/HKUDS/DeepCode.git
cd DeepCode/
cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml
# Edit secrets with API keys
./deepcode_docker/run_docker.sh
# Access at http://localhost:8000
```

## Configuration

See `deepcode_config.yaml` in this directory for imaging-101 specific settings.

## Comparison with Other Frameworks

| Feature          | ReAct      | Multi-Agent  | Claude Code | DeepCode     |
|------------------|------------|--------------|-------------|--------------|
| Agent type       | Single     | Pipeline     | Black-box   | Autonomous   |
| Planning         | Implicit   | Explicit     | Agent-driven| Hierarchical |
| Code search      | No         | No           | Agent IDE   | CodeRAG      |
| Auto-debug       | Yes (loop) | Yes (Judge)  | Agent IDE   | Yes (loop)   |
| Requires API     | Yes        | Yes          | No (manual) | Yes          |
| MCP integration  | No         | No           | No          | Yes          |
