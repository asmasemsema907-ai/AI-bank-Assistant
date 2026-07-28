# AI Bank Assistant

Same lab sequence as the original reference lab, adapted to a banking Q&A
knowledge base (1,000 rows) instead of the course-support example docs.

```text
01_documents.py
02_preprocessing.py
03_chunking.py
04_vector_representation.py
05_create_chroma_store.py
06_retrieve_context.py
07_prompting.py
streamlit_app.py
```

Final retrieval:

```text
hybrid = 0.4 * BM25 + 0.6 * all-MiniLM-L6-v2 embeddings
```

## What changed vs. the original reference lab

- **`01_documents.py`** now loads the 1,000 documents from
  `banking_knowledge_base_1000.csv` instead of a hardcoded list (same
  `documents` interface either way — every other file is unaffected).
- **`05_create_chroma_store.py`**: the original hardcoded an absolute
  Windows path (`E:/Courses/.../chroma_db`), which only works on that one
  machine. Changed to a path relative to the script itself
  (`Path(__file__).parent / "chroma_db"`), so it works unchanged on any
  machine or deployment target.
- **`02_preprocessing.py`** now points NLTK at a WordNet corpus bundled
  directly in this project (`nltk_data/corpora/wordnet/`), so lemmatization
  works fully offline — no `nltk.download()` call needed at deploy time,
  which is a common source of slow/failed cold starts on Streamlit Cloud.
  Note: only the **WordNet** data was provided/bundled — the `stopwords`
  corpus and `punkt` tokenizer data were not, so those still rely on the
  existing safe fallbacks already in the code (a small built-in stopword
  list, and a regex-based tokenizer) if their NLTK data isn't present.
  Functionally fine either way; if you want the exact NLTK stopword list
  instead of the fallback, add `nltk_data/corpora/stopwords/` the same way.
- **`streamlit_app.py`** now calls `create_vector_store()` once on startup
  (cached via `st.cache_resource`), so the Chroma store builds itself
  automatically on first run — you don't need to SSH into the deployment
  and manually run `05_create_chroma_store.py` first.
- Renamed to **AI Bank Assistant** throughout (page title, chat title, LLM
  persona in the prompt).
- **Bilingual support (English + Arabic)**: the prompt in `07_prompting.py`
  instructs the LLM to answer in whichever language the question was asked
  in, and chat bubbles in `streamlit_app.py` use `unicode-bidi: plaintext`
  so Arabic renders correctly right-to-left per message automatically.
  **Retrieval itself is now genuinely multilingual too**: the embedding
  model in `04_vector_representation.py` was switched from the English-only
  `all-MiniLM-L6-v2` to
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which maps
  a sentence and its translation to nearby vectors across ~50 languages
  including Arabic — so Arabic questions get real semantic retrieval against
  the English knowledge base, not just matches for the terms in
  `ARABIC_TERM_BRIDGE`. That dictionary is still kept as an extra lexical
  boost for BM25 (which is purely literal-token matching and has no
  cross-lingual ability on its own), but it's no longer the only thing
  making Arabic work. Same 384-dim output, so nothing downstream changed.
  Practical trade-off: this multilingual model is larger (~470MB vs. ~90MB
  for the English-only one) and has more layers (12 vs. 6). The app now
  loads it in local-files-only mode by default so production startup does
  not depend on outbound Hugging Face access. If the model is not cached,
  retrieval falls back to a local TF-IDF vectorizer instead of failing the
  dashboard. Set `ALLOW_MODEL_DOWNLOAD=true` only on servers where model
  downloads during setup are allowed.
- **`streamlit_app.py`**'s UI was upgraded from a plain text box + button to
  a floating chatbot-shaped widget (button bottom-right, opens a chat
  window) — same look as the earlier prototype screenshot, now wired to
  the real ChromaDB backend instead of a mockup. Requires
  `streamlit>=1.35.0` for `st.container(key=...)`, already pinned in
  `requirements.txt`.
- **Generation backend switched from OpenRouter to Ollama** (self-hosted,
  no API key/account needed at all). `07_prompting.py` now calls a local
  Ollama server's `/api/generate` endpoint instead of the OpenRouter/OpenAI
  SDK. See the important deployment note below — this changes what "real
  deployment" means for this project.

## Run locally

```bash
# 1. Install and start Ollama, and pull a model (one-time)
curl -fsSL https://ollama.com/install.sh | sh    # macOS/Linux; Windows: download from ollama.com
ollama pull llama3.1
ollama serve &                                    # runs on http://localhost:11434 by default

# 2. Set up and run this project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # Windows PowerShell: Copy-Item .env.example .env
# .env already defaults to OLLAMA_HOST=http://localhost:11434,
# OLLAMA_MODEL=llama3.1, and ALLOW_MODEL_DOWNLOAD=false.

streamlit run streamlit_app.py   # builds the Chroma store automatically on first run
```

No account, no API key, no cost — Ollama runs entirely on your machine.

## Deploying this for real — exact steps

**Important limitation first:** Ollama needs to run a real background
process holding a multi-gigabyte model in memory. **Streamlit Community
Cloud cannot do this** — its free tier has no mechanism to install/run a
persistent local LLM server. This isn't something the code can work around;
it's a platform limit. So "deploying this for real" now means one of two
different things depending on what you actually need:

### Option A — Just for yourself / a local team (simplest)
Run it locally exactly as in "Run locally" above, on a shared office
machine or your own laptop, and let people on the same network reach it via
your machine's local IP at `http://<your-ip>:8501` (Streamlit prints this as
the "Network URL" when it starts).

### Option B — A real public deployment (for actual customers)
You need a server you control running **both** Ollama and this app —
Streamlit Cloud alone won't work. The pattern:

1. Get a VPS (DigitalOcean, Hetzner, AWS EC2, etc. — 8GB+ RAM for a 7-8B
   model on CPU; a GPU instance is much faster if budget allows).
2. Install Ollama and this project on that server, run
   `ollama pull llama3.1` and `ollama serve` there.
3. Run `streamlit run streamlit_app.py --server.address 0.0.0.0` on the
   same server (or containerize both with Docker Compose, one service for
   Ollama and one for the Streamlit app on the same internal network).
4. Put a real domain + HTTPS in front of it (e.g. Caddy or Nginx) before
   giving the link to anyone.
5. If you want Ollama and the Streamlit app on *different* machines, set
   `OLLAMA_HOST` (via `.streamlit/secrets.toml` on the Streamlit-hosting
   side) to that other machine's reachable address — but exposing an Ollama
   port to the internet without any auth in front of it is a real security
   risk; put it behind a private network or a reverse proxy with auth if
   you do this.

I can't push to your GitHub account, provision a server, or click "Deploy"
on your behalf — I have no authenticated access to external infrastructure
from this environment. What I *can* do is make every remaining step here as
few and as precise as possible.

## API key rules (followed here)

- Ollama needs no API key at all — `07_prompting.py` just points at a host
  URL (`OLLAMA_HOST`) and a model name (`OLLAMA_MODEL`), both read from the
  environment via `python-dotenv` locally, and overridable from
  `st.secrets` when deployed (only relevant for Option B above, where
  Ollama runs on a different reachable host).
- `.env` and `.streamlit/secrets.toml` are both git-ignored regardless,
  since `OLLAMA_HOST` could point at private infrastructure you don't want
  public even without a "secret" in the traditional sense.

## Turning this into a sellable product

The architecture here is already real, not a toy — the CURRENT/OUTDATED
source distinction in particular is a genuinely useful pattern once you're
managing policy documents that change over time. To go further:
- Add a re-ingestion path (re-run `01`→`05` on a schedule or on document
  upload) instead of relying only on the on-startup auto-build.
- Add authentication in front of `streamlit_app.py` (Streamlit has none
  built in).
- Track server cost (Ollama needs real CPU/GPU compute running
  continuously) per customer if serving multiple clients from one
  deployment — no per-token API bill anymore, but the compute cost doesn't
  disappear, it just moves to infrastructure you pay for directly.
- Move off Streamlit Community Cloud's free tier once you have real usage
  volume or need guaranteed uptime — it's a great starting point, not a
  permanent production home for a paid product.
