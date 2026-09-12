# PakLegal AI

AI-powered legal information and complaint-drafting assistant for Pakistani citizens.

## Architecture

The Streamlit app uses several Groq-powered agents:

1. Legal Research Agent
2. Legal Action & Documents Agent
3. Complaint/Application Drafting Agent
4. Legal Quality Review Agent
5. Senior Legal Aid Synthesis Agent

A local RAG layer retrieves relevant material from the `knowledge/` directory.

## Important

This is a prototype for legal information, not legal advice. The knowledge base
must be populated with authoritative and current Pakistani legal sources before
real-world deployment.

Do not commit API keys.

## Local / Colab setup

```bash
pip install -r requirements.txt
```

Then set the key:

```python
import os
os.environ["GROQ_API_KEY"] = "YOUR_KEY"
```

Run:

```bash
streamlit run app.py
```

For Colab, Streamlit can be exposed with a suitable tunnel such as the one you
normally use for your testing environment.

## Knowledge base

Create:

```text
knowledge/
```

and place authoritative `.txt`, `.md`, or `.pdf` legal materials inside it.

Recommended future knowledge-base categories:

- Pakistani tenancy/rent laws, organized by province/territory
- Criminal procedure and FIR-related official material
- Pakistan Penal Code / relevant criminal-law material
- Cybercrime/online-fraud official material
- Police complaint procedures
- Court/legal-aid information
- Province-specific government/legal-aid procedures

Every document should retain its original title/source/date so the system can
show users where information came from.

## Production improvements

Before public deployment, add:

- authoritative source citations and source URLs
- document dates/versioning
- province/city routing
- stronger retrieval and reranking
- structured case intake
- human lawyer review/escalation
- audit logs with privacy controls
- PII redaction
- authentication/rate limiting
- tests for hallucination and incorrect legal claims
