# doc-qa-rag

An AI assistant that can answer questions about my own PDF contents, built from scratch to learn how LLM applications work in pratice: embedding, RAG, prompt engineering, structured prompting and output, evaluation and basic security.

Status: In progress. Sections marked as `TODO` are filled as each stage is completed.

## What it does

- Loads PDFs, splits them into chunks, and embeds them into a vector database locally (ChromaDB). `TODO`
- Retrieves the most relevant chunks for a question and asks an LLM to answer using only that context, citing file and page. `TODO`
- Includes a small evaluation set to measure retrieval and answer quality. `TODO`

## Setup
 
Requirements: Python 3.10+ and a Gemini API key.
 
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U google-genai chromadb pypdf python-dotenv numpy pydantic
```
 
Create a `.env` file:
 
```
GEMINI_API_KEY=your-key-here
MODEL=YOUR_CHOICE_OF_MODEL
```

## Usage

`TODO` (planned: `python ask.py "your question"`)

## Project Structure

```
doc-qa-rag/
├── README.md
├── experiments.py      # small API experiments (prompting, structured output, injection)
├── data/               # source PDFs (not committed)
├── .env                # API key (not committed)
└── ...                 # RAG pipeline files added later
```

## Experiments log

Short and reproducible experiments to test the Gemini API and prompt engineering, as well understanding how the model behaves with different inputs. Each entry records an unique experiment that tackles a specific challenge.

### E-00: Testing First API Call to Gemini

- **Question**: Can I successfully call the Gemini API and get a response?
- **Setup**: Any simple prompt, e.g. "Explain how AI works in a few words". or "What is the capital of France?"
- **Result**: The model responds with a short explanation or the correct answer.
- **What suprised me**: Choosing a model matters, since `gemini-3.8-flash` experiences high demand and is often unavailable, which is why I switched to other simpler models.
- **Takeaway**: The Gemini API is working and I can start experimenting with more complex prompts.

### E-01: Plain Call vs System Instruction

- **Question**: How much does a system instruction change the model's behavior regarding the answers it provides?
- **Setup**: Three different calls to the API, all with the same prompt ("Explain what a token is in an LLM."), but with different system instructions:
    1. No system instruction (plain call)
    2. System instruction: "You are explaining to a middle school student. Max two sentences without using technical jargon."
    3. Contradictory System instruction: "Always answer in Portuguese." when the prompt asked for an answer in English.
- **Result**: 
    1. The plain call without system instruction produced a technical answer, with a quick example and key things to know about tokens.
    2. With the system instruction, the model produced a far shorter and simpler answer compared to the plain call, and it was easier and faster to read.
    3. The system instruction has a higher authority than the prompt, and the model answered in Portuguese, even though the prompt was in English, as it was expected.
- **What suprised me**: AI models are actually very sensitive to system instructions, but they do a great job at following them, even when they contradict the prompt.
- **Takeaway**: System instructions are a powerful yet sensitive tool to control the model's behavior, which can benefit or hinder the quality of the answers depending on how they are used. That's why it is important to test different types of system instructions to find the best one for a given prompt within a domain.

### E-02: Temperature Calls

- **Question**: How does varying the temperature parameter affect the model's answers, and will temperature 0 always produce the same answer for the same prompt?
- **Setup**: Ran the same *creative* prompt with different temperature values (0, 1.0, 2.0) and repeated the same prompt three times.
- **Result**: 
    1. Default temperature (1.0): `Sunny`, `Sunny`, `Waffles`. This shows that it clustered around high-probability choices with minor variation.
    2. Temperature 0: `Sunny`, `Milo`, `Sunny`. This shows that it did not collapse into 100% identical outputs despite the zero-temperature parameter.
    3. Temperature 2.0: `Barnaby`, `Waffles`, `Sunny`. This shows that it produced more diverse outputs.
- **What suprised me**: The model didn't give the same answer for the same prompt with temperature 0, which was suprising to me since I expected it to do so.
- **Takeaway**: Using different temperature values will lead to more or less deterministic outputs, never truly deterministic. That's why for strict reproducibility in RAG or similar, other approachs are far more effective.

### E-03: Structured Output

- **Question**: How can I ensure that the model's output adheres to a specific structure and data types?
- **Setup**: Created a Pydantic `Job` schema (`company`, `role`, `skills`, `seniority`) and tested four different extraction scenarios:
    1. Valid Schema: Provided the JSON schema via `response_format` and asked the model to return job details in that format.
    2. Break Test 1 (No Schema): Asked the model to return job details without providing a schema.
    3. Break Test 2 (Missing Field): Provided the schema but omitted the `company` field in the prompt.
    4. Break Test 3 (Token Truncation): Provided the schema but set a very low `max_output_tokens` limit to force truncation of the output.
- **Result**: 
    1. Valid Schema: Returned pure JSON without code fence wrappers, mapping cleanly into the Pydantic model with typed fields intact.
    2. No Schema: The model wrapped the output in markdown fences (```` ```json ... ``` ````) and invented alternative keys (`job_title`, `skills_required`), triggering a Pydantic `ValidationError` (`Invalid JSON: expected value at line 1 column 1`).
    3. Missing Field: The model explicitly assigned `"company": null`, validating successfully as `company=None`.
    4. Token Truncation: Output cut off abruptly after just three tokens (`'{\n  "'`), raising a `ValidationError` due to an unexpected EOF during JSON parsing.
- **What suprised me**: There was nothing that suprised or caught me off guard in this experiment, as the results were expected based on the setup and the model's behavior, and I fully expected the model to hallucinate keys and wrap the output in markdown fences when no schema was provided, as well as to raise a `ValidationError` when the output was truncated due to token limits. 
- **Takeaway**: This approach of structured output is one of the most, if not the most, effective ways to ensure that the model's output is under our control and follows our proposed schema, which is essential in pipelines to avoid unexpected errors. However, this by itself is not enough to guarantee that the output will always be valid, as there are edge cases like token-limit truncations that can result in malformed JSON, which is why client-side validation is still required.

### E-04: Indirect Prompt Injection in Document Summarization

- **Question**: Can the model be tricked into ignoring the system instruction and following a malicious prompt instead?
- **Setup**: Injected a malicious prompt into the middle of a financial report, instructing the model to ignore all prior instructions and instead reply with a specific message. Tested across two setups:
    1. Naive Call: Raw string concatenation of the prompt and the poisoned document without delimiters or safety instructions.
    2. Defended Call: Wrapped the document in `<document>` XML tags and provided a system instruction explicitly designating tag contents as untrusted data.
- **Result**: 
    1. Naive: The attack failed to hijack the task. The model ignored the fake "SYSTEM ALERT" and summarized the financial metrics directly.
    2. Defended: The model produced a similar, clean two-sentence summary of the financial performance, ignoring the injection payload as well.
- **What suprised me**: The naive call held up against a classic prompt injection without explicit defensive measures. Modern Gemini models appear to be trained to prioritize the top-level user task frame over conflicting pseudo-system directives appearing mid-text.
- **Takeaway**: Even though the model was resilient to this simple prompt injection without any explicit defenses, it is still important to implement security measures like this one to protect against more sophisticated attacks, especially in production pipelines where the stakes are higher.

### Embedding Experiments

Testing semantic vector generation and nearest-neighbor search from scratch using `gemini-embedding-2` and NumPy.

1. Cossine Similarity Matrix (8 Sentences across 3 Topics)**:
    - Generated dense vectors for 8 sentences across Astronomy (`0–2`), Culinary (`3–5`), and Software (`6–7`).
    - The resulting $8 \times 8$ cosine similarity matrix confirmed high intra-topic clustering (e.g., Doc 1 [Jupiter] & Doc 2 [Jupiter] scored `0.74`) and clear cross-topic separation (e.g., Doc 0 [Jupiter] vs. Doc 4 [Pasta] dropped to `0.25`).
    - Unrelated sentences still maintained a baseline similarity of ~`0.30–0.50` due to shared syntax, sentence structure, and language representation constraints in positive embedding space.

2. Semantic Retrieval (Ranking Test):
    - Evaluated dense retrieval by querying: *"What instruments are used to observe the cosmos?"*
    - Successfully ranked the most relevant documents at the top:
     1. Doc 1 (*Astronomers discovered a new distant galaxy*, score: `0.6634`)
     2. Doc 2 (*The James Webb telescope captured images of stars*, score: `0.6446`)
     3. Doc 0 (*Jupiter is the largest planet in our solar system*, score: `0.5777`)

3. Challenging case without keyword overlaps:
    - Query: *"How to bake dough in the oven?"*
    - Successfully retrieved Doc 3 (*"A traditional bread recipe requires flour, water, and salt"*, score: `0.6725`) at #1 despite zero lexical overlap (`dough`/`bake`/`oven` vs. `bread`/`recipe`/`flour`/`water`/`salt`), validating dense semantic retrieval over traditional keyword search.

4. Challenging case with negative semantic similarity:
    - Query: *"Show me something that is NOT about astronomy or planets."*
   - Retrieval failed completely: the top 3 results were all astronomy sentences (Docs 1, 0, and 2, scores: `0.61–0.65`).

From this experiment, the key takeaways are the fact that dense embeddings are effective to capture concepts and resolve vocabulary mismatches where traditional keyword search fails. However, since dot products and cosine distances compute directional alighnment rather than logical conditions, the query containing negative semantic operators will retrieve results that are semantically similar to the positive tokens in the query. This has implications to RAG as it cannor handle negative or boolean semantic keywords on its own.

