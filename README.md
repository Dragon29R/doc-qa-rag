# doc-qa-rag

An AI assistant that answers questions grounded strictly in local PDF documents, built from scratch to explore how LLM applications work in practice: dense embeddings, vector search, RAG, prompt engineering, structured outputs, evaluation, and basic security.

## What it does

* **Ingests & Indexes PDFs Locally**: Extracts text page-by-page from PDFs in `data/`, passes through a cleaned and filtered process, splits them into overlapping 800-character chunks, embeds them with `gemini-embedding-001`, and stores them in a persistent local ChromaDB collection (`chroma_db/`).

* **Grounded QA with Citations**: Retrieves the closest chunks by cosine distance for a user query, where the user query is decomposed into sub-queries and instructs `gemini-3.5-flash-lite` (`temperature=0.1`) to answer using *only* the provided context while citing `[File: name.pdf, Page: X]`. Partial answers are allowed.

* **Hallucination Guardrails & Observability**: Refuses to answer when facts are missing from the retrieved chunks (`"I can't answer based on the given documents."`) and logs every call (timestamp, query, retrieved chunk IDs, cosine distances, and final answer) to `retrieval_logs.json` for debugging.

* **Automatic Evaluation**: Evaluates pipeline runs automatically against `eval_set.json`, measuring retrieval hit, cited hit, and negative refusal rates across pipeline and corpus iterations.

## How to run it

### 1. Setup

Requirements: Python 3.10+ and a Gemini API key.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U google-genai chromadb pypdf python-dotenv numpy pydantic

```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-key-here
MODEL=gemini-3.5-flash-lite

```

### 2. Index Documents & Query

1. Place your PDF files inside the `data/` directory.

2. Run the indexing cells in `rag.ipynb` (Steps 1–8) to build the local `chroma_db/` vector store (`docs` and `docs_clean`) and export `chunks.json`.

### 3. Run Automated Evaluation

Run the evaluation section in `rag.ipynb` (or the evaluation runner documented in `rag_eval.md`) against `eval_set.json`, specifying a run tag, the target ChromaDB collection (`docs` or `docs_clean`), and the retrieval mode (single retrieval vs. query decomposition). Per-run summaries and per-question logs are written to `results/`.

## Architecture in 5 Lines

1. **Load & Chunk (`pypdf`)**: Reads `data/*.pdf` page-by-page and splits text into word-boundary-aware chunks (`size=800`, `overlap=100`, `min_len=100`) tagged with `{id, source, page}`. Skips contents, acronym, reference, copyright, and index pages, and fixes hyphenation and stray newlines.

2. **Embed (`gemini-embedding-001`)**: Encodes document chunks in batches using task type `RETRIEVAL_DOCUMENT` and embeds incoming user questions with `RETRIEVAL_QUERY`.

3. **Store (`ChromaDB`)**: Persists vectors, raw chunk text, and file/page metadata locally in `chroma_db/` configured with cosine distance (`hnsw:space="cosine"`).

4. **Retrieve (**$k$**-NN Search)**: Queries ChromaDB for the $k=4$ nearest chunks to the question vector (or per decomposed sub-query), prints distances and previews, and records the trace to `retrieval_logs.json`.

5. **Generate (`gemini-3.5-flash-lite`)**: Formats retrieved chunks with `[File, Page]` headers into a low-temperature (`0.1`) system-prompted call that enforces strict context grounding, inline citations, and partial answers when only part of a question is supported.

## Project Structure

```
doc-qa-rag/
├── README.md                 # Project documentation, evaluation & experiment logs
├── rag.ipynb                 # End-to-end indexing, retrieval, and generation notebook
├── embeddings.py             # Standalone embedding & cosine similarity experiments
├── gemini_experiments.ipynb  # API experiments (system prompts, temperature, schemas, injection)
├── eval_set.json             # 15-question evaluation benchmark and expected outcomes
├── rag_eval.md               # Automatic evaluation using eval_set.json
├── chunks.json               # Exported text chunks with metadata
├── retrieval_logs.jsonl      # JSONL audit log of retrieved chunks and answers per query
├── chroma_db/                # Local persistent ChromaDB vector store
├── data/                     # Source PDFs (not committed)
├── results/                  # Results of each evaluation run
└── .env                      # API keys and configuration (not committed)

```

## Corpus Cleaning and Filtering

During the initial retrieval experiments and the Question 3 diagnosis, front and back matter (table of contents, lists of figures/tables, acronym lists, bibliographies, and indexes) repeatedly appeared in top-$k$ results because they pack high concentrations of domain keywords. At the same time, raw `pypdf` extraction left broken hyphenated words (`"nor-\nmally"`) and stray mid-sentence newlines across chunks.

To address both issues, a second collection (`docs_clean`) was built alongside the baseline collection (`docs`):

### 1. Skipped Page Ranges (`SKIP`)

Non-content pages were filtered out per PDF using 1-indexed inclusive PDF page ranges before chunking:

| PDF Document | Skipped PDF Pages (`SKIP`) | Pages Removed | Reason | 
| ----- | ----- | ----- | ----- | 
| `Automated_Dressing_Saturation_Analysis.pdf` | `(10, 16)`, `(85, 88)` | 11 | Contents, lists of figures/tables, acronyms (`10–16`), references (`85–88`) | 
| `league-of-legends-beginners-guide_compress.pdf` | `(2, 5)`, `(34, 36)` | 7 | Copyright & table of contents (`2–5`), bibliography & index (`34–36`) | 
| `Revolucao_cravos.pdf` | `(20, 22)` | 3 | References / bibliography (`20–22`) | 
| `python.pdf` | None (`[]`) | 0 | Short cheat-sheet / tutorial without separate front or back matter pages | 

### 2. Text Normalization & Corpus Impact

Before chunking, each kept page goes through text normalization to rejoin words split by end-of-line hyphens (`r"(\w)-\s*\n\s*(\w)"`) and collapse stray single newlines within paragraphs while preserving paragraph breaks.

* **Pages indexed**: `160` $\rightarrow$ `139` pages (`21` front/back matter pages skipped)

* **Total chunks**: `489` (`docs`) $\rightarrow$ `426` (`docs_clean`)

* **Hyphen line-breaks**: `235` $\rightarrow$ `0`

* **Gold text sanity check**: Exact-text queries confirmed that the Q3 gold strings (`"281 real"` for dataset composition and `"92.71"` + `"93.88"` for the best fold-0 $F_\beta$ pipeline) are still present in `docs_clean` after filtering and re-chunking.

| Metric | `docs` (Baseline) | `docs_clean` (Filtered & Cleaned) | 
| ----- | ----- | ----- | 
| Pages indexed | 160 | 139 | 
| Total chunks | 489 | 426 | 
| Hyphen breaks across chunks | 235 | 0 | 
| Q3 gold text (`281 real`, `92.71` + `93.88`) | Present | Present | 

### 3. Remaining Limitations & Caveats

* **Remaining PDF extraction artifacts**: Cleaning fixed hyphenation and line wraps, but `pypdf` still leaves missing spaces around italic spans, glued multi-column strings on dense pages, flattened table rows, and spaced-out math symbols ($F_\beta$ extracted as `F β`).

* **Confounded variables**: Because page filtering (`SKIP`) and text normalization were applied together when building `docs_clean` (which also shifts chunk boundaries across the entire corpus), their individual effects on retrieval ranking cannot be separated.

## Automated Evaluation

To iterate faster without manually grading every run, an automated evaluator was built around `eval_set.json` (15 questions: 12 in-scope answerable questions across the 4 PDFs and 3 unanswerable negative tests).

### How `eval_set.json` Works

* **Answer groups (`groups`)**: Each answerable question defines one or more groups of acceptable **PDF page numbers** (1-indexed PDF pages, not printed document page numbers). Each group represents **one required sub-answer** (for example, a two-part question can have two groups: one set of valid pages for part A, and another set of valid pages for part B). For a question to count as a hit, **every group** in that question must be satisfied by at least one page from the target source PDF.

* **Three metrics**:

  1. **Retrieval hit (`12` in-scope questions)**: All expected groups for the question have at least one matching `[source, page]` among the retrieved chunks.

  2. **Cited hit (`12` in-scope questions)**: All expected groups have at least one matching `[File: ..., Page: ...]` citation actually emitted in the model's generated answer.

  3. **Negatives refused (`3` out-of-scope questions)**: The model outputs the canonical abstention string (`"I can't answer based on the given documents"`).

### What Each Metric Misses

* **A retrieval hit is not a correct answer**: Retrieving the right PDF page only proves the page entered the context window; the chunk on that page might miss the specific sentence needed, or the LLM might still misread a table (e.g., citing the right results page on Q3 while naming the wrong pipeline variant).

* **Single-group two-part questions can pass with half the answer**: If a two-part question is encoded as a single group of acceptable pages (or if both parts live on the same page), retrieving/citing only one half still scores `1/1` on retrieval and cited hit.

* **Refusal is a string match**: Negative refusal checks for the canonical abstention phrase via string matching, so a hedged or differently worded refusal (or a partial refusal) is not captured the same way as a human reading the answer.

## Results

### Automated Evaluation Across Pipeline Runs

Note: The two `single_k4_*` runs below are labeled `single retrieval` after verifying `generate_answer.__code__` (confirmed by `avg_chunks = 4.0`, whereas decomposition retrieves per sub-query and deduplicates, yielding `4.9–5.2` average chunks).

| Run | Pipeline | Collection | Retrieval hit | Cited hit | Negatives refused | Avg chunks | Errors | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| `single_k4_docs` | single retrieval | `docs` | 11/12 | 11/12 | 3/3 | 4.0 | 0 | 
| `single_k4_docs_clean` | single retrieval | `docs_clean` | 11/12 | 11/12 | 3/3 | 4.0 | 0 | 
| `decomp_k4_docs` | decomposition | `docs` | 9/12 | 9/12 | 3/3 | 4.9 | 2 | 
| `decomp_k4_docs_clean` | decomposition | `docs_clean` | 10/12 | 10/12 | 3/3 | 5.2 | 1 | 

#### Question-by-Question Changes

* **`single_k4_docs` vs. `single_k4_docs_clean` (Equal totals: `11/12`)**: Both single-retrieval runs scored **11/12** on retrieval hit, **11/12** on cited hit, and **3/3** on negatives refused, with only Q3 failing in both collections (because Q3's dataset-size group sits at rank 30 for the blended query, outside $k=4$). Because the aggregate pass/fail totals are equal, this binary page-hit metric at $k=4$ is too coarse to show intermediate ranking changes or reduced front/back-matter noise within the retrieved chunks.

* **What corpus cleaning changed inside the retrieved sets (`decomp_k4_docs` vs. `decomp_k4_docs_clean`)**: Even when question-level pass/fail is unchanged, inspecting the per-question chunk lists shows that `docs_clean` eliminated the skipped front/back-matter pages that polluted `docs`:
  * **Q1**: `decomp_k4_docs` retrieved `Automated_Dressing_Saturation_Analysis.pdf` **p.87** (references); in `decomp_k4_docs_clean`, p.87 is gone and replaced by p.5 (abstract).
  * **Q8 & Q13**: `decomp_k4_docs` retrieved `league-of-legends-beginners-guide_compress.pdf` **p.34** (bibliography/index); in `decomp_k4_docs_clean`, p.34 is gone.
  * **Q15 (off-topic negative test)**: `decomp_k4_docs` retrieved 4 front-matter chunks (`Automated_Dressing_Saturation_Analysis.pdf` **p.10, p.11, p.14** and `league-of-legends` **p.4**); all 4 were filtered out in `decomp_k4_docs_clean`.

* **What changed on Q3 with Query Decomposition (`groups_retrieved: [true, false]`)**:
  * **Group 1 (Dataset size) flipped from Fail to Pass**: Splitting Q3 into sub-queries moved `Automated_Dressing_Saturation_Analysis.pdf` **p.39** (`459 images`) to the #1 retrieved slot in both `decomp_k4_docs` and `decomp_k4_docs_clean`, and the model answered and cited `[Page: 39]` in both runs.
  * **Group 2 (Best $F_\beta$ pipeline on fold 0) flipped from Pass to Fail**: As predicted by the [Question 3 Diagnosis](#question-3-diagnosis) (where the isolated sub-query `Q_B` pushed the gold $F_\beta$ chunk down to rank 19), decomposing Q3 at $k=4$ per sub-query retrieved pages `51, 76, 78/52, 81` instead of the abstract or Table 4.19/4.32 gold pages. Consequently, `groups_retrieved` was `[true, false]` (overall `retrieval_hit = false`), and the model cited Variant A on page 76 (`92.70%`) while stating that the fold 0 winner was not explicit in the retrieved context.

* **API 503 Errors account for the remaining drops in `decomp_k4_*`**:
  * Every non-errored answerable question besides Q3 passed (`1/1`) across all four runs.
  * In `decomp_k4_docs` (`9/12`, `2` errors), **Q7** and **Q9** failed solely due to transient `503 UNAVAILABLE` server spikes (`n_chunks = 0`). Both Q7 and Q9 succeeded in `decomp_k4_docs_clean` (retrieving and citing pages `9, 10` and page `24`, respectively).
  * In `decomp_k4_docs_clean` (`10/12`, `1` error), **Q4** failed solely due to a `503 UNAVAILABLE` spike (`n_chunks = 0`), whereas Q4 succeeded in `decomp_k4_docs` (retrieving and citing page `7`).

### Manual Evaluation, Baseline Pipeline (History)

*Note: The `14 / 15` score below is a **manual evaluation** of the original baseline pipeline (single retrieval over `docs`,* $k=4$*, strict all-or-nothing prompt). It measures **end-to-end answer correctness** judged by hand, whereas the automated table above measures page-level retrieval and citation hits, so the two tables are kept separate.*

Evaluated across the four documents (`Automated_Dressing_Saturation_Analysis.pdf`, `Revolucao_cravos.pdf`, `league-of-legends-beginners-guide_compress.pdf`, and `python.pdf`):

* **Overall Accuracy**: **14 / 15 (93.3%)**

* **Answerable Questions (In-Scope)**: **11 / 12 (91.7%)** answered accurately with correct `[File, Page]` citations.

* **Unanswerable Questions (Negative / Hallucination Tests)**: **3 / 3 (100%)** properly refused (`"I can't answer based on the given documents."`), successfully resisting domain traps where the model had pre-training knowledge but the PDFs did not.

| \# | Source Document / Category | Question Summary | Outcome | 
| ----- | ----- | ----- | ----- | 
| 1 | `Automated_Dressing_Saturation_Analysis.pdf` | Definition of a chronic wound & 4 ulcer categories | Pass (Correct answer & citations) | 
| 2 | `Automated_Dressing_Saturation_Analysis.pdf` | Two clinical criteria in the rule-based classifier | Pass (Correct answer & citations) | 
| 3 | `Automated_Dressing_Saturation_Analysis.pdf` | Dataset image counts & best pipeline $F_\beta$ score on fold 0 | **Fail** (Baseline abstained: dataset-size chunk ranked 30th; $F_\beta$ chunk ranked 2nd but strict prompt refused the whole answer. With partial-answer prompt + decomposition, dataset count `459` on p.39 is answered, but $F_\beta$ sub-query misses the fold 0 gold page and cites Variant A on p.76.) | 
| 4 | `Revolucao_cravos.pdf` | MFA's "three d's" & Decree Law 373–73 | Pass (Correct answer & citations) | 
| 5 | `Revolucao_cravos.pdf` | Five milestones of Portuguese democratization | Pass (Correct answer & citations) | 
| 6 | `Revolucao_cravos.pdf` | Structure of *Assembleia da República* & d'Hondt method | Pass (Correct answer & citations) | 
| 7 | `league-of-legends-beginners-guide_compress.pdf` | Riot Games founders (2006) & October 2009 release | Pass (Correct answer & citations) | 
| 8 | `league-of-legends-beginners-guide_compress.pdf` | Two effects of destroying an enemy inhibitor | Pass (Correct answer & citations) | 
| 9 | `league-of-legends-beginners-guide_compress.pdf` | Difference between Riot Points and Blue Essence | Pass (Correct answer & citations) | 
| 10 | `python.pdf` | `/` vs. `//` division & operator precedence | Pass (Correct answer & citations) | 
| 11 | `python.pdf` | Tuples vs. lists & default `None` return value | Pass (Correct answer & citations) | 
| 12 | `python.pdf` | Difference between a module and a package | Pass (Correct answer & citations) | 
| 13 | **Negative Test (Domain Trap — LoL)** | Buffs from Baron Nashor or Elemental Drakes | Pass (Correctly refused to answer) | 
| 14 | **Negative Test (Domain Trap — Python)** | List comprehensions & `async`/`await` syntax | Pass (Correctly refused to answer) | 
| 15 | **Negative Test (Off-Topic)** | Ingredients & oven temperature for chocolate cake | Pass (Correctly refused to answer) | 

## What Failed (and Why)

1. **False Abstention on Question #3 (`Automated_Dressing_Saturation_Analysis.pdf`)**

   * *Query*: *"How many images are in the dissertation's dataset, and which pipeline achieved the highest overall F-beta score on the fold 0 test set?"*

   * *What happened*: The baseline model returned `"I can't answer based on the given documents."`

   * *Root causes* (measured with the rank analysis in [Question 3 Diagnosis](#question-3-diagnosis)):

     * **Retrieval failure, dataset-size half**: for the full question the best chunk containing the dataset size ranked **30th** of 489, far outside $k=4$.

     * **Prompt failure, F-beta half**: a chunk with the F-beta answer (92.71%, recall 93.88%) ranked **2nd**, so it was already inside the top 4. The original prompt rule ("if the answer isn't explicit, I don't know") made the model refuse the whole question because only one half could be answered.

     * **Front & back matter pollution** is real (a List of Tables chunk appears in the top 8) but was not the main cause here: the abstract chunk still ranked above it.

     * **PDF math extraction** (`pypdf` writes $F_\beta$ as `F β`, which may weaken the match with "F-beta"): a hypothesis I have not tested.

2. **Other Pipeline Limitations Found During Debugging**

   * **Uncleaned PDF Text**: In the baseline collection, chunks preserved hyphenated line breaks (`"nor-\nmally"`), stray single newlines, and glued multi-column text on acronym lists (`"ACMAssociation for Computing Machinery"`). Hyphenation and stray newlines were fixed in `docs_clean`, though missing spaces in italic text and flattened table rows remain.

   * **No Hard Distance Cutoff in Retrieval**: `retrieve(question, k=4)` always returns 4 chunks even when cosine distances are poor (`> 0.44` on off-topic queries). While the generation prompt successfully caught all 3 negative tests, adding a distance threshold would save unnecessary LLM calls and prevent listing irrelevant `"Fontes recuperadas"` underneath `"I can't answer"` responses.

3. **Evaluation Incident Note: Rate Limits (HTTP 429), Server Spikes (503) & Run Labeling**

   * **Why rate limits hit during decomposition**: Query decomposition doubles the LLM generation calls per question (one call to decompose the question into sub-queries, plus one call to synthesize the final answer), whereas the Gemini API free tier allows only **15 requests per minute (RPM)**.

   * **Mitigation**: Incomplete runs that hit rate-limit errors were discarded. The evaluation harness now automatically retries on `429` responses with backoff and pauses between questions to stay under the 15 RPM ceiling (though occasional transient `503 UNAVAILABLE` high-demand spikes from the API can still occur, as seen on 1–2 questions in the `decomp_k4_*` logs).

   * **Run hygiene**: Two early evaluation files that were mislabeled during notebook state switches (before checking `generate_answer.__code__` and `avg_chunks`) were renamed to match the pipeline that actually ran.

### Question 3 Diagnosis

To understand why the pipeline failed on Question #3, I tracked the global rank of the "gold" chunks (the chunks that really contain the answer) across the whole collection of 489 chunks. Gold chunks were found by exact text search (`where_document={"$contains": ...}`), not by embeddings, so they do not depend on the retrieval being tested:

* **Dataset size**: chunks with `459` plus a marker that states the composition (e.g. `281 real`, `In total it contains`). 5 chunks: Section 3.3.1, the class-imbalance paragraph, the Limitations paragraph and two Conclusions chunks.

* **Best pipeline**: chunks with both `92.71` and `93.88`. 5 chunks: the two abstracts (Portuguese, English), the Results prose and Table 4.19, and Table 4.32.

* Excluded on purpose: a reference-list entry (`pp. 459–479`) and the Table 3.1 row `Total 459`, which have no context.

I then ranked all 489 chunks for the full question (`Q_FULL`) and for each half on its own (`Q_A`: dataset size, `Q_B`: best pipeline on fold 0):

| Query | Gold group | Gold ranks | Gold in top 4 / 10 / 20 | 
| ----- | ----- | ----- | ----- | 
| `Q_FULL` | dataset size | 30, 34, 67, 74, 137 | 0 / 0 / 0 | 
| `Q_FULL` | best pipeline | 2, 9, 13, 139, 152 | 1 / 2 / 3 | 
| `Q_A` | dataset size | 1, 9, 12, 13, 25 | 1 / 2 / 4 | 
| `Q_B` | best pipeline | 19, 20, 75, 117, 140 | 0 / 0 / 2 | 

The lists of ranks are sorted, so the table shows how many gold chunks sit where, not which chunk has which rank.

The results show the following:

1. **The two halves of Q3 failed for different reasons.**

   * *Dataset-size half: retrieval failure.* The best gold chunk is at rank 30 for the full question, so even $k=20$ finds none of the five.

   * *F-beta half: generation failure.* A gold chunk was at rank 2, inside the original top 4, yet the answer was a full refusal. The prompt rule that says "I don't know" when the answer is not explicit made the model discard the half it could answer.

2. **Query blending is the likely cause of the dataset-size miss.**

   * The best dataset-size chunk moves from rank **30** (full question) to rank **1** (`Q_A`), with the same chunks in the index.

   * A single embedding for two topics sits between both targets. I did not test this mechanism directly; the evidence is the change in rank when the question is split.

   * Splitting the question therefore looks necessary for multi-part questions.

3. **The isolated F-beta sub-question is worse than the full question.**

   * `Q_B` alone puts its best gold chunk at rank **19**, against rank **2** for the full question. The cause is not established.

   * Two hypotheses to test: many chunks discuss F-beta scores of the other approaches, so the topic is crowded; or the extra words in the full question (`dataset`, `dissertation`) pull the query towards the abstract.

   * Consequence: decomposition with a small $k$ per sub-query can lose the F-beta half. With decomposition at $k=8$ the sources contained none of the original pages and the answer gave Variant A as the best pipeline, instead of UPerNet.

4. **The distance curve is flat.**

   * From rank 1 to rank 20 the distance only moves by about `0.03` for the full question (`0.231` to `0.263`) and for `Q_B` (`0.244` to `0.276`), and by about `0.06` for `Q_A` (`0.226` to `0.283`).

   * With such small gaps the rank order is sensitive to wording, and a distance threshold cannot separate good chunks from bad ones.

5. **Front matter shows up but did not decide Q3.** In the later $k=8$ run a List of Tables chunk (PDF page 14) is retrieved in 6th position. This supports cleaning the corpus, but the abstract chunk still ranked above it.

6. **Page numbers need care.** The `page` stored in Chroma is the PDF page. In the body of this dissertation the printed page is the PDF page minus 16 (PDF page 39 is printed page 23), but the front matter uses roman numerals and a different offset. The evaluation set must therefore use PDF pages or chunk ids, not the printed numbers.

#### Fixes tried in the first iteration

* **Partial-answer prompt**: a new rule lets the model answer what the context supports and state which part it cannot answer; "I don't know" stays for questions where nothing relevant appears.

* **Larger** $k$: $k=8$ does not reach the dataset-size chunks (best rank 30), so it cannot fix the retrieval half.

* **Query decomposition**: `decompose_query` and `retrieve_multi` are implemented in the notebook.

These fixes solved Question 3 refusing the entire answer, allowing it to return the dataset-size half, though on the second half it retrieved and reported Variant A instead of UPerNet.

## Experiments Log

Short and reproducible experiments to test the Gemini API and prompt engineering, as well as understanding how the model behaves with different inputs. Each entry records a unique experiment that tackles a specific challenge. These experiments were done within the `gemini_experiments.ipynb` notebook, and the results are summarized below.

### E-00: Testing First API Call to Gemini

* **Question**: Can I successfully call the Gemini API and get a response?

* **Setup**: Any simple prompt, e.g. "Explain how AI works in a few words" or "What is the capital of France?"

* **Result**: The model responds with a short explanation or the correct answer.

* **What surprised me**: Choosing a model matters, since `gemini-3.8-flash` experiences high demand and is often unavailable, which is why I switched to other simpler models.

* **Takeaway**: The Gemini API is working and I can start experimenting with more complex prompts.

### E-01: Plain Call vs System Instruction

* **Question**: How much does a system instruction change the model's behavior regarding the answers it provides?

* **Setup**: Three different calls to the API, all with the same prompt ("Explain what a token is in an LLM."), but with different system instructions:

  1. No system instruction (plain call)

  2. System instruction: "You are explaining to a middle school student. Max two sentences without using technical jargon."

  3. Contradictory System instruction: "Always answer in Portuguese." when the prompt asked for an answer in English.

* **Result**:

  1. The plain call without system instruction produced a technical answer, with a quick example and key things to know about tokens.

  2. With the system instruction, the model produced a far shorter and simpler answer compared to the plain call, and it was easier and faster to read.

  3. The system instruction has a higher authority than the prompt, and the model answered in Portuguese, even though the prompt was in English, as it was expected.

* **What surprised me**: AI models are actually very sensitive to system instructions, but they do a great job at following them, even when they contradict the prompt.

* **Takeaway**: System instructions are a powerful yet sensitive tool to control the model's behavior, which can benefit or hinder the quality of the answers depending on how they are used. That's why it is important to test different types of system instructions to find the best one for a given prompt within a domain.

### E-02: Temperature Calls

* **Question**: How does varying the temperature parameter affect the model's answers, and will temperature 0 always produce the same answer for the same prompt?

* **Setup**: Ran the same *creative* prompt with different temperature values (0, 1.0, 2.0) and repeated the same prompt three times.

* **Result**:

  1. Default temperature (1.0): `Sunny`, `Sunny`, `Waffles`. This shows that it clustered around high-probability choices with minor variation.

  2. Temperature 0: `Sunny`, `Milo`, `Sunny`. This shows that it did not collapse into 100% identical outputs despite the zero-temperature parameter.

  3. Temperature 2.0: `Barnaby`, `Waffles`, `Sunny`. This shows that it produced more diverse outputs.

* **What surprised me**: The model didn't give the same answer for the same prompt with temperature 0, which was surprising to me since I expected it to do so.

* **Takeaway**: Using different temperature values will lead to more or less deterministic outputs, never truly deterministic. That's why for strict reproducibility in RAG or similar, other approaches are far more effective.

### E-03: Structured Output

* **Question**: How can I ensure that the model's output adheres to a specific structure and data types?

* **Setup**: Created a Pydantic `Job` schema (`company`, `role`, `skills`, `seniority`) and tested four different extraction scenarios:

  1. Valid Schema: Provided the JSON schema via `response_format` and asked the model to return job details in that format.

  2. Break Test 1 (No Schema): Asked the model to return job details without providing a schema.

  3. Break Test 2 (Missing Field): Provided the schema but omitted the `company` field in the prompt.

  4. Break Test 3 (Token Truncation): Provided the schema but set a very low `max_output_tokens` limit to force truncation of the output.

* **Result**:

  1. Valid Schema: Returned pure JSON without code fence wrappers, mapping cleanly into the Pydantic model with typed fields intact.

  2. No Schema: The model wrapped the output in markdown fences (```` ```json ... ``` ````) and invented alternative keys (`job_title`, `skills_required`), triggering a Pydantic `ValidationError` (`Invalid JSON: expected value at line 1 column 1`).

  3. Missing Field: The model explicitly assigned `"company": null`, validating successfully as `company=None`.

  4. Token Truncation: Output cut off abruptly after just three tokens (`'{\n  "'`), raising a `ValidationError` due to an unexpected EOF during JSON parsing.

* **What surprised me**: There was nothing that surprised or caught me off guard in this experiment, as the results were expected based on the setup and the model's behavior, and I fully expected the model to hallucinate keys and wrap the output in markdown fences when no schema was provided, as well as to raise a `ValidationError` when the output was truncated due to token limits.

* **Takeaway**: This approach of structured output is one of the most, if not the most, effective ways to ensure that the model's output is under our control and follows our proposed schema, which is essential in pipelines to avoid unexpected errors. However, this by itself is not enough to guarantee that the output will always be valid, as there are edge cases like token-limit truncations that can result in malformed JSON, which is why client-side validation is still required.

### E-04: Indirect Prompt Injection in Document Summarization

* **Question**: Can the model be tricked into ignoring the system instruction and following a malicious prompt instead?

* **Setup**: Injected a malicious prompt into the middle of a financial report, instructing the model to ignore all prior instructions and instead reply with a specific message. Tested across two setups:

  1. Naive Call: Raw string concatenation of the prompt and the poisoned document without delimiters or safety instructions.

  2. Defended Call: Wrapped the document in `<document>` XML tags and provided a system instruction explicitly designating tag contents as untrusted data.

* **Result**:

  1. Naive: The attack failed to hijack the task. The model ignored the fake "SYSTEM ALERT" and summarized the financial metrics directly.

  2. Defended: The model produced a similar, clean two-sentence summary of the financial performance, ignoring the injection payload as well.

* **What surprised me**: The naive call held up against a classic prompt injection without explicit defensive measures. Modern Gemini models appear to be trained to prioritize the top-level user task frame over conflicting pseudo-system directives appearing mid-text.

* **Takeaway**: Even though the model was resilient to this simple prompt injection without any explicit defenses, it is still important to implement security measures like this one to protect against more sophisticated attacks, especially in production pipelines where the stakes are higher.

### Embedding Experiments

Testing semantic vector generation and nearest-neighbor search from scratch using `gemini-embedding-2` and NumPy, all done in the `embeddings.py` file.

1. Cosine Similarity Matrix (8 Sentences across 3 Topics):

   * Generated dense vectors for 8 sentences across Astronomy (`0–2`), Culinary (`3–5`), and Software (`6–7`).

   * The resulting $8 \times 8$ cosine similarity matrix confirmed high intra-topic clustering (e.g., Doc 1 \[Jupiter\] & Doc 2 \[Jupiter\] scored `0.74`) and clear cross-topic separation (e.g., Doc 0 \[Jupiter\] vs. Doc 4 \[Pasta\] dropped to `0.25`).

   * Unrelated sentences still maintained a baseline similarity of \~`0.30–0.50` due to shared syntax, sentence structure, and language representation constraints in positive embedding space.

2. Semantic Retrieval (Ranking Test):

   * Evaluated dense retrieval by querying: *"What instruments are used to observe the cosmos?"*

   * Successfully ranked the most relevant documents at the top:

     1. Doc 1 (*Astronomers discovered a new distant galaxy*, score: `0.6634`)

     2. Doc 2 (*The James Webb telescope captured images of stars*, score: `0.6446`)

     3. Doc 0 (*Jupiter is the largest planet in our solar system*, score: `0.5777`)

3. Challenging case without keyword overlaps:

   * Query: *"How to bake dough in the oven?"*

   * Successfully retrieved Doc 3 (*"A traditional bread recipe requires flour, water, and salt"*, score: `0.6725`) at #1 despite zero lexical overlap (`dough`/`bake`/`oven` vs. `bread`/`recipe`/`flour`/`water`/`salt`), validating dense semantic retrieval over traditional keyword search.

4. Challenging case with negative semantic similarity:

   * Query: *"Show me something that is NOT about astronomy or planets."*

   * Retrieval failed completely: the top 3 results were all astronomy sentences (Docs 1, 0, and 2, scores: `0.61–0.65`).

From this experiment, the key takeaways are the fact that dense embeddings are effective to capture concepts and resolve vocabulary mismatches where traditional keyword search fails. However, since dot products and cosine distances compute directional alignment rather than logical conditions, the query containing negative semantic operators will retrieve results that are semantically similar to the positive tokens in the query. This has implications for RAG as it cannot handle negative or boolean semantic keywords on its own.

### Retrieval-Augmented Generation (RAG) Initial Retrieval Experiments

* **Question**: Does retrieval find the right passages, and what happens with an unanswerable question?

* **Setup**: `retrieve(question, k=4)` over the 489-chunk collection (cosine distance, lower is closer), with questions about three of the documents and one unrelated question.

* **Result**:

| Question | Best distance | Top results | 
| ----- | ----- | ----- | 
| Who is Ryze? | 0.311 | Bibliography (p.34) ranked first, the answer (p.9) second, the index (p.34) third | 
| What is a chronic wound? | 0.260 | Correct definition on p.17 ranked first and second, reference list (p.87) third | 
| What kind of political system does Portugal have? | 0.260 | All four from the Portugal paper, with the semi-presidential system page first | 
| How do I bake a cake? (unrelated) | 0.419 | Table of contents, Python code, a figure caption | 
| O que é uma ferida crónica? (different language) | 0.265 | Even with different languages, it captured the English chunks and in the fourth retrieval result, it captured the Portuguese abstract | 

* **What surprised me**: Some of the retrieved chunks contained index or bibliography contents, which can be explained by the fact that they contain keywords or acronyms that will flag similarity. In the future, this is an easy fix that will definitely improve the retrieved results.

* **Takeaway**: Retrieval finds the right document in all three answerable cases. Relevant questions had best distances of `0.26` to `0.33`, and the unrelated question `0.419`, so the gap is real but modest. Retrieval always returns $k$ results, so refusing to answer has to be handled by the generation prompt. Front and back matter (contents, references, index) takes ranking slots from real content, so filtering those pages is the next improvement.