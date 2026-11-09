# RAG Evaluation Q&A Set 1

This document contains 15 test questions designed to evaluate your RAG pipeline, divided into 12 answerable questions grounded in the source documents and 3 unanswerable questions to test hallucination resistance.

## Answerable Questions

### From `Automated_Dressing_Saturation_Analysis.pdf`

**1. Question:** How is a chronic wound defined, and into which four categories are chronic wounds usually classified?
*   **Expected Answer:** A chronic wound is a wound that does not follow a normal healing process, typically taking more than 4 to 12 weeks without reaching full anatomic and functional recovery. They are usually classified into four categories: pressure, diabetic, venous, and arterial ulcers.
*   **Expected Source:** `Automated_Dressing_Saturation_Analysis.pdf` (Chapter 1, Section 1.1, p. 1 / PDF p. 17).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**2. Question:** What are the two clinical criteria used by the rule-based post-processing classifier to flag that a wound dressing needs changing?
*   **Expected Answer:** A dressing is flagged as needing a change if either of two criteria is met: (1) the **saturation ratio is >= 0.60** (the fraction of the dressing surface occupied by exudate exceeds 60%), or (2) **edge proximity is <= d pixels** (any exudate pixel lies within $d$ pixels of the dressing boundary, indicating a risk of leakage).
*   **Expected Source:** `Automated_Dressing_Saturation_Analysis.pdf` (Chapter 3, Section 3.4.4, pp. 29–30).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**3. Question:** How many images are in the dissertation's dataset, and which pipeline achieved the highest overall F-beta score on the fold 0 test set?
*   **Expected Answer:** The dataset contains 459 images in total (281 real clinical images and 178 simulated images), with 154 requiring a dressing change and 305 marked as safe. The best overall performance on the fold 0 test set was achieved by the segmentation pipeline using the 3-class **UPerNet** model followed by the rule-based classifier, reaching an F-beta score of 92.71% and a recall of 93.88% (missing only 3 of 49 positive cases).
*   **Expected Source:** `Automated_Dressing_Saturation_Analysis.pdf` (Abstract, p. ii; Section 3.3.1, p. 23; Section 4.1.3 / Table 4.19, p. 50).
*  **Outcome:** Did not answer, "I can't answer based on the given documents." ❌

### From `Revolucao_cravos.pdf`

**4. Question:** What were the "three d's" promised by the Armed Forces Movement (MFA) during the 25 April 1974 Carnation Revolution, and which decree law initially sparked their professional grievances?
*   **Expected Answer:** The "April Captains" of the MFA promised the Portuguese people a political program known as the "three d's": **democracy, development, and decolonization**. The MFA formed principally in response to junior officers' demand to repeal **Decree Law 373–73**, which allowed conscripted officers (*milicianos*) to count all of their service toward seniority, moving ahead of professional junior officers.
*   **Expected Source:** `Revolucao_cravos.pdf` (Section *The Revolution of 25 April 1974*, p. 6).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**5. Question:** What are the five milestones on the road to Portuguese democratization identified by Paul Christopher Manuel?
*   **Expected Answer:** The five milestones are: (1) getting the new constitutional dual-executive arrangement up and running, (2) establishing civilian control of the military by abolishing the Council of the Revolution in the 1982 constitutional amendment, (3) integrating into the European Economic Community (EEC) on 1 January 1986, (4) building electoral alliances across ideological differences (such as in the 1986 presidential election), and (5) achieving the peaceful alternation of power among competing political parties.
*   **Expected Source:** `Revolucao_cravos.pdf` (Section *Five Milestones on the Road to Portuguese Democratization*, pp. 10–14).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**6. Question:** How is the Portuguese legislature (*Assembleia da República*) structured under the Constitution, and which method is used to allocate seats?
*   **Expected Answer:** Portugal has a unicameral legislature called the *Assembleia da República* with 230 seats, four-year terms, and 22 multi-seat constituencies (18 on the continent, one each for the Azores and Madeira, one for Portuguese in Europe, and one for Portuguese living elsewhere). Seats are allocated proportionally based on party votes using the **d'Hondt method**.
*   **Expected Source:** `Revolucao_cravos.pdf` (Section *Key Institutional Elements of Portuguese Democracy*, pp. 9–10).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

### From `league-of-legends-beginners-guide_compress.pdf`

**7. Question:** Who founded Riot Games in 2006, and when was *League of Legends* officially released?
*   **Expected Answer:** Riot Games was started in 2006 by college roommates **Brandon "Ryze" Beck** and **Marc "Tryndamere" Merrill**. After several years of work and beta tests, *League of Legends* was released as a free download in **October 2009**.
*   **Expected Source:** `league-of-legends-beginners-guide_compress.pdf` (Chapter 1, pp. 7–8).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**8. Question:** In *League of Legends*, what two things happen when your team destroys an enemy inhibitor on Summoner's Rift?
*   **Expected Answer:** First, destroying an inhibitor allows your team's Nexus to begin spawning more powerful minions called **super minions**. Second, it causes two special turrets called **Nexus turrets** to pop up near the opposing team's Nexus, which must be defeated before your team can damage the enemy Nexus.
*   **Expected Source:** `league-of-legends-beginners-guide_compress.pdf` (Chapter 2, pp. 13–14).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**9. Question:** What is the difference between Riot Points and Blue Essence in *League of Legends*?
*   **Expected Answer:** Both are used to get in-game cosmetic items, but **Riot Points** cost real-life money to purchase in the in-game store, whereas **Blue Essence** is earned for free by playing matches and completing missions.
*   **Expected Source:** `league-of-legends-beginners-guide_compress.pdf` (Chapter 3, *Dressed for Success* sidebar, p. 22).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

### From `python.pdf`

**10. Question:** In Python, what is the difference between the `/` and `//` division operators, and what is the order of operator precedence?
*   **Expected Answer:** The `/` operator divides and returns a floating-point number (`float`), while `//` divides and returns an integer (`int`). Operator precedence from highest to lowest is: (1) parentheses, (2) exponentiation (`**`), (3) multiplication / division, and (4) addition / subtraction.
*   **Expected Source:** `python.pdf` (Section *Arithmetic Operations*, p. 7).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**11. Question:** How do tuples differ from lists in Python, and what value does a Python function return by default if it has no `return` statement?
*   **Expected Answer:** Tuples are like read-only lists; once defined, you cannot add, remove, or modify items in a tuple. If a function does not include a `return` statement, it returns `None` by default, which represents the absence of a value.
*   **Expected Source:** `python.pdf` (Sections *Tuples* and *Functions*, pp. 9–10).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅


**12. Question:** What is the difference between a module and a package in Python?
*   **Expected Answer:** A module is a single file with Python code used to organize a program into smaller files, whereas a package is a directory containing a special `__init__.py` file that can hold one or more modules.
*   **Expected Source:** `python.pdf` (Sections *Modules* and *Packages*, pp. 12–13).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅
---

## Unanswerable Questions (Negative Tests)

*These questions are designed to test if your RAG pipeline correctly admits it does not know the answer, preventing hallucinations.*

**13. Question (Domain Trap — League of Legends):** What buffs does a team receive for slaying Baron Nashor or an Elemental Drake in *League of Legends*?
*   **Expected Answer:** The RAG system must state that it **does not know** based on the provided context (e.g., *"Não sei responder com base no contexto fornecido"*), because `league-of-legends-beginners-guide_compress.pdf` only mentions generic neutral monsters in the jungle and never mentions Baron Nashor or Elemental Drakes.
*  **Outcome:** Refused to answer to the question. ✅

**14. Question (Domain Trap — Python):** How do you write a list comprehension or define an asynchronous function using `async` and `await` in Python?
*   **Expected Answer:** The RAG system must state that it **does not know** based on the provided context, because `python.pdf` only covers beginner constructs (`for` loops, basic functions, classes, and modules) and does not contain information on list comprehensions or `async`/`await`.
*  **Outcome:** Refused to answer to the question. ✅

**15. Question (Completely Off-Topic):** What ingredients and oven temperature are needed to bake a traditional chocolate cake?
*   **Expected Answer:** The RAG system must state that it **does not know** based on the provided context, as none of the indexed documents contain cooking recipes.
*  **Outcome:** Refused to answer to the question. ✅

# RAG Evaluation Q&A Set 2

This document contains 15 test questions designed to evaluate your RAG pipeline, divided into 12 answerable questions grounded in the source documents and 3 unanswerable questions to test hallucination resistance. This was done with the addition of query decomposition and partial answering to the RAG pipeline.

## Answerable Questions

### From `Automated_Dressing_Saturation_Analysis.pdf`

**1. Question:** How is a chronic wound defined, and into which four categories are chronic wounds usually classified?
*   **Expected Answer:** A chronic wound is a wound that does not follow a normal healing process, typically taking more than 4 to 12 weeks without reaching full anatomic and functional recovery. They are usually classified into four categories: pressure, diabetic, venous, and arterial ulcers.
*   **Expected Source:** `Automated_Dressing_Saturation_Analysis.pdf` (Chapter 1, Section 1.1, p. 1 / PDF p. 17).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**2. Question:** What are the two clinical criteria used by the rule-based post-processing classifier to flag that a wound dressing needs changing?
*   **Expected Answer:** A dressing is flagged as needing a change if either of two criteria is met: (1) the **saturation ratio is >= 0.60** (the fraction of the dressing surface occupied by exudate exceeds 60%), or (2) **edge proximity is <= d pixels** (any exudate pixel lies within $d$ pixels of the dressing boundary, indicating a risk of leakage).
*   **Expected Source:** `Automated_Dressing_Saturation_Analysis.pdf` (Chapter 3, Section 3.4.4, pp. 29–30).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**3. Question:** How many images are in the dissertation's dataset, and which pipeline achieved the highest overall F-beta score on the fold 0 test set?
*   **Expected Answer:** The dataset contains 459 images in total (281 real clinical images and 178 simulated images), with 154 requiring a dressing change and 305 marked as safe. The best overall performance on the fold 0 test set was achieved by the segmentation pipeline using the 3-class **UPerNet** model followed by the rule-based classifier, reaching an F-beta score of 92.71% and a recall of 93.88% (missing only 3 of 49 positive cases).
*   **Expected Source:** `Automated_Dressing_Saturation_Analysis.pdf` (Abstract, p. ii; Section 3.3.1, p. 23; Section 4.1.3 / Table 4.19, p. 50).
*  **Outcome:** Did partially answer, only when k was increased to k=8, answering the total number of images, but still didn't answer the overall highest F-beta. ❌

### From `Revolucao_cravos.pdf`

**4. Question:** What were the "three d's" promised by the Armed Forces Movement (MFA) during the 25 April 1974 Carnation Revolution, and which decree law initially sparked their professional grievances?
*   **Expected Answer:** The "April Captains" of the MFA promised the Portuguese people a political program known as the "three d's": **democracy, development, and decolonization**. The MFA formed principally in response to junior officers' demand to repeal **Decree Law 373–73**, which allowed conscripted officers (*milicianos*) to count all of their service toward seniority, moving ahead of professional junior officers.
*   **Expected Source:** `Revolucao_cravos.pdf` (Section *The Revolution of 25 April 1974*, p. 6).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**5. Question:** What are the five milestones on the road to Portuguese democratization identified by Paul Christopher Manuel?
*   **Expected Answer:** The five milestones are: (1) getting the new constitutional dual-executive arrangement up and running, (2) establishing civilian control of the military by abolishing the Council of the Revolution in the 1982 constitutional amendment, (3) integrating into the European Economic Community (EEC) on 1 January 1986, (4) building electoral alliances across ideological differences (such as in the 1986 presidential election), and (5) achieving the peaceful alternation of power among competing political parties.
*   **Expected Source:** `Revolucao_cravos.pdf` (Section *Five Milestones on the Road to Portuguese Democratization*, pp. 10–14).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**6. Question:** How is the Portuguese legislature (*Assembleia da República*) structured under the Constitution, and which method is used to allocate seats?
*   **Expected Answer:** Portugal has a unicameral legislature called the *Assembleia da República* with 230 seats, four-year terms, and 22 multi-seat constituencies (18 on the continent, one each for the Azores and Madeira, one for Portuguese in Europe, and one for Portuguese living elsewhere). Seats are allocated proportionally based on party votes using the **d'Hondt method**.
*   **Expected Source:** `Revolucao_cravos.pdf` (Section *Key Institutional Elements of Portuguese Democracy*, pp. 9–10).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

### From `league-of-legends-beginners-guide_compress.pdf`

**7. Question:** Who founded Riot Games in 2006, and when was *League of Legends* officially released?
*   **Expected Answer:** Riot Games was started in 2006 by college roommates **Brandon "Ryze" Beck** and **Marc "Tryndamere" Merrill**. After several years of work and beta tests, *League of Legends* was released as a free download in **October 2009**.
*   **Expected Source:** `league-of-legends-beginners-guide_compress.pdf` (Chapter 1, pp. 7–8).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**8. Question:** In *League of Legends*, what two things happen when your team destroys an enemy inhibitor on Summoner's Rift?
*   **Expected Answer:** First, destroying an inhibitor allows your team's Nexus to begin spawning more powerful minions called **super minions**. Second, it causes two special turrets called **Nexus turrets** to pop up near the opposing team's Nexus, which must be defeated before your team can damage the enemy Nexus.
*   **Expected Source:** `league-of-legends-beginners-guide_compress.pdf` (Chapter 2, pp. 13–14).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**9. Question:** What is the difference between Riot Points and Blue Essence in *League of Legends*?
*   **Expected Answer:** Both are used to get in-game cosmetic items, but **Riot Points** cost real-life money to purchase in the in-game store, whereas **Blue Essence** is earned for free by playing matches and completing missions.
*   **Expected Source:** `league-of-legends-beginners-guide_compress.pdf` (Chapter 3, *Dressed for Success* sidebar, p. 22).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

### From `python.pdf`

**10. Question:** In Python, what is the difference between the `/` and `//` division operators, and what is the order of operator precedence?
*   **Expected Answer:** The `/` operator divides and returns a floating-point number (`float`), while `//` divides and returns an integer (`int`). Operator precedence from highest to lowest is: (1) parentheses, (2) exponentiation (`**`), (3) multiplication / division, and (4) addition / subtraction.
*   **Expected Source:** `python.pdf` (Section *Arithmetic Operations*, p. 7).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅

**11. Question:** How do tuples differ from lists in Python, and what value does a Python function return by default if it has no `return` statement?
*   **Expected Answer:** Tuples are like read-only lists; once defined, you cannot add, remove, or modify items in a tuple. If a function does not include a `return` statement, it returns `None` by default, which represents the absence of a value.
*   **Expected Source:** `python.pdf` (Sections *Tuples* and *Functions*, pp. 9–10).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅


**12. Question:** What is the difference between a module and a package in Python?
*   **Expected Answer:** A module is a single file with Python code used to organize a program into smaller files, whereas a package is a directory containing a special `__init__.py` file that can hold one or more modules.
*   **Expected Source:** `python.pdf` (Sections *Modules* and *Packages*, pp. 12–13).
*  **Outcome:** Answered correctly to the question, with correct sources. ✅
---

## Unanswerable Questions (Negative Tests)

*These questions are designed to test if your RAG pipeline correctly admits it does not know the answer, preventing hallucinations.*

**13. Question (Domain Trap — League of Legends):** What buffs does a team receive for slaying Baron Nashor or an Elemental Drake in *League of Legends*?
*   **Expected Answer:** The RAG system must state that it **does not know** based on the provided context (e.g., *"Não sei responder com base no contexto fornecido"*), because `league-of-legends-beginners-guide_compress.pdf` only mentions generic neutral monsters in the jungle and never mentions Baron Nashor or Elemental Drakes.
*  **Outcome:** Refused to answer to the question. ✅

**14. Question (Domain Trap — Python):** How do you write a list comprehension or define an asynchronous function using `async` and `await` in Python?
*   **Expected Answer:** The RAG system must state that it **does not know** based on the provided context, because `python.pdf` only covers beginner constructs (`for` loops, basic functions, classes, and modules) and does not contain information on list comprehensions or `async`/`await`.
*  **Outcome:** Refused to answer to the question. ✅

**15. Question (Completely Off-Topic):** What ingredients and oven temperature are needed to bake a traditional chocolate cake?
*   **Expected Answer:** The RAG system must state that it **does not know** based on the provided context, as none of the indexed documents contain cooking recipes.
*  **Outcome:** Refused to answer to the question. ✅