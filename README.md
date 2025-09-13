<h1 align="center"><img src="https://readme-typing-svg.demolab.com?font=Chakra+Petch&weight=500&size=29&duration=1&pause=1000&color=000000&background=601EF9&vCenter=true&repeat=false&width=990&lines=Conversational+Agent+for+Medical+Question-Answering+Using+RAG+and+LLM" alt="Typing SVG" /></h1>

<div align="center">
  <img src="https://github.com/Laoode/Medical-QA-Agent/blob/main/assets/Banner.gif" alt="Banner">
</div>

---
# Table of Contents
1. [Overview](#overview)
2. [Background](#background)
3. [Methodology](#methodology)
4. [Data Collection](#data-collection)
5. [Implementation](#implementation)
6. [Conclusion](#conclusion)
7. [Citation](#citation)
    
## Overview
<p>
  This  study  analyzes  the  application  of  the  RAG  concept  alongside  an  LLM  in  the  context  of  PubMed  QA  data  to  augment question-answering capabilities in the medical context.For answering questions relevant to private healthcare institutions, the Mistral 7B model was utilized.To limit hallucinations, an embedding model was used for document indexing, ensuring that the LLM answers based on the provided context information.The analysis was conducted using five embedding models, two of which are specialized medical models, PubMedBERT-base and BioLORD-2023, as well as three general models, GIST-large-Embedding-v0, b1ade-embed-kd, and all-MiniLM-L6-v2. As the results showed, general models performed better than  domain specific models, especially GIST-large-Embedding-v0 and  b1ade-embed-kd, which underscores the dominance of general-purpose training datasets in terms of fundamental semantic retrieval, even in medical domains. The outcome of this research study demonstrates that applying RAG and LLM locally can safeguard privacy while still  responding  to  medical  queries  with  appropriate  precision,  thus  establishing  a  foundation  for  a  dependable  medical  question-answering system.
</p>

## Background
Since the 1950s, conversational agents, or chatbots, have evolved significantly from the concept of the Turing test to modern systems built on Large Language Models (LLMs), like OpenAI's ChatGPT. While these transformer-based models are powerful, their use, particularly in healthcare, raises significant privacy and data protection concerns, as sensitive patient information must be sent to external, third-party services with limited transparency. A promising solution is the use of locally deployable LLMs, which can be trained and operated on-premise, ensuring medical data remains secure and private. However, LLMs have their own drawbacks, such as generating non-factual or "hallucinated" information. To mitigate this, Retrieval-Augmented Generation (RAG) is a proposed solution that enhances LLMs by integrating external knowledge sources, such as up-to-date medical data, to improve accuracy. The performance of a RAG system heavily relies on the quality of its embedding model, which is crucial for retrieving relevant information. This study implements a RAG pipeline for medical question-answering using the PubMed QA dataset, employing various embedding models and utilizing Mistral 7B as the local LLM to leverage its strong performance in medical tasks while maintaining data security.

## Methodology
This  section  explains  the  research  methods  applied  in  this  study  for  implementing  a  RAG  pipeline  in  a medical  question-answering system  with  Mistral  7B  as  the  local  large  language  model.The  focus  of  this  research  is  to  identify  the  best  embedding  models  for semantic  retrieval  in  the  medical  domain  using  the  PubMed  QA  dataset,  with  semantic  answer  similarity  evaluator  (SAS),  mean reciprocal rank (MRR), and Faithfulness as evaluation metrics.

<div align="center">
  <img src="https://github.com/Laoode/Medical-QA-Agent/blob/main/assets/Research_workflow.png" alt="Methodology">
</div>

## Data Collection
The data used consists of text collected from Hugging Face [PubMedQA_instruction](https://huggingface.co/datasets/vblagoje/PubMedQA_instruction/) with a total of 273k rows, separated into 272k rows as train and 1k rows as test.The original data has 4 columns: instruction refers to the question, context as the document reference which contains  abstracts  from  PubMed  articles,  response  is  the  correct  answer  based  on  context,  and  category  is  designated  for  the question-answering  that  is  already  done  with  `closed_qa`.

<div align="center">
  <img src="https://github.com/Laoode/Medical-QA-Agent/blob/main/assets/PubMedQA_instruction_dataset.png" alt="Methodology">
</div>

In this process, related columns were selected to ensure that the model can generate answers based on the context.This research down-sampled  the  first  10,000  rows  from  the  train  sample  of  the  PubMedQA  instruction  dataset,  with  selected  columns:  instruction  as  all questions to test the model later, context as all documents for retrieval to embedding models, and response as the ground truth answer to evaluate similarity from the generated LLM answer.

## Embedding Model
The 10,000  documents  (from  the  context  column)  were  processed  to  create  vector  representations  using  five  embedding  models  for comparison,  namely  PubMedBERT-base,  GIST-large-Embedding-v0,  BioLORD-2023,  b1ade-embed-kd,  and  all-MiniLM-L6-v2. The selection of these models is based on their  prevalence in the field of biomedicine and general NLP, and their performance to allow a comprehensive comparison of the quality of retrieval and answer generation results in the medical domain.

| Embedding Model | Window Size (Tokens) | Parameters (Millions) | Dimensions |
| :--- | :---: | :---: | :---: |
| PubMedBERT-base | 512 | 110 | 768 |
| GIST-large-Embedding-v0 | 512 | 335 | 1024 |
| BioLORD-2023 | 512 | 109 | 768 |
| b-tade-embed-kd | 512 | 335 | 1024 |
| all-MiniLM-L6-v2 | 256 | 22.7 | 384 |

All embedded documents are then stored as embeddings along with their respective documents in the vector database. A vector database is a type of database that stores data as high-dimensional vectors, which are mathematical representations of features or attributes [19]. `SentenceTransformDocumentEmbedder` from Haystack was used to convert text into dense vectors, which were then stored in `InMemoryDocumentStore` using `DocumentWriter`.

## Implementation
After the document indexing process is complete and the entire medical context is successfully converted into vector form, the next step is the implementation of the RAG pipeline. The pipeline has a few main steps:

1. **Query Embedding:** Each question was embedded using the same model as the documents (e.g., PubMedBERT for the PubMedBERT store) to ensure consistency.
2. **Document Retrieval:** `InMemoryEmbeddingRetriever` was used to retrieve the top-3 documents based on cosine similarity between query and document embeddings.
3. **Prompt Building:** `ChatPromptBuilder` creates a prompt with the retrieved documents and question to guide the LLM. In this case, the template for the LLM makes it answer like this:

    ```python
    # Define the prompt template
    template = [
        ChatMessage.from_user(
            """
            You are a medical expert answering questions based on the provided context. Use only the context to answer the question accurately.
    
            Context:
            {% for document in documents %}
                {{ document.content }}
            {% endfor %}
    
            Question: {{question}}
            Answer:
            """
        )
    ]
    ```

The LLM was implemented to answer based on the context provided from document retrieval, where the LLM only answers the question using the top 3 documents with the highest similarity. This ensures that the LLM does not hallucinate when answering the question, as the LLM is provided with the relevant document context.

4. **Answer Generation:** Mistral 7B, run locally with Ollama, was used to generate an answer from the prompt.
5. **Final Answer:** `AnswerBuilder` cleans up the generated answer for evaluation.

## Model Evaluation

To evaluate system performance, 100 questions were randomly selected from the test set using `random.seed(42)` to ensure reproducibility. For each embedding model, the RAG pipeline generated answers based on the retrieved documents, which were then evaluated using three metrics: **Mean Reciprocal Rank (MRR)** to assess document retrieval quality, **Semantic Answer Similarity (SAS)** to measure alignment with ground-truth answers, and **FaithfulnessEvaluator** to verify factual consistency with retrieved contexts. Below is an example of answers generated by different embedding models for the question:

"Do high levels of procalcitonin in the early phase after pediatric liver transplantation indicate poor postoperative outcome?"

| Embedding Model | Generated Answer |
| :--- | :--- |
| PubMedBERT-base | Yes, according to the provided context, high levels of procalcitonin in the early phase after pediatric liver transplantation do indicate a poor postoperative outcome. Patients with high PCT levels on postoperative day 2 had higher International Normalized Ratio values or P-OD 5, suffered more often from primary graft non-function, had a longer stay in the pediatric intensive care unit and on mechanical ventilation. However, there was no correlation between PCT elevation and systemic infection. |
| GIST-large-Embedding-v0 | Yes, according to the provided context, high levels of procalcitonin (PCT) on postoperative day 2 in pediatric liver transplant recipients are associated with a higher International Normalized Ratio value on postoperative day 5, more instances of primary graft non-function, longer stays in the pediatric intensive care unit, and longer duration on mechanical ventilation. However, it is important to note that PCT levels were not correlated with systemic infection. |
| BioLORD-2023 | Yes, according to the provided context, high levels of procalcitonin (PCT) in the early phase after pediatric liver transplantation are associated with a poorer postoperative outcome. Specifically, patients with high PCT levels on postoperative day 2 had higher International Normalized Ratio values on postoperative day 5, suffered more often from primary graft non-function, had a longer stay in the pediatric intensive care unit and on mechanical ventilation. However, it is important to note that there was no correlation between PCT elevation and systemic infection. |
| b-tade-embed-kd | Yes, based on the provided context, patients with high PCT levels on postoperative day 2 had a higher incidence of primary graft non-function, longer stay in the pediatric intensive care unit, and a longer duration on mechanical ventilation. They also had higher International Normalized Ratio values on postoperative day 5. Therefore, high PCT levels in the early phase after pediatric liver transplantation appear to indicate a poor postoperative outcome. |
| all-MiniLM-L6-v2 | Yes, based on the provided context, high levels of procalcitonin (PCT) in the early phase after pediatric liver transplantation are associated with a higher International Normalized Ratio (INR) value on postoperative day 5, a longer stay in the pediatric intensive care unit (p<0.01), and a longer duration of mechanical ventilation (p<0.001). Additionally, patients with high PCT levels had a higher incidence of primary graft non-function (p<0.05). These findings suggest that high levels of PCT in the early phase after pediatric liver transplantation may indicate a poor postoperative outcome. However, it is important to note that there was no correlation between PCT elevation and systemic infection in this study. |

Summarizes the average scores across 100 questions for each embedding model, which can be seen in Table below.

| Embedding Model | MRR | SAS | Faithfulness |
| :--- | :---: | :---: | :---: |
| PubMedBERT-base | 0.985 | 0.718 | 0.949 |
| GIST-large-Embedding-v0 | **1.000** | 0.814 | **0.950** |
| BioLORD-2023 | 0.910 | 0.676 | 0.939 |
|  b1ade-embed-kd | 0.866 | **0.855** | 0.916 |
| all-MiniLM-L6-v2 | 0.975 | 0.715 | 0.932 |

## Conclusion 
In this study, the RAG concept with LLM was successfully applied to generate answers in question-answering using PubMed QA data to address medical-related questions.The LLM with the Mistral 7B model was utilized locally to generate answers, focusing on private use for  healthcare  institutions.In  addition,  the  embedding  model  is  used  to  index  the  documents  so  that  the  LLM  model  can  only  answer based  on  the  available  context  data  to  avoid  hallucinatory  answers.In  the  application,  5  different  embedding  models  are  used: PubMedBERT-base  and  BioLORD-2023  models  are  models  trained  on  medical  field  corpora,  and  GIST-large-Embedding-v0,   b1ade-embed-kd,  and  all-MiniLM-L6-v2  models  are  models  trained  on  a  variety  of  different  domains.It  was  found  that  general  embedding models, specifically GIST-large-Embedding-v0 and  b1ade-embed-kd, outperformed domain-specific models such as PubMedBERT and BioLORD-2023   in   terms   of   MRR,   SAS,   and   Faithfulness   metrics.GIST-large-Embedding-v0   demonstrated   superior   retrieval performance  with  perfect  MRR  scores  (1.000)  and  high  faithfulness  (0.950),  while   b1ade-embed-kd  excelled  in  semantic  similarity (0.855),  both  significantly  outperforming  specialized  medical  models. This  suggests  that  broader  training  data  improves  semantic retrieval  and  answer  generation  in  a  medical  context.Leveraging  RAG  and  LLM  locally  can  minimize  privacy  problems  while  still maintaining high-quality responses, which can be a solution for a safe and reliable medical question-answering system.

## Citation
If you use this work, please cite:
```
@article{prayitno2025conversational,
  author = {La Ode Muhammad Yudhy Pryaitno and Annisa Nurfadilah and Septiyani Bayu Saudi and Widya Dwi Tsunami and Adha Mashur Sajiah},
  title = {Conversational Agent for Medical Question-Answering Using RAG and LLM},
  journal = {Journal of Artificial Intelligence and Engineering Applications},
  volume = {4},
  number = {3},
  pages = {1077},
  year = {2025},
  month = {June},
  issn = {2808-4519},
  doi = {10.59934/jaiea.v4i3.1077},
  url = {https://ioinformatic.org/index.php/JAIEA/article/view/1077/751},
  publisher = {Yayasan Kita Menulis}
}
```
