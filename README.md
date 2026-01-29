RAG Knowledge Forge
===================

1. For Embeddings (Current Implementation):

- Using sentence-transformers/all-MiniLM-L6-v2 for vector embeddings 
- 384 dimension vectors for similarity search 
- MongoDB for vector storage


2. For Text Generation/Summarization (Current Implementation):
- Primary: facebook/bart-large-cnn via HuggingFace pipeline
- Backup: LSA (Latent Semantic Analysis) summarizer
- Final Fallback: Basic extractive summarization


## Directory structure and necessary files:

```bash
├── 0-Instalation.sh
├── 1-Large-Text-Chunking.py
├── 2-RAG-Indexer.py
├── 3-MongoDB-Explorer.py
├── 4-RAG-Search.py
├── core
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   ├── data_ingestion.py
│   ├── query.py
│   ├── README.md
│   └── vectorization.py
├── data
│   ├── large_text_chunks.json
│   └── The-Gerson-Therapy-Reduced.json
├── logs
│   ├── chunking.log
│   ├── indexing.log
│   └── search.log
├── __pycache__
│   └── utils.cpython-311.pyc
├── README.md
├── source
│   └── The-Gerson-Therapy-Reduced.txt
└── utils.py
```

## Question and Result
```
+------------------------------------------------------------------------+
| Orthomolecular Medicine Search                                         |
| ====================================================================== |
| Using GPU: NVIDIA GeForce RTX 3060                                     |
| Enter 'exit' to quit                                                   |
+------------------------------------------------------------------------+

Enter your question: Benefits of Vitamin C


📚 Search Results:
====================================================================================================

+------------------------------------------------------------------------------------------------------+
| Generated Response                                                                                   |
| ---------------------------------------------------------------------------------------------------- |
| Based on the orthomolecular medicine text:                                                           |
|                                                                                                      |
| Gerson found that such liver therapy brings about the restoration of new red blood corpuscles        |
| (reticulocytes) Vitamin C is used supplementally as a tool for fighting infection, and as part of a  |
| pain-relieving triad of natural and nontoxic medications. Never use calcium or sodium ascorbate,     |
| since these two particular products will bring about serious detrimental effects. The best defense   |
| apparatus is a 100 percent functioning metabolism and reabsorption in the intestinal tract in        |
| combination with a healthy liver.                                                                    |
+------------------------------------------------------------------------------------------------------+

```


