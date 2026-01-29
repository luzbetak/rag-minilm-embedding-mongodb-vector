#!/bin/bash
#
pip install loguru pymongo sentence-transformers
pip install transformers torch sumy nltk rouge
pip install transformers torch sentence-transformers pymongo loguru python-dotenv

mkdir -p core data logs
touch core/__init__.py
