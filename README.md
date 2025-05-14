# DSE_697_Final_Project

Agricultural Extension work is responsible for sharing agricultural research and knowledge with farmers and producers. While extension agents are available, it would be convenient to have a tool which can aid both agents and stakeholders. This project aims to create an extension tool for Tennessee which can answer questions and disseminate knowledge in a reliable format. This tool can be adapted to other regions by changing the RAG database with regional extension sources.

## Finetuning
The LLaMA3-8B model was finetuned using an agricultural QA dataset using SFT and LoRA.
Usage:
- run 'fetch_model.py' to download model
- run 'fetch_data.py' to download data
- run 'training_script.sh' to train model with data using python script file 'train_model.py'
    - Output and error are in 'training-6354.out' and training-6354.err' respectively
    - Model is located in 'TennAG_llama_1ep'
- run 'run_inf.sh' to run inference on trained model using python script file 'inference.py'
    - Output and error of example run are in 'inference-6413.out' and 'inference-6413.err' respectively

## Retrieval Augmented Generation (RAG)
- Extension document PDFs in the pdf folder are converted to text using the PyMuPDF library
    - Only one example file included in GitHub, but 560 were used, downloaded from https://utextension.tennessee.edu/publications/
    - This data is saved to a JSON file (pdfs_as_docs.json)
- The model BAAI/bge-base-en-v1.5 is used for embedding this text into chunks of length 512 with 30 token overlap and a retriever is made to access this information
- A pipeline is created for the LLM
- A prompt template allows the question, context, and instructions to be passed to the LLM
- A RAG chain is created linking together the retriever with the LLM via the prompt template
- Finally, each question/answer pair is saved to a file for later assessment (question_answer.json)
