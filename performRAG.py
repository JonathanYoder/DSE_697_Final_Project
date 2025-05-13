from langchain_core.documents import Document
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import utils as chromautils

from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

# The query for finetuned model, both with and without RAG
question = 'What are some common illnesses in chickens?'

print('Imported, reading docs')

# Read in PDF information and store as list of Documents
with open('./pdfs_as_docs.json','r') as f:
    doc_dict = json.load(f)
docs = [Document(**doc) for doc in doc_dict]

# Chunk the data
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
chunked_docs = splitter.split_documents(docs)
chunked_docs = chromautils.filter_complex_metadata(chunked_docs)
print('Num doc chunks:',len(chunked_docs))

print('Getting embeddings')

# Get embeddings for chunked data and create a retriever for RAG usage
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", cache_folder='./.cache/HF_Embeddings')
vectorstore = Chroma.from_documents(chunked_docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Get retrieved data for given query & print part of response
response = retriever.invoke(question)
for i in range(len(response)):
    print(i,':',response[i].page_content[:100],'...') 


print('Getting model')

# Use finetuned model
model_name = "../TennAG_model_1ep"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
# )

model = AutoModelForCausalLM.from_pretrained(model_name)#, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained('/gpfs/wolf2/olcf/trn040/proj-shared/knelms3_jyoder5/huggingface/meta-llama/Meta-Llama-3-8B')#model_name)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

# Create pipeline for generating answers using finetuned model
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    #lr_scheduler_type="constant",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=512,
    eos_token_id=terminators,
    pad_token_id = tokenizer.pad_token_id,
)

print('Making pipeline')

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Prompt for giving the LLM instructions, a question, and context from retrieved data
prompt_template = """
<|start_header_id|>user<|end_header_id|>
You are an assistant for answering questions using provided context.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
Question: {input}
Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=prompt_template,
)
# # Several prompt structures were tested
# system_prompt = (
#     "Use the given context to answer the question. "
#     "If you don't know the answer, say you don't know. "
#     "Use three sentences maximum and keep the answer concise. "
#     "Context: {context}"
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# Create a RAG chain with the LLM pipeline and retriever
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

# Answer the question directly using the LLM
print()
print('No RAG:')
no_rag_answer = llm.invoke(question)
print(no_rag_answer)
# Answer the question using RAG
print()
print('RAG:')
rag_answer = chain.invoke({"input": question})
print(rag_answer)

# Summarize the question and responses
new_entry = {}
new_entry['question'] = question
new_entry['no_rag'] = no_rag_answer
new_entry['rag'] = {'context':str(rag_answer['context']), 
                    'answer':rag_answer['answer']}

# Update file with this query and the responses generated
with open('question_answer.json', 'r') as file:
    qa_data = json.load(file)
qa_data.append(new_entry)
with open('question_answer.json', 'w') as file:
    json.dump(qa_data, file, indent=4)