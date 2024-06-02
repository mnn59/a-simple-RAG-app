# LLM

## Get Started
- Download **llama-2** model from [here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin)
- Run **`pip install -r requirements.txt`** to install requirements
- Put **pdf** files in **`data`** directory
- Run **`python ingest.py`** to create vector database
- Run **`uvicorn app:app --reload`** to start chatbot
    - You can run in different host with this command: **`uvicorn app:app --reload --host 192.168.x.x --port x`**

```md
├───data
|   └───*.pdf
├───templates
|   └───index.html
├───vectorstore
│   └───db_faiss
├───app.py
├───ingest.py
└───llama-2-7b-chat.ggmlv3.q8_0.bin
```

## Notes
- If you have a compatible GPU installed, modify the `model_kwargs` in `ingest.py` and `app.py` to use `'cuda:0'` instead of `'cpu'`.
- For systems with high RAM, modify the `max_new_tokens` in `app.py` to `1024` to get more context.


