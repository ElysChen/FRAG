from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
import os
import shutil
from tushare_func import opt_basic, fut_basic, sge_basic
import json

# ---- 懒加载全局变量 ----
embeddings = None
func_db = None
detail_db = None

function_map = {
    "fut_basic": fut_basic,
    "sge_basic": sge_basic,
    "opt_basic": opt_basic,
}

def read_data(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def get_chunks(filename: str) -> list[str]:
    content = read_data(filename)
    chunks = content.split('\n')
    result = []
    header = ""
    count = 0
    for c in chunks:
        if '#' in c:
            if count > 0:
                result.append(header)
                header = ""
            header += f"{c}\n"
            count += 1
        else:
            header += f"{c}\n"
    if header != "":
        result.append(header)
    return result

def build_vector_db(filename: str, db_name: str):
    chunks = get_chunks(filename)
    documents = [Document(page_content=chunk) for chunk in chunks]
    if os.path.exists(db_name):
        shutil.rmtree(db_name)
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_name
    )
    vector_db.persist()

def lazy_init():
    global embeddings, func_db, detail_db
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )
    if not os.path.exists("./func_db"):
        build_vector_db("./func_db.md", "./func_db")

    if not os.path.exists("./parameter_db"):
        build_vector_db("./parameter_db.md", "./parameter_db")

    if func_db is None:
        func_db = Chroma(persist_directory="./func_db", embedding_function=embeddings)

    if detail_db is None:
        detail_db = Chroma(persist_directory="./parameter_db", embedding_function=embeddings)

        
from openai import OpenAI
def ask_chatgpt4(user_prompt: str, system_role: str = "你是一个善于整理金融知识的助手，会根据用户提供的上下文判断所需填入的参数") -> str:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "sk-WR7SVoynldaKd0dNbd05EPXw0Bg3p2FhCozf3dtKqcNEOtWS"),
        base_url="https://api.chatanywhere.tech/v1"
    )
    try:
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"请求失败: {str(e)}"

def answer_question(question: str) -> str:
    lazy_init()  # 确保初始化仅第一次执行
    results = func_db.similarity_search(question, k=1)
    func = "".join([doc.page_content for doc in results])
    results = detail_db.similarity_search(question + func, k=1)
    details = "".join([doc.page_content for doc in results])

    user_input = f"""
    [用户问题]
    {question}

    [需要调用的函数及其要求]
    {details}

    [回答格式要求]
    请根据用户的问题，判断应该调用哪个函数，并给出该函数所需的参数及其值，输出格式如下：
    {{"function": "函数名", "parameters": {{"参数1": "值1", "参数2": "值2", ...}}}}
    注意：请严格使用JSON格式输出，不要输出多余内容。
    """

    response_text = ask_chatgpt4(user_input)
    try:
        call_info = json.loads(response_text)
        func_name = call_info["function"]
        params = call_info.get("parameters", {})
        result = function_map[func_name](**params)
        return str(result)
    except Exception as e:
        return f"调用失败: {str(e)}\n模型回复: {response_text}"
