from flask import Flask, render_template, request, jsonify, Response
import os
import base64
from openai import OpenAI
import json
import tiktoken  # 新增
import importlib.util
from hug_build import answer_question
app = Flask(__name__)




client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-WR7SVoynldaKd0dNbd05EPXw0Bg3p2FhCozf3dtKqcNEOtWS"),
    base_url="https://api.chatanywhere.tech/v1"
)

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_data_from_script(script_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_data()

def get_all_data(question):
    results = []
    try:
        results.append("数据:\n" + answer_question(question))
    except Exception as e:
        results.append("[数据获取失败] " + str(e))
    return "\n\n".join(results)


def num_tokens_from_messages(messages, model="gpt-4.1"):
    # gpt-4.1兼容gpt-4
    encoding = tiktoken.encoding_for_model("gpt-4")
    num_tokens = 0
    for message in messages:
        if isinstance(message.get("content"), list):
            # 带图片的情况
            for part in message["content"]:
                if isinstance(part, dict) and part.get("type") == "text":
                    num_tokens += len(encoding.encode(part.get("text", "")))
        else:
            num_tokens += len(encoding.encode(message.get("content", "")))
        num_tokens += 4  # 每条消息的role等开销
    num_tokens += 2  # 额外开销
    return num_tokens

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.form
        image = request.files.get('image')
        history = data.get('history')
        if history:
            messages = json.loads(history)
        else:
            messages = [
                {"role": "system", "content": "你是一个金融分析助手，你可以有思考过程，但不要给我看，我只要看答案。你的唯一信息来源是下方提供的数据，所有回答都必须基于这些数据，不允许凭空编造，不允许瞎想。不要出现根据与“提供数据”类似的字眼，就好像是你自己提供的数据一样，从而增加用户沉浸感。如果你看不懂我的数据，你需要自己联想，但不要把联想的过程呈现给我。"}
            ]
        if image:
            base64_image = encode_image(image)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }}
                ]
            })
        else:
            user_input = data.get('message', '').strip()
            if user_input:
                # 获取所有数据
                all_data = get_all_data(user_input)
                compound_prompt = (
                    f"用户问题：{user_input}\n\n"
                    f"【请严格仅根据以下数据作答，不允许凭空猜测】\n"
                    f"{all_data}"
                )
                messages.append({"role": "user", "content": compound_prompt})

        # 统计输入tokens
        prompt_tokens = num_tokens_from_messages(messages)

        def generate():
            ai_text = ''
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )
            for chunk in response:
                if not chunk.choices or len(chunk.choices) == 0:
                    continue  # 跳过空choices
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    ai_text += delta.content
                    yield delta.content


            encoding = tiktoken.encoding_for_model("gpt-4")
            completion_tokens = len(encoding.encode(ai_text))

            total_tokens = prompt_tokens + completion_tokens
            stats = {
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }
            yield f"\n[[[TOKEN_STATS]]]{json.dumps(stats)}"

        return Response(generate(), mimetype='text/plain')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
