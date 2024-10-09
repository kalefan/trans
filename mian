import os
from flask import Flask, request, jsonify
from transformers import MarianTokenizer, MarianMTModel

# 指定模型下载或保存的目录
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)  # 创建目录如果不存在

# 加载预训练的MT0模型和分词器
tokenizer = MarianTokenizer.from_pretrained("./models/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6")
model = MarianMTModel.from_pretrained("./models/models--Helsinki-NLP--opus-mt-zh-en/snapshots/cf109095479db38d6df799875e34039d4938aaa6")

app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    chinese_text = data.get('text')

    if not chinese_text:
        return jsonify({"error": "No text provided"}), 400

    # 进行翻译
    translated_tokens = model.generate(**tokenizer.prepare_seq2seq_batch([chinese_text], return_tensors="pt"))
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

    return jsonify({"translation": translated_text[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11123)
