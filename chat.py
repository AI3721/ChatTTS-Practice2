import datetime
import gradio as gr
from utils import clear_gpu_cache
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


model_path = "models/Qwen-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)


def get_chat_response(prompt, history=[], temperature=0.5):
    answer, history = model.chat(
        tokenizer, query=prompt, history=history,
        top_p=0.7, temperature=0.9, max_new_tokens=1024)
    # 获取当前时间，打印日志
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"【{time}】 Q: {prompt}  A: {answer}")
    clear_gpu_cache()
    return answer

def respond(prompt, chatbot, temperature):
    answer = get_chat_response(prompt, chatbot, temperature)
    chatbot.append([prompt, answer])
    return "", chatbot

def create_chat_block():
    with gr.Blocks() as chat_block:
        with gr.Column():
            # gr.Markdown("<h1 style='text-align: center; font-size: 2em'>Chat Bot</h1>")
            chatbot = gr.Chatbot(label="AI 3721", avatar_images=['image/ME.png', 'image/AI.png'])
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, value=0.9)
            prompt = gr.Textbox(label="Prompt", lines=3, max_lines=3, autofocus=True)

            with gr.Row():
                clear_btn = gr.ClearButton(components=[prompt, chatbot], value="清空")
                submit_btn = gr.Button("提交", variant='primary')

            submit_btn.click(fn=respond,inputs=[prompt, chatbot, temperature], outputs=[prompt, chatbot])
            prompt.submit(fn=respond,inputs=[prompt, chatbot, temperature], outputs=[prompt, chatbot])
    
    return chat_block