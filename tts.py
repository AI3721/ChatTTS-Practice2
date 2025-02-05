import ex
import sys
import torch
import random
import torchaudio
import numpy as np
import gradio as gr
from models import ChatTTS
from utils import TorchSeedContext, clear_gpu_cache, load_audio
from utils import get_logger, float_to_int16, pcm_arr_to_mp3_view

INT16_MAX = np.iinfo(np.int16).max
has_interrupted = False
seed_max = 99999999
seed_min = 1
speakers = {
    "Default": {"seed": 21},
    "Speaker1": {"seed": 1111},
    "Speaker2": {"seed": 2222},
    "Speaker3": {"seed": 3333},
    "Speaker4": {"seed": 4444},
    "Speaker5": {"seed": 5555},
    "Speaker6": {"seed": 6666},
    "Speaker7": {"seed": 7777},
    "Speaker8": {"seed": 8888},
    "Speaker9": {"seed": 9999},
}

# 加载模型
logger = get_logger("TTS")
chat = ChatTTS.Chat(get_logger("ChatTTS"))
logger.info("Initializing ChatTTS...")
if chat.load():
    logger.info("Models loaded successfully.")
else:
    logger.error("Models load failed.")
    sys.exit(1)

# js插入文本
def js_insert(content):
    return f"""value => {{
        var content = ' {content} ';
        var text_div = document.getElementById('input_text');
        var textarea = text_div.querySelector('textarea');
        var cursorPosition = textarea.selectionStart;
        var value = textarea.value;
        var new_value = value.substring(0, cursorPosition) + content + value.substring(cursorPosition);
        textarea.value = new_value;
        textarea.selectionStart = cursorPosition + content.length;
        textarea.selectionEnd = cursorPosition + content.length;
        textarea.focus();
        return new_value
    }}"""
# 更换sample
def change_sample(sample_audio):
    if sample_audio is None:
        return ""
    file = open(sample_audio, 'rb')
    sample_audio = load_audio(file, 24000)
    return chat.sample_audio_speaker(sample_audio)
# 更新spk_emb
def update_emb(audio_seed):
    with TorchSeedContext(audio_seed):
        return chat.sample_random_speaker()
# 更新dvae_coef
def update_dvae(coef):
    if not has_interrupted:
        return coef
    chat.unload()
    chat.load()
    return chat.coef
# 中断生成
def interrupt():
    global has_interrupted
    has_interrupted = True
    chat.interrupt()
# 细化文本
def refine_text(text, text_seed, text_refine, temperature, top_P, top_K):
    if not text or not text_refine:
        return text
    # 自定义的文本正则优化方法
    from norm import normalize
    if isinstance(text, list):
        text = [normalize(t) for t in text]
    else:
        text = normalize(text)
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        temperature=temperature, top_P=top_P, top_K=top_K)
    with TorchSeedContext(text_seed):
        clear_gpu_cache() # 清理缓存
        texts = chat.infer(
            text=text,
            refine_text_only=True,
            skip_refine_text=False,
            params_refine_text=params_refine_text)
    if len(texts) == 1: # 不显示中括号['']
        return texts[0]
    return texts
# 生成音频
def generate_audio(text, audio_seed, stream, temperature, top_P, top_K, spk_emb, sample_emb, sample_text):
    if not text or not spk_emb.startswith("蘁淰"):
        return None
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        temperature=temperature, top_P=top_P, top_K=top_K, spk_emb=spk_emb)
    if sample_emb and sample_text:
        params_infer_code.spk_emb = None
        params_infer_code.spk_smp = sample_emb
        params_infer_code.txt_smp = sample_text
    with TorchSeedContext(audio_seed):
        clear_gpu_cache() # 清理缓存
        wavs = chat.infer(
            text=text,
            stream=stream,
            skip_refine_text=True,
            params_infer_code=params_infer_code)
    if stream:
        for wav_list in wavs: # 这里wavs是迭代器
            wav = wav_list[0] # 这里列表长度为 1
            if wav is not None and len(wav) > 0:
                yield 24000, float_to_int16(wav)
    elif len(wavs) == 1: # 只有一段音频返回
        yield 24000, float_to_int16(wavs[0])
    else: # 因为在TTS生成的多元数组中短音频会自动补零
        def drop_fill(list):
            while list[-1]==0: # 所以要去除多余的零
                list.pop()
            if len(list) < 24000: # 防止短音频太短
                add = [0]*((24000-len(list))//2)
                list = add + list + add
            return list
        wavs = [float_to_int16(wav).tolist() for wav in wavs]
        yield [np.array(drop_fill(wav), dtype=np.int16) for wav in wavs]
# 增强音频
def enhance_audio(audio, audio_denoise, audio_enhance, CFM_method, CFM_temperature, CFM_strength, CFM_eval):
    from enhance.enhancer.inference import denoise, enhance
    if isinstance(audio, str):
        dwav, sr = torchaudio.load(audio) # 双声道, float32
        dwav = dwav.mean(dim=0) # 转为单声道
        if audio_denoise:
            new_dwav, new_sr = denoise(dwav, sr, device='cuda')
        if audio_enhance:
            new_dwav, new_sr = enhance(dwav, sr, device='cuda', solver=CFM_method.lower(),
                                       nfe=CFM_eval, lambd=CFM_strength, tau=CFM_temperature)
        new_dwav = new_dwav.unsqueeze(0).cpu() # 转为双声道
        new_audio = "/tmp/gradio/enhanced_audio.wav" #
        torchaudio.save(new_audio, new_dwav, new_sr)
        torch.cuda.empty_cache()
        yield new_audio
    if isinstance(audio, tuple):
        sr, dwav = audio # int16, 单声道
        dwav = dwav.astype(np.float32)/INT16_MAX # 转为float32
        dwav = torch.tensor(dwav,dtype=torch.float32)
        if audio_denoise:
            new_dwav, new_sr = denoise(dwav, sr, device='cuda')
        if audio_enhance:
            new_dwav, new_sr = enhance(dwav, sr, device='cuda', solver=CFM_method.lower(),
                                       nfe=CFM_eval, lambd=CFM_strength, tau=CFM_temperature)
        new_dwav = new_dwav.cpu().numpy() # 转移到cpu内存，再转换为numpy数组
        new_dwav = (new_dwav*INT16_MAX).astype(np.int16) # 转为int16
        new_audio = new_sr, new_dwav
        torch.cuda.empty_cache()
        yield new_audio


# 更换speaker
def change_speaker(speaker):
    return speakers.get(speaker)["seed"]
# 随机speaker
def random_speaker():
    return gr.update(value=random.choices(list(speakers.keys()))[0])
# 随机种子
def random_seed():
    return gr.update(value=random.randint(seed_min, seed_max))
# 更换按钮
def change_button(generate_btn, interrupt_btn, is_visible):
    interrupt_btn = gr.update(visible=not is_visible)
    generate_btn = gr.update(visible=is_visible)
    return generate_btn, interrupt_btn
# 设置生成前按钮
def set_button_before(generate_btn, interrupt_btn):
    global has_interrupted
    has_interrupted = False
    return change_button(generate_btn, interrupt_btn, is_visible=has_interrupted)
# 设置生成后按钮
def set_button_after(generate_btn, interrupt_btn):
    global has_interrupted
    has_interrupted = True
    return change_button(generate_btn, interrupt_btn, is_visible=has_interrupted)

####################################################################################################

# 创建tts交互界面
def create_tts_block():
    with gr.Blocks() as tts_block:
        with gr.Column():
            # gr.Markdown("<h1 style='text-align: center; font-size: 2em'>Chat TTS</h1>")
            # Input
            with gr.Row():
                with gr.Column(min_width=0, scale=2):
                    with gr.Tab(label="Input Text"):
                        input_text = gr.Textbox(label="", lines=12, max_lines=12, show_copy_button=True, interactive=True, value=ex.input_text, elem_id="input_text")
                    with gr.Tab(label="Sample Text"):
                        sample_text = gr.Textbox(label="", lines=12, max_lines=12, show_copy_button=True, interactive=True, value="空")
                with gr.Column(min_width=0, scale=1):
                    with gr.Tab(label="Sample Audio"):
                        sample_audio = gr.Audio(label="", type="filepath", waveform_options=gr.WaveformOptions(sample_rate=24000))
                    with gr.Tab(label="Emb"):
                        sample_emb = gr.Textbox(label="", lines=12, max_lines=12, show_copy_button=True, interactive=True)
            # Button
            with gr.Row():
                with gr.Column(min_width=0, scale=2):
                    with gr.Row(): # 文本插入辅助控制词
                        laugh_btn = gr.Button("笑声 + [laugh]", min_width=0)
                        lbreak_btn = gr.Button("长停顿 + [lbreak]", min_width=0)
                        uv_break_btn = gr.Button("短停顿 + [uv_break]", min_width=0)
                with gr.Column(min_width=0, scale=1):
                    with gr.Row(): # 音频降噪或音频增强
                        is_ture = gr.Checkbox(value=True, visible=False)
                        is_false = gr.Checkbox(value=False, visible=False)
                        denoise_btn = gr.Button("音频降噪", min_width=0, interactive=True)
                        enhance_btn = gr.Button("音频增强", min_width=0, interactive=True)
            # 可调参数
            with gr.Accordion(label="可调参数：点击这里，可以尝试生成不同音色的声音！", open=False):
                # Checkbox
                with gr.Row():
                    with gr.Column(min_width=0):
                        CFM_method = gr.Dropdown(label="CFM_Method", choices=["Midpoint", "Euler", "RK4"], value="Midpoint", interactive=True, container=False)
                    with gr.Column(min_width=0):
                        audio_denoise = gr.Checkbox(label="Audio Denoise", value=False, interactive=True)
                    with gr.Column(min_width=0):
                        audio_enhance = gr.Checkbox(label="Audio Enhance", value=True, interactive=True)
                # Slider
                with gr.Row():
                    CFM_temperature = gr.Slider(label="CFM_Temperature", minimum=0, maximum=1, step=0.01, value=0.3, interactive=True, min_width=0)
                    CFM_strength = gr.Slider(label="CFM_Strength", minimum=0, maximum=1, step=0.05, value=0.7, interactive=True, min_width=0)
                    CFM_eval = gr.Slider(label="CFM_Eval", minimum=1, maximum=128, step=1, value=64, interactive=True, min_width=0)
                # Checkbox
                with gr.Row():
                    with gr.Column(min_width=0):
                        auto_play = gr.Checkbox(label="Auto Play", value=False, interactive=True, min_width=0)
                    with gr.Column(min_width=0):
                        stream_mode = gr.Checkbox(label="Stream Mode", value=False, interactive=True, min_width=0)
                    with gr.Column(min_width=0):
                        text_refine = gr.Checkbox(label="Text Refine", value=True, interactive=True, min_width=0)
                # Slider
                with gr.Row():
                    temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.01, value=0.3, interactive=True, min_width=0)
                    top_p = gr.Slider(label="top_P", minimum=0, maximum=1, step=0.05, value=0.7, interactive=True, min_width=0)
                    top_k = gr.Slider(label="top_K", minimum=1, maximum=20, step=1, value=10, interactive=True, min_width=0)
                # Tab
                with gr.Row():
                    with gr.Column(min_width=0):
                        with gr.Tab(label="Speaker"):
                            speaker = gr.Dropdown(show_label=False, choices=speakers.keys(), value="Default", interactive=True)
                        speaker_btn = gr.Button("\U0001F3B2")
                    with gr.Column(min_width=0):
                        with gr.Tab(label="Audio Seed"):
                            audio_seed = gr.Number(show_label=False, minimum=seed_min, maximum=seed_max, value=21, interactive=True)
                        with gr.Tab(label="Emb"):
                            spk_emb = gr.Textbox(show_label=False, max_lines=1, value=update_emb(audio_seed.value), interactive=True)
                        audio_seed_btn = gr.Button("\U0001F3B2")
                    with gr.Column(min_width=0):
                        with gr.Tab(label="Text Seed"):
                            text_seed = gr.Number(show_label=False, minimum=seed_min, maximum=seed_max, value=21, interactive=True)
                        with gr.Tab(label="DVAE"):
                            dvae_coef = gr.Textbox(show_label=False, max_lines=1, interactive=True)
                        text_seed_btn = gr.Button("\U0001F3B2")
                reload_btn = gr.Button("Reload (Update DVAE)")
            # Button
            generate_btn = gr.Button("Generate", variant="primary")
            interrupt_btn = gr.Button("Interrupt", variant="stop", visible=False)
            output_text = gr.Textbox(label="Output Text", lines=4, max_lines=4, show_copy_button=True, interactive=False)
            # 响应事件
            laugh_btn.click(fn=None, outputs=input_text, js=js_insert("[laugh]"))
            lbreak_btn.click(fn=None, outputs=input_text, js=js_insert("[lbreak]"))
            uv_break_btn.click(fn=None, outputs=input_text, js=js_insert("[uv_break]"))
            audio_denoise.change(fn=lambda a,b: (a!=True)&b, inputs=[audio_denoise, audio_enhance], outputs=audio_enhance, show_progress=False)
            audio_enhance.change(fn=lambda a,b: (a!=True)&b, inputs=[audio_enhance, audio_denoise], outputs=audio_denoise, show_progress=False)
            denoise_btn.click(fn=enhance_audio, inputs=[sample_audio, is_ture, is_false, CFM_method, CFM_temperature, CFM_strength, CFM_eval], outputs=sample_audio)
            enhance_btn.click(fn=enhance_audio, inputs=[sample_audio, is_false, is_ture, CFM_method, CFM_temperature, CFM_strength, CFM_eval], outputs=sample_audio)
            # 响应事件
            sample_audio.change(fn=change_sample, inputs=sample_audio, outputs=sample_emb)
            speaker_btn.click(fn=random_speaker, outputs=speaker)
            speaker.change(fn=change_speaker, inputs=speaker, outputs=audio_seed)
            audio_seed_btn.click(fn=random_seed, outputs=audio_seed)
            audio_seed.change(fn=update_emb, inputs=audio_seed, outputs=spk_emb)
            text_seed_btn.click(fn=random_seed, outputs=text_seed)
            reload_btn.click(fn=update_dvae, inputs=dvae_coef, outputs=dvae_coef)
            interrupt_btn.click(fn=interrupt)
            # output
            @gr.render(inputs=[auto_play, stream_mode])
            def make_audio(autoplay, streaming):
                output_audio = gr.Audio(
                    autoplay=autoplay, streaming=streaming,
                    label="Output Audio", interactive=False,
                    format='mp3' if not streaming else 'wav',
                    waveform_options=gr.WaveformOptions(sample_rate=24000))
                enhanced_audio = gr.Audio(
                    autoplay=autoplay, streaming=streaming,
                    label="Enhanced Audio", interactive=False,
                    format='mp3' if not streaming else 'wav',
                    waveform_options=gr.WaveformOptions(sample_rate=44100))
                generate_btn.click(
                    fn=set_button_before, 
                    inputs=[generate_btn, interrupt_btn],
                    outputs=[generate_btn, interrupt_btn]
                ).then(
                    fn=refine_text,
                    inputs=[input_text, text_seed, text_refine, temperature, top_p, top_k],
                    outputs=output_text
                ).then(
                    fn=generate_audio,
                    inputs=[output_text, audio_seed, stream_mode, temperature, top_p, top_k, spk_emb, sample_emb, sample_text],
                    outputs=output_audio
                ).then(
                    fn=enhance_audio,
                    inputs=[output_audio, audio_denoise, audio_enhance, CFM_method, CFM_temperature, CFM_strength, CFM_eval],
                    outputs=enhanced_audio
                ).then(
                    fn=set_button_after,
                    inputs=[generate_btn, interrupt_btn],
                    outputs=[generate_btn, interrupt_btn]
                )
            
            # 更多玩法
            with gr.Accordion(label="更多玩法：点击这里，可以体验更多新奇有趣的玩法！", open=False):
                from tts_more import create_tts_more_block
                create_tts_more_block(temperature, top_p, top_k)
    
    return tts_block