# 1:安裝必要的 Python 套件

!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q googletrans==4.0.0rc1
!pip install -q scikit-image
!pip install -q opencv-python # 圖像分析
!pip install -q diffusers transformers accelerate invisible-watermark safetensors
!pip install -q ipywidgets

print("1:安裝完成。")

# 2:匯入專案中會用到的所有 Python 模組

import torch
import clip
from PIL import Image, ImageDraw, ImageFont
from googletrans import Translator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab
import cv2
from diffusers import StableDiffusionPipeline
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
import time
import textwrap
import matplotlib.font_manager as fm

print("2:函式庫匯入完成。")

# 3:設定 Matplotlib 字體

print("3:正在設定 Matplotlib CJK 字體...")
font_path_cjk = '/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf'
if os.path.exists(font_path_cjk):
    try:
        fm.fontManager.addfont(font_path_cjk)
        prop = fm.FontProperties(fname=font_path_cjk)
        font_name = prop.get_name()
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False
        print(f"  已成功設定 CJK 字體為: {font_name}")
    except Exception as e:
        print(f"  設定字體 '{font_path_cjk}' 時發生錯誤: {e}. 退到通用列表。")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
else:
    print(f"  指定的 CJK 字體文件路徑不存在: {font_path_cjk}. 退到通用列表。")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
print("3:Matplotlib CJK 字體設定完畢。")

# Cell 4: 設定 PyTorch 的運算設備

if torch.cuda.is_available(): device_to_use = "cuda"
else: device_to_use = "cpu"
print(f"4:本次運行的計算設備是: {device_to_use}")
if device_to_use == "cpu": print("警告：未使用GPU！運行大型AI模型會非常慢。")

# 5:載入 OpenAI CLIP 模型

clip_model_name_to_load = "ViT-B/32"
clip_model_instance, clip_image_preprocess_fn = None, None
print(f"5:準備載入 CLIP 模型 ({clip_model_name_to_load})...")
if device_to_use == "cuda":
    try:
        clip_model_instance, clip_image_preprocess_fn = clip.load(clip_model_name_to_load, device=device_to_use)
        clip_model_instance.eval()
        print(f"  CLIP 模型 ({clip_model_name_to_load}) 已成功載入到 {device_to_use}！")
        torch.cuda.empty_cache()
    except Exception as e: print(f"  載入CLIP模型時發生錯誤: {e}")
else: print(f"  未實際載入CLIP模型，因為當前運算設備是 {device_to_use}。")

# 6:載入 Stable Diffusion 圖像生成模型

sd_model_id_to_load = "runwayml/stable-diffusion-v1-5"
image_generation_pipeline = None
print(f"6:準備載入 Stable Diffusion 模型 ({sd_model_id_to_load})...")
if device_to_use == "cuda":
    try:
        image_generation_pipeline = StableDiffusionPipeline.from_pretrained(sd_model_id_to_load, torch_dtype=torch.float16)
        image_generation_pipeline = image_generation_pipeline.to(device_to_use)
        print(f"  Stable Diffusion 模型 ({sd_model_id_to_load}) 已成功載入到 {device_to_use}！")
        torch.cuda.empty_cache()
    except Exception as e: print(f"  載入Stable Diffusion模型時發生錯誤: {e}")
else: print(f"  未實際載入Stable Diffusion模型，因為當前運算設備是 {device_to_use}。")

# 7:初始化 Google 翻譯工具

google_translator_instance = None
print("7:正在初始化 Google 翻譯工具...")
try:
    google_translator_instance = Translator()
    test_text = "你好"; translated_test_text = google_translator_instance.translate(test_text, src='zh-cn', dest='en').text
    print(f"  Google 翻譯工具初始化成功！測試翻譯 '{test_text}' -> '{translated_test_text}'")
except Exception as e: print(f"  Google 翻譯工具初始化失敗: {e}")

# 8:定義專案要分析的詞彙概念及多語言翻譯

word_concepts_list = [
    {"concept_name": "cool_ambiguous", "base_chinese": "酷 / 涼爽", "translations": {"zh": "冰涼的飲料，酷炫的風格，冷靜的態度", "en": "cool refreshing drink, cool stylish look, calm and cool attitude", "ja": "冷たい飲み物、かっこいいスタイル、冷静な態度", "ko": "시원한 음료, 멋진 스타일, 침착한 태도"}},
    {"concept_name": "soft_ambiguous", "base_chinese": "柔軟", "translations": {"zh": "柔軟，溫和的觸感，蓬鬆的雲朵", "en": "soft, gentle touch, fluffy clouds", "ja": "柔らかい、優しい手触り、ふわふわの雲", "ko": "부드럽다, 순한 촉감, 푹신한 구름"}},
    {"concept_name": "bright_ambiguous", "base_chinese": "明亮", "translations": {"zh": "明亮的光線，充滿希望的未來，聰明睿智的想法", "en": "bright light, hopeful future, intelligent idea", "ja": "明るい光、希望に満ちた未来、賢いアイデア", "ko": "밝은 빛, 희망찬 미래, 똑똑한 생각"}},
    {"concept_name": "dark_ambiguous", "base_chinese": "黑暗", "translations": {"zh": "漆黑的夜晚，神秘的森林深處，陰沉憂鬱的心情", "en": "dark night, mysterious deep forest, gloomy mood", "ja": "漆黒の夜、神秘的な森の奥、陰鬱な気分", "ko": "칠흑 같은 밤, 신비로운 깊은 숲, 침울한 기분"}},
    {"concept_name": "pure_ambiguous", "base_chinese": "純潔", "translations": {"zh": "純潔的心靈，單純的想法，純淨無瑕的白色", "en": "pure heart, simple idea, pristine flawless white", "ja": "純粋な心、単純な考え、清らかで無垢な白", "ko": "순수한 마음, 단순한 생각, 깨끗하고 티없는 흰색"}},
    {"concept_name": "warm_ambiguous", "base_chinese": "溫暖", "translations": {"zh": "溫暖的陽光灑在身上，壁爐裡溫暖的火焰，熱情的款待", "en": "warm sunshine on the skin, cozy fireplace flames, enthusiastic hospitality", "ja": "肌に注ぐ暖かい日差し、暖炉の暖かい炎、情熱的なもてなし", "ko": "피부에 내리쬐는 따뜻한 햇살, 벽난로의 따뜻한 불꽃, 열정적인 환대"}},
    {"concept_name": "happy_clear", "base_chinese": "快樂", "translations": {"zh": "陽光明媚的日子裡孩子們快樂的笑容，五彩繽紛的派對氣球", "en": "children's happy smiles on a sunny day, colorful party balloons", "ja": "晴れた日の子供たちの幸せな笑顔、カラフルなパーティーバルーン", "ko": "화창한 날 아이들의 행복한 미소, 다채로운 파티 풍선"}},
    {"concept_name": "angry_clear", "base_chinese": "生氣", "translations": {"zh": "因為不公正而生氣的表情，緊握的拳頭，火山爆發的瞬間", "en": "angry expression due to injustice, clenched fists, moment of volcanic eruption", "ja": "不正に対する怒りの表情、握りしめた拳、火山噴火の瞬間", "ko": "불의로 인한 화난 표정, 꽉 쥔 주먹, 화산 폭발의 순간"}},
    {"concept_name": "sad_clear", "base_chinese": "傷心", "translations": {"zh": "因為失去而傷心的眼淚，陰雨連綿的窗外，孤獨的流浪貓", "en": "sad tears shed for a loss, continuous rainy weather outside the window, a lonely stray cat", "ja": "喪失による悲しい涙、窓の外の長引く雨天、孤独な野良猫", "ko": "상실감에 흘리는 슬픈 눈물, 창밖의 계속되는 비 오는 날씨, 외로운 길고양이"}},
    {"concept_name": "surprised_clear", "base_chinese": "驚訝", "translations": {"zh": "收到意想不到的生日驚喜時驚訝的表情，魔術師的奇妙戲法", "en": "surprised expression upon receiving an unexpected birthday surprise, a magician's wonderful trick", "ja": "予期せぬ誕生日サプライズに驚いた表情、マジシャンの見事な手品", "ko": "예상치 못한 생일 서프라이즈를 받았을 때 놀란 표정, 마술사의 멋진 마술"}},
    {"concept_name": "hungry_clear", "base_chinese": "餓", "translations": {"zh": "肚子餓得咕咕叫，餐桌上豐盛美味的晚餐", "en": "stomach rumbling with hunger, a hearty and delicious dinner on the table", "ja": "お腹が空いてグーグー鳴る、食卓の上の豊かで美味しい夕食", "ko": "배가 고파 꼬르륵 소리가 나다, 식탁 위의 푸짐하고 맛있는 저녁 식사"}},
    {"concept_name": "tired_clear", "base_chinese": "疲憊", "translations": {"zh": "長時間工作後感到身心疲憊，舒適柔軟的床鋪", "en": "feeling mentally and physically exhausted after long hours of work, a comfortable and soft bed", "ja": "長時間の仕事の後で心身ともに疲れ果てた、快適で柔らかいベッド", "ko": "장시간 작업 후 심신이 지친 상태, 편안하고 부드러운 침대"}},
]
# word_concepts_to_process = [word_concepts_list[0], word_concepts_list[2]] # 測試時用少量詞彙
word_concepts_to_process = word_concepts_list
print(f"8:定義了 {len(word_concepts_to_process)} 個詞彙概念用於本次分析。")

# 9:獲取文本的 CLIP 語義嵌入向量
# 這個函式使用 CLIP 模型為輸入的各語言文本計算其語義嵌入向量。

def get_clip_text_embeddings_vector(text_prompts_dict, loaded_clip_model, computation_device):
    if loaded_clip_model is None:
        return {lang_code: np.zeros(512, dtype=np.float32) for lang_code in text_prompts_dict}
    text_embeddings_result_dict = {}
    with torch.no_grad():
        for lang_tag, text_content in text_prompts_dict.items():
            try:
                tokenized_input_text = clip.tokenize([text_content]).to(computation_device)
                text_semantic_features = loaded_clip_model.encode_text(tokenized_input_text)
                text_semantic_features /= text_semantic_features.norm(dim=-1, keepdim=True)
                text_embeddings_result_dict[lang_tag] = text_semantic_features.cpu().numpy().flatten()
            except Exception as e:
                print(f"為 '{lang_tag}':'{text_content[:30]}...' 生成CLIP嵌入時出錯: {e}")
                text_embeddings_result_dict[lang_tag] = np.zeros(512, dtype=np.float32)
    return text_embeddings_result_dict
print("9:定義完成。")

# 10:計算並打印 CLIP 嵌入向量之間的餘弦相似度

def calculate_and_print_embedding_similarity(embeddings_dict, reference_lang='en'):
    similarity_scores_result = {}
    if reference_lang not in embeddings_dict or np.all(np.isclose(embeddings_dict[reference_lang], 0)):
        return {f"{reference_lang}_vs_{lang}": None for lang in embeddings_dict if lang != reference_lang}
    print(f"  CLIP文本嵌入向量餘弦相似度 (vs '{reference_lang.upper()}'):")
    ref_embedding = embeddings_dict[reference_lang].reshape(1, -1)
    for lang, emb in embeddings_dict.items():
        if lang == reference_lang: continue
        sim_val_str, sim_num = "N/A (嵌入無效)", None
        if not np.all(np.isclose(emb, 0)):
            sim_num = cosine_similarity(ref_embedding, emb.reshape(1, -1))[0][0]
            sim_val_str = f"{sim_num:.3f}"
        similarity_scores_result[f"{reference_lang}_vs_{lang}"] = sim_num
        print(f"    - 與 {lang.upper()}: {sim_val_str}")
    return similarity_scores_result
print("10:定義完成。")

# 11:使用 Stable Diffusion 生成真實圖像

def generate_actual_image_with_sd(prompt_text, sd_pipeline, random_seed=42, inference_steps=30, cfg_scale=7.5, computation_device="cuda"):
    if sd_pipeline is None:
        placeholder_img = Image.new('RGB', (512,512), color='silver'); draw = ImageDraw.Draw(placeholder_img)
        try: font = ImageFont.truetype("DejaVuSans.ttf",18)
        except: font = ImageFont.load_default()
        draw.text((10,10), f"模型未載入\n提示:\n{prompt_text[:70]}...", fill=(60,60,60), font=font); return placeholder_img
    try:
        gen = torch.Generator(device=computation_device).manual_seed(random_seed)
        with torch.no_grad(): img = sd_pipeline(prompt_text, num_inference_steps=inference_steps, guidance_scale=cfg_scale, generator=gen).images[0]
        return img
    except Exception as e:
        print(f"  生成圖像時出錯 ('{prompt_text[:40]}...'): {e}"); error_img = Image.new('RGB', (512,512), color='lightcoral'); draw=ImageDraw.Draw(error_img)
        try: font = ImageFont.truetype("DejaVuSans.ttf",15)
        except: font = ImageFont.load_default()
        draw.text((10,10), f"圖像生成錯誤:\n{prompt_text[:60]}...\n錯誤:\n{str(e)[:100]}", fill=(0,0,0),font=font); return error_img
print("11:定義完成。")

# 12:從圖像中提取主要顏色 (RGB 和 Lab)

def extract_dominant_colors_from_image(pil_img, num_colors=5):
    if pil_img is None or pil_img.width < num_colors or pil_img.height < num_colors:
        rgb = np.array([[128,128,128]]*num_colors, dtype=int); lab = rgb2lab(rgb/255.0).reshape(-1,3); return rgb, lab
    try:
        img_rgb = pil_img.convert('RGB'); max_dim = 150
        ratio = max_dim/max(img_rgb.width, img_rgb.height); new_size=(max(1,int(img_rgb.width*ratio)),max(1,int(img_rgb.height*ratio)))
        img_res = img_rgb.resize(new_size, Image.Resampling.LANCZOS); pixels = np.array(img_res).reshape(-1,3)
        if pixels.shape[0] < num_colors:
            rgb = np.zeros((num_colors,3),dtype=int); actual_rgb = pixels.astype(int)
            rgb[:actual_rgb.shape[0]] = actual_rgb
            if actual_rgb.shape[0] < num_colors: rgb[actual_rgb.shape[0]:] = np.array([128,128,128])
        else:
            kmeans = KMeans(n_clusters=num_colors,random_state=0,n_init='auto',max_iter=200).fit(pixels)
            rgb = kmeans.cluster_centers_.astype(int)
        lab = rgb2lab(rgb.reshape((num_colors,1,3))/255.0).reshape((num_colors,3))
        return rgb, lab
    except Exception as e:
        print(f"  提取主色調時出錯: {e}"); rgb_err = np.array([[100,100,100]]*num_colors,dtype=int); lab_err = rgb2lab(rgb_err/255.0).reshape(-1,3); return rgb_err, lab_err
print("12:定義完成。")

# 13:分析圖像的全局特徵 (亮度、對比度、飽和度)

def analyze_global_image_features(pil_image):
    if pil_image is None: return {"avg_brightness": "N/A", "contrast_std": "N/A", "avg_saturation": "N/A"}
    try:
        cv_bgr = np.array(pil_image.convert('RGB'))[:,:,::-1].copy()
        gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
        brightness = round(np.mean(gray), 2)
        contrast = round(np.std(gray), 2)
        hsv = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2HSV)
        saturation = round(np.mean(hsv[:,:,1]), 2)
        return {"avg_brightness": brightness, "contrast_std": contrast, "avg_saturation": saturation}
    except Exception as e: print(f"  分析全局圖像特徵時出錯: {e}"); return {"avg_brightness":"Err","contrast_std":"Err","avg_saturation":"Err"}
print("13:定義完成。")

# 14:為生成的圖像準備中文解釋文本

def generate_explanation_for_image(chinese_concept, lang_prompt, pil_img, dom_colors_hex=None, global_feats=None):
    expl = f"\n--- 解釋模板 for 概念:【{chinese_concept}】| 語言提示: 【{lang_prompt[:70]}...】 ---\n"
    if dom_colors_hex: expl += f"圖像主要色票 (HEX): {', '.join(dom_colors_hex[:3])} ...\n"
    if global_feats:
        expl += f"全局圖像特徵: 亮度={global_feats.get('avg_brightness','N/A')}, "
        expl += f"對比度={global_feats.get('contrast_std','N/A')}, 飽和度={global_feats.get('avg_saturation','N/A')}\n"
    expl += f"\n原因推測與圖像描述 (請您填充)：\n"
    expl += f"   [請結合以上客觀指標和您的觀察，詳細闡述：\n"
    expl += f"    a. 圖像視覺風格與氛圍？\n"
    expl += f"    b. 主要元素與提示詞的關聯？\n"
    expl += f"    c. 色彩運用如何詮釋提示詞？\n"
    expl += f"    d. (特定語言)文化背景的可能影響？(例如韓國圖像為何常出現人物?)\n"
    expl += f"    e. 與其他語言生成圖像的差異及可能原因？]\n"
    expl += f"--------------------------------------------------------------------------\n"
    return expl
print("14:定義完成。")

# 15:輔助函式 - 繪製單個詞彙概念的綜合結果圖表

def plot_full_concept_results_chart(concept_id, base_chinese, prompts_dict, images_dict, colors_dict, similarities_dict, global_features_dict=None):
    langs = list(prompts_dict.keys()); num_langs = len(langs)
    h_ratio, w_ratio = 2.8, 4.0; total_h, total_w = h_ratio*2, w_ratio*num_langs
    fig, axs = plt.subplots(2, num_langs, figsize=(total_w, total_h), gridspec_kw={'height_ratios':[0.78,0.22]})
    if num_langs==1: axs=axs.reshape(2,1)

    title = f"概念分析: '{base_chinese}' ({concept_id})\nCLIP相似度(vs EN): "
    sim_strs = [f"{k.split('_vs_')[-1].upper()}: {v:.2f}" if isinstance(v,(float,np.floating)) else f"{k.split('_vs_')[-1].upper()}: {v}" for k,v in similarities_dict.items()]
    fig.suptitle(title + ", ".join(sim_strs), fontsize=11, y=1.04)

    for i, lang in enumerate(langs):
        img, colors = images_dict.get(lang), colors_dict.get(lang)
        global_feats_this_lang = global_features_dict.get(lang, {}) if global_features_dict else {}

        ax_img = axs[0,i]; ax_color = axs[1,i]
        if img: ax_img.imshow(img)
        else: ax_img.text(0.5,0.5,'圖像未生成',ha='center',va='center',transform=ax_img.transAxes)
        img_title = f"{lang.upper()}: \"{prompts_dict[lang][:30]}\"..."
        if global_feats_this_lang: # 在圖像標題下方添加簡要的全局特徵
            img_title += f"\n亮:{global_feats_this_lang.get('avg_brightness','-')} "
            img_title += f"對比:{global_feats_this_lang.get('contrast_std','-')} "
            img_title += f"飽:{global_feats_this_lang.get('avg_saturation','-')}"
        ax_img.set_title(img_title, fontsize=7.5)
        ax_img.axis('off')

        if colors:
            rgb_patch, lab_patch = colors; n_patch=len(rgb_patch)
            patch_canvas = np.zeros((25,100,3),dtype=np.uint8)
            patch_w = 100//n_patch if n_patch>0 else 100
            for j,rgb_c in enumerate(rgb_patch): patch_canvas[:,j*patch_w:(j+1)*patch_w] = rgb_c
            ax_color.imshow(patch_canvas)
            lab_str = "\n".join([f"L{l:.0f} a{a:.0f} b{b:.0f}" for l,a,b in lab_patch[:min(3,n_patch)]])
            ax_color.set_title(f"Lab(Top{min(3,n_patch)}):\n{lab_str}", fontsize=6)
        else: ax_color.text(0.5,0.5,'無顏色',ha='center',va='center',transform=ax_color.transAxes)
        ax_color.axis('off')
    plt.tight_layout(rect=[0,0,1,0.94]); plt.subplots_adjust(hspace=0.5, wspace=0.3); plt.show()
print("Cell 14: 輔助函式 `plot_full_concept_results_chart` 定義完成。")

# 16: 主程式

print(f"Cell 15: 即將開始執行主流程，處理 {len(word_concepts_to_process)} 個詞彙概念...")
BASE_SEED = 20240101 # 可以更改基礎種子以獲得不同的圖像系列
SD_STEPS = 22 # 推斷步數可以適當減少以加速，20-25通常是不錯的平衡
SD_CFG = 7.0  # 指導比例
NUM_DOM_COLORS = 5
SAVE_IMAGES_FLAG = True
IMAGES_OUT_DIR = "project_outputs_final_v4"

if SAVE_IMAGES_FLAG and not os.path.exists(IMAGES_OUT_DIR):
    os.makedirs(IMAGES_OUT_DIR); print(f"  已建立圖像儲存目錄: {IMAGES_OUT_DIR}")

results_collection = []

for concept_idx, concept_detail in enumerate(tqdm(word_concepts_to_process, desc="總體概念處理")):
    concept_id = concept_detail["concept_name"]; base_zh = concept_detail["base_chinese"]
    prompts = concept_detail["translations"]
    print(f"\n\n處理概念 #{concept_idx+1}: '{base_zh}' ({concept_id})")

    print("  [1. CLIP嵌入分析]")
    embeddings = get_clip_text_embeddings_vector(prompts, clip_model_instance, device_to_use)
    similarities = calculate_and_print_embedding_similarity(embeddings, reference_lang='en')

    print("  [2. 圖像生成、顏色與全局特徵分析]")
    concept_images = {}; concept_colors = {}; concept_global_features = {}
    concept_explanations_str = ""

    for lang_idx, (lang, prompt) in enumerate(tqdm(prompts.items(), desc=f"  '{concept_id}'語言處理", leave=False)):
        print(f"    -> {lang.upper()}: '{prompt}'")
        img_seed = BASE_SEED + concept_idx*100 + lang_idx*10
        pil_img = generate_actual_image_with_sd(prompt,image_generation_pipeline,img_seed,SD_STEPS,SD_CFG,device_to_use)
        concept_images[lang] = pil_img

        if SAVE_IMAGES_FLAG and pil_img:
            try:
                fname = f"{concept_id}_{lang}_s{img_seed}.png"; fpath = os.path.join(IMAGES_OUT_DIR,fname)
                pil_img.save(fpath)
            except Exception as e: print(f"      儲存圖像'{fname}'失敗: {e}")

        rgb_cs, lab_cs = extract_dominant_colors_from_image(pil_img, NUM_DOM_COLORS)
        concept_colors[lang] = (rgb_cs, lab_cs)
        global_feats = analyze_global_image_features(pil_img) # 分析全局特徵
        concept_global_features[lang] = global_feats # 儲存全局特徵

        expl_text = generate_explanation_for_image(base_zh, f"{lang.upper()}: {prompt}", pil_img,
                                                 [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in rgb_cs],
                                                 global_feats) # 傳遞全局特徵給解釋模板
        print(expl_text)
        concept_explanations_str += expl_text

        if device_to_use=="cuda": torch.cuda.empty_cache(); time.sleep(0.05)

    print("\n  [3. 繪製結果圖表]")
    plot_full_concept_results_chart(concept_id,base_zh,prompts,concept_images,concept_colors,similarities,concept_global_features)

    results_collection.append({"concept":concept_id,"base_chinese":base_zh, "prompts":prompts, "similarities":similarities,
                               "global_features": concept_global_features, # 保存全局特徵
                               "explanation_prompts":concept_explanations_str}) # 保存解釋模板
    print(f"  概念 '{concept_id}' 分析完畢。")
    if device_to_use=="cuda": torch.cuda.empty_cache()

print("\n\n所有詞彙概念處理完成！解釋模板已在上方打印。")
# 可選: 保存 results_collection 到 JSON
# ... (參考之前版本的JSON保存代碼，記得處理NpEncoder)

# 17: 清理模型和資源以釋放

print("正在清理模型資源...")
if 'clip_model_instance' in globals(): del clip_model_instance
if 'clip_image_preprocess_fn' in globals(): del clip_image_preprocess_fn
if 'image_generation_pipeline' in globals(): del image_generation_pipeline
if device_to_use=="cuda": torch.cuda.empty_cache()
print("  模型和資源清理操作已執行。")