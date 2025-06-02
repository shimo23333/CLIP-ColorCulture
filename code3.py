# 1:安裝必要的 Python 套件 (假設這些已在環境中安裝，或在腳本外部執行)
# !pip install -q git+https://github.com/openai/CLIP.git
# !pip install -q scikit-image
# !pip install -q opencv-python
# !pip install -q diffusers transformers accelerate invisible-watermark safetensors
# !pip install -q ipywidgets
# !pip install -q sentencepiece sacremoses

# print("1:套件安裝步驟已略過，假設已完成。")

# 2:匯入專案中會用到的所有 Python 模組
import torch
import clip
from PIL import Image, ImageDraw, ImageFont
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
import matplotlib.font_manager as fm
from transformers import pipeline
import re # 用於生成 concept_name

print("2:函式庫匯入完成。")

# --- Class定義 ---

class ConfigManager:
    def __init__(self, font_path_cjk='/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf',
                 images_out_dir="project_outputs_final_v4_oop_hf"): # 修改輸出目錄名以區分
        self.device = self._get_device()
        self.font_path_cjk = font_path_cjk
        self.images_out_dir = images_out_dir
        self._setup_matplotlib_font()
        self._setup_output_directory()
        self.hf_device_id = 0 if self.device == "cuda" else -1

    def _get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"4:本次運行的計算設備是: {device}")
        if device == "cpu":
            print("警告：未使用GPU！運行大型AI模型會非常慢。")
        return device

    def _setup_matplotlib_font(self):
        print("3:正在設定 Matplotlib CJK 字體...")
        if os.path.exists(self.font_path_cjk):
            try:
                fm.fontManager.addfont(self.font_path_cjk)
                prop = fm.FontProperties(fname=self.font_path_cjk)
                font_name = prop.get_name()
                plt.rcParams['font.family'] = font_name
                plt.rcParams['axes.unicode_minus'] = False
                print(f"  已成功設定 CJK 字體為: {font_name}")
            except Exception as e:
                print(f"  設定字體 '{self.font_path_cjk}' 時發生錯誤: {e}. 退到通用列表。")
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
        else:
            print(f"  指定的 CJK 字體文件路徑不存在: {self.font_path_cjk}. 退到通用列表。")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        print("3:Matplotlib CJK 字體設定完畢。")

    def _setup_output_directory(self):
        if not os.path.exists(self.images_out_dir):
            os.makedirs(self.images_out_dir)
            print(f"  已建立圖像儲存目錄: {self.images_out_dir}")


class ModelManager:
    def __init__(self, device, hf_device_id):
        self.device = device
        self.hf_device_id = hf_device_id
        self.clip_model = None
        self.clip_preprocess = None
        self.sd_pipeline = None
        self.translation_pipelines = {}

    def load_clip_model(self, model_name="ViT-B/32"):
        print(f"5:準備載入 CLIP 模型 ({model_name})...")
        if self.device == "cuda":
            try:
                self.clip_model, self.clip_preprocess = clip.load(model_name, device=self.device)
                self.clip_model.eval()
                print(f"  CLIP 模型 ({model_name}) 已成功載入到 {self.device}！")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  載入CLIP模型時發生錯誤: {e}")
                self.clip_model, self.clip_preprocess = None, None
        else:
            print(f"  未實際載入CLIP模型，因為當前運算設備是 {self.device}。")
        return self.clip_model, self.clip_preprocess

    def load_sd_model(self, model_id="runwayml/stable-diffusion-v1-5"):
        print(f"6:準備載入 Stable Diffusion 模型 ({model_id})...")
        if self.device == "cuda":
            try:
                self.sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
                self.sd_pipeline = self.sd_pipeline.to(self.device)
                print(f"  Stable Diffusion 模型 ({model_id}) 已成功載入到 {self.device}！")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  載入Stable Diffusion模型時發生錯誤: {e}")
                self.sd_pipeline = None
        else:
            print(f"  未實際載入Stable Diffusion模型，因為當前運算設備是 {self.device}。")
        return self.sd_pipeline

    def init_translation_models(self, target_languages=['en', 'ja', 'ko']):
        print("7:正在初始化 Hugging Face 翻譯模型...")
        # 你可能需要根據你的中文輸入類型（簡體/繁體）選擇更合適的模型
        # 例如 'Helsinki-NLP/opus-mt-ZH-PLACEHOLDER' (ZH 代表廣泛中文)
        # 或者確保你的輸入與模型期望的中文方言一致
        model_map = {
            'en': 'Helsinki-NLP/opus-mt-zh-en',
            'ja': 'Helsinki-NLP/opus-mt-zh-ja', # 假設存在且適用
            'ko': 'Helsinki-NLP/opus-mt-zh-ko'  # 假設存在且適用
        }
        loaded_any_model = False
        for lang_code in target_languages:
            if lang_code in model_map:
                model_name = model_map[lang_code]
                try:
                    print(f"  載入翻譯模型 for zh -> {lang_code} ({model_name})...")
                    translator = pipeline(f"translation_zh_to_{lang_code}", # 任務名可能需要調整
                                          model=model_name,
                                          device=self.hf_device_id)
                    self.translation_pipelines[lang_code] = translator
                    print(f"    翻譯模型 for zh -> {lang_code} ({model_name}) 載入成功。")
                    loaded_any_model = True
                except Exception as e:
                    print(f"    載入翻譯模型 for zh -> {lang_code} ({model_name}) 失敗: {e}")
                    self.translation_pipelines[lang_code] = None
            else:
                print(f"  未找到針對 zh -> {lang_code} 的預定義翻譯模型。")
        if not loaded_any_model:
            print("警告: 未能成功載入任何Hugging Face翻譯模型。自動翻譯功能將受限。")
        return self.translation_pipelines

    def cleanup(self):
        print("正在清理模型資源...")
        if self.clip_model: del self.clip_model
        if self.clip_preprocess: del self.clip_preprocess
        if self.sd_pipeline: del self.sd_pipeline
        if self.translation_pipelines:
            for lang, pipe in self.translation_pipelines.items():
                if pipe: del pipe # 釋放pipeline物件
            self.translation_pipelines.clear()
            if self.device == "cuda": # 清理pipeline可能佔用的VRAM
                torch.cuda.empty_cache()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("  模型和資源清理操作已執行。")

class ConceptDataProvider:
    def __init__(self, clear_default_concepts=False): # 新增參數
        self.word_concepts_list = [] # 默認清空，由用戶在main中添加
        if not clear_default_concepts:
            # 如果需要，可以保留原有的預設列表
            self.word_concepts_list = [
                {"concept_name": "cool_ambiguous", "base_chinese": "酷 / 涼爽", "translations": {"zh": "冰涼的飲料，酷炫的風格，冷靜的態度", "en": "cool refreshing drink, cool stylish look, calm and cool attitude", "ja": "冷たい飲み物、かっこいいスタイル、冷静な態度", "ko": "시원한 음료, 멋진 스타일, 침착한 태도"}},
                # ... (其他預設概念可以放在這裡)
            ]
        self.word_concepts_to_process = self.word_concepts_list
        print(f"8:初始定義了 {len(self.word_concepts_list)} 個詞彙概念。")

    def _generate_concept_name(self, base_chinese):
        """根據中文詞彙生成一個簡化的英文概念名"""
        # 移除特殊字符，只保留字母和數字，並用下劃線連接
        # 這是一個非常基礎的實現，可能需要更複雜的邏輯來生成好的英文名
        # 或者，要求用戶為每個詞彙也提供一個英文 concept_name
        name = re.sub(r'[^\w]', '', base_chinese) # 移除非字母數字字符
        name = name[:20] # 限制長度
        if not name: name = "unnamed_concept"
        return f"{name}_auto_translated"


    def add_concept_with_auto_translation(self, translation_pipelines, base_chinese,
                                          chinese_description_for_prompt=None, target_languages=['en', 'ja', 'ko']):
        """
        新增一個概念，並使用 Hugging Face pipeline 自動翻譯其描述性句子。
        如果 chinese_description_for_prompt 未提供，則直接翻譯 base_chinese。
        """
        concept_name = self._generate_concept_name(base_chinese) # 自動生成 concept_name
        text_to_translate = chinese_description_for_prompt if chinese_description_for_prompt else base_chinese

        print(f"\n  準備新增概念 '{concept_name}' (基於 '{base_chinese}')...")
        print(f"    將翻譯: '{text_to_translate[:50]}...'")

        if not translation_pipelines or not any(translation_pipelines.values()):
            print("    錯誤：翻譯 pipelines 未提供或均未成功載入，無法自動翻譯。將使用原文作為提示。")
            translations = {lang: text_to_translate for lang in target_languages}
            translations['zh'] = text_to_translate
        else:
            translations = {"zh": text_to_translate} # 中文提示使用原始描述或詞彙
            for lang_code in target_languages:
                translator_pipeline = translation_pipelines.get(lang_code)
                if translator_pipeline:
                    try:
                        translated_result = translator_pipeline(text_to_translate)
                        translated_text = translated_result[0]['translation_text']
                        translations[lang_code] = translated_text
                        print(f"      -> {lang_code.upper()}: {translated_text}")
                    except Exception as e:
                        print(f"      使用Hugging Face模型翻譯到 {lang_code.upper()} 失敗: {e}")
                        translations[lang_code] = f"翻譯失敗: {text_to_translate}" # 回退
                else:
                    print(f"      未找到 {lang_code.upper()} 的翻譯 pipeline，使用原文。")
                    translations[lang_code] = text_to_translate # 回退

        new_concept = {
            "concept_name": concept_name,
            "base_chinese": base_chinese, # 核心詞彙
            "translations": translations  # 用於 prompt 的翻譯文本
        }
        self.word_concepts_list.append(new_concept)
        print(f"  新概念 '{concept_name}' 已成功添加。現有 {len(self.word_concepts_list)} 個概念。")
        # 更新 self.word_concepts_to_process，因為它與 self.word_concepts_list 是同一個物件
        print(f"8(更新):定義了 {len(self.word_concepts_to_process)} 個詞彙概念用於本次分析。")

    def get_concepts_to_process(self):
        if not self.word_concepts_to_process:
            print("警告：沒有定義任何詞彙概念進行處理。")
        return self.word_concepts_to_process

# ... (TextAnalyzer, ImageProcessor, ReportGenerator, AnalysisPipeline 保持不變，僅確保其彈性) ...
class TextAnalyzer:
    def __init__(self, clip_model, device):
        self.clip_model = clip_model
        self.device = device
    def get_clip_text_embeddings(self, text_prompts_dict):
        if self.clip_model is None:
            print("  警告: CLIP 模型未載入，文本嵌入將為零向量。")
            return {lang_code: np.zeros(512, dtype=np.float32) for lang_code in text_prompts_dict}
        text_embeddings_result_dict = {}
        with torch.no_grad():
            for lang_tag, text_content in text_prompts_dict.items():
                try:
                    tokenized_input_text = clip.tokenize([text_content]).to(self.device)
                    text_semantic_features = self.clip_model.encode_text(tokenized_input_text)
                    text_semantic_features /= text_semantic_features.norm(dim=-1, keepdim=True)
                    text_embeddings_result_dict[lang_tag] = text_semantic_features.cpu().numpy().flatten()
                except Exception as e:
                    print(f"為 '{lang_tag}':'{text_content[:30]}...' 生成CLIP嵌入時出錯: {e}")
                    text_embeddings_result_dict[lang_tag] = np.zeros(512, dtype=np.float32)
        return text_embeddings_result_dict
    def calculate_embedding_similarity(self, embeddings_dict, reference_lang='en'):
        similarity_scores_result = {}
        if reference_lang not in embeddings_dict or embeddings_dict.get(reference_lang) is None or np.all(np.isclose(embeddings_dict[reference_lang], 0)):
            print(f"  參考語言 '{reference_lang.upper()}' 的嵌入向量無效或不存在，無法計算相似度。")
            return {f"{reference_lang}_vs_{lang}": None for lang in embeddings_dict if lang != reference_lang}
        print(f"  CLIP文本嵌入向量餘弦相似度 (vs '{reference_lang.upper()}'):")
        ref_embedding = embeddings_dict[reference_lang].reshape(1, -1)
        for lang, emb in embeddings_dict.items():
            if lang == reference_lang: continue
            sim_val_str = "N/A (嵌入無效)"
            sim_num = None
            if emb is not None and not np.all(np.isclose(emb, 0)):
                sim_num = cosine_similarity(ref_embedding, emb.reshape(1, -1))[0][0]
                sim_val_str = f"{sim_num:.3f}"
            similarity_scores_result[f"{reference_lang}_vs_{lang}"] = sim_num
            print(f"    - 與 {lang.upper()}: {sim_val_str}")
        return similarity_scores_result

class ImageProcessor:
    def __init__(self, sd_pipeline, device):
        self.sd_pipeline = sd_pipeline
        self.device = device
    def generate_image_with_sd(self, prompt_text, random_seed=42, inference_steps=30, cfg_scale=7.5):
        if self.sd_pipeline is None:
            placeholder_img = Image.new('RGB', (512, 512), color='silver')
            draw = ImageDraw.Draw(placeholder_img)
            try: font = ImageFont.truetype("DejaVuSans.ttf", 18)
            except IOError: font = ImageFont.load_default()
            draw.text((10, 10), f"SD模型未載入\n提示:\n{prompt_text[:70]}...", fill=(60, 60, 60), font=font)
            return placeholder_img
        try:
            gen = torch.Generator(device=self.device).manual_seed(random_seed)
            with torch.no_grad():
                img = self.sd_pipeline(prompt_text, num_inference_steps=inference_steps, guidance_scale=cfg_scale, generator=gen).images[0]
            return img
        except Exception as e:
            print(f"  生成圖像時出錯 ('{prompt_text[:40]}...'): {e}")
            error_img = Image.new('RGB', (512, 512), color='lightcoral')
            draw = ImageDraw.Draw(error_img)
            try: font = ImageFont.truetype("DejaVuSans.ttf", 15)
            except IOError: font = ImageFont.load_default()
            draw.text((10, 10), f"圖像生成錯誤:\n{prompt_text[:60]}...\n錯誤:\n{str(e)[:100]}", fill=(0, 0, 0), font=font)
            return error_img
    def extract_dominant_colors(self, pil_img, num_colors=5):
        if pil_img is None or (hasattr(pil_img, 'width') and pil_img.width < num_colors) or \
           (hasattr(pil_img, 'height') and pil_img.height < num_colors): # 更安全的檢查
            rgb_fallback = np.array([[128, 128, 128]] * num_colors, dtype=int)
            lab_fallback = rgb2lab(rgb_fallback.reshape((num_colors, 1, 3)) / 255.0).reshape((num_colors, 3))
            return rgb_fallback, lab_fallback
        try:
            img_rgb = pil_img.convert('RGB')
            max_dim = 150
            ratio = max_dim / max(img_rgb.width, img_rgb.height)
            new_size = (max(1, int(img_rgb.width * ratio)), max(1, int(img_rgb.height * ratio)))
            img_res = img_rgb.resize(new_size, Image.Resampling.LANCZOS)
            pixels = np.array(img_res).reshape(-1, 3)
            if pixels.shape[0] < num_colors:
                rgb_colors = np.zeros((num_colors, 3), dtype=int)
                actual_extracted_colors = pixels.astype(int)
                rgb_colors[:actual_extracted_colors.shape[0]] = actual_extracted_colors
                if actual_extracted_colors.shape[0] < num_colors:
                    rgb_colors[actual_extracted_colors.shape[0]:] = np.array([128,128,128])
            else:
                kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init='auto', max_iter=200).fit(pixels)
                rgb_colors = kmeans.cluster_centers_.astype(int)
            if rgb_colors.shape[0] < num_colors:
                padded_colors = np.full((num_colors, 3), 128, dtype=int)
                padded_colors[:rgb_colors.shape[0]] = rgb_colors
                rgb_colors = padded_colors
            lab_colors = rgb2lab(rgb_colors.reshape((num_colors, 1, 3)) / 255.0).reshape((num_colors, 3))
            return rgb_colors, lab_colors
        except Exception as e:
            print(f"  提取主色調時出錯: {e}")
            rgb_err = np.array([[100, 100, 100]] * num_colors, dtype=int)
            lab_err = rgb2lab(rgb_err.reshape((num_colors,1,3)) / 255.0).reshape((num_colors, 3))
            return rgb_err, lab_err
    def analyze_global_features(self, pil_image):
        if pil_image is None: return {"avg_brightness": "N/A", "contrast_std": "N/A", "avg_saturation": "N/A"}
        try:
            cv_bgr = np.array(pil_image.convert('RGB'))[:, :, ::-1].copy()
            gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
            brightness = round(np.mean(gray), 2)
            contrast = round(np.std(gray), 2)
            hsv = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2HSV)
            saturation = round(np.mean(hsv[:, :, 1]), 2)
            return {"avg_brightness": brightness, "contrast_std": contrast, "avg_saturation": saturation}
        except Exception as e:
            print(f"  分析全局圖像特徵時出錯: {e}")
            return {"avg_brightness": "Err", "contrast_std": "Err", "avg_saturation": "Err"}

class ReportGenerator:
    def generate_explanation_template(self, chinese_concept, lang_prompt_info, dom_colors_hex=None, global_feats=None):
        expl = f"\n--- 解釋模板 for 概念:【{chinese_concept}】| 語言提示: 【{lang_prompt_info[:70]}...】 ---\n"
        if dom_colors_hex: expl += f"圖像主要色票 (HEX): {', '.join(dom_colors_hex[:3])} ...\n"
        if global_feats:
            expl += f"全局圖像特徵: 亮度={global_feats.get('avg_brightness', 'N/A')}, "
            expl += f"對比度={global_feats.get('contrast_std', 'N/A')}, 飽和度={global_feats.get('avg_saturation', 'N/A')}\n"
        expl += f"\n原因推測與圖像描述 (請您填充)：\n"
        expl += f"   [請結合以上客觀指標和您的觀察，詳細闡述：\n"
        expl += f"    a. 圖像視覺風格與氛圍？\n    b. 主要元素與提示詞的關聯？\n    c. 色彩運用如何詮釋提示詞？\n"
        expl += f"    d. (特定語言)文化背景的可能影響？\n    e. 與其他語言生成圖像的差異及可能原因？]\n"
        expl += f"--------------------------------------------------------------------------\n"
        return expl
    def plot_concept_results(self, concept_id, base_chinese, prompts_dict, images_dict,colors_dict, similarities_dict, global_features_dict=None):
        langs = list(prompts_dict.keys())
        num_langs = len(langs)
        if num_langs == 0:
            print(f"概念 '{concept_id}' 無 prompts 可繪製。")
            return
        h_ratio, w_ratio = 2.8, 4.0
        total_h, total_w = h_ratio * 2, w_ratio * num_langs if num_langs > 0 else w_ratio
        fig, axs = plt.subplots(2, max(1, num_langs), figsize=(total_w, total_h), gridspec_kw={'height_ratios': [0.78, 0.22]})
        if num_langs == 1: axs = axs.reshape(2, 1)

        title_base = f"概念分析: '{base_chinese}' ({concept_id})\nCLIP相似度(vs EN): "
        sim_strs = []
        if similarities_dict:
            sim_strs = [f"{k.split('_vs_')[-1].upper()}: {v:.2f}" if isinstance(v, (float, np.floating)) else f"{k.split('_vs_')[-1].upper()}: {v}"
                        for k, v in similarities_dict.items()]
        fig.suptitle(title_base + ", ".join(sim_strs), fontsize=11, y=1.04)
        for i, lang in enumerate(langs):
            img = images_dict.get(lang)
            colors_data = colors_dict.get(lang)
            global_feats_this_lang = (global_features_dict or {}).get(lang, {})
            ax_img = axs[0, i]; ax_color = axs[1, i]
            if img: ax_img.imshow(img)
            else: ax_img.text(0.5, 0.5, '圖像未生成', ha='center', va='center', transform=ax_img.transAxes)
            img_title_prompt = prompts_dict.get(lang, "N/A")
            img_title = f"{lang.upper()}: \"{img_title_prompt[:30]}\"..."
            if global_feats_this_lang:
                img_title += f"\n亮:{global_feats_this_lang.get('avg_brightness', '-')} 對比:{global_feats_this_lang.get('contrast_std', '-')} 飽:{global_feats_this_lang.get('avg_saturation', '-')}"
            ax_img.set_title(img_title, fontsize=7.5); ax_img.axis('off')
            if colors_data:
                rgb_patch, lab_patch = colors_data
                if rgb_patch is not None and len(rgb_patch) > 0 :
                    n_patch = len(rgb_patch)
                    patch_canvas = np.zeros((25, 100, 3), dtype=np.uint8)
                    patch_w = 100 // n_patch
                    for j, rgb_c in enumerate(rgb_patch): patch_canvas[:, j * patch_w:(j + 1) * patch_w] = rgb_c
                    ax_color.imshow(patch_canvas)
                    lab_str_parts = []
                    if lab_patch is not None:
                        for l_val, a_val, b_val in lab_patch[:min(3,n_patch)]: lab_str_parts.append(f"L{l_val:.0f} a{a_val:.0f} b{b_val:.0f}")
                        lab_str = "\n".join(lab_str_parts)
                        ax_color.set_title(f"Lab(Top{min(3,n_patch)}):\n{lab_str}", fontsize=6)
                    else: ax_color.set_title(f"RGB顏色", fontsize=6)
                else: ax_color.text(0.5,0.5,'無顏色數據',ha='center',va='center',transform=ax_color.transAxes, fontsize=6)
            else: ax_color.text(0.5, 0.5, '無顏色', ha='center', va='center', transform=ax_color.transAxes, fontsize=6)
            ax_color.axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.subplots_adjust(hspace=0.5, wspace=0.3); plt.show()

class AnalysisPipeline:
    def __init__(self, config_manager, model_manager, concept_provider, text_analyzer, image_processor, report_generator):
        self.config = config_manager; self.models = model_manager; self.concepts_provider = concept_provider
        self.text_analyzer = text_analyzer; self.image_processor = image_processor; self.reporter = report_generator
        self.base_seed = 20240101; self.sd_steps = 20; self.sd_cfg = 7.0 # 減少步數以加速
        self.num_dom_colors = 5; self.save_images_flag = True
    def run_analysis(self):
        print(f"Cell 15: 即將開始執行主流程...")
        concepts_to_process = self.concepts_provider.get_concepts_to_process()
        if not concepts_to_process:
            print("沒有概念需要處理，流程結束。")
            return []
        print(f"  將處理 {len(concepts_to_process)} 個詞彙概念...")
        results_collection = []
        for concept_idx, concept_detail in enumerate(tqdm(concepts_to_process, desc="總體概念處理")):
            concept_id = concept_detail["concept_name"]; base_zh = concept_detail["base_chinese"]; prompts = concept_detail["translations"]
            print(f"\n\n處理概念 #{concept_idx + 1}: '{base_zh}' ({concept_id})")
            print("  [1. CLIP嵌入分析]")
            embeddings = self.text_analyzer.get_clip_text_embeddings(prompts)
            similarities = self.text_analyzer.calculate_embedding_similarity(embeddings, reference_lang='en')
            print("  [2. 圖像生成、顏色與全局特徵分析]")
            concept_images = {}; concept_colors = {}; concept_global_features = {}; concept_explanations_str = ""
            for lang_idx, (lang, prompt) in enumerate(tqdm(prompts.items(), desc=f"  '{concept_id}'語言處理", leave=False)):
                print(f"    -> {lang.upper()}: '{prompt}'")
                img_seed = self.base_seed + concept_idx * 100 + lang_idx * 10
                pil_img = self.image_processor.generate_image_with_sd(prompt, random_seed=img_seed, inference_steps=self.sd_steps, cfg_scale=self.sd_cfg)
                concept_images[lang] = pil_img
                if self.save_images_flag and pil_img:
                    try:
                        fname = f"{concept_id}_{lang}_s{img_seed}.png"; fpath = os.path.join(self.config.images_out_dir, fname)
                        pil_img.save(fpath)
                    except Exception as e: print(f"      儲存圖像'{fname}'失敗: {e}")
                rgb_cs, lab_cs = self.image_processor.extract_dominant_colors(pil_img, self.num_dom_colors)
                concept_colors[lang] = (rgb_cs, lab_cs)
                global_feats = self.image_processor.analyze_global_features(pil_img)
                concept_global_features[lang] = global_feats
                hex_colors = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in rgb_cs] if rgb_cs is not None and len(rgb_cs)>0 else []
                expl_text = self.reporter.generate_explanation_template(base_zh, f"{lang.upper()}: {prompt}", hex_colors, global_feats)
                print(expl_text); concept_explanations_str += expl_text
                if self.config.device == "cuda": torch.cuda.empty_cache(); time.sleep(0.05)
            print("\n  [3. 繪製結果圖表]")
            self.reporter.plot_concept_results(concept_id, base_zh, prompts, concept_images, concept_colors, similarities, concept_global_features)
            results_collection.append({"concept": concept_id, "base_chinese": base_zh, "prompts": prompts, "similarities": similarities, "global_features": concept_global_features, "explanation_prompts_combined": concept_explanations_str})
            print(f"  概念 '{concept_id}' 分析完畢。")
            if self.config.device == "cuda": torch.cuda.empty_cache()
        print("\n\n所有詞彙概念處理完成！解釋模板已在上方打印。")
        return results_collection

# --- 主執行流程 ---
if __name__ == '__main__':
    # 1. 初始化配置管理器
    config_mgr = ConfigManager()

    # 2. 初始化模型管理器
    model_mgr = ModelManager(device=config_mgr.device, hf_device_id=config_mgr.hf_device_id)
    clip_model, _ = model_mgr.load_clip_model()
    sd_pipeline = model_mgr.load_sd_model()
    translation_pipelines = model_mgr.init_translation_models(target_languages=['en', 'ja', 'ko'])

    # 3. 初始化數據提供者，並清空預設概念
    concept_provider = ConceptDataProvider(clear_default_concepts=True)

    # --- 定義你想要自動翻譯和分析的中文詞彙列表 ---
    # 每個元素可以是：
    #   - 一個簡單的中文詞 (str): 程式會直接翻譯這個詞作為各語言的 prompt
    #   - 一個元組 (str, str): (核心中文詞, 用於翻譯生成 prompt 的中文描述句)
    chinese_terms_to_analyze = [
        "酷",
        "柔軟",
        "明亮",
        "黑暗",
        "純潔", 
        "溫暖", 
        "快樂",
        "生氣",
        "傷心",
        "驚訝",
        "餓",
        "疲憊",
    ]
    # --- -------------------------------------- ---

    print(f"\n--- 開始為用戶定義的 {len(chinese_terms_to_analyze)} 個詞彙添加概念 ---")
    if not any(translation_pipelines.values()): # 再次檢查翻譯模型是否真的載入
        print("警告：沒有可用的翻譯模型，將主要使用中文原文作為提示。")

    for term_index, term_input in enumerate(chinese_terms_to_analyze):
        core_chinese_word = ""
        chinese_description = None

        if isinstance(term_input, str):
            core_chinese_word = term_input
            # chinese_description = term_input # 如果希望直接翻譯詞彙，可以這樣設定
            # 或者，你可以為單詞創建一個更豐富的預設描述模板
            chinese_description = f"一張描繪'{term_input}'的圖片，展現其典型特徵和氛圍"
        elif isinstance(term_input, tuple) and len(term_input) == 2:
            core_chinese_word = term_input[0]
            chinese_description = term_input[1]
        else:
            print(f"跳過無效的輸入格式: {term_input}")
            continue

        print(f"  處理輸入 #{term_index + 1}: 核心詞='{core_chinese_word}', 描述='{chinese_description}'")
        concept_provider.add_concept_with_auto_translation(
            translation_pipelines=translation_pipelines,
            base_chinese=core_chinese_word,
            chinese_description_for_prompt=chinese_description, # 傳遞描述句
            target_languages=['en', 'ja', 'ko']
        )
    print("--- 用戶定義詞彙概念添加完成 ---")

    text_analyzer = TextAnalyzer(clip_model=clip_model, device=config_mgr.device)
    image_processor = ImageProcessor(sd_pipeline=sd_pipeline, device=config_mgr.device)
    report_generator = ReportGenerator()

    # 4. 初始化並執行分析流程
    pipeline = AnalysisPipeline(
        config_manager=config_mgr,
        model_manager=model_mgr,
        concept_provider=concept_provider,
        text_analyzer=text_analyzer,
        image_processor=image_processor,
        report_generator=report_generator
    )

    if not concept_provider.get_concepts_to_process():
        print("沒有任何概念被定義，無法執行分析流程。請檢查 `chinese_terms_to_analyze`。")
    else:
        all_results = pipeline.run_analysis()
        # 可以選擇在這裡處理 all_results

    # 5. 清理資源
    model_mgr.cleanup()

    print("腳本執行完畢。")