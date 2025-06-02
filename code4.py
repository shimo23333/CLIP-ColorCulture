# 1:安裝必要的 Python 套件 (假設這些已在環境中安裝，或在腳本外部執行)
# !pip install -q git+https://github.com/openai/CLIP.git
# !pip install -q googletrans==4.0.0rc1 # 確保是這個版本或能正常工作的版本
# !pip install -q scikit-image
# !pip install -q opencv-python
# !pip install -q diffusers transformers accelerate invisible-watermark safetensors
# !pip install -q ipywidgets
# print("1:套件安裝步驟已略過，假設已完成。")

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
import matplotlib.font_manager as fm

print("函式庫匯入完成。")

# --- Class定義 ---

class ConfigManager:
    """
    職責：管理設定，如運算設備、字體路徑和輸出目錄。
    說明：這就像是專案的「設定面板」，集中管理一些全域會用到的設定值。
    """
    def __init__(self, font_path_cjk='/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf',
                 images_out_dir="project_outputs_oop_googletrans"): # 修改目錄名以區分版本
        self.device = self._get_device()
        self.font_path_cjk = font_path_cjk
        self.images_out_dir = images_out_dir
        self._setup_matplotlib_font()
        self._setup_output_directory()

    def _get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"本次運行的計算設備是: {device}")
        if device == "cpu":
            print("警告：未使用GPU！運行大型AI模型會非常慢。")
        return device

    def _setup_matplotlib_font(self):
        print("正在設定 Matplotlib CJK 字體...")
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
        plt.rcParams['axes.unicode_minus'] = False # 再次確保
        print("Matplotlib CJK 字體設定完畢。")

    def _setup_output_directory(self):
        if not os.path.exists(self.images_out_dir):
            os.makedirs(self.images_out_dir)
            print(f"  已建立圖像儲存目錄: {self.images_out_dir}")

class ModelManager:
    """
    職責：載入和管理AI模型 (CLIP, Stable Diffusion) 及翻譯工具。
    說明：這就像是專案的「AI模型與工具倉庫管理員」。
    """
    def __init__(self, device):
        self.device = device
        self.clip_model = None
        self.clip_preprocess = None
        self.sd_pipeline = None
        self.translator = None

    def load_clip_model(self, model_name="ViT-B/32"):
        print(f"準備載入 CLIP 模型 ({model_name})...")
        if self.device == "cuda":
            try:
                self.clip_model, self.clip_preprocess = clip.load(model_name, device=self.device)
                self.clip_model.eval()
                print(f"  CLIP 模型 ({model_name}) 已成功載入到 {self.device}！")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  載入CLIP模型時發生錯誤: {e}")
                self.clip_model, self.clip_preprocess = None, None # 確保失敗時設為None
        else:
            print(f"  未實際載入CLIP模型，因為當前運算設備是 {self.device}。")
        return self.clip_model, self.clip_preprocess

    def load_sd_model(self, model_id="runwayml/stable-diffusion-v1-5"):
        print(f"準備載入 Stable Diffusion 模型 ({model_id})...")
        if self.device == "cuda":
            try:
                self.sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
                self.sd_pipeline = self.sd_pipeline.to(self.device)
                print(f"  Stable Diffusion 模型 ({model_id}) 已成功載入到 {self.device}！")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  載入Stable Diffusion模型時發生錯誤: {e}")
                self.sd_pipeline = None # 確保失敗時設為None
        else:
            print(f"  未實際載入Stable Diffusion模型，因為當前運算設備是 {self.device}。")
        return self.sd_pipeline

    def init_translator(self):
        print("正在初始化 Google 翻譯工具...")
        try:
            self.translator = Translator()
            # 進行一次測試翻譯以確認服務可用性
            test_text = "你好"
            translated_test_text = self.translator.translate(test_text, src='zh-cn', dest='en').text
            print(f"  Google 翻譯工具初始化成功！測試翻譯 '{test_text}' -> '{translated_test_text}'")
        except Exception as e:
            print(f"  Google 翻譯工具初始化失敗: {e}")
            self.translator = None # 確保失敗時設為None
        return self.translator

    def cleanup(self):
        print("正在清理模型資源...")
        if self.clip_model: del self.clip_model
        if self.clip_preprocess: del self.clip_preprocess
        if self.sd_pipeline: del self.sd_pipeline
        if self.translator: del self.translator # 清理 translator 物件
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print("  模型和資源清理操作已執行。")

class ConceptDataProvider:
    """
    職責：提供專案要分析的詞彙概念及多語言翻譯。
    說明：這就像是專案的「詞彙字典」，存放著所有要研究的詞和它們的翻譯。
            此版本使用預定義的高質量翻譯。
    """
    def __init__(self):
        self.word_concepts_list = [
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
        # 為了測試方便，可以只選取部分詞彙，例如:
        # self.word_concepts_to_process = [self.word_concepts_list[0], self.word_concepts_list[6]]
        self.word_concepts_to_process = self.word_concepts_list
        print(f"定義了 {len(self.word_concepts_to_process)} 個詞彙概念用於本次分析。")

    def get_concepts_to_process(self):
        return self.word_concepts_to_process

class TextAnalyzer:
    """
    職責：處理文本，如獲取CLIP嵌入向量和計算相似度。
    說明：這就像是專案的「文本分析師」，專門理解文字的意義並比較它們的相似程度。
    """
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
                    print(f"  為 '{lang_tag}':'{text_content[:30]}...' 生成CLIP嵌入時出錯: {e}")
                    text_embeddings_result_dict[lang_tag] = np.zeros(512, dtype=np.float32)
        return text_embeddings_result_dict

    def calculate_embedding_similarity(self, embeddings_dict, reference_lang='en'):
        similarity_scores_result = {}
        if reference_lang not in embeddings_dict or embeddings_dict.get(reference_lang) is None or \
           np.all(np.isclose(embeddings_dict[reference_lang], 0)):
            print(f"  參考語言 '{reference_lang.upper()}' 的嵌入向量無效或不存在，無法計算相似度。")
            return {f"{reference_lang}_vs_{lang}": None for lang in embeddings_dict if lang != reference_lang}

        print(f"  CLIP文本嵌入向量餘弦相似度 (vs '{reference_lang.upper()}'):")
        ref_embedding = embeddings_dict[reference_lang].reshape(1, -1)

        for lang, emb in embeddings_dict.items():
            if lang == reference_lang:
                continue
            sim_val_str = "N/A (嵌入無效)"
            sim_num = None
            if emb is not None and not np.all(np.isclose(emb, 0)):
                sim_num = cosine_similarity(ref_embedding, emb.reshape(1, -1))[0][0]
                sim_val_str = f"{sim_num:.3f}"
            similarity_scores_result[f"{reference_lang}_vs_{lang}"] = sim_num
            print(f"    - 與 {lang.upper()}: {sim_val_str}")
        return similarity_scores_result

class ImageProcessor:
    """
    職責：處理圖像，如生成圖像、提取顏色、分析全局特徵。
    說明：這就像是專案的「圖像處理大師」，會畫圖，也會分析圖的顏色和整體感覺。
    """
    def __init__(self, sd_pipeline, device):
        self.sd_pipeline = sd_pipeline
        self.device = device

    def generate_image_with_sd(self, prompt_text, random_seed=42, inference_steps=30, cfg_scale=7.5):
        if self.sd_pipeline is None:
            print("  警告: Stable Diffusion 模型未載入，將返回佔位符圖像。")
            placeholder_img = Image.new('RGB', (512, 512), color='silver')
            draw = ImageDraw.Draw(placeholder_img)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 18)
            except IOError:
                font = ImageFont.load_default()
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
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 15)
            except IOError:
                font = ImageFont.load_default()
            draw.text((10, 10), f"圖像生成錯誤:\n{prompt_text[:60]}...\n錯誤:\n{str(e)[:100]}", fill=(0, 0, 0), font=font)
            return error_img

    def extract_dominant_colors(self, pil_img, num_colors=5):
        if pil_img is None or (hasattr(pil_img, 'width') and pil_img.width < num_colors) or \
           (hasattr(pil_img, 'height') and pil_img.height < num_colors):
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
                    rgb_colors[actual_extracted_colors.shape[0]:] = np.array([128,128,128]) # 用灰色填充
            else:
                kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init='auto', max_iter=200).fit(pixels)
                rgb_colors = kmeans.cluster_centers_.astype(int)

            # 確保返回的rgb_colors總是有num_colors個元素
            if rgb_colors.shape[0] < num_colors:
                padded_colors = np.full((num_colors, 3), [128,128,128], dtype=int) # 預設填充灰色
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
        if pil_image is None:
            return {"avg_brightness": "N/A", "contrast_std": "N/A", "avg_saturation": "N/A"}
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
    """
    職責：產生分析報告和繪製結果圖表。
    說明：這就像是專案的「報告撰寫員與設計師」，負責整理結果並用圖表展示。
    """
    def generate_explanation_template(self, chinese_concept, lang_prompt_info, dom_colors_hex=None, global_feats=None):
        expl = f"\n--- 解釋模板 for 概念:【{chinese_concept}】| 語言提示: 【{lang_prompt_info[:70]}...】 ---\n"
        if dom_colors_hex:
            expl += f"圖像主要色票 (HEX): {', '.join(dom_colors_hex[:3])} ...\n"
        if global_feats:
            expl += f"全局圖像特徵: 亮度={global_feats.get('avg_brightness', 'N/A')}, "
            expl += f"對比度={global_feats.get('contrast_std', 'N/A')}, 飽和度={global_feats.get('avg_saturation', 'N/A')}\n"
        expl += f"\n原因推測與圖像描述 (請您填充)：\n"
        expl += f"   [請結合以上客觀指標和您的觀察，詳細闡述：\n"
        expl += f"    a. 圖像視覺風格與氛圍？\n"
        expl += f"    b. 主要元素與提示詞的關聯？\n"
        expl += f"    c. 色彩運用如何詮釋提示詞？\n"
        expl += f"    d. (特定語言)文化背景的可能影響？\n"
        expl += f"    e. 與其他語言生成圖像的差異及可能原因？]\n"
        expl += f"--------------------------------------------------------------------------\n"
        return expl

    def plot_concept_results(self, concept_id, base_chinese, prompts_dict, images_dict,
                               colors_dict, similarities_dict, global_features_dict=None):
        langs = list(prompts_dict.keys())
        num_langs = len(langs)
        if num_langs == 0:
            print(f"概念 '{concept_id}' 無 prompts 可繪製。")
            return

        h_ratio, w_ratio = 2.8, 4.0 # 調整比例以適應更多內容
        total_h, total_w = h_ratio * 2, w_ratio * num_langs if num_langs > 0 else w_ratio # 避免 num_langs 為 0
        fig, axs = plt.subplots(2, max(1, num_langs), figsize=(total_w, total_h), gridspec_kw={'height_ratios': [0.78, 0.22]})

        if num_langs == 1: # 處理 axs 不是二維陣列的情況
            axs = axs.reshape(2, 1)

        title_base = f"概念分析: '{base_chinese}' ({concept_id})\nCLIP相似度(vs EN): "
        sim_strs = []
        if similarities_dict:
            sim_strs = [f"{k.split('_vs_')[-1].upper()}: {v:.2f}" if isinstance(v, (float, np.floating)) else f"{k.split('_vs_')[-1].upper()}: {v}"
                        for k, v in similarities_dict.items()]
        fig.suptitle(title_base + ", ".join(sim_strs), fontsize=11, y=1.04) # 調整 y 以免重疊

        for i, lang in enumerate(langs):
            img = images_dict.get(lang)
            colors_data = colors_dict.get(lang) # (rgb_patch, lab_patch)
            global_feats_this_lang = (global_features_dict or {}).get(lang, {})

            ax_img = axs[0, i]
            ax_color = axs[1, i]

            if img:
                ax_img.imshow(img)
            else:
                ax_img.text(0.5, 0.5, '圖像未生成', ha='center', va='center', transform=ax_img.transAxes)

            img_title_prompt = prompts_dict.get(lang, "N/A")
            img_title = f"{lang.upper()}: \"{img_title_prompt[:30]}\"..." # 調整截斷長度
            if global_feats_this_lang:
                img_title += f"\n亮:{global_feats_this_lang.get('avg_brightness', '-')} "
                img_title += f"對比:{global_feats_this_lang.get('contrast_std', '-')} "
                img_title += f"飽:{global_feats_this_lang.get('avg_saturation', '-')}"
            ax_img.set_title(img_title, fontsize=7.5)
            ax_img.axis('off')

            if colors_data:
                rgb_patch, lab_patch = colors_data
                if rgb_patch is not None and len(rgb_patch) > 0 :
                    n_patch = len(rgb_patch)
                    patch_canvas = np.zeros((25, 100, 3), dtype=np.uint8)
                    patch_w = 100 // n_patch if n_patch > 0 else 100 # 避免除以零
                    for j, rgb_c in enumerate(rgb_patch):
                        patch_canvas[:, j * patch_w:(j + 1) * patch_w] = rgb_c
                    ax_color.imshow(patch_canvas)

                    lab_str_parts = []
                    if lab_patch is not None:
                        for l_val, a_val, b_val in lab_patch[:min(3,n_patch)]: # 最多顯示前3個 Lab 值
                            lab_str_parts.append(f"L{l_val:.0f} a{a_val:.0f} b{b_val:.0f}")
                        lab_str = "\n".join(lab_str_parts)
                        ax_color.set_title(f"Lab(Top{min(3,n_patch)}):\n{lab_str}", fontsize=6)
                    else:
                        ax_color.set_title(f"RGB顏色", fontsize=6) # 如果沒有Lab數據
                else:
                    ax_color.text(0.5,0.5,'無顏色數據',ha='center',va='center',transform=ax_color.transAxes, fontsize=6)
            else:
                ax_color.text(0.5, 0.5, '無顏色', ha='center', va='center', transform=ax_color.transAxes, fontsize=6)
            ax_color.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.94]) # 調整 rect 以免標題被裁切
        plt.subplots_adjust(hspace=0.5, wspace=0.3) # 調整子圖間距
        plt.show()

class AnalysisPipeline:
    """
    職責：協調所有組件，執行完整的分析流程。
    說明：這就像是專案的「總指揮」，負責調度各個專家（其他類別）按順序完成工作。
    """
    def __init__(self, config_manager, model_manager, concept_provider,
                 text_analyzer, image_processor, report_generator):
        self.config = config_manager
        self.models = model_manager
        self.concepts_provider = concept_provider
        self.text_analyzer = text_analyzer
        self.image_processor = image_processor
        self.reporter = report_generator

        # 主流程中的參數
        self.base_seed = 20240101 # 基礎隨機種子
        self.sd_steps = 25        # Stable Diffusion 推理步數
        self.sd_cfg = 7.2         # CFG Scale
        self.num_dom_colors = 5   # 提取主要顏色的數量
        self.save_images_flag = True # 是否保存生成的圖像

    def run_analysis(self):
        print(f"即將開始執行主流程...")
        concepts_to_process = self.concepts_provider.get_concepts_to_process()
        if not concepts_to_process:
            print("警告：沒有定義任何詞彙概念進行處理。流程結束。")
            return []
        print(f"  將處理 {len(concepts_to_process)} 個詞彙概念...")

        results_collection = []

        for concept_idx, concept_detail in enumerate(tqdm(concepts_to_process, desc="總體概念處理進度")):
            concept_id = concept_detail["concept_name"]
            base_zh = concept_detail["base_chinese"]
            prompts = concept_detail["translations"] # 這裡的 prompts 應該是已經準備好的多語言提示

            print(f"\n\n處理概念 #{concept_idx + 1}: '{base_zh}' ({concept_id})")

            print("  [1. CLIP嵌入分析]")
            embeddings = self.text_analyzer.get_clip_text_embeddings(prompts)
            similarities = self.text_analyzer.calculate_embedding_similarity(embeddings, reference_lang='en')

            print("  [2. 圖像生成、顏色與全局特徵分析]")
            concept_images = {}
            concept_colors = {}
            concept_global_features = {}
            concept_explanations_str = ""

            for lang_idx, (lang, prompt_text) in enumerate(tqdm(prompts.items(), desc=f"  '{concept_id}'語言處理", leave=False)):
                print(f"    -> {lang.upper()}: '{prompt_text}'")
                img_seed = self.base_seed + concept_idx * 100 + lang_idx * 10 # 為每個圖像生成唯一的種子

                pil_img = self.image_processor.generate_image_with_sd(
                    prompt_text,
                    random_seed=img_seed,
                    inference_steps=self.sd_steps,
                    cfg_scale=self.sd_cfg
                )
                concept_images[lang] = pil_img

                if self.save_images_flag and pil_img:
                    try:
                        fname = f"{concept_id}_{lang}_s{img_seed}.png"
                        fpath = os.path.join(self.config.images_out_dir, fname)
                        pil_img.save(fpath)
                        # print(f"      圖像已儲存至: {fpath}") # 可選的詳細輸出
                    except Exception as e:
                        print(f"      儲存圖像'{fname}'失敗: {e}")

                rgb_cs, lab_cs = self.image_processor.extract_dominant_colors(pil_img, self.num_dom_colors)
                concept_colors[lang] = (rgb_cs, lab_cs)

                global_feats = self.image_processor.analyze_global_features(pil_img)
                concept_global_features[lang] = global_feats

                hex_colors = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in rgb_cs] if rgb_cs is not None and len(rgb_cs)>0 else []
                expl_text = self.reporter.generate_explanation_template(
                    base_zh, f"{lang.upper()}: {prompt_text}", hex_colors, global_feats
                )
                print(expl_text)
                concept_explanations_str += expl_text

                if self.config.device == "cuda":
                    torch.cuda.empty_cache()
                    time.sleep(0.05)

            print("\n  [3. 繪製結果圖表]")
            self.reporter.plot_concept_results(
                concept_id, base_zh, prompts, concept_images,
                concept_colors, similarities, concept_global_features
            )

            results_collection.append({
                "concept": concept_id,
                "base_chinese": base_zh,
                "prompts": prompts, # 保存用於生成的提示
                "similarities": similarities,
                "global_features": concept_global_features,
                "dominant_colors_data": concept_colors, # 保存更詳細的顏色數據
                "explanation_templates_output": concept_explanations_str
            })
            print(f"  概念 '{concept_id}' 分析完畢。")
            if self.config.device == "cuda":
                torch.cuda.empty_cache()

        print("\n\n所有詞彙概念處理完成！")
        return results_collection

# --- 主執行流程 ---
if __name__ == '__main__':
    print("主腳本開始執行...")

    # 1. 初始化配置管理器
    config_mgr = ConfigManager()

    # 2. 初始化模型管理器
    model_mgr = ModelManager(device=config_mgr.device)
    clip_model, _ = model_mgr.load_clip_model()
    sd_pipeline = model_mgr.load_sd_model()
    # translator = model_mgr.init_translator() # 如果你的 ConceptDataProvider 依賴動態翻譯，則取消註釋

    # 3. 初始化數據和工具
    # ConceptDataProvider 此版本直接使用內部定義的列表，不需要 translator
    concept_provider = ConceptDataProvider()
    text_analyzer = TextAnalyzer(clip_model=clip_model, device=config_mgr.device)
    image_processor = ImageProcessor(sd_pipeline=sd_pipeline, device=config_mgr.device)
    report_generator = ReportGenerator()

    # 4. 初始化並執行分析流程
    main_pipeline = AnalysisPipeline(
        config_manager=config_mgr,
        model_manager=model_mgr,
        concept_provider=concept_provider,
        text_analyzer=text_analyzer,
        image_processor=image_processor,
        report_generator=report_generator
    )

    if not concept_provider.get_concepts_to_process():
        print("主腳本：沒有任何概念被定義，無法執行分析流程。請檢查 ConceptDataProvider。")
    elif config_mgr.device == "cuda" and (not clip_model or not sd_pipeline):
         print("主腳本：核心AI模型（CLIP 或 Stable Diffusion）未能成功載入到GPU。分析流程可能無法正確執行或結果無意義。")
    else:
        all_run_results = main_pipeline.run_analysis()
        # print("\n最終收集到的所有結果數據:", all_run_results) # 可選：打印最終數據結構

    # 5. 清理資源
    model_mgr.cleanup()

    print("\n主腳本執行完畢。")