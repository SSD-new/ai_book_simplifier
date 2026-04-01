#Модель
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#model_id = "mistralai/Mistral-Nemo-Instruct-2407"
#model_id = "Qwen/Qwen2.5-7B-Instruct"
model_id = "IlyaGusev/saiga_llama3_8b"

# Квантизация
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left' 

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},           
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    
)


print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

#Парсер
import re
def parse_war_and_peace(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    volumes = re.split(r'^Том\s+', content, flags=re.MULTILINE | re.IGNORECASE)
    
    structured_data = []
    
    for v_idx, vol in enumerate(volumes[1:], 1):
        parts = re.split(r'^Часть\s+', vol, flags=re.MULTILINE | re.IGNORECASE)
        
        for p_idx, part in enumerate(parts[1:], 1):
            # Главы разделяем по римским цифрам
            chapters = re.split(r'^[IVXLCDM]+\.?\s*$', part, flags=re.MULTILINE)
            
            for c_idx, chapter in enumerate(chapters[1:], 1):
                text = chapter.strip()
                if text:
                    structured_data.append({
                        "vol": v_idx,
                        "part": p_idx,
                        "chap": c_idx,
                        "text": text
                    })
    return structured_data

# Основной класс

import torch
import gc

class TolstoyProcessor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        self.characters_lore = """
[БАЗА ДАННЫХ ПЕРСОНАЖЕЙ - ЖЕСТКОЕ ПРАВИЛО]:
[БЕЗУХОВЫ]: Пьер (незаконный сын графа Кирилла Безухова). Пьер НЕ РОСТОВ и НЕ КУРАГИН!
[РОСТОВЫ]: Граф Илья (отец), графиня Наталья (мать). Дети: Николай (жив, не полковник!), Наташа, Вера, Петя. Соня (племянница).
[БОЛКОНСКИЕ]: Старый князь Николай. Дети: Андрей (муж Лизы), княжна Марья. Лиза (беременна, жена Андрея).
[КУРАГИНЫ]: Князь Василий (хитрый дальний родственник, ОН НЕ ОТЕЦ ПЬЕРУ!). Дети: Элен, Анатоль, Ипполит.
[ДРУГИЕ]: Борис Друбецкой (сын Анны Михайловны). Долохов. Денисов.
ВРЕМЯ ДЕЙСТВИЯ: 1805 год (НЕ 1812!).
"""

    def clean_truncate(self, text, max_chars=5000):
        """Обрезаем текст по последней точке, чтобы не было обрывков фраз.
        5000 символов безопасно влезают в 10GB VRAM вместе с промптом."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_dot = truncated.rfind('.')
        if last_dot != -1:
            return truncated[:last_dot + 1]
        return truncated

    def _build_prompt(self, chapter_text):
        system_content = (
            "Ты — строгий исторический архивариус. Твоя задача — извлечь 100% достоверные факты из текста.\n"
            "КРИТИЧЕСКИЕ ПРАВИЛА (ЗА НАРУШЕНИЕ - ШТРАФ):\n"
            "1. СТРОГО СВЕРЯЙ ФАМИЛИИ с [БАЗОЙ ДАННЫХ]. Князь Василий — КУРАГИН. Илья — РОСТОВ.\n"
            "2. ЗАПРЕЩЕНО убивать героев! Николай Ростов ВЫЖИВАЕТ при Шенграбене.\n"
            "3. ЗАПРЕЩЕНО додумывать романтику! Княжна Марья ОТКАЗЫВАЕТ Анатолю Курагину.\n"
            "4. Хронология: Сейчас 1805 год.\n"
            "5. Если текст обрывается до того, как персонаж принял решение или выжил, напиши: 'Чем закончилась сцена — в данном фрагменте не указано.'\n"
            "6. Пиши сухим языком фактов (3-5 предложений).\n"
            f"{self.characters_lore}"
        )
        
        user_content = f"Напиши краткое содержание этого текста:\n\n{chapter_text}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def process_chapter(self, chapter_text):
        
        safe_text = self.clean_truncate(chapter_text, max_chars=7500)
        
        prompt = self._build_prompt(safe_text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000).to("cuda")
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=350,        
                temperature=0.1,           
                repetition_penalty=1.15,  
                do_sample=True,
                top_p=0.85,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        new_tokens = output_tokens[0][inputs.input_ids.shape[-1]:]
        summary = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Очистка памяти
        del inputs, output_tokens
        gc.collect()
        torch.cuda.empty_cache()
        
        return summary

#Основной цикл
import os
import re

output_file = "war_and_peace_summarized_llama_test.md"
chapters_list = parse_war_and_peace("tolstoy_voyna-i-mir.txt")


last_vol, last_part, last_chap = 0, 0, 0

if os.path.exists(output_file):
    print(f"Файл {output_file} найден. Ищу последнюю обработанную главу...")
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()
        # Ищем все заголовки вида "## Том 1, Часть 1, Глава 5"
        found_chapters = re.findall(r"## Том (\d+), Часть (\d+), Глава (\d+)", content)
        
        if found_chapters:
            # Берем самую последнюю найденную главу
            last_vol, last_part, last_chap = map(int, found_chapters[-1])
            print(f"Продолжаем с: Том {last_vol}, Часть {last_part}, Глава {last_chap}")
else:
    # Если файла нет, создаем его и пишем заголовок
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Война и мир: Краткое содержание (AI Generated)\n\n")
    print("Создан новый файл для содержания.")


processor = TolstoyProcessor(model, tokenizer)

for item in chapters_list:
    v, p, c = item['vol'], item['part'], item['chap']
    
    # Пропускаем главы, которые уже обработаны
    if (v < last_vol) or \
       (v == last_vol and p < last_part) or \
       (v == last_vol and p == last_part and c <= last_chap):
        continue

    print(f"Обработка: Том {v}, Часть {p}, Глава {c}...")
    
    try:
        summary = processor.process_chapter(item['text'])
        
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"## Том {v}, Часть {p}, Глава {c}\n")
            f.write(f"{summary}\n\n")
            f.write("---\n\n")
            
    except Exception as e:
        print(f"!!! Ошибка на Том {v}, Часть {p}, Глава {c}: {e}")
        print("Прекращаю работу. Данные сохранены.")
        break

    if c % 5 == 0:
        print(f"--- Готово 5 глав. Последнее: {summary[:50]}... ---")

print("Обработка завершена или достигнут конец списка.")