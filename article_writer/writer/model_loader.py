"""
Загрузка дообученной модели Qwen3 и инференс.
Базовая модель: unsloth/qwen3-4b-unsloth-bnb-4bit.
Адаптер (LoRA): из папки qwen3-style-model, если там есть adapter_config.json.
"""
import os
from pathlib import Path

# Кэш Hugging Face в папке проекта (избегаем PermissionError в ~/.cache)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HF_CACHE_DIR = _PROJECT_ROOT / ".cache" / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Базовая модель из README чекпоинта
BASE_MODEL_ID = "unsloth/qwen3-4b-unsloth-bnb-4bit"

# Возможные имена папки с адаптером (поддержка qwen3-style-model и qwen3_style_model)
_WRITER_DIR = Path(__file__).resolve().parent
_ADAPTER_FOLDER_NAMES = ("qwen3-style-model", "qwen3_style_model")


def _get_adapter_dir() -> Path:
    """Папка с дообученным адаптером (та, в которой есть adapter_config.json)."""
    for name in _ADAPTER_FOLDER_NAMES:
        p = (_WRITER_DIR / name).resolve()
        if (p / "adapter_config.json").exists():
            return p
    return (_WRITER_DIR / _ADAPTER_FOLDER_NAMES[0]).resolve()


def _adapter_present() -> bool:
    return (_get_adapter_dir() / "adapter_config.json").exists()


def load_tokenizer():
    """Загружает токенизатор базовой модели (тот же, что и у адаптера)."""
    return AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        cache_dir=str(HF_CACHE_DIR),
    )


def load_model(device_map: str = "auto"):
    """
    Загружает модель: базу с Hub и при наличии — PEFT-адаптер из qwen3_style_model.
    При нехватке VRAM часть слоёв выгружается на CPU (max_memory).
    """
    tokenizer = load_tokenizer()
    # Разрешаем выгрузку части модели на CPU при нехватке VRAM (ноутбуки 8 GB и т.п.)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 0:
        max_memory = {}
        for i in range(n_gpus):
            vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            # Оставляем ~1.5 GB запаса на GPU, остальное может уехать на CPU
            max_memory[i] = f"{max(4, int(vram_gb - 1.5))}GiB"
        max_memory["cpu"] = "20GiB"
    else:
        max_memory = None
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
        cache_dir=str(HF_CACHE_DIR),
    )

    if _adapter_present():
        from peft import PeftModel
        adapter_path = str(_get_adapter_dir())
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
        model.eval()
    else:
        model.eval()

    return model, tokenizer


def generate_article(
    model,
    tokenizer,
    topic: str,
    word_count: int,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """
    Генерирует статью по теме и ориентиру по количеству слов.

    topic: содержание/тема статьи
    word_count: ориентировочное количество слов
    """
    prompt = (
        f"Напиши статью на тему: {topic}\n"
        f"Ориентировочный объём: примерно {word_count} слов. Пиши развёрнуто и по делу."
    )
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Декодируем только сгенерированную часть
    new_ids = output_ids[0][inputs.input_ids.shape[1] :].tolist()
    answer = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Обрезаем по первому стоп-токену или лишнему префиксу
    if "<|im_end|>" in answer:
        answer = answer.split("<|im_end|>")[0].strip()
    return answer.strip()
