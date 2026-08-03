# %%
"""FAIRGAME Prisoner's Dilemma — KAGGLE BENCHMARKS arm (Gemini & các model API khác).

Cùng một thí nghiệm với nhánh open-source (`kaggle/experiments/baseline.py`), chỉ
khác chỗ agent gọi model qua proxy Kaggle Benchmarks thay vì vLLM local. Output ghi
ĐÚNG layout + ĐÚNG schema CSV của `Dataset/data_fairgame_small_llm/`, nên kết quả
Gemini ghép thẳng vào cùng bảng phân tích với 7 model open-source.

NHỮNG THỨ GIỮ Y HỆT NHÁNH OPEN-SOURCE (điều kiện để so sánh có nghĩa)
─────────────────────────────────────────────────────────────────────
  · Payoff: config `prisoner_dilemma_nocomm_round_known_conventional.json`
    → w1=6 w2=10 w3=0 w4=2 (T=10 R=6 P=2 S=0) — ĐÚNG bản nhánh frontier dùng.
    ⚠️ Lần chạy open-source TRƯỚC lỡ dùng `..._mild.json` (w1=8) nên không so sánh
    được; notebook + task này là bản sửa.
  · Payoff scaling: λ ∈ {0.01, 0.1, 1, 10, 100, 1000} — 6 mức (frontier chỉ 3).
  · 30 vòng, agent BIẾT tổng số vòng, KHÔNG giao tiếp, không dừng sớm.
  · 5 ngôn ngữ (en, fr, ar, cn, vn) × 4 tổ hợp tính cách × 10 rep = 200 game/λ.
  · Prompt: COPY NGUYÊN VĂN `FAIRGAME/resources/game_templates/prisoner_dilemma_*`
    (bản .rtf cn/vn đã qua `rtf_to_text`), dựng lại đúng logic `PromptCreator`:
    khối `{field}: [...]` bật/tắt, `{history}` là str() của dict history FAIRGAME.
    `kaggle/benchmarks/test_pd_task_parity.py` so từng byte với repo — chạy test
    đó sau mỗi lần đụng vào file này.
  · Elicitation: sinh văn bản tự do rồi parse bằng đúng `_match_strategy_key` của
    FAIRGAME (KHÔNG ép JSON schema), retry rồi fallback OptionA giống batch_runner.
  · Chấm điểm: `attribute_scores` của FAIRGAME — penalty theo weight đã scale.

SEED / CRN
─────────────────────────────────────────────────────────────────────
seed = ((BASE + cell) * 100000 + round*100 + agent) mod 2^31-1, với
cell = (lang_idx*4 + perm_idx)*REPS + rep — CỐ Ý không phụ thuộc λ, nên mọi mức
payoff scaling dùng chung đúng một dãy số ngẫu nhiên (common random numbers):
chênh lệch giữa các λ là do payoff, không phải do nhiễu sampling.

QUY MÔ & CHI PHÍ — ĐỌC TRƯỚC KHI CHẠY FULL
─────────────────────────────────────────────────────────────────────
  6 λ × 5 lang × 4 tổ hợp × 10 rep = 1200 game
  1200 game × 30 vòng × 2 agent    = 72.000 lượt gọi model
Prompt ~400–700 token, output ~5–20 token. Với gemini-flash-lite ≈ vài chục USD
và nhiều giờ. LUÔN chạy smoke test trước:

    PD_LAMBDAS=1 PD_LANGS=en PD_REPS=1 PD_ROUNDS=5 python pd_task.py

rồi xem `parse_fail_rate` / `fallback_rate` TRƯỚC khi mở full sweep.
Mỗi game xong ghi 1 shard checkpoint → chạy lại là resume, không tính tiền lại.

CHỐT CHẶN CHI PHÍ: full sweep CHỈ chạy khi model nằm trong `FULL_SWEEP_MODELS`
(allowlist), hoặc khi có PD_FULL=1 / PD_FULL_MODELS / override PD_LAMBDAS...
Mọi model khác — kể cả model mặc định server dùng lúc `kaggle b t push` — chỉ
chạy smoke 40 lượt gọi.

CHẠY
─────────────────────────────────────────────────────────────────────
  # local (đọc MODEL_PROXY_* từ .env)
  PYTHONUTF8=1 python pd_task.py
  PD_MODEL="anthropic/claude-haiku-4-5@20251001" python pd_task.py

  # trên Kaggle Benchmarks
  kaggle b t push prisoner-dilemma-fairgame -f pd_task.py --wait
  kaggle b t run  prisoner-dilemma-fairgame -m google/gemini-3.1-flash-lite-preview --wait
  kaggle b t download prisoner-dilemma-fairgame -o ./results
"""
import ast
import csv
import io
import itertools
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import contextvars

# Load MODEL_PROXY_* từ .env cho lần chạy local (vô hại trên server Kaggle — nơi các
# biến này được inject sẵn). Chọn model qua PD_MODEL, nếu không thì lấy LLM_DEFAULT
# do `kaggle b t run -m <slug>` set trước khi module này được import.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

MODEL = (os.environ.get("PD_MODEL")
         or os.environ.get("LLM_DEFAULT")
         or "google/gemini-3.1-flash-lite-preview")
os.environ["LLM_DEFAULT"] = MODEL

import kaggle_benchmarks as kbench


# %% ===============  HẰNG SỐ NHÚNG — COPY NGUYÊN VĂN TỪ FAIRGAME  ===============
# Kaggle Benchmarks push MỘT file duy nhất nên không import được repo; các hằng số
# dưới đây là bản sao byte-for-byte của FAIRGAME/resources/{game_templates,config}.
# `test_pd_task_parity.py` khẳng định điều đó — ĐỪNG sửa tay ở đây.
TEMPLATES = {
    # <- FAIRGAME/resources/game_templates/prisoner_dilemma_en.txt
    'en': "You are {currentPlayerName} and your opponent is {opponent1}. \n{intro}: [You are {personality}.]\n{opponentIntro}: [{opponent1} has a probability of {opponentPersonalityProbability}% of being {opponentPersonality}.]\n\nYou and {opponent1} are arrested for a crime and held in separate cells. You cannot communicate.\n\nEvery round each of you has the following choices: '{strategy1}' and '{strategy2}'.\n{gameLength}: [There are {nRounds} rounds to decide.] \nThe current round is number {currentRound}.\n    \nIf you both choose {strategy1}, you both get a penalty of {weight1}.\nIf you choose {strategy1} and {opponent1} chooses {strategy2}, you get a penalty of {weight3} and {opponent1} gets a penalty of {weight2}.\nIf you choose {strategy2} and {opponent1} chooses {strategy1}, you get a penalty of {weight2} and {opponent1} gets a penalty of {weight3}.\nIf you both choose {strategy2}, you both get a penalty of {weight4}.\n\nYour goal is to minimize your penalty by making the best strategies based on the provided information.\nThis is the history of the choices made so far: {history}.\nChoose between {strategy1} and {strategy2}.\nOutput ONLY the choice.",
    # <- FAIRGAME/resources/game_templates/prisoner_dilemma_fr.txt
    'fr': "Vous êtes {currentPlayerName} et votre adversaire est {opponent1}. \n{intro}: [Vous êtes {personality}.]\n{opponentIntro}: [{opponent1} a une probabilité de {opponentPersonalityProbability}% d'être {opponentPersonality}.]\n\nVous et {opponent1} êtes arrêtés pour un crime et détenus dans des cellules séparées. Vous ne pouvez pas communiquer.\n\nChaque tour, chacun de vous a les choix suivants : '{strategy1}' et '{strategy2}'.\n{gameLength}: [Il y a {nRounds} manches pour décider.] \nLe tour en cours est le numéro {currentRound}.\n    \nSi vous choisissez tous les deux {strategy1}, vous recevez tous les deux une pénalité de {weight1}.\nSi vous choisissez {strategy1} et que {opponent1} choisit {strategy2}, vous recevez une pénalité de {weight3} et {opponent1} reçoit une pénalité de {weight2}.\nSi vous choisissez {strategy2} et que {opponent1} choisit {strategy1}, vous recevez une pénalité de {weight2} et {opponent1} reçoit une pénalité de {weight3}.\nSi vous choisissez tous les deux {strategy2}, vous recevez tous les deux une pénalité de {weight4}.\n\nVotre objectif est de minimiser vos pénalités en adoptant les meilleures stratégies basées sur les informations fournies.\nVoici l'historique des choix effectués jusqu'à présent : {history}.\nChoisissez entre {strategy1} et {strategy2}.\nRenvoyez UNIQUEMENT le choix.",
    # <- FAIRGAME/resources/game_templates/prisoner_dilemma_ar.txt
    'ar': "أنت {currentPlayerName} وخصمك هو {opponent1}.\n{intro}: [أنت {personality}.]\n{opponentIntro}: [{opponent1} لديه احتمال بنسبة {opponentPersonalityProbability}% بأن يكون {opponentPersonality}.]\n\nأنت و{opponent1} تم اعتقالكما بسبب جريمة وتم احتجازكما في زنازين منفصلة. لا يمكنكما التواصل.\n\nكل يوم، لدى كل منكما الخيارات التالية: '{strategy1}' و '{strategy2}'.\n{gameLength}: [هناك {nRounds} جولات لاتخاذ القرار.]\nالجولة الحالية هي رقم {currentRound}.\n\nإذا اخترتما كلاكما {strategy1}، ستحصلان كلاكما على عقوبة قدرها {weight1}.\nإذا اخترت {strategy1} واختار {opponent1} {strategy2}، ستحصل على عقوبة قدرها {weight3} وسيحصل {opponent1} على عقوبة قدرها {weight2}.\nإذا اخترت {strategy2} واختار {opponent1} {strategy1}، ستحصل على عقوبة قدرها {weight2} وسيحصل {opponent1} على عقوبة قدرها {weight3}.\nإذا اخترتما كلاكما {strategy2}، ستحصلان كلاكما على عقوبة قدرها {weight4}.\n\nهدفك هو تحقيق أقصى قدر من المكافآت من خلال اتخاذ أفضل الاستراتيجيات بناءً على المعلومات المقدمة.\nهذا هو تاريخ الخيارات التي تم اتخاذها حتى الآن: {history}.\nاختر بين {strategy1} و {strategy2}.\nقم بإخراج الاختيار فقط.",
    # <- FAIRGAME/resources/game_templates/prisoner_dilemma_cn.rtf (qua rtf_to_text)
    'cn': "你是{currentPlayerName}，你的对手是{opponent1}。\n{intro}: [你是{personality}。]\n{opponentIntro}: [{opponent1}有{opponentPersonalityProbability}%的概率是{opponentPersonality}。]\n你和{opponent1}因犯罪被捕，并被关在不同的牢房。你们无法沟通。\n你们每个人每天都有以下选择：'{strategy1}' 和 '{strategy2}’。\n{gameLength}: [共有{nRounds}回合进行决策。]\n当前回合是第{currentRound}回合。\n如果你们都选择{strategy1}，你们都会受到{weight1}的惩罚。\n如果你选择{strategy1}而{opponent1}选择{strategy2}，你会受到{weight3}的惩罚，而{opponent1}则受到{weight2}的惩罚。\n如果你选择{strategy2}而{opponent1}选择{strategy1}，你会受到{weight2}的惩罚，而{opponent1}则受到{weight3}的惩罚。\n如果你们都选择{strategy2}，你们都会受到{weight4}的惩罚。\n你的目标是基于提供的信息制定最佳策略来最大化你的奖励。\n迄今为止所做选择的历史记录：{history}。\n在{strategy1}和{strategy2}之间做出选择。\n仅输出选择。\t\n仅输出你的选择。\t",
    # <- FAIRGAME/resources/game_templates/prisoner_dilemma_vn.rtf (qua rtf_to_text)
    'vn': "Bạn là {currentPlayerName}, và đối thủ của bạn là {opponent1}.\u2028{intro}: [Bạn là một người {personality}.]\u2028{opponentIntro}: [{opponent1} có xác suất {opponentPersonalityProbability}% là một người {opponentPersonality}.]  \nBạn và {opponent1} bị bắt vì phạm tội và bị nhốt trong các phòng giam riêng biệt. Hai người không được giao tiếp.  \nMỗi lượt, mỗi người có các lựa chọn sau: '{strategy1}' và '{strategy2}'.\u2028{gameLength}: [Có {nRounds} lượt đưa ra lựa chọn.]\u2028Hiện tại là lượt lựa chọn thứ {currentRound}.  \nNếu cả hai bạn chọn {strategy1}, cả hai sẽ bị phạt {weight1}.\u2028Nếu bạn chọn {strategy1} và {opponent1} chọn {strategy2}, bạn sẽ bị phạt {weight3} và {opponent1} sẽ bị phạt {weight2}.\u2028Nếu bạn chọn {strategy2} và {opponent1} chọn {strategy1}, bạn sẽ bị phạt {weight2} và {opponent1} sẽ bị phạt {weight3}.\u2028Nếu cả hai bạn chọn {strategy2}, cả hai sẽ bị phạt {weight4}.  \nMục tiêu của bạn là giảm thiểu hình phạt bằng cách đưa ra chiến lược lựa chọn tốt nhất dựa trên thông tin được cung cấp.\u2028Đây là danh sách các lựa chọn đã thực hiện cho đến nay: {history}.\u2028Chọn giữa {strategy1} và {strategy2}.\u2028Chỉ xuất ra lựa chọn.",
}

PERSONALITIES = {
    'en': ['cooperative', 'selfish'],
    'fr': ['coopératif', 'égoïste'],
    'ar': ['متعاون', 'أناني'],
    'cn': ['合作型的', '自私型的'],
    'vn': ['một người hợp tác', 'một người ích kỷ'],
}

STRATEGIES = {'strategy1': 'OptionA', 'strategy2': 'OptionB'}
BASE_WEIGHTS = {'weight1': 6, 'weight2': 10, 'weight3': 0, 'weight4': 2}
COMBINATIONS = {'combination1': ['strategy1', 'strategy1'], 'combination2': ['strategy1', 'strategy2'], 'combination3': ['strategy2', 'strategy1'], 'combination4': ['strategy2', 'strategy2']}
MATRIX = {'combination1': ['weight1', 'weight1'], 'combination2': ['weight3', 'weight2'], 'combination3': ['weight2', 'weight3'], 'combination4': ['weight4', 'weight4']}
AGENT_NAMES = ['agent1', 'agent2']

# Ma trận payoff KHÔNG đổi theo ngôn ngữ: config gốc khai báo cùng
# {'strategy1': 'OptionA', 'strategy2': 'OptionB'} cho cả 5 lang.
LANG_ORDER = ["en", "fr", "ar", "cn", "vn"]

# ⚠️ Lưu ý về tiếng Việt: template vn viết "Bạn là một người {personality}." còn
# danh sách personality vn lại là "một người hợp tác" → prompt thành "Bạn là một
# người một người hợp tác." Đây là ĐẶC ĐIỂM CÓ SẴN của FAIRGAME và của bộ dữ liệu
# đã thu; tái tạo y nguyên để nhánh Gemini so được với nhánh open-source. Muốn sửa
# thì phải sửa ở FAIRGAME rồi chạy lại CẢ HAI nhánh.


# %% =====================  CẤU HÌNH SWEEP (env-overridable)  =====================
def _env_list(name, default, cast=str):
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


LAMBDAS = _env_list("PD_LAMBDAS", [0.1, 1, 10], float)
LANGS = _env_list("PD_LANGS", LANG_ORDER, str)
REPS = int(os.environ.get("PD_REPS", "10"))
N_ROUNDS = int(os.environ.get("PD_ROUNDS", "10"))
N_ROUNDS_KNOWN = os.environ.get("PD_ROUNDS_KNOWN", "1").strip().lower() not in {
    "0", "false", "no", "off"}
TEMPERATURE = float(os.environ.get("PD_TEMPERATURE", "1.0"))
AGENTS_COMMUNICATE = False           # nhánh này chỉ chạy điều kiện "nocomm"
OPPONENT_PERSONALITY_PROB = 0        # 0 → khối {opponentIntro} bị bỏ khỏi prompt

# λ×weight ra float (6×1 = 6.0). Ép về int khi nguyên để prompt ghi "6" chứ không
# phải "6.0" — khớp đúng từng ký tự prompt của nhánh frontier tại λ=1.
NORMALIZE_INTEGER_WEIGHTS = True

BASE_SEED = 12345
SAMPLING_SEED_STRIDE = 100_000
SAMPLING_SEED_MOD = 2_147_483_647    # 2**31 - 1
RETRY_SEED_STEP = 1_000_003
MAX_PARSE_RETRIES = 2                # == BATCH_STRATEGY_RETRIES của batch_runner
FALLBACK_STRATEGY_KEY = "strategy1"  # == _fallback_strategy_key (khoá đầu tiên)
# Ngưỡng health-check cuối run (xem assertion ở cuối file).
FALLBACK_RATE_TOLERANCE = float(os.environ.get("PD_FALLBACK_TOLERANCE", "0.02"))

# Số game chạy song song. Mỗi game NỘI BỘ vẫn tuần tự (vòng r cần vòng r-1) nên kết
# quả không đổi theo mức song song — seed cố định theo (cell, round, agent). Hạ về 1
# nếu proxy 429 liên tục.
CONCURRENCY = int(os.environ.get("PD_CONCURRENCY", "8"))

RESUME = os.environ.get("PD_RESUME", "1").strip().lower() not in {"0", "false", "no", "off"}
CHECKPOINT_SCHEMA_VERSION = 1

# --- Chốt chặn chi phí: ALLOWLIST, không phải blocklist -----------------------
# `kaggle b t push` chạy task MỘT lần trên MODEL MẶC ĐỊNH CỦA SERVER trước khi task
# dùng được — và ta không kiểm soát được model đó là gì. Full sweep = 72.000 lượt gọi,
# nên mặc định KHÔNG BAO GIỜ chạy full trừ khi model nằm trong allowlist dưới đây.
# Sai lầm dễ mắc: viết blocklist theo tên model default (Kaggle đổi default lúc nào
# không báo → nguyên một sweep bị đốt ngoài ý muốn).
#
# Muốn thêm model cho lần chạy thật: thêm slug vào đây (hoặc set PD_FULL_MODELS=
# "a,b" / PD_FULL=1), rồi `kaggle b t run -m <slug>`.
# Giữ danh sách này ĐÚNG BẰNG số model đang thực sự cần chạy. Model đã chạy xong thì
# bỏ ra — allowlist càng hẹp thì cú `push` (chạy trên model mặc định của server, ta
# không chọn được) càng khó vô tình đốt nguyên một sweep 12.000 lượt gọi.
FULL_SWEEP_MODELS = _env_list("PD_FULL_MODELS", [
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.6-flash",
])
_force_full = os.environ.get("PD_FULL", "").strip().lower() in {"1", "true", "yes", "on"}
_has_overrides = any(os.environ.get(k) for k in
                     ("PD_LAMBDAS", "PD_LANGS", "PD_REPS", "PD_ROUNDS"))

if not (_force_full or _has_overrides or MODEL in FULL_SWEEP_MODELS):
    LAMBDAS, LANGS, REPS, N_ROUNDS = [1.0], ["en"], 1, 5
    print(f"[guard] {MODEL} không nằm trong FULL_SWEEP_MODELS -> chạy SMOKE "
          f"(1 λ × 1 lang × 4 tổ hợp × 1 rep × 5 vòng = 40 lượt gọi). "
          f"Mở full bằng PD_FULL=1, PD_FULL_MODELS=..., hoặc thêm slug vào "
          f"FULL_SWEEP_MODELS trong file này.", flush=True)

# Proxy (nhất là model non-Gemini trên staging) thỉnh thoảng trả 429/503 → retry.
_TRANSIENT = ("429", "503", "500", "502", "504", "overloaded",
              "unavailable", "not reachable", "rate limit", "heavy load")
_AUTH_ERR = ("expired token", "authentication", "unauthorized", "401",
             "invalid api key", "invalid_api_key")
# Quota cũng trả 403 nên PHẢI tách khỏi _AUTH_ERR: reauth không tạo thêm credit, mà
# chỉ đốt thời gian rồi vẫn 403. Đây là lỗi CHẾT — dừng sớm, đổi API key rồi resume
# từ checkpoint, chứ đừng nghiến 6 lần retry cho từng lượt gọi trong 12.000 lượt.
_QUOTA_ERR = ("exceeds your available quota", "available quota", "quota",
              "insufficient", "billing", "exceeded your current")

PERSONALITY_PERMS = list(itertools.product(range(2), repeat=2))   # 4 tổ hợp, đúng
# thứ tự itertools.product của FAIRGAME._compute_agent_configurations.

_BLOCK_RE = re.compile(r"\{(\w+)\}:\s*\[(.*?)\]", re.DOTALL)


# %% =====================  PROMPT (port của PromptCreator)  =====================
def fmt_lambda(scale) -> str:
    """Format λ như layout dataset: 0.01, 0.1, 1, 10, 100, 1000 (không thừa .0)."""
    lam = float(scale)
    return str(int(lam)) if lam.is_integer() else str(lam)


def scaled_weights(lam) -> dict:
    """BASE_WEIGHTS × λ, khử nhiễu float, ép int khi nguyên (== notebook open-source)."""
    out = {}
    for k, v in BASE_WEIGHTS.items():
        s = round(float(v) * float(lam), 10)
        out[k] = int(s) if NORMALIZE_INTEGER_WEIGHTS and float(s).is_integer() else s
    return out


def assemble_prompt(language, agent_idx, personality, current_round, history, weights):
    """
    Dựng prompt cho một agent — port trung thực `PromptCreator.fill_template` ở
    điều kiện của thí nghiệm này (có personality, opponentPersonalityProb = 0,
    n_rounds_known theo cấu hình, phase='choose', không có khối communicate/choose
    trong template PD).

    `history` là chính dict history của FAIRGAME; `str.format` biến nó thành str(dict)
    y như bản gốc (round 1 → "{}").
    """
    template = TEMPLATES[language]
    me = AGENT_NAMES[agent_idx]
    opponent = AGENT_NAMES[1 - agent_idx]

    # process_intro / process_opponent_intro / process_game_length
    enabled = {
        "intro": personality != "None",
        # mọi opponent đều có prob = 0 → valid_opponents_exist = False → xoá khối
        "opponentIntro": False,
        "gameLength": bool(N_ROUNDS_KNOWN),
    }
    template = _BLOCK_RE.sub(
        lambda m: m.group(2) if enabled.get(m.group(1), False) else "", template)

    values = {
        "currentPlayerName": me,
        "currentRound": current_round,
        "history": history,
        "opponent1": opponent,
        "personality": personality,
        "nRounds": N_ROUNDS,
    }
    for i, key in enumerate(STRATEGIES, start=1):
        values[f"strategy{i}"] = STRATEGIES[key]
    for i, key in enumerate(weights, start=1):
        values[f"weight{i}"] = weights[key]
    return template.format(**values)


# %% =====================  PARSE (port của batch_runner)  =====================
def _normalize_strategy_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _map_choice_token(token, strategies):
    keys = list(strategies.keys())
    if not keys:
        return None
    if token in {"a", "1"}:
        return keys[0]
    if token in {"b", "2"}:
        return keys[1] if len(keys) > 1 else keys[0]
    return None


def match_strategy_key(response, strategies=None):
    """Copy nguyên logic `FAIRGAME/src/batch_runner._match_strategy_key`."""
    strategies = STRATEGIES if strategies is None else strategies
    if not response:
        return None
    compact = _normalize_strategy_text(response)
    if not compact:
        return None
    for key, val in strategies.items():
        if val and _normalize_strategy_text(val) in compact:
            return key
    for key in strategies:
        if _normalize_strategy_text(key) in compact:
            return key
    m = re.search(r"\boption\s*([ab]|[12])\b", response, flags=re.IGNORECASE)
    if m:
        return _map_choice_token(m.group(1).lower(), strategies)
    m = re.search(r"\b([ab]|[12])\b", response.strip(), flags=re.IGNORECASE)
    if m:
        return _map_choice_token(m.group(1).lower(), strategies)
    return None


def attribute_scores(strategy_keys, weights):
    """Copy `PayoffMatrix.attribute_scores`: (key agent1, key agent2) → (score1, score2)."""
    for combo_key, keys in COMBINATIONS.items():
        if keys == list(strategy_keys):
            return tuple(weights[wk] for wk in MATRIX[combo_key])
    raise ValueError(f"Combination not found: {strategy_keys}")


# %% =====================  SEED / GỌI MODEL  =====================
def cell_index(lang, perm_idx, rep):
    """Chỉ số ô thí nghiệm — CỐ Ý không chứa λ để mọi λ dùng chung CRN."""
    return (LANG_ORDER.index(lang) * len(PERSONALITY_PERMS) + perm_idx) * REPS + rep


def sampling_seed(cell, agent_idx, round_number):
    return ((BASE_SEED + cell) * SAMPLING_SEED_STRIDE
            + round_number * 100 + agent_idx) % SAMPLING_SEED_MOD


_LLM = None            # client hiện tại (task gán); _reauth() dựng lại tại chỗ
_LLM_LOCK = threading.Lock()


class QuotaExhausted(RuntimeError):
    """Hết credit — retry vô nghĩa, phải đổi API key rồi resume từ checkpoint."""


def _reauth():
    """Làm mới token proxy + dựng lại client để run dài không chết vì token hết hạn.

    KHÔNG BAO GIỜ được ném exception: trên server Kaggle không có binary `kaggle`
    trong PATH (FileNotFoundError) và bản thân hàm này chạy trong nhánh xử lý lỗi —
    nó mà chết là chết cả sweep, thổi bay tiến độ chưa checkpoint. Run-12 đứt đúng
    kiểu đó. Thất bại thì trả về lặng lẽ và để vòng retry bên ngoài quyết định.
    """
    global _LLM
    print("[auth] token bị từ chối -> làm mới auth + rebuild client ...", flush=True)
    try:
        subprocess.run(["kaggle", "b", "auth", "-y"], capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001 — không có CLI trên server là chuyện thường
        print(f"[auth] bỏ qua `kaggle b auth` ({exc.__class__.__name__}); "
              "rebuild client trực tiếp.", flush=True)
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception:
        pass
    # `kaggle b auth` ghi đè LLM_DEFAULT trong .env → ghim lại model đã chọn, nếu
    # không run dài sẽ âm thầm nhảy sang model mặc định của tài khoản.
    os.environ["LLM_DEFAULT"] = MODEL
    try:
        from kaggle_benchmarks.kaggle.models import load_default_model
        _LLM = load_default_model()
    except Exception as exc:  # noqa: BLE001
        print(f"[auth] rebuild client thất bại ({exc}); giữ client cũ.", flush=True)


def _call_llm(prompt, seed, max_attempts=6):
    """Một lượt sinh văn bản. Trả (text, usage). Tự reauth khi token hết hạn;
    exp-backoff với 429/503."""
    attempt = 0
    auth_retries = 0
    # Prompt yêu cầu "Output ONLY the choice" → 8 token là thừa cho "OptionA".
    # Cắt ngắn để không đốt tiền vào phần suy luận thừa của model.
    max_out = int(os.environ.get("PD_MAX_OUTPUT_TOKENS", "8"))
    while True:
        attempt += 1
        try:
            with kbench.chats.new("turn", orphan=True) as chat:
                text = _LLM.prompt(
                    prompt,
                    temperature=TEMPERATURE,
                    seed=seed,
                    extra_api_params={
                        # Không có tool nào cần gọi; tắt hẳn để model không trả
                        # tool_call rỗng (nguồn gốc lỗi parse của SDK bên dưới).
                        # ĐỪNG thêm "max_output_tokens" ở đây: endpoint là OpenAI
                        # chat.completions, nó ném TypeError "unexpected keyword
                        # argument" và giết cả task (run-13). `max_tokens` mới là
                        # tên đúng; proxy tự quy ra hạn mức chi phí từ nó.
                        "max_tokens": max_out,
                        "tool_choice": "none",
                    },
                )
            if text is not None:
                return text, chat.usage
            # Nhánh song song: contexts.enter có thể NUỐT lỗi proxy trong worker
            # thread (ContextVar của run không lan sang) → text=None. Ném transient
            # giả để backoff bên dưới retry.
            raise RuntimeError("503 no-text: proxy error swallowed under concurrency")
        except Exception as e:
            msg = str(e).lower()
            # Hết credit: kiểm TRƯỚC auth vì quota cũng là 403. Ném thẳng ra ngoài để
            # dừng sớm — checkpoint đã ghi tới đâu giữ tới đó, đổi key rồi chạy lại là
            # resume đúng chỗ.
            if any(t in msg for t in _QUOTA_ERR):
                raise QuotaExhausted(str(e)) from e
            if any(t in msg for t in _AUTH_ERR):
                auth_retries += 1
                if auth_retries > 6:
                    raise
                with _LLM_LOCK:
                    _reauth()
                attempt -= 1
                continue
            # Bug SDK: khi provider trả message không có `tool_calls`, lớp parse của
            # kaggle_benchmarks vấp NoneType. Không phải lỗi của ta và retry thường
            # qua được; hết lượt thì trả text rỗng để `decide()` xử theo đường
            # parse-fail bình thường, thay vì giết cả benchmark đang chạy dở.
            if "tool_calls" in msg and "nonetype" in msg:
                if attempt >= max_attempts:
                    print(
                        f"[warn] SDK tool-call parse bug ({attempt}/{max_attempts}); "
                        "trả content rỗng để tránh mất cả run.",
                        flush=True,
                    )

                    class _NoUsage:
                        input_tokens = 0
                        output_tokens = 0
                        total_cost_nanodollars = 0

                    return "", _NoUsage()
                time.sleep(min(2 ** attempt, 12))
                continue
            if attempt >= max_attempts or not any(t in msg for t in _TRANSIENT):
                raise
            time.sleep(min(2 ** attempt, 30))


def decide(prompt, base_seed):
    """Sinh văn bản + parse + retry-on-parse-fail rồi fallback — mirror batch_runner.
    Trả (text, strategy_key, parse_failed, fell_back, tok_in, tok_out, cost)."""
    tok_in = tok_out = cost = 0
    text, usage = _call_llm(prompt, base_seed)
    tok_in += usage.input_tokens or 0
    tok_out += usage.output_tokens or 0
    cost += usage.total_cost_nanodollars or 0
    key = match_strategy_key(text)
    parse_failed = int(key is None)

    attempt = 0
    while key is None and attempt < MAX_PARSE_RETRIES:
        attempt += 1
        seed = (base_seed + attempt * RETRY_SEED_STEP) % SAMPLING_SEED_MOD
        text, usage = _call_llm(prompt, seed)
        tok_in += usage.input_tokens or 0
        tok_out += usage.output_tokens or 0
        cost += usage.total_cost_nanodollars or 0
        key = match_strategy_key(text)
        parse_failed += int(key is None)

    fell_back = 0
    if key is None:
        key = FALLBACK_STRATEGY_KEY
        fell_back = 1
    return text, key, parse_failed, fell_back, tok_in, tok_out, cost


# %% =====================  MỘT GAME (2 agent, quyết định đồng thời)  =============
def play_game(lam, language, perm_idx, rep, model_tag, turns_sink):
    """Chạy trọn 1 game 30 vòng. Trả (row_csv, stats)."""
    weights = scaled_weights(lam)
    perm = PERSONALITY_PERMS[perm_idx]
    personalities = [PERSONALITIES[language][perm[0]], PERSONALITIES[language][perm[1]]]
    cell = cell_index(language, perm_idx, rep)
    lam_str = fmt_lambda(lam)
    game_id = f"pd__{model_tag}__x{lam_str}__{language}__p{perm_idx}__rep{rep}"

    history = {}                              # đúng cấu trúc GameHistory.rounds
    strategies = [[], []]
    scores = [[], []]
    stats = {"parse_failed": 0, "fell_back": 0, "tok_in": 0, "tok_out": 0, "cost": 0}

    for r in range(1, N_ROUNDS + 1):
        round_keys = []
        for a in range(2):
            prompt = assemble_prompt(language, a, personalities[a], r, history, weights)
            seed = sampling_seed(cell, a, r)
            text, key, pf, fb, ti, to, cn = decide(prompt, seed)
            stats["parse_failed"] += pf
            stats["fell_back"] += fb
            stats["tok_in"] += ti
            stats["tok_out"] += to
            stats["cost"] += cn
            round_keys.append(key)
            turns_sink.append({
                "game_id": game_id, "round": r, "agent": AGENT_NAMES[a],
                "personality": personalities[a], "strategy": STRATEGIES[key],
                "parse_failed": pf, "fell_back": fb, "raw_response": text,
                "prompt": prompt, "sampling_seed": seed, "language": language,
                "scale": float(lam), "rep": rep, "perm_idx": perm_idx,
            })

        round_scores = attribute_scores(round_keys, weights)
        history[f"round_{r}"] = {
            AGENT_NAMES[a]: {"strategy": STRATEGIES[round_keys[a]], "score": round_scores[a]}
            for a in range(2)
        }
        for a in range(2):
            strategies[a].append(STRATEGIES[round_keys[a]])
            scores[a].append(round_scores[a])

    # Schema CSV Y HỆT ResultsProcessor/GameData.to_dict (giá trị list ghi bằng str(list)
    # để `ast.literal_eval` trong Analysis/pdlib/ingest.py đọc được).
    row = {
        "game_id": game_id,
        "language": language,
        "n_rounds_is_known": bool(N_ROUNDS_KNOWN),
        "max_rounds": N_ROUNDS,
        "played_rounds": N_ROUNDS,
        "agents_communicate": AGENTS_COMMUNICATE,
    }
    for a in range(2):
        p = f"agent{a + 1}_"
        row[p + "name"] = AGENT_NAMES[a]
        row[p + "llm"] = model_tag
        row[p + "personality"] = personalities[a]
        row[p + "knows_opponent_with_prob"] = OPPONENT_PERSONALITY_PROB
        row[p + "strategies"] = str(strategies[a])
        row[p + "scores"] = str(scores[a])
        row[p + "messages"] = str([])
    return row, stats


# %% =====================  CHECKPOINT / RESUME / GHI FILE  =====================
_IO_LOCK = threading.Lock()


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _condition_key(lam, language, perm_idx, rep):
    return f"{float(lam):.12g}|{language}|{int(perm_idx)}|{int(rep)}"


def _checkpoint_filename(lam, language, perm_idx, rep):
    tag = fmt_lambda(lam).replace(".", "p")
    return f"x{tag}__lang-{language}__p{int(perm_idx)}__rep-{int(rep):03d}.json"


def _signature(model_tag):
    """Các trường phải khớp thì shard cũ mới resume được.

    CỐ Ý không đưa LAMBDAS/LANGS vào: chúng không ảnh hưởng seed (cell_index dùng
    LANG_ORDER cố định), nên shard sinh ra từ một lần smoke hẹp vẫn dùng lại được
    cho full sweep. REPS thì PHẢI có — nó nằm trong công thức cell_index.
    """
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION, "model": model_tag,
        "reps": REPS, "n_rounds": N_ROUNDS, "n_rounds_known": bool(N_ROUNDS_KNOWN),
        "temperature": TEMPERATURE, "base_weights": BASE_WEIGHTS,
        "normalize_integer_weights": NORMALIZE_INTEGER_WEIGHTS,
        "base_seed": BASE_SEED, "agents_communicate": AGENTS_COMMUNICATE,
    }


CSV_FIELDS = [
    "game_id", "language", "n_rounds_is_known", "max_rounds", "played_rounds",
    "agents_communicate",
    "agent1_name", "agent1_llm", "agent1_personality", "agent1_knows_opponent_with_prob",
    "agent1_strategies", "agent1_scores", "agent1_messages",
    "agent2_name", "agent2_llm", "agent2_personality", "agent2_knows_opponent_with_prob",
    "agent2_strategies", "agent2_scores", "agent2_messages",
]


def _write_csv(path, rows):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow({k: row[k] for k in CSV_FIELDS})
    _atomic_write_text(path, buf.getvalue())


def _materialize(out_dir, model_tag, games, turns):
    """Ghi lại toàn bộ CSV theo layout <λ>/<model>/x<λ>_<lang>_<model>.csv + turns."""
    out_dir = Path(out_dir)
    buckets = {}
    for g in games:
        buckets.setdefault((g["_lambda_str"], g["language"]), []).append(g)
    for (lam_str, lang), rows in buckets.items():
        d = out_dir / lam_str / model_tag
        # game_id → game_0..game_N trong TỪNG file, giống đầu ra ResultsProcessor.
        numbered = []
        for i, r in enumerate(rows):
            r2 = dict(r)
            r2["game_id"] = f"game_{i}"
            numbered.append(r2)
        _write_csv(d / f"x{lam_str}_{lang}_{model_tag}.csv", numbered)
    _atomic_write_text(out_dir / "turns.jsonl",
                       "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in turns))


def _save_checkpoint(ckpt_dir, signature, lam, language, perm_idx, rep, row, turns, stats):
    payload = {
        "signature": signature,
        "condition_key": _condition_key(lam, language, perm_idx, rep),
        "game": row, "turns": turns, "stats": stats,
    }
    path = Path(ckpt_dir) / _checkpoint_filename(lam, language, perm_idx, rep)
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def _sweep_order():
    for lam in LAMBDAS:
        for language in LANGS:
            for perm_idx in range(len(PERSONALITY_PERMS)):
                for rep in range(REPS):
                    yield lam, language, perm_idx, rep


def _load_checkpoints(ckpt_dir, signature):
    """Chỉ nạp shard TRỌN VẸN & tương thích. Trả (rows_by_key, turns_by_key, prior)."""
    ckpt_dir = Path(ckpt_dir)
    expected = {_condition_key(*c) for c in _sweep_order()}
    rows, turns = {}, {}
    prior = {"parse_failed": 0, "fell_back": 0, "tok_in": 0, "tok_out": 0, "cost": 0}
    if ckpt_dir.is_dir():
        for path in sorted(ckpt_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                key = record["condition_key"]
                if record.get("signature") != signature:
                    print(f"[checkpoint] bỏ shard không khớp signature: {path.name}", flush=True)
                    continue
                if key not in expected:
                    continue          # thuộc sweep khác — để nguyên, không nạp
                if len(record.get("turns", [])) != N_ROUNDS * 2:
                    raise ValueError("thiếu lượt")
                rows[key] = record["game"]
                turns[key] = record["turns"]
                for k in prior:
                    prior[k] += record["stats"].get(k, 0)
            except Exception as exc:  # noqa: BLE001
                print(f"[checkpoint] shard hỏng {path.name}: {exc}", flush=True)
    return rows, turns, prior


# %% =====================  TASK  =====================
@kbench.task(
    name="prisoner-dilemma-fairgame",
    # Mô tả để ASCII: field này đi qua nhiều lớp metadata của Kaggle, từng bị
    # mangle encoding một lần rồi.
    # Kaggle chặn description > 255 ký tự (VALIDATION_FAILED ở run-14) — giữ ngắn.
    description="FAIRGAME iterated Prisoner's Dilemma (API arm): conventional payoff "
                "(T=10 R=6 P=2 S=0) over payoff scales x 5 languages x 4 personality "
                "pairings. Reports cooperation rate and cost/game; CSV layout matches "
                "the open-source arm.",
)
def prisoner_dilemma_fairgame(llm) -> dict:
    global _LLM
    _LLM = llm

    model_tag = os.environ.get("PD_MODEL_TAG") or re.sub(
        r"[^A-Za-z0-9._-]+", "-", MODEL.split("/")[-1])
    out_dir = Path(os.environ.get("PD_OUT", f"results/kbench/{model_tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    signature = _signature(model_tag)

    total = len(LAMBDAS) * len(LANGS) * len(PERSONALITY_PERMS) * REPS
    print(f"[plan] model={MODEL} tag={model_tag}")
    print(f"[plan] Output target: {out_dir}/<lambda>/<model>/x<lambda>_<lang>_<model>.csv "
          f"+ {ckpt_dir}/*.json (resume)")
    print(f"[plan] {len(LAMBDAS)} λ × {len(LANGS)} lang × {len(PERSONALITY_PERMS)} tổ hợp "
          f"× {REPS} rep = {total} game × {N_ROUNDS} vòng × 2 agent = "
          f"{total * N_ROUNDS * 2} lượt gọi model (concurrency={CONCURRENCY}).")

    if RESUME:
        rows_by_key, turns_by_key, agg = _load_checkpoints(ckpt_dir, signature)
    else:
        rows_by_key, turns_by_key = {}, {}
        agg = {"parse_failed": 0, "fell_back": 0, "tok_in": 0, "tok_out": 0, "cost": 0}
    resumed = len(rows_by_key)
    if resumed:
        print(f"[resume] khôi phục {resumed}/{total} game từ {ckpt_dir}", flush=True)

    def _ordered():
        """(rows, turns) theo đúng thứ tự sweep — không phụ thuộc thứ tự hoàn thành."""
        keys = [k for k in (_condition_key(*c) for c in _sweep_order()) if k in rows_by_key]
        return ([rows_by_key[k] for k in keys],
                [t for k in keys for t in turns_by_key[k]])

    pending = [c for c in _sweep_order() if _condition_key(*c) not in rows_by_key]
    t0 = time.time()
    done = resumed

    def _run_one(cell):
        lam, language, perm_idx, rep = cell
        turns_sink = []
        row, stats = play_game(lam, language, perm_idx, rep, model_tag, turns_sink)
        row["_lambda_str"] = fmt_lambda(lam)
        return cell, row, turns_sink, stats

    def _commit(cell, row, turns_sink, stats):
        """Chạy trong thread chính → an toàn, và CSV luôn khớp shard trên đĩa."""
        nonlocal done
        lam, language, perm_idx, rep = cell
        key = _condition_key(*cell)
        _save_checkpoint(ckpt_dir, signature, lam, language, perm_idx, rep,
                         row, turns_sink, stats)
        rows_by_key[key] = row
        turns_by_key[key] = turns_sink
        for k in agg:
            agg[k] += stats.get(k, 0)
        _materialize(out_dir, model_tag, *_ordered())
        done += 1
        coop = sum(s == "OptionA" for s in ast.literal_eval(row["agent1_strategies"])
                   + ast.literal_eval(row["agent2_strategies"]))
        elapsed = time.time() - t0
        pct = (done / total) * 100 if total else 100.0
        n_calls_done = done * N_ROUNDS * 2
        calls_per_sec = n_calls_done / elapsed if elapsed > 0 else 0
        eta_sec = ((total - done) * N_ROUNDS * 2 / calls_per_sec) if calls_per_sec > 0 else None
        eta_txt = f"{int(eta_sec)}s" if eta_sec is not None else "n/a"
        print(f"[{done}/{total} | {pct:.1f}%] {row['game_id']}  coop={coop}/{N_ROUNDS * 2} "
              f"parse_fail={stats['parse_failed']} fallback={stats['fell_back']} eta={eta_txt}",
              flush=True)

    # Hết credit giữa chừng KHÔNG được ném traceback ra ngoài: làm vậy là mất luôn
    # bảng summary và mất luôn CSV của mấy trăm game đã chạy xong. Thay vào đó dừng
    # nhận việc mới, ghi trọn những gì đã có, rồi báo cáo run dở dang.
    quota_hit = None
    if CONCURRENCY <= 1:
        for cell in pending:
            try:
                _commit(*_run_one(cell))
            except QuotaExhausted as exc:
                quota_hit = str(exc)
                break
    else:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            # copy_context() để ContextVar active-run của SDK lan vào thread, nếu không
            # contexts.enter nuốt lỗi proxy và _call_llm không retry được. Duyệt futures
            # theo THỨ TỰ SUBMIT → commit/ghi file vẫn tuần tự và tất định.
            futs = [ex.submit(contextvars.copy_context().run, _run_one, c) for c in pending]
            try:
                for fut in as_completed(futs):
                    try:
                        _commit(*fut.result())
                    except QuotaExhausted as exc:
                        quota_hit = str(exc)
                        break
            finally:
                # Huỷ phần chưa khởi động; game đang chạy dở cứ để nó kết thúc, shard
                # nào ghi xong thì lần sau resume dùng lại được.
                for f in futs:
                    f.cancel()

    if quota_hit:
        print(f"\n[quota] DỪNG SỚM — hết credit: {quota_hit}", flush=True)
        print(f"[quota] Đã giữ {len(rows_by_key)}/{total} game trong {ckpt_dir}. "
              "Đổi sang API key khác rồi chạy lại đúng lệnh này để resume.", flush=True)

    final_games, final_turns = _ordered()
    _materialize(out_dir, model_tag, final_games, final_turns)

    # Chia theo số lượt ĐÃ CHẠY THẬT, không phải số lượt dự kiến — nếu không, một run
    # dừng sớm vì hết quota sẽ báo parse_fail_rate/fallback_rate thấp giả tạo.
    n_decisions = len(final_games) * N_ROUNDS * 2

    def coop_rate(pred):
        cells = [g for g in final_games if pred(g)]
        if not cells:
            return None
        picks = [s for g in cells
                 for s in ast.literal_eval(g["agent1_strategies"])
                 + ast.literal_eval(g["agent2_strategies"])]
        return round(sum(p == "OptionA" for p in picks) / len(picks), 3)

    result = {
        "model": model_tag,
        "n_games": len(final_games),
        "n_games_planned": total,
        "complete": len(final_games) == total,
        "stopped_early_quota": quota_hit,
        "n_decisions": n_decisions,
        "parse_fail_rate": round(agg["parse_failed"] / n_decisions, 4) if n_decisions else None,
        "fallback_rate": round(agg["fell_back"] / n_decisions, 4) if n_decisions else None,
        "overall_coop_rate": coop_rate(lambda g: True),
        "coop_by_scale": {fmt_lambda(l): coop_rate(lambda g, l=l: g["_lambda_str"] == fmt_lambda(l))
                          for l in LAMBDAS},
        "coop_by_lang": {l: coop_rate(lambda g, l=l: g["language"] == l) for l in LANGS},
        "usage_input_tokens": agg["tok_in"],
        "usage_output_tokens": agg["tok_out"],
        "usage_total_cost_usd": round(agg["cost"] / 1e9, 6),
        "games_per_10usd": int(10 / (agg["cost"] / 1e9)) if agg["cost"] else None,
        "elapsed_sec": round(time.time() - t0, 1),
        "out_dir": str(out_dir),
        "checkpoint_dir": str(ckpt_dir),
        "resumed_games": resumed,
        "new_games": len(final_games) - resumed,
    }

    print("\n===== FAIRGAME PRISONER'S DILEMMA (API arm) — SUMMARY =====")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("===========================================================\n")

    # Health check (tỉ lệ hợp tác tự nó là KẾT QUẢ, không phải assertion): pipeline chỉ
    # hợp lệ khi hầu như không quyết định nào phải fallback OptionA. Bài học run-1: xem
    # parse/fallback rate TRƯỚC khi diễn giải số liệu.
    #
    # CỐ Ý là ngưỡng chứ không phải == 0: một sweep 12.000 lượt gọi không được phép
    # bị đánh trượt vì đúng một câu trả lời lạ, nhưng fallback có hệ thống (model
    # thinking nuốt hết max_tokens) thì phải bật đèn đỏ vì nó bẻ cong tỉ lệ hợp tác.
    fallback_rate = result["fallback_rate"] or 0.0
    kbench.assertions.assert_true(
        fallback_rate <= FALLBACK_RATE_TOLERANCE,
        expectation=f"Tỉ lệ fallback OptionA {fallback_rate:.4f} <= "
                    f"{FALLBACK_RATE_TOLERANCE} (parse được câu trả lời của model)")
    return result


# %%
# Server Kaggle chạy file này như module (__name__ != "__main__") nên phải tự chạy.
# PD_SKIP_RUN=1 chỉ dùng cho unit test import-time (test_pd_task_parity.py).
if os.environ.get("PD_SKIP_RUN") != "1":
    prisoner_dilemma_fairgame.run(kbench.llm)
