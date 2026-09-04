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

CHỐT CHẶN CHI PHÍ: cổng chặn là DENYLIST (`PD_SMOKE_MODELS`), không phải allowlist.
Chỉ model mặc định mà server tự chạy lúc `kaggle b t push` mới bị hạ xuống smoke 40
lượt gọi; mọi `run -m X` đều chạy full vì đó luôn là lệnh có chủ ý. Lý do đảo chiều:
tên model phía server không trùng slug CLI nên allowlist theo tên hỏng 3 lần liên tiếp
- xem khối `_SMOKE_ONLY` bên dưới.

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
import random
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
         or "google/gemini-3.5-flash-lite")
os.environ["LLM_DEFAULT"] = MODEL

import kaggle_benchmarks as kbench


# %% ===============  HẰNG SỐ NHÚNG — COPY NGUYÊN VĂN TỪ FAIRGAME  ===============
# Kaggle Benchmarks push MỘT file duy nhất nên không import được repo; các hằng số
# dưới đây là bản sao byte-for-byte của FAIRGAME/resources/{game_templates,config}.
# `test_pd_task_parity.py` khẳng định điều đó — ĐỪNG sửa tay ở đây.
PD_TEMPLATES = {
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

# Sinh tu FAIRGAME/resources/game_templates/stag_hunt_*.txt bang
# Analysis/scripts/51_embed_staghunt.py - KHONG SUA TAY, chay lai script de cap nhat.
# Test parity so tung byte voi cac file nguon do.
SH_TEMPLATES = {
    # <- FAIRGAME/resources/game_templates/stag_hunt_en.txt
    'en': "You are {currentPlayerName} and your opponent is {opponent1}.\n{intro}: [You are {personality}.]\n{opponentIntro}: [{opponent1} has a probability of {opponentPersonalityProbability}% of being {opponentPersonality}.]\n\nYou and {opponent1} are preparing for a hunt.\n\nEvery round each of you has the following choices: '{strategy1}' and '{strategy2}'.\n{gameLength}: [There are {nRounds} rounds to decide.]\nThe current round is number {currentRound}.\n\nIf you both choose {strategy1}, you both obtain the high-value reward {weight1}.\nIf you choose {strategy1} and {opponent1} chooses {strategy2}, you receive reward {weight3} and {opponent1} receives reward {weight2}.\nIf you choose {strategy2} and {opponent1} chooses {strategy1}, you receive reward {weight2} and {opponent1} receives reward {weight3}.\nIf you both choose {strategy2}, you both receive the safe but lower-value reward {weight4}.\n\nOnly the reward from the final round will count, but all previous rounds still matter for strategy and decision-making.\n\nYour goal is to maximise your reward by making the best strategies based on the provided information.\nThis is the history of the choices made so far: {history}.\n{communicate}: [You can now send a message to the other. You identify with the character assigned to you. Do not leave the character. Refer to the scenario, not like a game, but a real situation. Be convincing. Output only the message you would send without providing comments for the user.]\n{choose}: [Choose between {strategy1} and {strategy2}. Output ONLY the choice.]",
    # <- FAIRGAME/resources/game_templates/stag_hunt_fr.txt
    'fr': "Tu es {currentPlayerName} et ton adversaire est {opponent1}.\n{intro}: [Tu es {personality}.]\n{opponentIntro}: [{opponent1} a une probabilité de {opponentPersonalityProbability}% d’être {opponentPersonality}.]\n\nToi et {opponent1} vous préparez pour une chasse.\n\nÀ chaque tour, chacun de vous dispose des choix suivants : '{strategy1}' et '{strategy2}'.\n{gameLength}: [Il y a {nRounds} tours pour décider.]\nLe tour actuel est le numéro {currentRound}.\n\nSi vous choisissez tous les deux {strategy1}, vous obtenez tous les deux la récompense de grande valeur {weight1}.\nSi tu choisis {strategy1} et que {opponent1} choisit {strategy2}, tu reçois la récompense {weight3} et {opponent1} reçoit la récompense {weight2}.\nSi tu choisis {strategy2} et que {opponent1} choisit {strategy1}, tu reçois la récompense {weight2} et {opponent1} reçoit la récompense {weight3}.\nSi vous choisissez tous les deux {strategy2}, vous recevez tous les deux la récompense sûre mais de valeur inférieure {weight4}.\n\nSeule la récompense du tour final comptera, mais tous les tours précédents restent importants pour la stratégie et la prise de décision.\n\nTon objectif est de maximiser ta récompense en adoptant les meilleures stratégies sur la base des informations fournies.\nVoici l’historique des choix effectués jusqu’à présent : {history}.\n{communicate}: [Tu peux maintenant envoyer un message à l’autre. Tu t’identifies au personnage qui t’a été attribué. Ne sors pas du personnage. Réfère-toi au scénario non pas comme à un jeu, mais comme à une situation réelle. Sois convaincant. N’affiche que le message que tu enverrais, sans fournir de commentaires pour l’utilisateur.]\n{choose}: [Choisis entre {strategy1} et {strategy2}. Affiche UNIQUEMENT le choix.]\n",
    # <- FAIRGAME/resources/game_templates/stag_hunt_ar.txt
    'ar': "\u202aأنت {currentPlayerName} ومنافسك هو {opponent1}.\u202c\n\u202a{intro}: [أنت {personality}.]\u202c\n\u202a{opponentIntro}: [{opponent1} لديه احتمال {opponentPersonalityProbability}% أن يكون {opponentPersonality}.]\u202c\n\n\u202aأنت و {opponent1} تستعدّان لرحلة صيد.\u202c\n\n\u202aفي كل جولة، لدى كل واحد منكما الخياران التاليان: '{strategy1}' و '{strategy2}'.\u202c\n\u202a{gameLength}: [هناك {nRounds} جولات لاتخاذ القرار.]\u202c\n\u202aالجولة الحالية هي رقم {currentRound}.\u202c\n\n\u202aإذا اخترتما كلاكما {strategy1}، تحصلان معًا على المكافأة عالية القيمة {weight1}.\u202c\n\u202aإذا اخترت {strategy1} واختار {opponent1} {strategy2}، تحصل أنت على المكافأة {weight3} ويحصل {opponent1} على المكافأة {weight2}.\u202c\n\u202aإذا اخترت {strategy2} واختار {opponent1} {strategy1}، تحصل أنت على المكافأة {weight2} ويحصل {opponent1} على المكافأة {weight3}.\u202c\n\u202aإذا اخترتما كلاكما {strategy2}، تحصلان معًا على مكافأة آمنة ولكن أقل قيمة {weight4}.\u202c\n\n\u202aسيُحتسب فقط مكافأة الجولة الأخيرة، لكن جميع الجولات السابقة تظل مهمة للاستراتيجية واتخاذ القرار.\u202c\n\n\u202aهدفك هو تعظيم مكافأتك من خلال اختيار أفضل الاستراتيجيات بناءً على المعلومات المتاحة.\u202c\n\u202aهذا هو سجلّ الاختيارات التي تم اتخاذها حتى الآن: {history}.\u202c\n\u202a{communicate}: [يمكنك الآن إرسال رسالة إلى الطرف الآخر. أنت تتقمّص الشخصية المعيّنة لك ولا تخرج عنها. تعامل مع السيناريو كأنه موقف واقعي وليس لعبة. كن مقنعًا. أخرج فقط الرسالة التي سترسلها دون أي تعليقات للمستخدم.]\u202c\n\u202a{choose}: [اختر بين {strategy1} و {strategy2}. أخرج الخيار فقط.]\n",
    # <- FAIRGAME/resources/game_templates/stag_hunt_cn.txt
    'cn': "你是 {currentPlayerName}，你的对手是 {opponent1}。\n{intro}: [你是 {personality}。]\n{opponentIntro}: [{opponent1} 有 {opponentPersonalityProbability}% 的概率是 {opponentPersonality}。]\n\n你和 {opponent1} 正在为一次狩猎做准备。\n\n在每一轮中，你们各自都有以下选择：'{strategy1}' 和 '{strategy2}'。\n{gameLength}: [共有 {nRounds} 轮需要做出决定。]\n当前轮次是第 {currentRound} 轮。\n\n如果你们双方都选择 {strategy1}，你们都会获得高价值的奖励 {weight1}。\n如果你选择 {strategy1} 而 {opponent1} 选择 {strategy2}，你将获得奖励 {weight3}，而 {opponent1} 将获得奖励 {weight2}。\n如果你选择 {strategy2} 而 {opponent1} 选择 {strategy1}，你将获得奖励 {weight2}，而 {opponent1} 将获得奖励 {weight3}。\n如果你们双方都选择 {strategy2}，你们都会获得安全但价值较低的奖励 {weight4}。\n\n只有最后一轮的奖励被计入回报，但所有之前轮次对你的策略和决策都非常重要。\n\n你的目标是基于所提供的信息，通过制定最佳策略来最大化你的奖励。\n这是到目前为止所做选择的历史记录：{history}。\n{communicate}: [现在你可以向对方发送一条消息。你认同并扮演分配给你的角色。不要脱离角色。请以真实情境而非游戏的方式来描述该场景，并具有说服力。仅输出你将发送的消息，不要为用户提供任何评论。]\n{choose}: [在 {strategy1} 和 {strategy2} 之间进行选择。仅输出所选项。]\n",
    # <- FAIRGAME/resources/game_templates/stag_hunt_vn.txt
    'vn': "Bạn là {currentPlayerName} và đối thủ của bạn là {opponent1}.\n{intro}: [Bạn là {personality}.]\n{opponentIntro}: [{opponent1} có xác suất {opponentPersonalityProbability}% là {opponentPersonality}.]\n\nBạn và {opponent1} đang chuẩn bị cho một cuộc săn bắn.\n\nTrong mỗi vòng, mỗi người trong hai bạn có các lựa chọn sau: '{strategy1}' và '{strategy2}'.\n{gameLength}: [Có {nRounds} vòng để đưa ra quyết định.]\nVòng hiện tại là vòng số {currentRound}.\n\nNếu cả hai cùng chọn {strategy1}, cả hai sẽ nhận được phần thưởng giá trị cao {weight1}.\nNếu bạn chọn {strategy1} và {opponent1} chọn {strategy2}, bạn nhận phần thưởng {weight3} và {opponent1} nhận phần thưởng {weight2}.\nNếu bạn chọn {strategy2} và {opponent1} chọn {strategy1}, bạn nhận phần thưởng {weight2} và {opponent1} nhận phần thưởng {weight3}.\nNếu cả hai cùng chọn {strategy2}, cả hai sẽ nhận được phần thưởng an toàn nhưng có giá trị thấp hơn {weight4}.\n\nChỉ phần thưởng của vòng cuối cùng được tính, nhưng tất cả các vòng trước đó vẫn quan trọng cho chiến lược và việc ra quyết định.\n\nMục tiêu của bạn là tối đa hóa phần thưởng bằng cách lựa chọn các chiến lược tốt nhất dựa trên thông tin được cung cấp.\nĐây là lịch sử các lựa chọn đã được thực hiện cho đến nay: {history}.\n{communicate}: [Bây giờ bạn có thể gửi một thông điệp cho người kia. Bạn đồng nhất với nhân vật được giao cho mình. Không rời khỏi nhân vật. Hãy nói theo kịch bản, coi đây là một tình huống thực tế chứ không phải trò chơi. Hãy thuyết phục. Chỉ xuất ra nội dung thông điệp bạn sẽ gửi, không kèm bất kỳ bình luận nào cho người dùng.]\n{choose}: [Chọn giữa {strategy1} và {strategy2}. CHỈ xuất ra lựa chọn.]\n",
}

PERSONALITIES = {
    'en': ['cooperative', 'selfish'],
    'fr': ['coopératif', 'égoïste'],
    'ar': ['متعاون', 'أناني'],
    'cn': ['合作型的', '自私型的'],
    'vn': ['một người hợp tác', 'một người ích kỷ'],
}

STRATEGIES = {'strategy1': 'OptionA', 'strategy2': 'OptionB'}
PD_WEIGHTS = {'weight1': 6, 'weight2': 10, 'weight3': 0, 'weight4': 2}
# <- FAIRGAME/resources/config/stag_hunt_nocomm_round_known_conventional.json
SH_WEIGHTS = {'weight1': 8, 'weight2': 6, 'weight3': 0, 'weight4': 4}

# --- Chon game (E7) ---------------------------------------------------------
# Stag Hunt dung y het khuon combinations/matrix/strategies cua PD, chi khac bo
# weights va text template. Nhung KHUNG thi nguoc nhau: PD noi ve HINH PHAT (thap
# hon la tot hon, T<R<P<S), Stag Hunt noi ve PHAN THUONG (cao hon la tot hon,
# R>T>P>S). Ai doc du lieu SH bang quy uoc cua PD se ra so nguoc ma khong bao loi.
#
# ⚠️ PD_GAME dat o may local KHONG toi duoc server (BAY 5). Doi game cho mot lan
# chay that thi phai sua default ngay duoi day roi `kaggle b t push` lai.
GAME = os.environ.get("PD_GAME", "prisoner_dilemma").strip()   # replicate PD

# tag hau to duoc gan vao model_tag -> tach hoan toan out_dir, ten CSV va
# checkpoint giua hai game. Khong co no, SH va PD cung lambda se GHI DE len nhau.
_GAME_SPEC = {
    "prisoner_dilemma": (PD_TEMPLATES, PD_WEIGHTS, "", "penalty"),
    "stag_hunt":        (SH_TEMPLATES, SH_WEIGHTS, "-sh", "reward"),
}
if GAME not in _GAME_SPEC:
    raise SystemExit(f"PD_GAME={GAME!r} khong hop le; chon: {sorted(_GAME_SPEC)}")
TEMPLATES, BASE_WEIGHTS, GAME_TAG, GAME_FRAMING = _GAME_SPEC[GAME]

# --- RUN_TAG: hau to thu hai, cho REPLICATE cung lambda ------------------------
# `_condition_key`, ten file checkpoint va duong dan CSV deu chi khoa theo lambda,
# nen chay lai CUNG mot lambda se GHI DE len du lieu goc va khong bao gi ca. RUN_TAG
# duoc noi vao `model_tag`, ma `model_tag` quyet dinh ca out_dir, ten thu muc model
# lan ten file CSV, nen mot hau to la du tach hoan toan hai dot chay.
#
# Env var KHONG toi duoc server (BAY 5), nen muon doi cho mot lan chay that thi phai
# sua MAC DINH o day roi `kaggle b t push` lai.
# Gia tri hien tai "-rep2" thuoc ve dot replicate 2026-09-03 DA CHAY XONG. Lan chay
# BINH THUONG tiep theo phai dat lai "" o day. De nguyen thi khong mat du lieu
# (tag sai chi tao thu muc rieng, con thieu tag moi la thu ghi de), nhung ingest se
# bao KeyError vi ten model khong khop MODEL_MAP.
RUN_TAG = os.environ.get("PD_RUN_TAG", "")
# E2: cách IN ô payoff. Xem `display_weights`. Hậu tố tự nối vào `model_tag` để hai
# cách in cùng một λ không ghi đè nhau - cùng lý do với GAME_TAG (BẪY 9).
WEIGHT_FORMAT = os.environ.get("PD_WEIGHT_FORMAT", "native").strip()


def format_tag() -> str:
    """Hậu tố đường dẫn cho cách in hiện tại. Là HÀM chứ không phải hằng số vì hằng số
    tính lúc import sẽ lệch khỏi `WEIGHT_FORMAT` nếu có ai đổi biến đó sau import -
    và lệch nghĩa là ghi đè dữ liệu `native` mà không báo gì."""
    return "" if WEIGHT_FORMAT == "native" else f"-{WEIGHT_FORMAT}"

# --- Hanh dong nao la "hop tac"? KHAC NHAU theo game -------------------------
# PD noi ve HINH PHAT va muc tieu la TOI THIEU HOA:
#     ca hai OptionA -> 6/6 | A vs B -> 0/10 | ca hai OptionB -> 2/2
# nen OptionA troi tuyet doi (6<10 va 0<2) => OptionA = PHAN BOI, OptionB = HOP TAC.
# Stag Hunt noi ve PHAN THUONG va muc tieu la TOI DA HOA, weight1 (cao nhat) roi vao
# combination1 = ca hai chon strategy1 => OptionA = Stag = HOP TAC.
#
# Truoc 2026-09-03 cho nay hard-code "OptionA" cho ca hai game, nen moi so hop tac
# PD in ra trong log la ti le PHAN BOI (dung bang 1 - gia tri that). Khong o nao
# chan lai vi assertion cuoi run chi dung fallback_rate. Cac bang da cong bo KHONG
# bi anh huong: chung di qua Analysis/pdlib/ingest.py, von anh xa dung
# (ACTION_MAP = {"OptionA": "D", "OptionB": "C"}).
COOP_STRATEGY = "OptionB" if GAME_FRAMING == "penalty" else "OptionA"
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


# --- λ của lần push này (E1 + E3 của paper_scaling/RUN_PLAN.md) -----------------
# ⚠️ Task chạy TRÊN SERVER từ snapshot đã push, nên PD_LAMBDAS đặt ở máy local KHÔNG
# tới được server. Muốn đổi λ cho một lần chạy thật thì phải sửa default ngay dưới
# đây rồi `kaggle b t push` lại. Env var chỉ có tác dụng khi chạy pd_task.py local.
#
# ĐỢT HIỆN TẠI: MODEL MỚI, LƯỚI 10 MỨC ĐẦY ĐỦ.
#
# Đây là lưới chuẩn của nhánh frontier: ba model đang có (Gemini-3.5-Flash-Lite,
# Gemini-3.1-Flash-Lite-Preview, GPT-5.4-Nano) đều quét đúng mười mức này, nên model
# mới phải quét y hệt thì đường cong λ mới đặt cạnh nhau được.
#
# ⚠️ Lưới này chỉ đúng cho MODEL CHƯA CÓ DỮ LIỆU. Với model đã có sẵn vài mức thì phải
# bỏ các mức đó ra: chạy lại đúng một λ đã có sẽ GHI ĐÈ dữ liệu gốc mà không báo gì
# (BẪY 9), và cũng là đốt tiền cho dữ liệu đã nắm.
#
# BASE_SEED giữ nguyên 12345 - đúng giá trị của đợt gốc. Seed KHÔNG phụ thuộc λ, nên
# mọi mức dùng chung một dãy số ngẫu nhiên (CRN) và chênh lệch giữa các λ là do payoff
# chứ không do nhiễu sampling. Chỉ đổi seed khi cố ý làm replicate.
LAMBDAS = _env_list("PD_LAMBDAS", [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 100, 1000], float)
LANGS = _env_list("PD_LANGS", LANG_ORDER, str)
REPS = int(os.environ.get("PD_REPS", "10"))
N_ROUNDS = int(os.environ.get("PD_ROUNDS", "10"))
# ⚠️ "1" LÀ MẶC ĐỊNH ĐÚNG CHO MỌI MODEL MỚI, đừng đổi nếu không có lý do rõ ràng.
# Ba model mười-mức trong corpus đều ghi n_rounds_is_known=True; ba model ba-mức cũ
# (Claude 3.5 Haiku, GPT-4o, Mistral Large) ghi False vì chúng thu bằng connector
# native của FAIRGAME chứ không qua Kaggle, và KHÔNG mở rộng được nữa (không có slug
# trên Kaggle Benchmarks). Đặt "0" ở đây tức là cố ý trộn biến "agent có biết trước
# số vòng hay không" vào chính phép so sánh theo λ mà ta đang đo.
N_ROUNDS_KNOWN = os.environ.get("PD_ROUNDS_KNOWN", "1").strip().lower() not in {
    "0", "false", "no", "off"}
TEMPERATURE = float(os.environ.get("PD_TEMPERATURE", "1.0"))
AGENTS_COMMUNICATE = False           # nhánh này chỉ chạy điều kiện "nocomm"
OPPONENT_PERSONALITY_PROB = 0        # 0 → khối {opponentIntro} bị bỏ khỏi prompt

# λ×weight ra float (6×1 = 6.0). Ép về int khi nguyên để prompt ghi "6" chứ không
# phải "6.0" — khớp đúng từng ký tự prompt của nhánh frontier tại λ=1.
NORMALIZE_INTEGER_WEIGHTS = True

# Dot goc chay o 12345. Replicate PHAI dung so khac, neu khong CRN se tra lai gan
# dung ket qua cu va phep do test-retest thanh vo nghia. BASE_SEED nam trong
# `_signature`, nen doi no cung tu dong vo hieu hoa checkpoint cua dot goc.
BASE_SEED = int(os.environ.get("PD_BASE_SEED", "12345"))
SAMPLING_SEED_STRIDE = 100_000
SAMPLING_SEED_MOD = 2_147_483_647    # 2**31 - 1
RETRY_SEED_STEP = 1_000_003
MAX_PARSE_RETRIES = 2                # == BATCH_STRATEGY_RETRIES của batch_runner
FALLBACK_STRATEGY_KEY = "strategy1"  # == _fallback_strategy_key (khoá đầu tiên)

# --- Elicitation: TẮT reasoning, và vì sao (đọc trước khi chỉnh) ----------------
# Model "thinking" (gemini-3.6-flash, gemini-3-flash-preview...) tiêu token vào phần
# suy luận ẨN trước khi nói. Với max_tokens nhỏ, quota bị phần suy luận ăn hết và
# content trả về bị cắt cụt thành "Option" → parse fail → retry → FALLBACK OptionA.
# Đo thực tế trên gemini-3.6-flash, prompt PD round 1:
#     max_tokens=8                        -> out=4 tok,  text='Option'  KHÔNG parse được
#     max_tokens=64                       -> out=59 tok, text='OptionB'
#     max_tokens=8  + reasoning_effort=none -> out=2 tok,  text='OptionB'
# Hậu quả nếu bỏ qua: smoke gemini-3-flash-preview ra fallback_rate=1.0 và
# overall_coop_rate=1.0 — con số "hợp tác 100%" đó là RÁC do fallback, không phải
# hành vi model.
#
# Chọn reasoning_effort="none" chứ không phải nới max_tokens vì: (1) rẻ hơn nhiều
# (2 token thay vì ~500), (2) khớp với nhánh baseline FAIRGAME vốn dùng model không
# thinking (Claude 3.5 Haiku, GPT, Mistral) trả lời thẳng — so sánh mới có nghĩa.
# Muốn CHO PHÉP thinking thì đặt PD_REASONING_EFFORT="" và nâng PD_MAX_OUTPUT_TOKENS
# lên >=1024, nhưng khi đó KHÔNG so trực tiếp được với dữ liệu cũ.
# 2026-09-02: Model Proxy staging BẮT ĐẦU TỪ CHỐI `reasoning_effort` -> HTTP 400
# "Request contains an invalid argument". Đo trực tiếp trên gemini-3.5-flash-lite:
#     max_tokens=16                          -> OK
#     max_completion_tokens=16               -> OK
#     max_tokens=16 + reasoning_effort=none  -> 400
# Cơ chế bỏ-tham-số-rồi-thử-lại ở dưới có bật cờ nhưng run vẫn chết, nên mặc định
# chuyển sang KHÔNG GỬI tham số này. Bù lại phải nới max_tokens (xem ngay dưới), vì
# reasoning_effort="none" trước đây chính là thứ giữ cho model thinking khỏi tiêu hết
# quota token vào phần suy luận.
REASONING_EFFORT = os.environ.get("PD_REASONING_EFFORT", "").strip()
# 16 (không phải 8): thừa cho "OptionA" mà vẫn còn biên nếu model nào đó không chịu
# reasoning_effort và lỡ nói thêm vài token.
# 16 chỉ an toàn khi có reasoning_effort="none". Không gửi được tham số đó nữa
# thì phải nới lên 64 - mức đã đo là đủ cho gemini-3.6-flash trả lời trọn vẹn
# (59 token) thay vì bị cắt thành 'Option'. max_tokens là TRẦN chứ không phải
# lượng bị tính tiền, nên model không-thinking vẫn chỉ tốn ~2 token như cũ.
MAX_OUTPUT_TOKENS = int(os.environ.get("PD_MAX_OUTPUT_TOKENS", "128"))
# Provider nào không nhận `reasoning_effort` thì bật cờ này và thôi gửi kèm.
_NO_REASONING_PARAM = threading.Event()
_UNSUPPORTED_PARAM = ("reasoning_effort", "unsupported", "unrecognized",
                      "unexpected keyword", "invalid_request_error")
# `tool_choice="none"` được gửi để model khỏi trả tool_call rỗng (nguồn lỗi parse của
# SDK). Nhưng endpoint OpenAI/xAI TỪ CHỐI tham số này khi không kèm `tools`:
#     Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools'
#     are specified.
# Đó là thứ đã chặn `gpt-5.4-nano` và `grok-4.20` (RUN_PLAN F2). Không phải lỗi SDK -
# chính task này gửi tham số đó.
_NO_TOOL_CHOICE = threading.Event()
_TOOL_CHOICE_ERR = ("tool_choice",)
# 400 KHÔNG nêu tên tham số. Proxy Google chỉ nói "Request contains an invalid
# argument" (đo 2026-09-02 với reasoning_effort). Bản trước chỉ khớp theo TÊN tham số
# nên cờ không bao giờ bật và cả run chết vì một tham số tuỳ chọn.
_BAD_ARG_ERR = ("invalid argument", "invalid_argument", "400")
# Prompt của lượt gọi khởi động (xem chỗ gọi, ngay trước ThreadPoolExecutor). Cố ý
# ngắn và không liên quan gì tới trò chơi: nó chỉ để dò xem provider có nhận các tham
# số tuỳ chọn không, kết quả bị vứt đi và không vào dữ liệu.
WARMUP_PROMPT = "Reply with exactly one word: OK"

# Ngưỡng health-check cuối run (xem assertion ở cuối file).
FALLBACK_RATE_TOLERANCE = float(os.environ.get("PD_FALLBACK_TOLERANCE", "0.02"))

# Số game chạy song song. Mỗi game NỘI BỘ vẫn tuần tự (vòng r cần vòng r-1) nên kết
# quả không đổi theo mức song song — seed cố định theo (cell, round, agent). Hạ về 1
# nếu proxy 429 liên tục.
CONCURRENCY = int(os.environ.get("PD_CONCURRENCY", "8"))

RESUME = os.environ.get("PD_RESUME", "1").strip().lower() not in {"0", "false", "no", "off"}
CHECKPOINT_SCHEMA_VERSION = 1

# --- Chốt chặn chi phí ---------------------------------------------------------
# `kaggle b t push` chạy task MỘT lần trên MODEL MẶC ĐỊNH CỦA SERVER trước khi task
# dùng được, và ta không chọn được model đó. Đó là lượt chạy duy nhất cần chặn.
# Chi tiết vì sao là denylist chứ không phải allowlist: xem khối `_SMOKE_ONLY` dưới.
_force_full = os.environ.get("PD_FULL", "").strip().lower() in {"1", "true", "yes", "on"}
_has_overrides = any(os.environ.get(k) for k in
                     ("PD_LAMBDAS", "PD_LANGS", "PD_REPS", "PD_ROUNDS"))

def _slug(name: str) -> str:
    """Ten model rut gon: bo tien to provider, chuan hoa ky tu la ve '-'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name.split("/")[-1])


# --- Cong chan chi phi: DENYLIST chu khong phai allowlist -----------------------
# Muc dich cua cong chan nay CHI la: `kaggle b t push` tu dong chay task mot lan tren
# model MAC DINH cua server, va ta khong chon duoc model do -> phai cho no roi ve smoke
# thay vi dot mot sweep 20.000 luot goi.
#
# Truoc day dung allowlist theo TEN, va no hong ba lan lien vi ten server KHONG trung
# slug CLI:
#     CLI  claude-haiku-4-5-20251001  ->  server  anthropic/claude-haiku-4-5@20251001
#     CLI  gemini-3.5-flash-lite      ->  server  google/gemini-3.5-flash-lite
#     CLI  gemma-4-26b-a4b-it         ->  server  google/gemma-4-26b-a4b      (mat '-it')
# Moi lan lech la sweep AM THAM tut xuong 40 luot goi ma van bao Completed.
# Khong the doan truoc dang ten, nen dao chieu: mot lenh `run -m X` luon la co y, con
# `push` thi khong. Chi can chan dung model mac dinh cua server.
# Ngoai model mac dinh cua server, danh sach nay con giu cac UNG VIEN dang do: cho
# chung roi ve smoke 40 luot goi de biet co chay duoc server-side khong ma khong doi
# rui ro dot mot sweep. Do xong thi BO SLUG DO RA rồi push lai truoc khi chay that.
_SMOKE_ONLY = {_slug(m) for m in _env_list("PD_SMOKE_MODELS", [
    "gemini-3-flash-preview",
    "qwen3-next-80b-a3b-instruct",
    "deepseek-v3.1",
    "glm-5",
    "claude-haiku-4-5-20251001",
])}
if not (_force_full or _has_overrides) and _slug(MODEL) in _SMOKE_ONLY:
    LAMBDAS, LANGS, REPS, N_ROUNDS = [1.0], ["en"], 1, 5
    print(f"[guard] {MODEL} nằm trong PD_SMOKE_MODELS -> chạy SMOKE "
          f"(1 λ × 1 lang × 4 tổ hợp × 1 rep × 5 vòng = 40 lượt gọi). "
          f"Mở full bằng PD_FULL=1 hoặc override PD_LAMBDAS/PD_LANGS/PD_REPS.",
          flush=True)

# Proxy (nhất là model non-Gemini trên staging) thỉnh thoảng trả 429/503 → retry.
_TRANSIENT = ("429", "503", "500", "502", "504", "overloaded",
              "unavailable", "not reachable", "rate limit", "heavy load")
# Lỗi tạm thời đáng chờ lâu hơn hẳn lỗi thật: chờ CHÍNH LÀ cách xử lý đúng cho nó, còn
# với lỗi thật thì retry chỉ tổ đốt thời gian. Dùng chung một ngân sách 6 lượt là thứ
# đã giết run grok-4.20 (1040035) sau khi nó đã chạy được game đầu tiên.
TRANSIENT_MAX_ATTEMPTS = int(os.environ.get("PD_TRANSIENT_ATTEMPTS", "14"))
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
    """BASE_WEIGHTS × λ, khử nhiễu float, ép int khi nguyên (== notebook open-source).

    Trả về SỐ, không phải chuỗi: `attribute_scores` dùng chính dict này để chấm điểm.
    Việc in ra prompt là chuyện khác - xem `display_weights`.
    """
    out = {}
    for k, v in BASE_WEIGHTS.items():
        s = round(float(v) * float(lam), 10)
        out[k] = int(s) if NORMALIZE_INTEGER_WEIGHTS and float(s).is_integer() else s
    return out


def display_weights(weights) -> dict:
    """Ô payoff ĐÚNG NHƯ prompt in ra. Đây là chỗ E2 tách notation khỏi magnitude.

    Trên lưới decade, "λ < 1" và "ô in ra có dấu chấm thập phân" là CÙNG MỘT sự kiện,
    nên không phân biệt được hai giả thuyết. `PD_WEIGHT_FORMAT` cắt đúng chỗ đó:

    - ``native`` (mặc định) - y như FAIRGAME, `6` và `0.6`. Giữ nguyên để test parity
      byte-exact không đổi và mọi dữ liệu cũ vẫn tái tạo được.
    - ``decN``  - luôn N chữ số thập phân: `dec2` cho `6.00`, `0.60`, `60.00`.

    Hai nhánh thí nghiệm dựng từ đó:

    - **magnitude cố định, notation đổi**: cùng λ=1, chạy `native` (`6`) và `dec2`
      (`6.00`). Payoff y hệt nhau, chỉ khác cách in. Chênh lệch nào cũng là NOTATION.
    - **notation cố định, magnitude đổi**: λ ∈ {0.1, 1, 10} đều in `dec2`, nên ô nào
      cũng có dấu chấm. Chênh lệch nào cũng là MAGNITUDE.

    Không đụng tới điểm số: chấm điểm vẫn dùng `scaled_weights`, tức là số thật.
    """
    if WEIGHT_FORMAT == "native":
        return weights
    m = re.fullmatch(r"dec(\d+)", WEIGHT_FORMAT)
    if not m:
        raise ValueError(f"PD_WEIGHT_FORMAT không hợp lệ: {WEIGHT_FORMAT!r} "
                         f"(chỉ nhận 'native' hoặc 'decN')")
    nd = int(m.group(1))
    return {k: f"{float(v):.{nd}f}" for k, v in weights.items()}


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
        # Template Stag Hunt boc cau lenh chon vao khoi {choose} va co them khoi
        # {communicate}. FAIRGAME voi phase='choose' GIU 'choose', BO 'communicate'
        # (prompt_creator.py:157-167). Khong khai bao 'choose' o day thi khoi mac
        # dinh bi xoa -> prompt mat han cau yeu cau chon, va model van tra loi gi do
        # nen loi khong lo ra o dau ca.
        "choose": True,
        "communicate": bool(AGENTS_COMMUNICATE),
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
    shown = display_weights(weights)
    for i, key in enumerate(shown, start=1):
        values[f"weight{i}"] = shown[key]
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


def _drop_optional_param(msg: str) -> str:
    """Provider từ chối một tham số tuỳ chọn. Trả:

    - ``"dropped"`` - vừa gỡ được một tham số, thử lại ngay và KHÔNG tính lượt;
    - ``"stale"``   - đúng lỗi đó nhưng tham số đã bị gỡ từ trước, nên đây là request
      bay song song được gửi trước lúc cờ kịp bật -> retry có backoff là qua;
    - ``""``        - không phải lỗi tham số.

    Vì sao không chỉ khớp theo tên tham số: hai provider báo lỗi theo hai kiểu.

    - OpenAI/xAI CÓ nêu tên:
        "Invalid value for 'tool_choice': 'tool_choice' is only allowed when
         'tools' are specified"
    - Proxy Google KHÔNG nêu tên, chỉ nói "Request contains an invalid argument"
      (đo 2026-09-02 với `reasoning_effort`).

    Bản trước chỉ khớp theo tên nên với Google cờ không bao giờ bật: mọi lượt gọi đều
    400 và cả run Errored sau ~10 giây. Vì thế khi thấy 400 kiểu chung chung thì gỡ
    DẦN từng tham số tuỳ chọn - mỗi tham số một lần cho cả run, tối đa hai lượt phí.
    """
    def _has(tokens):
        return any(t in msg for t in tokens)

    if not (_has(_UNSUPPORTED_PARAM) or _has(_TOOL_CHOICE_ERR)
            or _has(_BAD_ARG_ERR)):
        return ""
    # Lỗi nêu đích danh `tool_choice` thì đừng đụng tới `reasoning_effort`.
    if (REASONING_EFFORT and not _NO_REASONING_PARAM.is_set()
            and not _has(_TOOL_CHOICE_ERR)):
        _NO_REASONING_PARAM.set()
        print("[warn] provider từ chối `reasoning_effort` -> bỏ tham số này. "
              "CẢNH BÁO: model thinking có thể ăn hết max_tokens rồi trả text "
              "cụt -> kiểm tra fallback_rate cuối run.", flush=True)
        return "dropped"
    if not _NO_TOOL_CHOICE.is_set():
        _NO_TOOL_CHOICE.set()
        print("[warn] provider từ chối `tool_choice` -> bỏ tham số này. Model có thể "
              "trả tool_call rỗng; nhánh 'SDK tool-call parse bug' bên dưới đỡ.",
              flush=True)
        return "dropped"
    # Cờ đã bật mà vẫn thấy đúng lỗi đó: CONCURRENCY worker cùng bay, những request
    # gửi đi trước lúc cờ kịp bật vẫn mang tham số cũ. Đây là thứ đã giết run 1034690
    # (gpt-5.4-nano): worker đầu gỡ được `tool_choice`, worker thứ hai nhận cùng 400
    # nhưng không còn gì để gỡ -> rơi thẳng ra ngoài. Phải coi là tạm thời.
    return "stale"


def _call_llm(prompt, seed, max_attempts=6):
    """Một lượt sinh văn bản. Trả (text, usage). Tự reauth khi token hết hạn;
    exp-backoff với 429/503."""
    attempt = 0
    auth_retries = 0
    while True:
        attempt += 1
        try:
            params = {
                # ĐỪNG thêm "max_output_tokens" ở đây: endpoint là OpenAI
                # chat.completions, nó ném TypeError "unexpected keyword argument"
                # và giết cả task (run-13). `max_tokens` mới là tên đúng.
                "max_tokens": MAX_OUTPUT_TOKENS,
            }
            # Không có tool nào cần gọi; tắt hẳn để model không trả tool_call rỗng
            # (nguồn gốc lỗi parse của SDK bên dưới). Endpoint OpenAI/xAI từ chối
            # tham số này khi không kèm `tools` -> gỡ ra, xem _drop_optional_param.
            if not _NO_TOOL_CHOICE.is_set():
                params["tool_choice"] = "none"
            if REASONING_EFFORT and not _NO_REASONING_PARAM.is_set():
                params["reasoning_effort"] = REASONING_EFFORT
            with kbench.chats.new("turn", orphan=True) as chat:
                text = _LLM.prompt(
                    prompt,
                    temperature=TEMPERATURE,
                    seed=seed,
                    extra_api_params=params,
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
            # Provider từ chối một tham số TUỲ CHỌN: gỡ nó ra rồi thử lại, thay vì
            # để cả sweep chết vì thứ không ảnh hưởng tới nội dung câu trả lời.
            _param_status = _drop_optional_param(msg)
            if _param_status == "dropped":
                attempt -= 1
                continue
            if _param_status == "stale":
                if attempt >= max_attempts:
                    raise
                time.sleep(min(2 ** attempt, 12))
                continue
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
            transient = any(t in msg for t in _TRANSIENT)
            budget = TRANSIENT_MAX_ATTEMPTS if transient else max_attempts
            if attempt >= budget or not transient:
                raise
            # Jitter: không có nó thì CONCURRENCY worker cùng ngủ `2**attempt` giây rồi
            # thức dậy cùng một lúc và lại đâm vào nhau, kéo dài đúng cái quá tải đang
            # cố chờ cho qua.
            time.sleep(min(2 ** attempt, 45) * (0.5 + random.random()))


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

    model_tag = os.environ.get("PD_MODEL_TAG") or (re.sub(
        r"[^A-Za-z0-9._-]+", "-", MODEL.split("/")[-1])
        + GAME_TAG + format_tag() + RUN_TAG)
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
        coop = sum(s == COOP_STRATEGY
                   for s in ast.literal_eval(row["agent1_strategies"])
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
    # Một lượt gọi KHỞI ĐỘNG tuần tự trước khi mở pool. Mục đích duy nhất: để các cờ
    # tham số (`_NO_TOOL_CHOICE`, `_NO_REASONING_PARAM`) ổn định xong mới chạy song
    # song. Không có nó thì CONCURRENCY worker cùng bay vào cùng một lỗi 400, worker
    # đầu gỡ được tham số còn các worker sau nhận lỗi đã hết cách gỡ - đó là thứ đã
    # giết run 1034690 (gpt-5.4-nano). Rẻ: đúng một lượt gọi, và nếu provider từ chối
    # tham số nào thì thấy ngay ở dòng log đầu tiên thay vì sau vài trăm game.
    if pending and CONCURRENCY > 1:
        try:
            _call_llm(WARMUP_PROMPT, BASE_SEED)
            print("[warmup] 1 lượt gọi tuần tự OK -> mở pool.", flush=True)
        except QuotaExhausted:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[warmup] lượt gọi khởi động lỗi ({type(exc).__name__}: {exc}); "
                  f"vẫn mở pool vì retry trong _call_llm có thể qua được.", flush=True)

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
        return round(sum(p == COOP_STRATEGY for p in picks) / len(picks), 3)

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
