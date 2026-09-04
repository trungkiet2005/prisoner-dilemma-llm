"""Parity test: `pd_task.py` (nhánh Kaggle Benchmarks) == FAIRGAME (nhánh open-source).

Kaggle Benchmarks chỉ push MỘT file, nên `pd_task.py` phải NHÚNG bản sao của prompt
template + config payoff và VIẾT LẠI logic dựng prompt / parse / chấm điểm. Test này
là thứ giữ cho bản sao đó không trôi khỏi bản gốc: nếu ai sửa template trong
`FAIRGAME/resources/` mà quên đồng bộ `pd_task.py` (hoặc ngược lại), test đỏ.

Chạy:
    cd <repo root>
    PYTHONUTF8=1 python -m pytest kaggle/benchmarks/test_pd_task_parity.py -q
    # hoặc không cần pytest:
    PYTHONUTF8=1 python kaggle/benchmarks/test_pd_task_parity.py
"""
import contextlib
import itertools
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FAIRGAME_DIR = REPO_ROOT / "FAIRGAME"
CONFIG_PATH = (FAIRGAME_DIR / "resources" / "config"
               / "prisoner_dilemma_nocomm_round_known_conventional.json")
TEMPLATE_DIR = FAIRGAME_DIR / "resources" / "game_templates"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ngăn pd_task tự gọi .run() (sẽ đụng proxy model) khi chỉ import để test.
os.environ["PD_SKIP_RUN"] = "1"
# Ghim sweep nhỏ + tất định để test không phụ thuộc env của máy chạy.
os.environ.setdefault("PD_REPS", "10")
os.environ.setdefault("PD_ROUNDS", "30")

import pd_task  # noqa: E402

from FAIRGAME.src.batch_runner import _match_strategy_key as fg_match  # noqa: E402
from FAIRGAME.src.fairgame_factory import FairGameFactory  # noqa: E402
from FAIRGAME.src.game_round import GameRound  # noqa: E402
from FAIRGAME.src.payoff_matrix import PayoffMatrix  # noqa: E402
from FAIRGAME.src.results_processing.results_processor import ResultsProcessor  # noqa: E402
from FAIRGAME.src.utils.rtf_to_text import rtf_to_text  # noqa: E402


@pytest.fixture(autouse=True)
def _default_to_prisoner_dilemma():
    """Mọi test chạy ở chế độ PD, notation gốc - trừ khi tự vào sh_mode()/weight_format().

    pd_task chọn game VÀ cách in payoff ở thời điểm import, và cả hai default đó ĐỔI
    theo lần push đang chuẩn bị (E7 đặt game thành stag_hunt; E2 đặt cách in thành
    dec2). Không ghim lại thì cả bộ test parity hỏng mỗi khi ai đó cấu hình một lần
    chạy khác - tức là test suite phụ thuộc vào đúng thứ nó phải kiểm tra độc lập.
    """
    old = (pd_task.TEMPLATES, pd_task.BASE_WEIGHTS, pd_task.GAME_TAG, pd_task.GAME,
           pd_task.WEIGHT_FORMAT)
    pd_task.TEMPLATES = pd_task.PD_TEMPLATES
    pd_task.BASE_WEIGHTS = pd_task.PD_WEIGHTS
    pd_task.GAME_TAG = ""
    pd_task.GAME = "prisoner_dilemma"
    pd_task.WEIGHT_FORMAT = "native"
    try:
        yield
    finally:
        (pd_task.TEMPLATES, pd_task.BASE_WEIGHTS, pd_task.GAME_TAG, pd_task.GAME,
         pd_task.WEIGHT_FORMAT) = old


LANGS = ["en", "fr", "ar", "cn", "vn"]
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def fairgame_template(lang: str) -> str:
    txt = TEMPLATE_DIR / f"prisoner_dilemma_{lang}.txt"
    rtf = TEMPLATE_DIR / f"prisoner_dilemma_{lang}.rtf"
    if txt.exists():
        return txt.read_text(encoding="utf-8")
    return rtf_to_text(rtf.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 1. Hằng số nhúng
# --------------------------------------------------------------------------
def test_templates_match_fairgame_sources():
    assert set(pd_task.TEMPLATES) == set(LANGS)
    for lang in LANGS:
        assert pd_task.TEMPLATES[lang] == fairgame_template(lang), (
            f"template {lang} lệch — chạy lại bước nhúng cho pd_task.py")


def test_payoff_and_agents_match_config():
    pm = CONFIG["payoffMatrix"]
    assert pd_task.BASE_WEIGHTS == pm["weights"]
    assert pd_task.COMBINATIONS == pm["combinations"]
    assert pd_task.MATRIX == pm["matrix"]
    assert pd_task.AGENT_NAMES == CONFIG["agents"]["names"]
    for lang in LANGS:
        assert pd_task.PERSONALITIES[lang] == CONFIG["agents"]["personalities"][lang]
        # config khai báo cùng bộ strategy cho mọi ngôn ngữ -> pd_task nhúng 1 bản chung
        assert pd_task.STRATEGIES == pm["strategies"][lang]


def test_conventional_payoff_is_the_frontier_one():
    """Chốt chặn cho chính lỗi đã xảy ra: w1 PHẢI là 6 (conventional), không phải 8 (mild)."""
    assert pd_task.BASE_WEIGHTS == {"weight1": 6, "weight2": 10, "weight3": 0, "weight4": 2}


# --------------------------------------------------------------------------
# 2. Dựng prompt
# --------------------------------------------------------------------------
def _fairgame_game(lang, lam, n_rounds, rounds_known):
    """Dựng FairGame thật từ config + template gốc, weights đã scale như pd_task."""
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["llm"] = "LocalModel"
    cfg.pop("llms", None)
    cfg["languages"] = [lang]
    cfg["nRounds"] = n_rounds
    cfg["nRoundsIsKnown"] = rounds_known
    cfg["promptTemplate"] = {lang: fairgame_template(lang)}
    cfg["payoffMatrix"]["weights"] = pd_task.scaled_weights(lam)

    factory = FairGameFactory()
    processed = factory.io_manager.process_and_validate_configuration(cfg)
    return factory.create_games(processed)   # 4 game = 4 tổ hợp tính cách


def test_prompt_matches_fairgame_prompt_creator():
    """So từng ký tự prompt của pd_task với prompt FAIRGAME thật sinh ra."""
    checked = 0
    for lang, lam, rounds_known in itertools.product(LANGS, [1, 0.01, 1000], [True, False]):
        n_rounds = 30
        games = _fairgame_game(lang, lam, n_rounds, rounds_known)
        weights = pd_task.scaled_weights(lam)

        old_known = pd_task.N_ROUNDS_KNOWN
        old_n = pd_task.N_ROUNDS
        pd_task.N_ROUNDS_KNOWN = rounds_known
        pd_task.N_ROUNDS = n_rounds
        try:
            for perm_idx, game in enumerate(games):
                agents = list(game.agents.values())
                # round 1: history rỗng; round 3: history 2 vòng đã ghi.
                for current_round in (1, 3):
                    game.current_round = current_round
                    game.history.rounds = {}
                    for r in range(1, current_round):
                        game.history.rounds[f"round_{r}"] = {
                            agents[0].name: {"strategy": "OptionA", "score": weights["weight1"]},
                            agents[1].name: {"strategy": "OptionB", "score": weights["weight2"]},
                        }
                    runner = GameRound(game)
                    for a_idx, agent in enumerate(agents):
                        expected = runner.create_prompt(agent, phase="choose")
                        got = pd_task.assemble_prompt(
                            lang, a_idx, agent.personality, current_round,
                            game.history.rounds, weights)
                        assert got == expected, (
                            f"prompt lệch: lang={lang} λ={lam} known={rounds_known} "
                            f"perm={perm_idx} agent={a_idx} round={current_round}\n"
                            f"--- pd_task ---\n{got!r}\n--- FAIRGAME ---\n{expected!r}")
                        checked += 1
        finally:
            pd_task.N_ROUNDS_KNOWN = old_known
            pd_task.N_ROUNDS = old_n
    assert checked == len(LANGS) * 3 * 2 * 4 * 2 * 2


def test_personality_permutation_order_matches_factory():
    """Thứ tự 4 tổ hợp tính cách của pd_task phải trùng itertools.product của factory."""
    for lang in LANGS:
        games = _fairgame_game(lang, 1, 30, True)
        fg_order = [tuple(a.personality for a in g.agents.values()) for g in games]
        pd_order = [tuple(pd_task.PERSONALITIES[lang][i] for i in perm)
                    for perm in pd_task.PERSONALITY_PERMS]
        assert pd_order == fg_order, f"lang={lang}: {pd_order} != {fg_order}"


# --------------------------------------------------------------------------
# 3. Parse + chấm điểm
# --------------------------------------------------------------------------
RESPONSES = [
    "OptionA", "OptionB", " optionb ", "I choose OptionA.", "**OptionB**",
    "Option A", "A", "B", "1", "2", "strategy1", "strategy2",
    "我选择OptionA", "Je choisis OptionB.", "", "hmm let me think about it",
    "The best move here is clearly to defect.", "Answer: option b",
]


def test_strategy_parsing_matches_batch_runner():
    for resp in RESPONSES:
        assert pd_task.match_strategy_key(resp) == fg_match(resp, pd_task.STRATEGIES), (
            f"parse lệch cho {resp!r}")


def test_scores_match_payoff_matrix():
    for lam in (0.01, 0.1, 1, 10, 100, 1000):
        weights = pd_task.scaled_weights(lam)
        matrix_data = dict(CONFIG["payoffMatrix"])
        matrix_data["weights"] = weights
        pm = PayoffMatrix(matrix_data, "en")

        class _A:
            def __init__(self):
                self.scores = []

            def add_score(self, s):
                self.scores.append(s)

        for keys in itertools.product(["strategy1", "strategy2"], repeat=2):
            agents = [_A(), _A()]
            pm.attribute_scores(agents, list(keys))
            expected = (agents[0].scores[0], agents[1].scores[0])
            assert pd_task.attribute_scores(list(keys), weights) == expected


def test_lambda_scaling_is_integer_clean():
    assert pd_task.scaled_weights(1) == {"weight1": 6, "weight2": 10, "weight3": 0, "weight4": 2}
    assert pd_task.scaled_weights(0.1)["weight1"] == 0.6      # không phải 0.6000000000000001
    assert pd_task.scaled_weights(1000)["weight2"] == 10000
    assert [pd_task.fmt_lambda(x) for x in (0.01, 0.1, 1, 10, 100, 1000)] == \
        ["0.01", "0.1", "1", "10", "100", "1000"]


def test_e1_offdecade_lambda_notation():
    """E1: λ ngoài decade phải in ra ĐÚNG dạng đã thiết kế.

    Cả thí nghiệm E1 dựa trên một sự thật về cách in số: λ=0.5 có độ lớn dưới đơn vị
    nhưng in ra TOÀN SỐ NGUYÊN, trong khi λ=0.25 vẫn in ra thập phân. Đó là thứ phá
    vỡ đồng nhất thức "λ < 1 <=> ô có dấu thập phân" của lưới decade, và là lý do
    duy nhất để chạy bốn mức này.

    Nếu NORMALIZE_INTEGER_WEIGHTS bị tắt, hoặc ai đó thêm format spec vào template,
    thì λ=0.5 sẽ in ra "0.0 / 1.0 / 3.0 / 5.0" và ô quyết định biến mất trong im lặng
    - sweep vẫn chạy, vẫn ra số, nhưng không còn trả lời được câu hỏi nào. Test này
    chặn đúng trường hợp đó.
    """
    expected = {
        0.25: ("0 / 0.5 / 1.5 / 2.5", True),
        0.5: ("0 / 1 / 3 / 5", False),        # <- ô quyết định
        2: ("0 / 4 / 12 / 20", False),
        5: ("0 / 10 / 30 / 50", False),
    }
    for lam, (cells_exp, frac_exp) in expected.items():
        w = pd_task.scaled_weights(lam)
        cells = [str(w[k]) for k in ("weight3", "weight4", "weight1", "weight2")]
        assert " / ".join(cells) == cells_exp, f"lambda={lam}: in ra {cells}"
        assert any("." in c for c in cells) is frac_exp, f"lambda={lam}: thap phan sai"

        # Phép nhân λ chỉ là null có định lý bảo chứng khi thứ tự PD còn nguyên.
        T, R, P, S = w["weight2"], w["weight1"], w["weight4"], w["weight3"]
        assert T > R > P > S, f"lambda={lam}: thu tu PD gay"
        assert 2 * R > T + S, f"lambda={lam}: 2R > T+S gay"

    # Tên thư mục phải parse được bằng float(), vì ingest.py đọc scale từ tên thư mục.
    assert [pd_task.fmt_lambda(x) for x in (0.25, 0.5, 2, 5)] == ["0.25", "0.5", "2", "5"]


def test_e1_configs_match_pd_task_scaling():
    """Config E1 sinh cho nhánh FAIRGAME native phải khớp với pd_task.

    Hai nhánh nhân λ bằng hai đoạn code khác nhau (50_make_e1_configs.scale_weight và
    pd_task.scaled_weights). Nếu chúng lệch nhau thì prompt của nhánh native khác
    prompt của nhánh proxy, và không nhánh nào so được với nhánh kia.
    """
    import json as _json

    cfg_dir = REPO_ROOT / "FAIRGAME" / "resources" / "config" / "e1_offdecade"
    if not cfg_dir.exists():
        return  # chưa chạy Analysis/scripts/50_make_e1_configs.py

    for lam in (0.25, 0.5, 2, 5):
        name = ("prisoner_dilemma_nocomm_round_known_conventional_"
                f"x{pd_task.fmt_lambda(lam)}.json")
        path = cfg_dir / name
        assert path.exists(), f"thieu {name} - chay Analysis/scripts/50_make_e1_configs.py"
        cfg = _json.loads(path.read_text(encoding="utf-8"))
        assert cfg["payoffMatrix"]["weights"] == pd_task.scaled_weights(lam), \
            f"lambda={lam}: config native lech voi pd_task.scaled_weights"


# --------------------------------------------------------------------------
# 4. Schema CSV
# --------------------------------------------------------------------------
def test_csv_fields_match_results_processor():
    """Cột CSV của pd_task phải TRÙNG THỨ TỰ với đầu ra ResultsProcessor của FAIRGAME."""
    games = _fairgame_game("en", 1, 2, True)
    game = games[0]
    agents = list(game.agents.values())
    for r in (1, 2):
        for agent in agents:
            agent.add_strategy("OptionA")
            agent.add_score(6)
        game.history.rounds[f"round_{r}"] = {
            a.name: {"strategy": "OptionA", "score": 6} for a in agents}
    game.description.pop("payoff_matrix", None)
    results = {"game_0": {"description": game.description,
                          "history": game.history.describe()}}
    df = ResultsProcessor().process(results)
    assert list(df.columns) == pd_task.CSV_FIELDS


def test_csv_fields_match_existing_dataset():
    """Và phải trùng luôn với dữ liệu đã thu trong Dataset/ (nếu có sẵn trong repo)."""
    import csv as _csv

    sample = next((REPO_ROOT / "Dataset").rglob("x1_en_*.csv"), None)
    if sample is None:
        return
    with open(sample, encoding="utf-8", newline="") as f:
        header = next(_csv.reader(f))
    assert header == pd_task.CSV_FIELDS, f"{sample.name} có schema khác"


# --------------------------------------------------------------------------
# 5. Stag Hunt (E7)
# --------------------------------------------------------------------------
SH_CONFIG_PATH = (REPO_ROOT / "FAIRGAME" / "resources" / "config"
                  / "stag_hunt_nocomm_round_known_conventional.json")


def sh_template(lang: str) -> str:
    return (TEMPLATE_DIR / f"stag_hunt_{lang}.txt").read_text(encoding="utf-8")


@contextlib.contextmanager
def sh_mode():
    """Đổi tạm pd_task sang Stag Hunt.

    pd_task chọn game ở thời điểm import (module-level), nên không thể vừa test PD
    vừa test SH trong cùng một tiến trình mà không đổi tráo như thế này. Cùng kiểu
    với cách các test khác đổi tạm N_ROUNDS.
    """
    old = (pd_task.TEMPLATES, pd_task.BASE_WEIGHTS, pd_task.GAME_TAG, pd_task.GAME)
    pd_task.TEMPLATES = pd_task.SH_TEMPLATES
    pd_task.BASE_WEIGHTS = pd_task.SH_WEIGHTS
    pd_task.GAME_TAG = "-sh"
    pd_task.GAME = "stag_hunt"
    try:
        yield
    finally:
        (pd_task.TEMPLATES, pd_task.BASE_WEIGHTS,
         pd_task.GAME_TAG, pd_task.GAME) = old


def test_staghunt_templates_match_fairgame_sources():
    """5 template Stag Hunt nhúng trong pd_task phải trùng TỪNG BYTE với file gốc.

    Đây là cùng cái bẫy đã làm hỏng bản PD một lần: mở/lưu bằng editor không phải
    UTF-8 làm hỏng CHÍNH PROMPT gửi cho model, không chỉ comment. Block được sinh
    bằng Analysis/scripts/51_embed_staghunt.py - đừng sửa tay.
    """
    assert set(pd_task.SH_TEMPLATES) == set(LANGS)
    for lang in LANGS:
        assert pd_task.SH_TEMPLATES[lang] == sh_template(lang), (
            f"template stag_hunt {lang} lệch - chạy lại 51_embed_staghunt.py")


def test_staghunt_payoff_matches_config():
    pm = json.loads(SH_CONFIG_PATH.read_text(encoding="utf-8"))["payoffMatrix"]
    assert pd_task.SH_WEIGHTS == pm["weights"]
    # Stag Hunt dùng y hệt khuôn combinations/matrix/strategies của PD nên hai game
    # chia sẻ được các hằng số đó. Test này chốt điều kiện ấy: nếu FAIRGAME đổi
    # khuôn thì pd_task phải tách chúng ra theo game.
    assert pd_task.COMBINATIONS == pm["combinations"]
    assert pd_task.MATRIX == pm["matrix"]
    for lang in LANGS:
        assert pd_task.STRATEGIES == pm["strategies"][lang]


def test_staghunt_is_a_stag_hunt_not_a_dilemma():
    """Cấu trúc phải đúng Stag Hunt, và bất biến λ phải là định lý.

    R > T > P > S (không có hành động trội, hai cân bằng thuần), và Hare trội về
    rủi ro vì P-S > R-T. Cả hai đều là tỉ số của hiệu nên λ-bất biến.
    """
    with sh_mode():
        for lam in (0.1, 0.5, 1, 10, 100):
            w = pd_task.scaled_weights(lam)
            R, T, P, S = w["weight1"], w["weight2"], w["weight4"], w["weight3"]
            assert R > T > P > S, f"λ={lam}: không phải Stag Hunt, có {R},{T},{P},{S}"
            assert (P - S) > (R - T), f"λ={lam}: Hare phải trội về rủi ro"
            p_star = (P - S) / ((R - T) + (P - S))
            assert abs(p_star - 2 / 3) < 1e-9, f"λ={lam}: p*={p_star}, phải là 2/3"


def test_staghunt_lambda_half_prints_integers():
    """λ=0.5 in ra số nguyên dù độ lớn dưới đơn vị - ô quyết định của E1, lặp lại
    được ở game thứ hai."""
    with sh_mode():
        w = pd_task.scaled_weights(0.5)
        cells = [str(w[k]) for k in ("weight3", "weight4", "weight2", "weight1")]
        assert " / ".join(cells) == "0 / 2 / 3 / 4", cells
        assert not any("." in c for c in cells)


def test_staghunt_prompt_matches_fairgame_prompt_creator():
    """So từng ký tự prompt Stag Hunt với prompt FAIRGAME thật sinh ra.

    Quan trọng nhất là khối {choose}: template Stag Hunt bọc câu lệnh chọn vào đó,
    còn `enabled` trong assemble_prompt mặc định XOÁ mọi khối không khai báo. Quên
    khai báo 'choose' thì prompt mất hẳn câu yêu cầu chọn mà model vẫn trả lời gì
    đó, nên lỗi không lộ ra ở đâu ngoài test này.
    """
    checked = 0
    with sh_mode():
        for lang, lam, known in itertools.product(LANGS, [1, 0.5, 100], [True, False]):
            n_rounds = 10
            cfg = json.loads(SH_CONFIG_PATH.read_text(encoding="utf-8"))
            cfg["llm"] = "LocalModel"
            cfg.pop("llms", None)
            cfg["languages"] = [lang]
            cfg["nRounds"] = n_rounds
            cfg["nRoundsIsKnown"] = known
            cfg["promptTemplate"] = {lang: sh_template(lang)}
            weights = pd_task.scaled_weights(lam)
            cfg["payoffMatrix"]["weights"] = weights

            factory = FairGameFactory()
            games = factory.create_games(
                factory.io_manager.process_and_validate_configuration(cfg))

            old_known, old_n = pd_task.N_ROUNDS_KNOWN, pd_task.N_ROUNDS
            pd_task.N_ROUNDS_KNOWN, pd_task.N_ROUNDS = known, n_rounds
            try:
                for game in games:
                    agents = list(game.agents.values())
                    for current_round in (1, 3):
                        game.current_round = current_round
                        game.history.rounds = {}
                        for r in range(1, current_round):
                            game.history.rounds[f"round_{r}"] = {
                                agents[0].name: {"strategy": "OptionA",
                                                 "score": weights["weight1"]},
                                agents[1].name: {"strategy": "OptionB",
                                                 "score": weights["weight2"]},
                            }
                        runner = GameRound(game)
                        for a_idx, agent in enumerate(agents):
                            expected = runner.create_prompt(agent, phase="choose")
                            got = pd_task.assemble_prompt(
                                lang, a_idx, agent.personality, current_round,
                                game.history.rounds, weights)
                            assert got == expected, (
                                f"prompt stag_hunt lệch: lang={lang} λ={lam} "
                                f"known={known} agent={a_idx} round={current_round}\n"
                                f"--- pd_task ---\n{got!r}\n--- FAIRGAME ---\n{expected!r}")
                            checked += 1
            finally:
                pd_task.N_ROUNDS_KNOWN, pd_task.N_ROUNDS = old_known, old_n
    assert checked == len(LANGS) * 3 * 2 * 4 * 2 * 2


def test_game_tag_separates_output_paths():
    """Hai game KHÔNG được ghi vào cùng đường dẫn.

    pd_task chỉ khoá output theo λ/lang/model. Không có hậu tố game thì Stag Hunt
    ở λ=1 ghi đè lên PD ở λ=1, và `resume` còn coi checkpoint của game kia là đã
    xong. Hỏng im lặng: run báo Completed, dữ liệu là của game khác.
    """
    assert pd_task._GAME_SPEC["stag_hunt"][2] == "-sh"
    assert pd_task._GAME_SPEC["prisoner_dilemma"][2] == ""
    tags = {spec[2] for spec in pd_task._GAME_SPEC.values()}
    assert len(tags) == len(pd_task._GAME_SPEC), "hậu tố game bị trùng"


@contextlib.contextmanager
def weight_format(fmt):
    """Đổi cách IN ô payoff trong phạm vi một test."""
    old = pd_task.WEIGHT_FORMAT
    pd_task.WEIGHT_FORMAT = fmt
    try:
        yield
    finally:
        pd_task.WEIGHT_FORMAT = old


def test_native_rendering_is_the_identity():
    """`native` phải trả về ĐÚNG dict số ban đầu, không phải bản sao đã format.

    Đây là thứ giữ cho mọi test parity byte-exact ở trên còn ý nghĩa: chúng chạy dưới
    fixture ghim `native`, nên nếu `native` lỡ format lại gì đó thì toàn bộ so sánh
    byte-exact với FAIRGAME thành so sánh với chính pd_task.
    """
    w = pd_task.scaled_weights(0.1)
    assert pd_task.display_weights(w) is w


def test_dec2_fixes_notation_while_magnitude_varies():
    """Nhánh 'notation cố định, magnitude đổi' của E2.

    Trên lưới decade, 'λ < 1' và 'ô in ra có dấu chấm' là cùng một sự kiện. `dec2` cắt
    ràng buộc đó: mọi λ đều in ra dấu chấm và đúng 2 chữ số thập phân, nên khác biệt
    hành vi còn lại chỉ có thể do độ lớn.
    """
    with weight_format("dec2"):
        seen = {}
        for lam in (0.1, 1, 10):
            cells = list(pd_task.display_weights(pd_task.scaled_weights(lam)).values())
            assert all("." in c for c in cells), (lam, cells)
            assert all(len(c.split(".")[1]) == 2 for c in cells), (lam, cells)
            seen[lam] = cells
        assert seen[0.1] != seen[1] != seen[10], "độ lớn phải vẫn khác nhau"


def test_native_and_dec2_same_lambda_differ_only_in_glyphs():
    """Nhánh 'magnitude cố định, notation đổi' của E2.

    Cùng λ=1: `native` in `6`, `dec2` in `6.00`. Giá trị SỐ y hệt nhau, nên mọi chênh
    lệch hành vi giữa hai ô này là hiệu ứng của cách in, không phải của payoff.
    """
    w = pd_task.scaled_weights(1)
    with weight_format("native"):
        native = [str(v) for v in pd_task.display_weights(w).values()]
    with weight_format("dec2"):
        dec2 = list(pd_task.display_weights(w).values())
    assert native == ["6", "10", "0", "2"]
    assert dec2 == ["6.00", "10.00", "0.00", "2.00"]
    assert [float(x) for x in native] == [float(x) for x in dec2]


def test_weight_format_never_touches_scoring():
    """Điểm số phải đi bằng SỐ, không bằng chuỗi đã format.

    `scaled_weights` được dùng cho cả prompt lẫn `attribute_scores`. Nếu format rò vào
    nhánh chấm điểm thì payoff thành chuỗi và điểm sai hoàn toàn - mà không có gì báo.
    """
    w = pd_task.scaled_weights(0.1)
    keys = ["strategy1", "strategy2"]   # attribute_scores dùng KEY, không dùng nhãn
    base = pd_task.attribute_scores(keys, w)
    with weight_format("dec2"):
        assert pd_task.attribute_scores(keys, w) == base
        assert all(isinstance(v, (int, float)) for v in w.values())


def test_format_tag_separates_output_paths():
    """`dec2` ở λ=1 KHÔNG được ghi đè `native` ở λ=1.

    Đường dẫn chỉ khoá theo λ/lang/model, nên hai cách in cùng một λ sẽ trỏ vào đúng
    một file. `format_tag()` là thứ duy nhất tách chúng ra - cùng vai trò với GAME_TAG.
    """
    tags = []
    for fmt in ("native", "dec1", "dec2", "dec3"):
        with weight_format(fmt):
            tags.append(pd_task.format_tag())
    assert tags[0] == "", "native phải giữ nguyên đường dẫn cũ"
    assert len(set(tags)) == 4, "hai cách in không được ra cùng một hậu tố"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}\n      {e}")
    print(f"\n{'ALL PASS' if not failed else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
