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
import itertools
import json
import os
import sys
from pathlib import Path

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
