"""Drive multiple FairGame instances in lockstep so the LLM can be called in
batches.

Within a single round, agents see only history from previous rounds, and
games are independent of one another. So at every round-step we can collect
(n_games × n_agents) prompts and issue them in one batched generate() call.

This module preserves the existing per-game state machine (history, scores,
stop conditions) — it only swaps the LLM call layer for a batched one.
"""

import os
import re

from FAIRGAME.src.game_round import GameRound

# Inlined to be robust against older offline_patch_assets/game_round.py
# snapshots that don't expose these helpers.
_STRATEGY_RETRIES = 10


def _verbose() -> bool:
    return os.environ.get("FAIRGAME_VERBOSE_LOGS", "0") == "1"


def _normalize_strategy_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _map_choice_token(token: str, strategies: dict):
    ordered_keys = list(strategies.keys())
    if not ordered_keys:
        return None
    if token in {"a", "1"}:
        return ordered_keys[0]
    if token in {"b", "2"}:
        return ordered_keys[1] if len(ordered_keys) > 1 else ordered_keys[0]
    return None


def _match_strategy_key(response: str, strategies: dict):
    response_compact = _normalize_strategy_text(response)
    if not response_compact:
        return None
    for key, val in strategies.items():
        if not val:
            continue
        strategy_compact = _normalize_strategy_text(val)
        if strategy_compact and strategy_compact in response_compact:
            return key
    for key in strategies.keys():
        key_compact = _normalize_strategy_text(key)
        if key_compact and key_compact in response_compact:
            return key
    match = re.search(r"\boption\s*([ab]|[12])\b", response, flags=re.IGNORECASE)
    if match:
        return _map_choice_token(match.group(1).lower(), strategies)
    match = re.search(r"\b([ab]|[12])\b", response.strip(), flags=re.IGNORECASE)
    if match:
        return _map_choice_token(match.group(1).lower(), strategies)
    return None


def _fallback_strategy_key(strategies: dict):
    for key in strategies.keys():
        return key
    return None


def _send_prompts_default(prompts):
    """Default send function — uses the local LLM connector singleton."""
    from FAIRGAME.src.llm_connectors.local_vllm_connector import send_prompts_global
    return send_prompts_global(prompts)


def run_games_batched(games, send_prompts=None, batch_size=0,
                      max_strategy_retries=2):
    """Run all games in lockstep with batched LLM calls.

    Args:
        games: list of FairGame instances (already constructed).
        send_prompts: callable(list[str]) -> list[str]. Defaults to the local
            LLM connector singleton.
        batch_size: optional max prompts per generate() call (0 = no chunking,
            run everything as one batch). Use this to cap VRAM if needed.
        max_strategy_retries: how many retry rounds to do for prompts whose
            response didn't match any strategy. After that, fall back to the
            first strategy (matches existing GameRound fallback behavior).
    """
    send = send_prompts or _send_prompts_default

    def _send(prompts):
        return send(prompts) if batch_size == 0 else send(prompts, batch_size)

    # Initialize round trackers — mirrors FairGame.run() but interleaved.
    runners_by_game = {id(g): GameRound(g) for g in games}

    step = 0
    while True:
        # Active games: still have rounds left and stop condition not met.
        active = [
            g for g in games
            if g.current_round <= g.n_rounds and not g.stop_condition_is_met()
        ]
        if not active:
            break

        # Refresh round runners for the new round number.
        for g in active:
            runners_by_game[id(g)] = GameRound(g)

        step += 1
        if _verbose():
            print(f"[batch] step {step}: {len(active)} games active")

        # === Communication phase (only if any active game has it on) ===
        comm_games = [g for g in active if g.agents_communicate]
        if comm_games:
            comm_prompts = []
            comm_meta = []  # (runner, agent, prompt)
            for g in comm_games:
                runner = runners_by_game[id(g)]
                for agent in g.agents.values():
                    prompt = runner.create_prompt(agent, phase='communicate')
                    comm_prompts.append(prompt)
                    comm_meta.append((runner, agent, prompt))
            comm_responses = _send(comm_prompts)
            for (runner, agent, prompt), response in zip(comm_meta, comm_responses):
                runner.game.history.update_round(
                    runner.round_number,
                    agent.name,
                    {'message_prompt': prompt, 'message': response},
                )

        # === Strategy phase ===
        strat_prompts = []
        strat_meta = []  # (game, runner, agent)
        for g in active:
            runner = runners_by_game[id(g)]
            for agent in g.agents.values():
                prompt = runner.create_prompt(agent, phase='choose')
                strat_prompts.append(prompt)
                strat_meta.append((g, runner, agent))

        strat_responses = _send(strat_prompts)
        # Per-meta strategy key (None = needs retry/fallback).
        keys = [None] * len(strat_meta)
        for i, ((g, _r, _a), response) in enumerate(zip(strat_meta, strat_responses)):
            keys[i] = _match_strategy_key(response, g.payoff_matrix.strategies)
            if _verbose() and keys[i] is None:
                print(f"[batch] invalid response (game {id(g)}): {response[:120]!r}")

        # Retry loop — only re-generate the failed indices.
        retries_done = 0
        max_retries = min(max_strategy_retries, _STRATEGY_RETRIES - 1)
        while retries_done < max_retries:
            failed = [i for i, k in enumerate(keys) if k is None]
            if not failed:
                break
            retry_prompts = [strat_prompts[i] for i in failed]
            retry_responses = _send(retry_prompts)
            for i, response in zip(failed, retry_responses):
                g = strat_meta[i][0]
                keys[i] = _match_strategy_key(response, g.payoff_matrix.strategies)
            retries_done += 1

        # Fallback for any still-invalid responses.
        for i, k in enumerate(keys):
            if k is None:
                g = strat_meta[i][0]
                fk = _fallback_strategy_key(g.payoff_matrix.strategies)
                keys[i] = fk
                if _verbose():
                    print(f"[batch] fallback → {fk} for game {id(g)}")

        # Apply strategies and update history per game.
        per_game_strategies = {id(g): [] for g in active}
        for i, (g, runner, agent) in enumerate(strat_meta):
            key = keys[i]
            agent.add_strategy(g.payoff_matrix.strategies[key])
            per_game_strategies[id(g)].append(key)

        for g in active:
            runner = runners_by_game[id(g)]
            round_strategies = per_game_strategies[id(g)]
            g.choices_made.append(round_strategies)
            g.payoff_matrix.attribute_scores(list(g.agents.values()), round_strategies)
            runner._update_round_history()
            g.current_round += 1
