import os
import re
import time

from FAIRGAME.src.prompt_creator import PromptCreator

_STRATEGY_RETRIES = 10
_STRATEGY_RETRY_DELAY_SEC = 1.0
_DEFAULT_FALLBACK_ENABLED = os.environ.get("FAIRGAME_STRATEGY_FALLBACK", "1") == "1"
_DEFAULT_FALLBACK_LOG = os.environ.get("FAIRGAME_STRATEGY_FALLBACK_LOG", "1") == "1"


def _verbose_logs_enabled() -> bool:
    return os.environ.get("FAIRGAME_VERBOSE_LOGS", "0") == "1"


def _normalize_strategy_text(text: str) -> str:
    """Normalize text for robust strategy matching (ignore case/punctuation)."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _match_strategy_key(response: str, strategies: dict) -> str | None:
    """Return the strategy key that best matches the response, if any."""
    response_compact = _normalize_strategy_text(response)
    if not response_compact:
        return None

    # Primary: match normalized strategy text inside normalized response.
    for key, val in strategies.items():
        if not val:
            continue
        strategy_compact = _normalize_strategy_text(val)
        if strategy_compact and strategy_compact in response_compact:
            return key

    # Secondary: match strategy key (e.g., "strategy1") if mentioned.
    for key in strategies.keys():
        key_compact = _normalize_strategy_text(key)
        if key_compact and key_compact in response_compact:
            return key

    # Regex-based parsing for explicit option or numeric answers.
    match = re.search(r"\boption\s*([ab]|[12])\b", response, flags=re.IGNORECASE)
    if match:
        token = match.group(1).lower()
        return _map_choice_token(token, strategies)

    match = re.search(r"\b([ab]|[12])\b", response.strip(), flags=re.IGNORECASE)
    if match:
        token = match.group(1).lower()
        return _map_choice_token(token, strategies)

    return None


def _map_choice_token(token: str, strategies: dict) -> str | None:
    """Map a short token like 'a', 'b', '1', or '2' to a strategy key."""
    ordered_keys = list(strategies.keys())
    if not ordered_keys:
        return None

    if token in {"a", "1"}:
        return ordered_keys[0]
    if token in {"b", "2"}:
        return ordered_keys[1] if len(ordered_keys) > 1 else ordered_keys[0]
    return None


def _fallback_strategy_key(strategies: dict) -> str | None:
    """Return the first available strategy key for a safe fallback."""
    for key in strategies.keys():
        return key
    return None


class GameRound:
    """
    Represents a single round in the game, handling both communication
    and strategy selection phases for all agents.
    """

    def __init__(self, game):
        """
        Initialize with a reference to the FairGame instance.

        Args:
            game (FairGame): The main game object containing all configuration,
                             agents, history, and the payoff matrix.
        """
        self.game = game
        self.round_number = game.current_round

    def run(self):
        """
        Execute one round of the game.

        If agents_communicate is True, they each receive a communication prompt
        and respond with a message. Afterwards, they receive a strategy prompt
        and choose a strategy.

        Returns:
            list of str: The list of strategy keys (not names) chosen by agents this round.
        """
        if self.game.agents_communicate:
            self._execute_communication_phase()

        round_strategies = []
        for agent in self.game.agents.values():
            prompt = self.create_prompt(agent, phase='choose')
            strategy = self._execute_agent_strategy(agent, prompt)
            round_strategies.append(strategy)

        return round_strategies

    def _execute_communication_phase(self):
        """
        Sends a communication prompt to each agent and records the resulting message
        in the game history.
        """
        for agent in self.game.agents.values():
            prompt = self.create_prompt(agent, phase='communicate')
            message = agent.execute_round(prompt)
            self.game.history.update_round(self.round_number, agent.name, {
                'message_prompt': prompt,
                'message': message
            })

    def create_prompt(self, agent, phase):
        """
        Create a prompt for an agent based on the current game state.

        Args:
            agent: The agent object to create the prompt for.
            phase (str): The phase of the round ('communicate' or 'choose').

        Returns:
            str: The prompt to be sent to the agent.
        """
        opponents = self._get_opponents(agent)
        prompt_creator = PromptCreator(
            self.game.language,
            self.game.prompt_template,
            self.game.n_rounds,
            self.game.n_rounds_known,
            self.game.payoff_matrix
        )
        return prompt_creator.fill_template(
            agent,
            opponents,
            self.round_number,
            self.game.history.rounds,
            phase
        )

    def _get_opponents(self, agent):
        """
        Retrieve all other agents in the game that are not the specified agent.

        Args:
            agent: The reference agent.

        Returns:
            list: A list of all agent objects except the reference agent.
        """
        return [a for a in self.game.agents.values() if a != agent]

    def _execute_agent_strategy(self, agent, prompt):
        """
        Retrieve a strategy from the agent after giving it a strategy prompt.

        Retries up to 10 times with 1s delay when the response is invalid or
        agent execution fails (same behavior as the former ``retry`` decorator).

        Args:
            agent: The agent object whose strategy is being determined.
            prompt (str): The strategy prompt.

        Returns:
            str: The strategy key selected by the agent.

        Raises:
            ValueError: If no matching strategy is found in the agent's response after all retries.
        """
        last_error = None
        for attempt in range(_STRATEGY_RETRIES):
            try:
                response = agent.execute_round(prompt)
                if _verbose_logs_enabled():
                    print("RESPONSE ", response)
                found_strategy = _match_strategy_key(
                    response,
                    self.game.payoff_matrix.strategies,
                )
                if found_strategy:
                    agent.add_strategy(self.game.payoff_matrix.strategies[found_strategy])
                    return found_strategy
                last_error = ValueError("No matching strategy found")
            except Exception as e:
                last_error = e
            if attempt < _STRATEGY_RETRIES - 1:
                time.sleep(_STRATEGY_RETRY_DELAY_SEC)
        if _DEFAULT_FALLBACK_ENABLED:
            fallback_key = _fallback_strategy_key(self.game.payoff_matrix.strategies)
            if fallback_key:
                if _DEFAULT_FALLBACK_LOG or _verbose_logs_enabled():
                    print(
                        "WARNING: No matching strategy found after retries; "
                        f"falling back to {fallback_key}."
                    )
                agent.add_strategy(self.game.payoff_matrix.strategies[fallback_key])
                return fallback_key
        raise last_error

    def _update_round_history(self):
        """
        Update the game history with each agent's final strategy and score for this round.
        """
        for agent in self.game.agents.values():
            self.game.history.update_round(self.round_number, agent.name, {
                'strategy': agent.last_strategy(),
                'score': agent.last_score()
            })
