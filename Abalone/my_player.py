from player_abalone import PlayerAbalone
from game_state_abalone import GameStateAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from math import inf

def manhattan_dist(self, A: list[int], B: list[int]) -> int:
    """
    Compute the Manhattan distance between two points.

    Args:
        A (list[int]): First point
        B (list[int]): Second point 

    Returns:
        int: Manhattan distance between the two points
    """
    mask1 = [(0, 2), (1, 3), (2, 4)]
    mask2 = [(0, 4)]
    diff = (abs(B[0] - A[0]), abs(B[1] - A[1]))
    dist = (abs(B[0] - A[0]) + abs(B[1] - A[1])) / 2
    if diff in mask1:
        dist += 1
    if diff in mask2:
        dist += 2
    return dist

class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "bob", time_limit: float = 60*15, *args) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name, time_limit, *args)

   

    def get_opponent(self, state: GameStateAbalone) -> PlayerAbalone:
        players = state.get_players()
        return players[1] if players[0] == self else players[0]

    def get_score_difference(self, state: GameStateAbalone, scores: dict[int, float]) -> int:
        heuristic = 0
        opponent = self.get_opponent(state)
        curr_move = state.get_step()
        weight = curr_move / 50

        previous_player_score = scores[self.get_id()]
        previous_opponent_score = scores[opponent.get_id()]

        player_score = state.scores[self.get_id()]
        opponent_player_score = state.scores[opponent.get_id()]

        if opponent_player_score == -6:
            heuristic += 300 

        if player_score == -6:
            heuristic -= 300 

        if opponent_player_score == -5:
            heuristic += 250

        if player_score < previous_player_score:
            heuristic -= 250
        if opponent_player_score < previous_opponent_score:
            heuristic += 200
        return heuristic * weight

    def get_center_proximity(self, state: GameStateAbalone) -> int:
        center = (8, 4)
        env = state.get_rep().get_env()
        dist = 0
        nb_p = 0

        for p in env.items():
            key, value = p
            if value.get_owner_id() == self.get_id():
                dist += manhattan_dist(center, key)
                nb_p += 1

        return dist / nb_p

    def heuristic(self, state: GameStateAbalone, scores: dict[int, float]) -> int:
        return (self.get_score_difference(state, scores)) - self.get_center_proximity(state) * 0.005 

    def cut_off(self, d: int):
        return lambda state, depth: depth > d

    def h_alpha_beta_search(self, state: GameStateAbalone, cutoff, h) -> tuple[int, Action]:
        scores = state.get_scores()

        def max_value(state: GameStateAbalone, alpha: int, beta: int, depth: int) -> tuple[int, Action]:
            if state.is_done():
                return state.scores[state.next_player.get_id()], None
            if cutoff(state, depth):
                return h(state, scores), None
            v, move = -inf, None
            for a in state.get_possible_actions():
                v2, _ = min_value(a.get_next_game_state(), alpha, beta, depth+1)
                if v2 > v:
                    v, move = v2, a
                    alpha = max(alpha, v)
                if v >= beta:
                    return v, move
            return v, move

        def min_value(state: GameStateAbalone, alpha: int, beta: int, depth: int) -> tuple[int, Action]:
            if state.is_done():
                return state.scores[state.next_player.get_id()], None
            if cutoff(state, depth):
                return h(state, scores), None
            v, move = inf, None
            for a in state.get_possible_actions():
                v2, _ = max_value(a.get_next_game_state(), alpha, beta, depth+1)
                if v2 < v:
                    v, move = v2, a
                    beta = min(beta, v)
                if v <= alpha:
                    return v, move
            return v, move
        return max_value(state, -inf, +inf, 0)

    def compute_action(self, current_state: GameStateAbalone, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """
        return self.h_alpha_beta_search(current_state, self.cut_off(2), self.heuristic)[1]
