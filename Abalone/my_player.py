from player_abalone import PlayerAbalone
from game_state_abalone import GameStateAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from math import inf

CENTER = (8, 4)
TIME_PER_MOVE = 50 # Just under 36 sec (time limit is 15 min -> 900 sec -> 900/25 = 36 sec)
STEP_DEPTH_THRESHOLD = 30 # Arbitrary number of steps to start to go deeper in the search

def manhattan_dist(A: list[int], B: list[int]) -> int:
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
        """
        Return the opponent player.

        Args:
            state (GameStateAbalone): Current game state representation

        Returns:
            PlayerAbalone: Opponent player
        """
        players = state.get_players()
        return players[1] if players[0] == self else players[0]

    def get_score_difference(self, state: GameStateAbalone, scores: dict[int, float]) -> int:
        """
        Compute the difference in scores between the player and the opponent.

        Args:
            state (GameStateAbalone): Current game state representation
            scores (dict[int, float]): Scores of the players

        Returns:
            int: Score difference between the player and the opponent
        """
        opponent = self.get_opponent(state)

        previous_player_score = scores[self.get_id()]
        previous_opponent_score = scores[opponent.get_id()]

        player_score = state.scores[self.get_id()]
        opponent_player_score = state.scores[opponent.get_id()]
        heuristic = player_score

        # If the player wins the game
        if opponent_player_score == -6:
            heuristic += 500

        # If the player loses a marble
        if player_score < previous_player_score:
            heuristic -= 100
            
        # If the opponent loses a marble
        if opponent_player_score < previous_opponent_score:
            heuristic += 200
        return heuristic

    def get_center_proximity(self, state: GameStateAbalone) -> int:
        """
        Compute the proximity of the player's marbles to the center of the board.

        Args:
            state (GameStateAbalone): Current game state representation

        Returns:
            int: Proximity of the player's marbles to the center of the board
        """
        env = state.get_rep().get_env()
        total_distance = 0
        nb_marbles = 0

        # If there are no marbles on the board
        if env.items() is None:
            return 0

        for marble in env.items():
            key, value = marble
            if value.get_owner_id() == self.get_id():
                total_distance += manhattan_dist(CENTER, key)
                nb_marbles += 1

        return total_distance / nb_marbles

    def calculate_heuristic(self, state: GameStateAbalone, scores: dict[int, float]) -> int:
        """
        Compute the heuristic value of the current game state.

        Args:
            state (GameStateAbalone): Current game state representation
            scores (dict[int, float]): Scores of the players

        Returns:
            int: Heuristic value of the current game state
        """
        return (self.get_score_difference(state, scores)) - self.get_center_proximity(state) * 0.005

    def get_possible_actions_intelligent(self, current_state: GameState) -> list[Action]:
        """
        Return the possible actions that are beneficial to the player (no suicide moves)

        Args:
            current_state (GameState): Current game state representation

        Returns:
            list[Action]: List of possible actions that are beneficial to the player
        """
        return [action for action in current_state.get_possible_actions()
                if action.get_next_game_state().get_player_score(self) >= current_state.get_player_score(self)]

    def cut_off(self, d: int):
        return lambda state, depth: depth > d

    def h_alpha_beta_search(self, state: GameStateAbalone, cutoff, h) -> tuple[int, Action]:
        """
        Alpha-Beta search algorithm.

        Args:   
            state (GameStateAbalone): Current game state
            player (PlayerAbalone): Current player
            cutoff (function): Cut off function
            h (function): Heuristic function

        Returns:
            tuple[int, Action]: Tuple with the best value and the best action
        """
        scores = state.get_scores()
        start_time = self.get_remaining_time()
        curr_move = state.get_step()
        if curr_move < STEP_DEPTH_THRESHOLD:
            cutoff = self.cut_off(2)

        def max_value(state: GameStateAbalone, alpha: int, beta: int, depth: int) -> tuple[int, Action]:
            if state.is_done():
                return state.scores[state.next_player.get_id()], None

            if cutoff(state, depth):
                return h(state, scores), None

            v, move = -inf, None
            for a in state.get_possible_actions():
                v2, _ = min_value(a.get_next_game_state(),
                                  alpha, beta, depth+1)
                if v2 > v:
                    v, move = v2, a
                    alpha = max(alpha, v)
                if v >= beta:
                    return v, move
                if (start_time - self.get_remaining_time()) > TIME_PER_MOVE:
                    print("Time limit reached")
                    return v, move

            return v, move

        def min_value(state: GameStateAbalone, alpha: int, beta: int, depth: int) -> tuple[int, Action]:
            if state.is_done():
                return state.scores[state.next_player.get_id()], None

            if cutoff(state, depth):
                return h(state, scores), None

            v, move = inf, None
            for a in state.get_possible_actions():
                v2, _ = max_value(a.get_next_game_state(),
                                  alpha, beta, depth+1)
                if v2 < v:
                    v, move = v2, a
                    beta = min(beta, v)
                if v <= alpha:
                    return v, move
                if (start_time - self.get_remaining_time()) > TIME_PER_MOVE:
                    print("Time limit reached")
                    return v, move

            return v, move

        return max_value(state, -inf, +inf, 0)

    def last_move(self, current_state: GameStateAbalone) -> Action:
        """
        Calculate the best action for the last move.
        
        Args:
            current_state (GameStateAbalone): Current game state representation
            
        Returns:
            Action: Best action for the last move
        """
        possible_actions = list(current_state.get_possible_actions())
        other_id = possible_actions[0].get_next_game_state(
        ).next_player.get_id()
        best_action = None
        best_score = current_state.max_score - 1
        for a in possible_actions:
            score = a.get_next_game_state(
            ).scores[self.id] - a.get_next_game_state().scores[other_id]
            if score > best_score:
                best_action = a
                best_score = score

        return best_action

    def compute_action(self, current_state: GameStateAbalone, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """
        if current_state.get_step() == current_state.max_step - 1:
            return self.last_move(current_state)
        return self.h_alpha_beta_search(current_state, self.cut_off(3), self.calculate_heuristic)[1]
