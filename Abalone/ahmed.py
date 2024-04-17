#  AHMED SABSABI (2055025)
#  HICHEM LAMRAOUI (1916853)

import heapq
import random
from math import inf
from typing import Optional, Literal
from seahorse.game.game_layout.board import Piece
from seahorse.player.player import Player
from board_abalone import BoardAbalone
from game_state_abalone import GameStateAbalone
from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState


class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "bob", time_limit: float = 60 * 15, *args) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name, time_limit, *args)
        self.max_depth: int = 4  # 5 depth levels, including zero
        self.max_width: int = 15
        self.cell_hash: Optional[dict] = None
        self.transposition_table: Optional[dict] = None

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """
        _, m = self.heuristic_minimax_with_type_b_strategy(current_state)
        return m

    ############################################################################################################
    #                                      ALGORITHM 1: MINIMAX SEARCH                                         #
    ############################################################################################################
    def minimax_search(self, current_state: GameState) -> tuple[float, Action]:
        """
            Executes the Minimax search algorithm to determine the optimal move in the given game state.

            Parameters:
            - current_state (GameState): The current state of the game.

            Returns:
            tuple[float, Action]: A tuple containing the evaluation score and the optimal action determined by the Minimax algorithm.

            Notes:
            The Minimax search algorithm explores the game tree to find the optimal move for the current player, considering
            both maximizing and minimizing strategies. The evaluation score represents the desirability of the current state
            for the current player, and the optimal action is the move recommended by the Minimax algorithm.

            The Minimax algorithm is implemented through the max_value function, which recursively explores the game tree.
        """

        return self.max_value(current_state)

    def max_value(self, current_state: GameState) -> tuple[float, Optional[Action]]:
        """
            Executes the maximizing phase of the Minimax algorithm to determine the best possible move.

            Parameters:
            - current_state (GameState): The current state of the game.

            Returns:
            tuple[float, Optional[Action]]: A tuple containing the maximum evaluation score and the corresponding optimal action.
            If the game is in a terminal state, the evaluation score is the player's score, and the action is set to None.

            Notes:
            The max_value function is a key component of the Minimax algorithm. It explores the game tree in the maximizing
            phase, considering possible actions and recursively evaluating the resulting states through the min_value function.

            The function returns a tuple containing the maximum evaluation score and the corresponding optimal action.
            If the game is in a terminal state, the evaluation score is the player's score, and the action is set to None.

            The get_possible_actions_without_those_where_player_attacks_himself function is utilized to filter out actions
            where the player attacks itself, ensuring that only valid and strategic moves are considered.
        """

        if current_state.is_done():
            return current_state.get_player_score(self), None

        v_star, m_star = -inf, None

        # Exclude actions where the player attacks himself.
        filtered_possible_actions = self.get_possible_actions_without_those_where_player_attacks_himself(current_state, self)

        for a in filtered_possible_actions:
            s_prime = a.get_next_game_state()
            v, _ = self.min_value(s_prime)

            if v > v_star:
                v_star = v
                m_star = a

        return v_star, m_star

    def min_value(self, current_state: GameState) -> tuple[float, Optional[Action]]:
        """
            Executes the minimizing phase of the Minimax algorithm to determine the best possible move.

            Parameters:
            - current_state (GameState): The current state of the game.

            Returns:
            tuple[float, Optional[Action]]: A tuple containing the minimum evaluation score and the corresponding optimal action.
            If the game is in a terminal state, the evaluation score is the player's score, and the action is set to None.

            Notes:
            The min_value function is a key component of the Minimax algorithm. It explores the game tree in the minimizing
            phase, considering possible actions and recursively evaluating the resulting states through the max_value function.

            The function returns a tuple containing the minimum evaluation score and the corresponding optimal action.
            If the game is in a terminal state, the evaluation score is the player's score, and the action is set to None.

            The get_possible_actions_without_those_where_player_attacks_himself function is utilized to filter out actions
            where the player attacks himself, ensuring that only valid and strategic moves are considered.
        """

        if current_state.is_done():
            return current_state.get_player_score(self), None

        v_star, m_star = inf, None

        # Exclude actions where the player attacks himself.
        filtered_possible_actions = self.get_possible_actions_without_those_where_player_attacks_himself(current_state, self)

        for a in filtered_possible_actions:
            s_prime = a.get_next_game_state()
            v, _ = self.max_value(s_prime)

            if v < v_star:
                v_star = v
                m_star = a

        return v_star, m_star

    ############################################################################################################
    #                              ALGORITHM 2: MINIMAX SEARCH WITH ALPHA BETA PRUNING                         #
    ############################################################################################################

    def minimax_search_with_alpha_beta_pruning(self, current_state: GameState) -> tuple[float, Action]:
        """
            Executes the Minimax search algorithm with alpha-beta pruning to determine the optimal move.

            Parameters:
            - current_state (GameState): The current state of the game.

            Returns:
            tuple[float, Action]: A tuple containing the evaluation score and the optimal action determined by the Minimax
            algorithm with alpha-beta pruning.

            Notes:
            The Minimax search algorithm with alpha-beta pruning is a variant of the classic Minimax algorithm. It optimizes
            the search process by pruning branches that cannot impact the final decision, thereby reducing the computational
            complexity.

            The algorithm is implemented through the max_value_with_alpha_beta_pruning function, which recursively explores
            the game tree while maintaining alpha-beta bounds.

            The function returns a tuple containing the evaluation score and the optimal action determined by the Minimax
            algorithm with alpha-beta pruning.
        """

        return self.max_value_with_alpha_beta_pruning(current_state, -inf, inf)

    def max_value_with_alpha_beta_pruning(self, current_state: GameState, alpha: float, beta: float) -> tuple[float, Optional[Action]]:
        """
            Executes the maximizing phase of the Minimax algorithm with alpha-beta pruning to determine the best possible move.

            Parameters:
            - current_state (GameState): The current state of the game.
            - alpha (float): The alpha parameter for alpha-beta pruning.
            - beta (float): The beta parameter for alpha-beta pruning.

            Returns:
            tuple[float, Optional[Action]]: A tuple containing the maximum evaluation score and the corresponding optimal action.
            If the game is in a terminal state, the evaluation score is the player's score, and the action is set to None.

            Notes:
            The max_value_with_alpha_beta_pruning function is a variant of the Minimax algorithm with alpha-beta pruning. It
            explores the game tree in the maximizing phase, considering possible actions and recursively evaluating the resulting
            states through the min_value_with_alpha_beta_pruning function. The alpha-beta pruning optimizes the search process
            by pruning branches that cannot impact the final decision.

            The function returns a tuple containing the maximum evaluation score and the corresponding optimal action.
            If the game is in a terminal state, the evaluation score is the player's score, and the action is set to None.

            The get_possible_actions_without_those_where_player_attacks_himself function is utilized to filter out actions
            where the player attacks himself, ensuring that only valid and strategic moves are considered.

            The alpha and beta parameters are used to maintain bounds for alpha-beta pruning. If the current best score is
            greater than or equal to beta, pruning occurs, and the function returns the current best score and action.
        """

        if current_state.is_done():
            return current_state.get_player_score(self), None

        v_star, m_star = -inf, None

        # Exclude actions where the player attacks himself.
        filtered_possible_actions = self.get_possible_actions_without_those_where_player_attacks_himself(current_state, self)

        for a in filtered_possible_actions:
            s_prime = a.get_next_game_state()
            v, _ = self.min_value_with_alpha_beta_pruning(s_prime, alpha, beta)

            if v > v_star:
                v_star = v
                m_star = a
                alpha = max(alpha, v_star)

            if v_star >= beta:
                return v_star, m_star

        return v_star, m_star

    def min_value_with_alpha_beta_pruning(self, current_state: GameState, alpha: float, beta: float) -> tuple[float, Optional[Action]]:
        """
            Executes the minimizing phase of the Minimax algorithm with alpha-beta pruning to determine the best possible move.

            Parameters:
            - current_state (GameState): The current state of the game.
            - alpha (float): The alpha parameter for alpha-beta pruning.
            - beta (float): The beta parameter for alpha-beta pruning.

            Returns:
            tuple[float, Optional[Action]]: A tuple containing the minimum evaluation score and the corresponding optimal action.
            If the game is in a terminal state, the evaluation score is the player's score, and the action is set to None.

            Notes:
            The min_value_with_alpha_beta_pruning function is a variant of the Minimax algorithm with alpha-beta pruning. It
            explores the game tree in the minimizing phase, considering possible actions and recursively evaluating the resulting
            states through the max_value_with_alpha_beta_pruning function. The alpha-beta pruning optimizes the search process
            by pruning branches that cannot impact the final decision.

            The function returns a tuple containing the minimum evaluation score and the corresponding optimal action.
            If the game is in a terminal state, the evaluation score is the player's score, and the action is set to None.

            The get_possible_actions_without_those_where_player_attacks_himself function is utilized to filter out actions
            where the player attacks himself, ensuring that only valid and strategic moves are considered.

            The alpha and beta parameters are used to maintain bounds for alpha-beta pruning. If the current best score is
            less than or equal to alpha, pruning occurs, and the function returns the current best score and action.
        """

        if current_state.is_done():
            return current_state.get_player_score(self), None

        v_star, m_star = inf, None

        # Exclude actions where the player attacks himself.
        filtered_possible_actions = self.get_possible_actions_without_those_where_player_attacks_himself(current_state, self)

        for a in filtered_possible_actions:
            s_prime = a.get_next_game_state()
            v, _ = self.max_value_with_alpha_beta_pruning(s_prime, alpha, beta)

            if v < v_star:
                v_star = v
                m_star = a
                beta = min(beta, v_star)

            if v_star <= alpha:
                return v_star, m_star

        return v_star, m_star

    ############################################################################################################
    #              ALGORITHM 3: HEURISTIC MINIMAX SEARCH (TYPE A STRATEGY + ALPHA BETA PRUNING)                #
    ############################################################################################################

    def heuristic_minimax(self, current_state: GameState) -> tuple[float, Action]:
        """
            Executes the Heuristic Minimax search algorithm with type A strategy and alpha-beta pruning to determine the optimal move.

            Parameters:
            - current_state (GameState): The current state of the game.

            Returns:
            tuple[float, Action]: A tuple containing the evaluation score and the optimal action determined by the Heuristic Minimax
            algorithm with type A strategy and alpha-beta pruning.

            Notes:
            The Heuristic Minimax search algorithm with type A strategy and alpha-beta pruning integrates heuristics for
            evaluating leaves at a fixed depth (type A strategy) to the classic Minimax algorithm. The alpha-beta pruning optimizes
            the search process by pruning branches that cannot impact the final decision, reducing computational complexity.

            The algorithm is implemented through the heuristic_max_value function, which recursively explores the game tree while
            maintaining alpha-beta bounds and considering heuristic evaluations at a fixed depth.

            The function returns a tuple containing the evaluation score and the optimal action determined by the Heuristic Minimax
            algorithm with type A strategy and alpha-beta pruning.
        """

        return self.heuristic_max_value(current_state, -inf, inf, self.max_depth)

    def heuristic_max_value(self, current_state: GameState, alpha: float, beta: float, depth: int) -> tuple[float, Optional[Action]]:
        """
            Executes the maximizing phase of the Heuristic Minimax algorithm with type A strategy and alpha-beta pruning.

            Parameters:
            - current_state (GameState): The current state of the game.
            - alpha (float): The alpha parameter for alpha-beta pruning.
            - beta (float): The beta parameter for alpha-beta pruning.
            - depth (int): The remaining depth to explore in the game tree.

            Returns:
            tuple[float, Optional[Action]]: A tuple containing the maximum evaluation score and the corresponding optimal action.
            If the game is in a terminal state, the evaluation score is the player's score, and the action is set to None.
            If the specified depth is reached, the heuristic evaluation is performed using the abalone_heuristic_version_1 function.

            Notes:
            The heuristic_max_value function is a key component of the Heuristic Minimax algorithm with type A strategy and
            alpha-beta pruning. It explores the game tree in the maximizing phase, considering possible actions and recursively
            evaluating the resulting states through the heuristic_min_value function. The alpha-beta pruning optimizes the search
            process by pruning branches that cannot impact the final decision.

            If the specified depth is reached, the heuristic evaluation is performed using the abalone_heuristic_version_1 function,
            providing a fixed-depth heuristic evaluation for leaves.

            The function returns a tuple containing the maximum evaluation score and the corresponding optimal action.
            If the game is in a terminal state, the evaluation score is the player's score, and the action is set to None.
        """

        if current_state.is_done():
            return current_state.get_player_score(self), None

        if not depth:
            return self.abalone_heuristic_version_1(current_state, self, 'Max'), None

        v_star, m_star = -inf, None

        # Exclude actions where the player attacks himself.
        filtered_possible_actions = self.get_possible_actions_without_those_where_player_attacks_himself(current_state, self)

        for a in filtered_possible_actions:
            s_prime = a.get_next_game_state()
            v, _ = self.heuristic_min_value(s_prime, alpha, beta, depth - 1)

            if v > v_star:
                v_star = v
                m_star = a
                alpha = max(alpha, v_star)

            if v_star >= beta:
                return v_star, m_star

        return v_star, m_star

    def heuristic_min_value(self, current_state: GameState, alpha: float, beta: float, depth: int) -> tuple[float, Optional[Action]]:
        """
            Executes the minimizing phase of the Heuristic Minimax algorithm with type A strategy and alpha-beta pruning.

            Parameters:
            - current_state (GameState): The current state of the game.
            - alpha (float): The alpha parameter for alpha-beta pruning.
            - beta (float): The beta parameter for alpha-beta pruning.
            - depth (int): The remaining depth to explore in the game tree.

            Returns:
            tuple[float, Optional[Action]]: A tuple containing the minimum evaluation score and the corresponding optimal action.
            If the game is in a terminal state, the evaluation score is the player's score, and the action is set to None.
            If the specified depth is reached, the heuristic evaluation is performed using the abalone_heuristic_version_1 function.

            Notes:
            The heuristic_min_value function is a key component of the Heuristic Minimax algorithm with type A strategy and
            alpha-beta pruning. It explores the game tree in the minimizing phase, considering possible actions and recursively
            evaluating the resulting states through the heuristic_max_value function. The alpha-beta pruning optimizes the search
            process by pruning branches that cannot impact the final decision.

            If the specified depth is reached, the heuristic evaluation is performed using the abalone_heuristic_version_1 function,
            providing a fixed-depth heuristic evaluation for leaves.

            The function returns a tuple containing the minimum evaluation score and the corresponding optimal action.
            If the game is in a terminal state, the evaluation score is the player's score, and the action is set to None.
        """

        if current_state.is_done():
            return current_state.get_player_score(self), None

        if not depth:
            return self.abalone_heuristic_version_1(current_state, self, 'Min'), None

        v_star, m_star = inf, None

        # Exclude actions where the player attacks himself.
        filtered_possible_actions = self.get_possible_actions_without_those_where_player_attacks_himself(current_state, self)

        for a in filtered_possible_actions:
            s_prime = a.get_next_game_state()
            v, _ = self.heuristic_max_value(s_prime, alpha, beta, depth - 1)

            if v < v_star:
                v_star = v
                m_star = a
                beta = min(beta, v_star)

            if v_star <= alpha:
                return v_star, m_star

        return v_star, m_star

    def abalone_heuristic_version_1(self, current_state: GameState, player: Player, max_or_min: Literal['Max', 'Min']) -> float:
        """
            Evaluates the heuristic score for a given player in the Abalone game.

            Parameters:
            - current_state (GameState): The current state of the game.
            - player (Player): The player for whom the heuristic score is being evaluated.
            - max_or_min (Literal['Max', 'Min']): Indicates whether the heuristic score is for the maximizing ('Max') or minimizing ('Min') player.

            Returns:
            float: The heuristic score for the specified player based on the given state and strategy.

            Notes:
            The abalone_heuristic_version_1 function calculates a heuristic score for a player in the Abalone game. The score
            is influenced by various factors, including being part of a cluster, position on the board edges, proximity to the
            center, and mobility. The bonuses and penalties are applied based on the specified player and strategy (maximizing or minimizing).

            The function returns a float representing the heuristic score for the specified player.
        """

        score = current_state.get_player_score(player)

        if max_or_min == 'Max':
            # Applying bonuses and penalties to the current player (Max).
            score += self.get_bonus_for_being_part_of_a_cluster(current_state, player)
            score += self.get_penalty_for_being_on_edges(current_state, player)
            score += self.get_bonus_for_being_close_to_center(current_state, player)
            score += self.get_bonus_for_mobility(current_state)
        else:
            # Applying bonuses and penalties to the current player (Min).
            score -= self.get_bonus_for_being_part_of_a_cluster(current_state, player)
            score -= self.get_penalty_for_being_on_edges(current_state, player)
            score -= self.get_bonus_for_being_close_to_center(current_state, player)
            score -= self.get_bonus_for_mobility(current_state)

        return score

    @staticmethod
    def get_bonus_for_being_part_of_a_cluster(current_state: GameState, player: Player) -> int:
        """
            Calculates the bonus for being part of a cluster for a specified player in the Abalone game.

            Parameters:
            - current_state (GameState): The current state of the game.
            - player (Player): The player for whom the bonus is calculated.

            Returns:
            int: The bonus for being part of a cluster based on the specified player and state.

            Raises:
            TypeError: If the provided current_state is not an instance of GameStateAbalone.

            Notes:
            The get_bonus_for_being_part_of_a_cluster function calculates the bonus for a player being part of a cluster in the
            Abalone game. It iterates through the marbles on the board, identifies the player's marbles, and evaluates the cluster
            size based on neighboring marbles of the same type.

            The function returns an integer representing the bonus for being part of a cluster.
        """

        if not isinstance(current_state, GameStateAbalone):
            raise TypeError(f"Expected {GameStateAbalone.__name__} instance, got {current_state.__class__.__name__} instead.")

        bonus_for_being_part_of_a_cluster = 0

        for marble in current_state.get_rep().get_env().items():
            piece_coordinates, piece = marble

            if isinstance(piece, Piece) and piece.owner_id == player.get_id():
                cluster_size = 0
                neighbors = current_state.get_neighbours(*piece_coordinates)

                for neighbor in neighbors.values():
                    if neighbor[0] == piece.piece_type:
                        cluster_size += 1

                bonus_for_being_part_of_a_cluster += cluster_size

        return bonus_for_being_part_of_a_cluster

    @staticmethod
    def get_penalty_for_being_on_edges(current_state: GameState, player: Player) -> int:
        """
            Calculates the penalty for being on the edges for a specified player in the Abalone game.

            Parameters:
            - current_state (GameState): The current state of the game.
            - player (Player): The player for whom the penalty is calculated.

            Returns:
            int: The penalty for being on the edges based on the specified player and state.

            Raises:
            TypeError: If the provided current_state is not an instance of GameStateAbalone.

            Notes:
            The get_penalty_for_being_on_edges function calculates the penalty for a player being on the edges of the Abalone board.
            It iterates through the marbles on the board, identifies the player's marbles, and evaluates the number of neighboring
            marbles located on the edges.

            The function returns an integer representing the penalty for being on the edges.
        """

        if not isinstance(current_state, GameStateAbalone):
            raise TypeError(f"Expected {GameStateAbalone.__name__} instance, got {current_state.__class__.__name__} instead.")

        penalty_for_being_on_edges = 0

        for marble in current_state.get_rep().get_env().items():
            piece_coordinates, piece = marble

            if isinstance(piece, Piece) and piece.owner_id == player.get_id():
                number_of_edges = 0
                neighbors = current_state.get_neighbours(*piece_coordinates)

                for neighbor in neighbors.values():
                    if neighbor[0] == 'OUTSIDE':
                        number_of_edges += 1

                penalty_for_being_on_edges -= number_of_edges

        return penalty_for_being_on_edges

    @staticmethod
    def get_bonus_for_mobility(current_state: GameState) -> int:
        """
            Calculates the bonus for mobility based on the number of possible actions in the Abalone game.

            Parameters:
            - current_state (GameState): The current state of the game.

            Returns:
            int: The bonus for mobility based on the number of possible actions.

            Raises:
            TypeError: If the provided current_state is not an instance of GameState.

            Notes:
            The get_bonus_for_mobility function calculates the bonus for mobility in the Abalone game. It considers the number of
            possible actions that can be taken in the current state as an indicator of increased mobility.

            The function returns an integer representing the bonus for mobility.
        """

        return len(current_state.get_possible_actions())

    @staticmethod
    def get_bonus_for_being_close_to_center(current_state: GameState, player: Player) -> int:
        """
            Calculates the bonus for being close to the center for a specified player in the Abalone game.

            Parameters:
            - current_state (GameState): The current state of the game.
            - player (Player): The player for whom the bonus is calculated.

            Returns:
            int: The bonus for being close to the center based on the specified player and state.

            Raises:
            TypeError: If the provided current_state is not an instance of GameStateAbalone or board_abalone is not an instance
            of BoardAbalone or player is not an instance of PlayerAbalone.

            Notes:
            The get_bonus_for_being_close_to_center function calculates the bonus for a player being close to the center in the
            Abalone game. It considers the Manhattan distance of the player's marbles to the center of the board.

            The function returns an integer representing the bonus for being close to the center.
        """

        board_abalone = current_state.get_rep()

        if not isinstance(board_abalone, BoardAbalone):
            raise TypeError(f"Expected {BoardAbalone.__name__} instance, got {board_abalone.__class__.__name__} instead.")

        grid_abalone = board_abalone.get_grid()
        bonus_for_being_close_to_center = 0
        center = (len(grid_abalone[0]) // 2, len(grid_abalone) // 2)

        if not isinstance(player, PlayerAbalone):
            raise TypeError(f"Expected {PlayerAbalone.__name__} instance, got {player.__class__.__name__} instead.")

        for i, row in enumerate(grid_abalone):
            for j, col in enumerate(row):
                if col == player.get_piece_type():
                    bonus_for_being_close_to_center -= abs(center[0] - i) + abs(center[1] - j)

        return bonus_for_being_close_to_center

    ###########################################################################################################################
    #       ALGORITHM 4: IMPROVED HEURISTIC MINIMAX SEARCH (TRANSPOSITION TABLE + TYPE A STRATEGY + ALPHA BETA PRUNING)       #
    ###########################################################################################################################

    @staticmethod
    def generate_cell_hash(current_state: GameState) -> dict[str, int]:
        """
            Generates a hash map for cells in the Abalone game grid.

            Parameters:
            - current_state (GameState): The current state of the game.

            Returns:
            dict[str, int]: A dictionary where each key is a string representation of a cell (including indices and cell value),
            and the corresponding value is a random 64-bit integer.

            Raises:
            TypeError: If the provided current_state is not an instance of GameStateAbalone or board_abalone is not an instance
            of BoardAbalone.

            Notes:
            The generate_cell_hash function generates a hash map for cells in the Abalone game grid. It assigns a unique random
            64-bit integer to each cell in the grid, identified by its row index, column index, and cell value.

            The function returns a dictionary containing string representations of cell positions as keys and corresponding random
            64-bit integers as values.
        """

        board_abalone = current_state.get_rep()

        if not isinstance(board_abalone, BoardAbalone):
            raise TypeError(f"Expected {BoardAbalone.__name__} instance, got {board_abalone.__class__.__name__} instead.")

        grid_abalone = board_abalone.get_grid()
        cell_hash = {}

        for i, row in enumerate(grid_abalone):
            for j, _ in enumerate(row):
                for cell_value in [0, 'B', 'W', 3]:
                    cell_hash[f'{i}, {j}, {cell_value}'] = random.getrandbits(64)

        return cell_hash

    @staticmethod
    def hash_board_abalone(cell_hash: dict[str, int], current_state: GameState) -> int:
        """
            Computes the hash key for the Abalone game board based on the provided cell hash map.

            Parameters:
            - cell_hash (dict[str, int]): A dictionary mapping cell positions to random 64-bit integers.
            - current_state (GameState): The current state of the game.

            Returns:
            int: The computed hash key for the Abalone game board.

            Raises:
            TypeError: If the provided current_state is not an instance of GameStateAbalone or board_abalone is not an instance
            of BoardAbalone.

            Notes:
            The hash_board_abalone function computes the hash key for the Abalone game board based on the provided cell hash map.
            It iterates through the cells in the game grid, XORing the corresponding random 64-bit integers from the cell hash map.

            The function returns an integer representing the computed hash key for the Abalone game board.
        """

        board_abalone = current_state.get_rep()

        if not isinstance(board_abalone, BoardAbalone):
            raise TypeError(f"Expected {BoardAbalone.__name__} instance, got {board_abalone.__class__.__name__} instead.")

        grid_abalone = board_abalone.get_grid()
        hash_key = 0

        for i, row in enumerate(grid_abalone):
            for j, col in enumerate(row):
                hash_key ^= cell_hash.get(f'{i}, {j}, {col}', 0)

        return hash_key

    def save_transposition_table(self, hash_key: int, value: float, action: Action, depth: int, node_type: Literal['EXACT', 'LOWER_BOUND', 'UPPER_BOUND']) -> None:
        """
            Saves an entry in the transposition table.

            Parameters:
            - hash_key (int): The hash key associated with the current state.
            - value (float): The minimax value associated with the current state.
            - action (Action): The action that led to the current state.
            - depth (int): The depth in the game tree at which the minimax value was computed.
            - node_type (Literal['EXACT', 'LOWER_BOUND', 'UPPER_BOUND']): The type of node in the minimax search tree.

            Returns:
            None

            Notes:
            The save_transposition_table function saves an entry in the transposition table. The entry includes the hash key,
            minimax value, action, depth, and node type. This information can be used for transposition table lookup during
            subsequent searches to avoid redundant computations.

            This method is typically called during the minimax search algorithm.
        """

        self.transposition_table[hash_key] = {'value': value, 'action': action, 'depth': depth, 'type': node_type}

    def get_transposition_table(self, hash_key: int) -> dict:
        """
            Retrieves an entry from the transposition table based on the provided hash key.

            Parameters:
            - hash_key (int): The hash key associated with the state to look up.

            Returns:
            dict or None: A dictionary containing information stored in the transposition table for the specified hash key.
            Returns None if the hash key is not found in the transposition table.

            Notes:
            The get_transposition_table function retrieves an entry from the transposition table based on the provided hash key.
            The returned dictionary includes information such as the minimax value, action, depth, and node type associated with
            the state.

            This method is typically called during the minimax search algorithm to check if a previously computed value is
            available in the transposition table.

        """

        return self.transposition_table.get(hash_key)

    def handle_transposition_table(self, current_state_hash: int, alpha: float, beta: float, depth: int):
        """
            Handles transposition table lookup and updates during the minimax search.

            Parameters:
            - current_state_hash (int): The hash key associated with the current state.
            - alpha (float): The alpha value for alpha-beta pruning.
            - beta (float): The beta value for alpha-beta pruning.
            - depth (int): The current depth in the game tree.

            Returns:
            tuple[float, Action, None, None] or tuple[None, None, float, float]: A tuple containing the minimax value, action,
            and potentially updated alpha and beta values. If a relevant entry is found in the transposition table, the first
            tuple format is returned with the corresponding values. Otherwise, the second tuple format is returned with None
            values for minimax value and action, and potentially updated alpha and beta values.

            Notes:
            The handle_transposition_table function handles transposition table lookup and updates during the minimax search. It
            checks if a relevant entry is present in the transposition table for the current state.

            This method is typically called during the minimax search algorithm to check for cached values in the transposition
            table and to determine whether pruning is possible.

        """

        transposition_table_entry = self.get_transposition_table(current_state_hash)

        if transposition_table_entry and transposition_table_entry['depth'] >= depth:
            if transposition_table_entry['type'] == 'EXACT':
                return transposition_table_entry['value'], transposition_table_entry['action'], None, None
            elif transposition_table_entry['type'] == 'LOWER_BOUND':
                alpha = max(alpha, transposition_table_entry['value'])
            elif transposition_table_entry['type'] == 'UPPER_BOUND':
                beta = min(beta, transposition_table_entry['value'])

            if alpha >= beta:
                return transposition_table_entry['value'], transposition_table_entry['action'], None, None

        return None, None, alpha, beta

    def heuristic_minimax_with_translation_table(self, current_state: GameState) -> tuple[float, Action]:
        """
            Performs an improved heuristic minimax search with a transposition table, type A strategy, and alpha-beta pruning.

            Parameters:
            - current_state (GameState): The current state of the game.

            Returns:
            tuple[float, Action]: A tuple containing the minimax value and the corresponding action.

            Notes:
            The heuristic_minimax_with_translation_table function performs an improved heuristic minimax search with a
            transposition table, type A strategy, and alpha-beta pruning. It uses a pre-generated cell hash for efficient
            computation and a transposition table to cache and retrieve previously computed values, reducing redundant
            computations.

            This method is part of the improved heuristic minimax algorithm (Algorithm 4) and is typically called to determine the
            best move in the Abalone game.

        """

        if not self.cell_hash:
            self.cell_hash = self.generate_cell_hash(current_state)

        if not self.transposition_table:
            self.transposition_table = {}

        return self.heuristic_max_value_with_translation_table(current_state, -inf, inf, self.max_depth)

    def heuristic_max_value_with_translation_table(self, current_state: GameState, alpha: float, beta: float, depth: int) -> tuple[float, Optional[Action]]:
        """
            Computes the maximized heuristic value for the given state using a transposition table, type A strategy, and alpha-beta pruning.

            Parameters:
            - current_state (GameState): The current state of the game.
            - alpha (float): The alpha value for alpha-beta pruning.
            - beta (float): The beta value for alpha-beta pruning.
            - depth (int): The current depth in the game tree.

            Returns:
            tuple[float, Optional[Action]]: A tuple containing the maximized heuristic value and the corresponding action.
            If the depth is 0 or the game is done, it returns the heuristic value and None for the action.

            Notes:
            The heuristic_max_value_with_translation_table function computes the maximized heuristic value for the given state using
            a transposition table, type A strategy, and alpha-beta pruning. It caches and retrieves previously computed values from
            the transposition table to avoid redundant computations.

            This method is typically called during the improved heuristic minimax search algorithm (Algorithm 4).

        """

        current_state_hash = self.hash_board_abalone(self.cell_hash, current_state)
        stored_value, stored_action, a, b = self.handle_transposition_table(current_state_hash, alpha, beta, depth)

        if stored_value is not None:
            if depth == self.max_depth:
                for possible_action in current_state.get_possible_actions():
                    if stored_action.next_game_state == possible_action.next_game_state:
                        return stored_value, possible_action
            else:
                return stored_value, stored_action

        alpha, beta = a, b

        if current_state.is_done():
            return current_state.get_player_score(self), None

        if not depth:
            return self.abalone_heuristic_version_1(current_state, self, 'Max'), None

        v_star, m_star = -inf, None

        # Exclude actions where the player attacks himself.
        filtered_possible_actions = self.get_possible_actions_without_those_where_player_attacks_himself(current_state, self)

        for a in filtered_possible_actions:
            s_prime = a.get_next_game_state()
            v, _ = self.heuristic_min_value_with_translation_table(s_prime, alpha, beta, depth - 1)

            if v > v_star:
                v_star = v
                m_star = a
                alpha = max(alpha, v_star)

            if v_star >= beta:
                self.save_transposition_table(current_state_hash, v_star, m_star, depth, 'LOWER_BOUND')
                return v_star, m_star

        self.save_transposition_table(current_state_hash, v_star, m_star, depth, 'EXACT')
        return v_star, m_star

    def heuristic_min_value_with_translation_table(self, current_state: GameState, alpha: float, beta: float, depth: int) -> tuple[float, Optional[Action]]:
        """
            Computes the minimized heuristic value for the given state using a transposition table, type A strategy, and alpha-beta pruning.

            Parameters:
            - current_state (GameState): The current state of the game.
            - alpha (float): The alpha value for alpha-beta pruning.
            - beta (float): The beta value for alpha-beta pruning.
            - depth (int): The current depth in the game tree.

            Returns:
            tuple[float, Optional[Action]]: A tuple containing the minimized heuristic value and the corresponding action.
            If the depth is 0 or the game is done, it returns the heuristic value and None for the action.

            Notes:
            The heuristic_min_value_with_translation_table function computes the minimized heuristic value for the given state using
            a transposition table, type A strategy, and alpha-beta pruning. It caches and retrieves previously computed values from
            the transposition table to avoid redundant computations.

            This method is typically called during the improved heuristic minimax search algorithm (Algorithm 4).

        """

        current_state_hash = self.hash_board_abalone(self.cell_hash, current_state)
        stored_value, stored_action, a, b = self.handle_transposition_table(current_state_hash, alpha, beta, depth)

        if stored_value is not None:
            if depth == self.max_depth:
                for possible_action in current_state.get_possible_actions():
                    if stored_action.next_game_state == possible_action.next_game_state:
                        return stored_value, possible_action
            else:
                return stored_value, stored_action

        alpha, beta = a, b

        if current_state.is_done():
            return current_state.get_player_score(self), None

        if not depth:
            return self.abalone_heuristic_version_1(current_state, self, 'Min'), None

        v_star, m_star = inf, None

        # Exclude actions where the player attacks himself.
        filtered_possible_actions = self.get_possible_actions_without_those_where_player_attacks_himself(current_state, self)

        for a in filtered_possible_actions:
            s_prime = a.get_next_game_state()
            v, _ = self.heuristic_max_value_with_translation_table(s_prime, alpha, beta, depth - 1)

            if v < v_star:
                v_star = v
                m_star = a
                beta = min(beta, v_star)

            if v_star <= alpha:
                self.save_transposition_table(current_state_hash, v_star, m_star, depth, 'UPPER_BOUND')
                return v_star, m_star

        self.save_transposition_table(current_state_hash, v_star, m_star, depth, 'EXACT')
        return v_star, m_star

    def to_json(self) -> dict:
        return {i: j for i, j in self.__dict__.items() if i not in ['timer', 'max_depth', 'cell_hash', 'transposition_table']}

    ###################################################################################################################################
    #           ALGORITHM 5: IMPROVED HEURISTIC MINIMAX SEARCH ( TYPE A STRATEGY + TYPE B STRATEGY + ALPHA BETA PRUNING)              #
    ###################################################################################################################################

    def heuristic_minimax_with_type_b_strategy(self, current_state: GameState) -> tuple[float, Action]:
        """
            Performs an improved heuristic minimax search with both type A and type B strategies, and alpha-beta pruning.

            Parameters:
            - current_state (GameState): The current state of the game.

            Returns:
            tuple[float, Action]: A tuple containing the minimax value and the corresponding action.

            Notes:
            The heuristic_minimax_with_type_b_strategy function performs an improved heuristic minimax search with both type A and
            type B strategies, and alpha-beta pruning. It considers both fixed depth and fixed width evaluation strategies to make
            informed decisions during the Abalone game.

            This method is part of the improved heuristic minimax algorithm (Algorithm 5) and is typically called to determine the
            best move in the Abalone game.

        """

        return self.heuristic_max_value_with_type_b_strategy(current_state, -inf, inf, self.max_depth, self.max_width)

    def heuristic_max_value_with_type_b_strategy(self, current_state: GameState, alpha: float, beta: float, depth: int, width: int) -> tuple[float, Optional[Action]]:
        """
            Computes the maximized heuristic value for the given state using both type A and type B strategies, and alpha-beta pruning.

            Parameters:
            - current_state (GameState): The current state of the game.
            - alpha (float): The alpha value for alpha-beta pruning.
            - beta (float): The beta value for alpha-beta pruning.
            - depth (int): The current depth in the game tree.
            - width (int): The fixed width for evaluating leaves in the game tree.

            Returns:
            tuple[float, Optional[Action]]: A tuple containing the maximized heuristic value and the corresponding action.
            If the depth is 0 or the game is done, it returns the heuristic value and None for the action.

            Notes:
            The heuristic_max_value_with_type_b_strategy function computes the maximized heuristic value for the given state using
            both type A and type B strategies, and alpha-beta pruning. It considers the top N possible actions based on the specified
            width, evaluating a subset of actions to improve efficiency (used the improved heuristic version 2).

            This method is typically called during the improved heuristic minimax algorithm (Algorithm 5) when considering type B
            strategy for evaluating leaves in the game tree.

        """

        if current_state.is_done():
            return current_state.get_player_score(self), None

        if not depth:
            return self.abalone_heuristic_version_2(current_state, self, 'Max'), None

        v_star, m_star = -inf, None

        top_n_possible_actions = self.get_top_n_of_possible_actions(current_state, width, self, 'Max')

        for a in top_n_possible_actions:
            s_prime = a.get_next_game_state()
            v, _ = self.heuristic_min_value_with_type_b_strategy(s_prime, alpha, beta, depth - 1, width)

            if v > v_star:
                v_star = v
                m_star = a
                alpha = max(alpha, v_star)

            if v_star >= beta:
                return v_star, m_star

        return v_star, m_star

    def heuristic_min_value_with_type_b_strategy(self, current_state: GameState, alpha: float, beta: float, depth: int, width: int) -> tuple[float, Optional[Action]]:
        """
            Computes the minimized heuristic value for the given state using both type A and type B strategies, and alpha-beta pruning.

            Parameters:
            - current_state (GameState): The current state of the game.
            - alpha (float): The alpha value for alpha-beta pruning.
            - beta (float): The beta value for alpha-beta pruning.
            - depth (int): The current depth in the game tree.
            - width (int): The fixed width for evaluating leaves in the game tree.

            Returns:
            tuple[float, Optional[Action]]: A tuple containing the minimized heuristic value and the corresponding action.
            If the depth is 0 or the game is done, it returns the heuristic value and None for the action.

            Notes:
            The heuristic_min_value_with_type_b_strategy function computes the minimized heuristic value for the given state using
            both type A and type B strategies, and alpha-beta pruning. It considers the top N possible actions based on the specified
            width, evaluating a subset of actions to improve efficiency (used the improved heuristic version 2).

            This method is typically called during the improved heuristic minimax algorithm (Algorithm 5) when considering type B
            strategy for evaluating leaves in the game tree.

        """

        if current_state.is_done():
            return current_state.get_player_score(self), None

        if not depth:
            return self.abalone_heuristic_version_2(current_state, self, 'Min'), None

        v_star, m_star = inf, None

        top_n_possible_actions = self.get_top_n_of_possible_actions(current_state, width, self, 'Min')

        for a in top_n_possible_actions:
            s_prime = a.get_next_game_state()
            v, _ = self.heuristic_max_value_with_type_b_strategy(s_prime, alpha, beta, depth - 1, width)

            if v < v_star:
                v_star = v
                m_star = a
                beta = min(beta, v_star)

            if v_star <= alpha:
                return v_star, m_star

        return v_star, m_star

    def get_top_n_of_possible_actions(self, current_state: GameState, n: int, player: Player, max_or_min: Literal['Max', 'Min']) -> list[Action]:
        """
            Retrieves the top N possible actions based on the specified heuristic and player type.

            Parameters:
            - current_state (GameState): The current state of the game.
            - n (int): The number of top actions to retrieve.
            - player (Player): The player for whom the actions are evaluated.
            - max_or_min (Literal['Max', 'Min']): The type of player ('Max' for maximizing, 'Min' for minimizing).

            Returns:
            list[Action]: A list containing the top N possible actions based on the specified heuristic and player type.

            Notes:
            The get_top_n_of_possible_actions function retrieves the top N possible actions based on the specified heuristic and
            player type. It excludes actions where the player attacks themselves and uses a heap-based algorithm to efficiently
            retrieve the top actions.

            This method is typically called during the improved heuristic minimax algorithm (Algorithm 5) when selecting a subset
            of actions to evaluate in the game tree.

        """

        # Exclude actions where the player attacks himself.
        filtered_possible_actions = self.get_possible_actions_without_those_where_player_attacks_himself(current_state, player)

        if max_or_min == 'Max':
            top_n = heapq.nlargest(n, filtered_possible_actions, key=lambda action: self.abalone_heuristic_version_2(action.get_next_game_state(), player, 'Max'))
        else:
            top_n = heapq.nsmallest(n, filtered_possible_actions, key=lambda action: self.abalone_heuristic_version_2(action.get_next_game_state(), player, 'Min'))

        return top_n

    @staticmethod
    def get_possible_actions_without_those_where_player_attacks_himself(current_state: GameState, player: Player) -> list[Action]:
        """
            Retrieves a list of possible actions without those where the player attacks themselves.

            Parameters:
            - current_state (GameState): The current state of the game.
            - player (Player): The player for whom actions are filtered.

            Returns:
            list[Action]: A list containing possible actions without those where the player attacks themselves.

            Notes:
            The get_possible_actions_without_those_where_player_attacks_himself function filters possible actions based on the
            condition that the player's score in the next state should be greater than or equal to their score in the current state.

            This method is typically used to exclude actions where the player attacks themselves during the abalone game.

        """

        return [action for action in current_state.get_possible_actions() if action.get_next_game_state().get_player_score(player) >= current_state.get_player_score(player)]

    def abalone_heuristic_version_2(self, current_state: GameState, player: Player, max_or_min: Literal['Max', 'Min']) -> float:
        """
            Calculates a heuristic score for the given game state and player using an extended set of bonuses and penalties.

            Parameters:
            - current_state (GameState): The current state of the game.
            - player (Player): The player for whom the heuristic score is calculated.
            - max_or_min (Literal['Max', 'Min']): The type of player ('Max' for maximizing, 'Min' for minimizing).

            Returns:
            float: The heuristic score for the specified player in the given game state.

            Notes:
            The abalone_heuristic_version_2 function calculates a heuristic score for the given game state and player. The
            heuristic includes bonuses and penalties for being part of a cluster, being close to the center, and being on edges. It
            additionally considers the opponent's state to assess the throwing out of opponent's marbles.

            This method is an enhanced version of the original abalone heuristic (abalone_heuristic_version_1) and is utilized in the
            improved heuristic minimax algorithm (Algorithm 5) for evaluating leaves at a fixed depth and selecting the best moves
            at each level of the game tree.

        """

        score = current_state.get_player_score(player)

        if max_or_min == 'Max':
            # Applying bonuses and penalties to the current player (Max).
            score += self.get_bonus_for_being_part_of_a_cluster(current_state, player)
            score += self.get_penalty_for_being_on_edges(current_state, player)
            score += self.get_bonus_for_being_close_to_center(current_state, player)

            # Applying bonuses and penalties to the opponent (Max).
            score -= self.get_bonus_for_being_part_of_a_cluster(current_state, current_state.get_next_player())
            score -= self.get_penalty_for_being_on_edges(current_state, current_state.get_next_player())
            score -= self.get_bonus_for_being_close_to_center(current_state, current_state.get_next_player())

            # Giving significant weight to the situation where the current player can throw out an opponent's marble (Max).
            score -= current_state.get_player_score(current_state.get_next_player()) * 50
        else:
            # Applying bonuses and penalties to the current player (Min).
            score -= self.get_bonus_for_being_part_of_a_cluster(current_state, player)
            score -= self.get_penalty_for_being_on_edges(current_state, player)
            score -= self.get_bonus_for_being_close_to_center(current_state, player)

            # Applying bonuses and penalties to the opponent (Min).
            score += self.get_bonus_for_being_part_of_a_cluster(current_state, current_state.get_next_player())
            score += self.get_penalty_for_being_on_edges(current_state, current_state.get_next_player())
            score += self.get_bonus_for_being_close_to_center(current_state, current_state.get_next_player())

            # Giving significant weight to the situation where the current player can throw out an opponent's marble (Min).
            score += current_state.get_player_score(current_state.get_next_player()) * 50

        return score
