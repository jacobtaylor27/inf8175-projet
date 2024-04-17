import math

from player_abalone import PlayerAbalone
from game_state_abalone import GameStateAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError


class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "bob", time_limit: float=60*15,*args) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type,name,time_limit,*args)


    def compute_action(self, current_state: GameStateAbalone, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """
        return self.alphabeta_search(current_state)[1]
    
    def cutoff_depth(d):
        """A cutoff function that searches to depth d."""
        return lambda current_state, depth: depth > d
    

    def getNearCenterScore(self, game_state: GameStateAbalone):
        other_id = game_state.next_player.get_id()
        current_rep = game_state.get_rep()
        board = current_rep.get_env()

        center_offset_0 = [(8,4)]
        center_offset_1 = [(7,3), (6,4), (7,5), (9,5), (10,4), (9,3)]
        center_offset_2 = [(6,2), (5,3), (4,4), (5,5), (6,6), (8,6), (10,6), (11,5), (12,4), (11,3), (10,2), (8,2)]
        center_offset_3 = [(5,1), (4,2), (3,3), (2,4), (3,5), (4,6), (5,7), (7,7), (9,7), (11,7), (12,6), (13,5), (14,4), (13,3), (12,2), (11,1), (9,1), (7,1)]
        center_offset_4 = [(4,0), (3,1), (2,2), (1,3), (0,4), (1,5), (2,6), (3,7), (4,8), (6,8), (8,8), (10,8), (12,8), (13,7), (14,6), (15,5), (16,4), (15,3), (14,2), (13,1), (12,0), (10,0), (8,0), (6,0)]

        player_pieces = []
        opponent_pieces = []

        for i, j in list(board.keys()):
            coordinate = board.get((i, j), None)
            if coordinate.get_owner_id() == self.id:
                player_pieces.append((i,j))
            if coordinate.get_owner_id() == other_id:
                opponent_pieces.append((i,j))
        
        center_control_weight = 0 # max = 1, min = 0

        total_weight = 0
        for position in player_pieces:
            if position in center_offset_0:
                total_weight += 1
            elif position in center_offset_1:
                total_weight += 0.8
            elif position in center_offset_2:
                total_weight += 0.6
            elif position in center_offset_3:
                total_weight += 0.3
            elif position in center_offset_4:
                total_weight += 0
        center_control_weight = total_weight / len(player_pieces)
        return center_control_weight
    
    def getScore(self, current_state: GameStateAbalone):
        other_id = current_state.next_player.get_id()
        score = current_state.scores[self.id] - current_state.scores[other_id]
        return score

    def heuristic(self, current_state: GameStateAbalone):
        return 0
        score = self.getScore(current_state)
        center = self.getNearCenterScore(current_state)
        weightScore = 0.8
        weightCenter = 0.2
        return weightScore * score + weightCenter * center

    def alphabeta_search(self, game_state: GameStateAbalone, cutoff=cutoff_depth(2)):

        def max_value(current_state: GameStateAbalone, alpha, beta, depth):

            if current_state.is_done() or cutoff(current_state, depth):
                return self.heuristic(current_state), None

            v_star = - infinity
            m_star = None

            possible_actions = list(current_state.get_possible_actions())
            for possible_action in possible_actions:
                new_state = possible_action.get_next_game_state()
                (v, _) = min_value(new_state, alpha, beta, depth + 1)
                if v > v_star:
                    v_star = v
                    m_star = possible_action
                    alpha = max(alpha, v_star)
                if v_star >= beta:
                    return (v_star, m_star)
            return (v_star, m_star)

        def min_value(current_state: GameStateAbalone, alpha, beta, depth):
            
            if current_state.is_done() or cutoff(current_state, depth):
                return self.heuristic(current_state), None

            v_star = infinity
            m_star = None

            possible_actions = list(current_state.get_possible_actions())
            for possible_action in possible_actions:
                new_state = possible_action.get_next_game_state()
                (v, _) = max_value(new_state, alpha, beta, depth + 1)
                if v < v_star:
                    v_star = v
                    m_star = possible_action
                    beta = min(beta, v_star)
                if v_star <= beta:
                    return (v_star, m_star)
            return (v_star, m_star)
        
        next_action = max_value(game_state, -infinity, +infinity, 0)
        return next_action

infinity = math.inf
