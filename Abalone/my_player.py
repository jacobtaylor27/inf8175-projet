from player_abalone import PlayerAbalone
from game_state_abalone import GameStateAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from seahorse.utils.custom_exceptions import MethodNotImplementedError

from math import inf


# ----------- Utils functions ------------ #
def manhattanDist(A: list[int], B: list[int]) -> int:
    """
    Compute the Manhattan distance between two points.

    Args:
        A (list[int]): First point
        B (list[int]): Second point 
    
    Returns:
        int: Manhattan distance between the two points
    """
    mask1 = [(0,2),(1,3),(2,4)]
    mask2 = [(0,4)]
    diff = (abs(B[0] - A[0]),abs(B[1] - A[1]))
    dist = (abs(B[0] - A[0]) + abs(B[1] - A[1]))/2
    if diff in mask1:
        dist += 1
    if diff in mask2:
        dist += 2
    return dist

# ----------- Heuristic function ------------ #
def heuristic(state: GameStateAbalone) -> int:
    """
    Heuristic function for the Abalone game.

    Args:
        state (GameStateAbalone): Current game state

    Returns:
        int: Heuristic value
    """

    player_id = state.next_player.get_id()
    other_id = state.get_players()[0].get_id() if player_id == state.get_players()[1].get_id() else state.get_players()[1].get_id()

    player_score = state.scores[player_id]
    other_player_score = state.scores[other_id]

    return player_score - other_player_score


# ----------- Alpha-Beta search ------------ #

def cut_off(d: int):
    """
    Cutoff function that searches to depth d.

    Args:
        d (int): Depth of the search

    Returns:
        function: Cutoff function
    """
    return lambda state, depth: depth > d

def h_alpha_beta_search(state: GameStateAbalone, player: PlayerAbalone, cutoff=cut_off(2), h=lambda s: 0) -> tuple[int, Action]:
     """
     Search game to determine best action; use alpha-beta pruning.
    
     Args:
          state (GameStateAbalone): Current game state
          cutoff (function, optional): Cutoff function (default is cut_off(6))
          h (function, optional): Heuristic function (default is lambda s: 0)
    
     Returns:
          tuple[int, Action]: Tuple with the best value and the best action
     """
    
     def max_value(state: GameStateAbalone, alpha: int, beta: int, depth: int) -> tuple[int, Action]:
        """
        Max value function.

        Args:
            state (GameStateAbalone): Current game state
            alpha (int): Alpha value
            beta (int): Beta value
            depth (int): Depth of the search

        Returns:
            tuple[int, Action]: Tuple with the best value and the best action
        """
        if state.is_done():
            return state.scores[state.next_player.get_id()], None
        if cutoff(state, depth):
            return h(state), None
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
          """
          Min value function.
    
          Args:
                state (GameStateAbalone): Current game state
                alpha (int): Alpha value
                beta (int): Beta value
                depth (int): Depth of the search
    
          Returns:
                tuple[int, Action]: Tuple with the best value and the best action
          """
          if state.is_done():
            return state.scores[state.next_player.get_id()], None
          if cutoff(state, depth):
            return h(state), None
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
        #TODO

        print("Current state: ", current_state)
        print("self ", self)
        

        if self.get_remaining_time() < 60:
            pass

        return h_alpha_beta_search(current_state, self, h=heuristic)[1]
