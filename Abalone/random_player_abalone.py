import random

from player_abalone import PlayerAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState

from game_state_abalone import GameStateAbalone

class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game that makes random moves.

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
        Function to implement the logic of the player (here random selection of a feasible solution).

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: Randomly selected feasible action
        """
        possible_actions = current_state.get_possible_actions()
        random.seed("seahorse")
        if kwargs:
            pass
        return random.choice(list(possible_actions))
