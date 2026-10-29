"""
AlphaZero on Connect-4 from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_empty_board
import numpy as np

def make_empty_board() -> np.ndarray:
    """Return a 6x7 integer numpy array of zeros representing an empty Connect-4 board."""
    return np.zeros((6, 7), dtype=np.int32)

# Step 2 - column_top_row
import numpy as np

def column_top_row(board : np.ndarray, column : int) -> int:
    """Return the lowest empty row in `column`, or -1 if the column is full."""
    return (board.shape[0] - 1) - int(np.count_nonzero(board[:, column]))

# Step 3 - drop_piece
def drop_piece(board : np.ndarray, column : int, player : int) -> np.ndarray:
    # TODO: place `player` in the lowest empty row of `column` and return the new board
    landing_row : int = column_top_row(board, column)

    if landing_row == -1: raise ValueError('Column is full cannot place mark.') 

    new_board = board.copy()
    new_board[landing_row, column] = player

    return new_board

# Step 4 - column_full
import numpy as np

def column_full(board : np.ndarray, column : int) -> bool:
    """Return True if `column` has no empty rows left."""
    return bool(np.count_nonzero(board[:, column] == 0) == 0)

# Step 5 - valid_moves
def valid_moves(board : np.ndarray) -> np.ndarray:
    # Return a list of column indices that still have at least one empty row
    return np.where(
        np.array([not column_full(board, col) for col in range(7)])
    )[0].tolist()

# Step 6 - four_in_a_row_horizontal
def four_in_a_row_horizontal(board : np.ndarray) -> int:
    # scan every row for four consecutive matching non-zero pieces horizontally

    first : np.ndarray = board[:, :-3]

    wins = (
        (first != 0)
        & (first == board[:, 1:-2])
        & (first == board[:, 2:-1])
        & (first == board[:, 3:])
    )

    return int(first[wins][0]) if np.any(wins) else 0

# Step 7 - four_in_a_row_vertical
def four_in_a_row_vertical(board : np.ndarray) -> int:
    # TODO: scan every column for four consecutive matching non-zero pieces vertically
    first : np.ndarray = board[:-3, :]

    wins = (
        (first != 0)
        & (first == board[1:-2, :])
        & (first == board[2:-1, :])
        & (first == board[3:, :])
    )

    return int(first[wins][0]) if np.any(wins) else 0

# Step 8 - four_in_a_row_diagonal_down_right
def four_in_a_row_diagonal_down_right(board : np.ndarray) -> int:
    # TODO: scan every down-right diagonal of the 6x7 board for four matching non-zero pieces
    first = board[:-3, :-3]

    wins = (
        (first != 0)
        & (first == board[1:-2, 1:-2])
        & (first == board[2:-1, 2:-1])
        & (first == board[3:,   3:])
    )

    return int(first[wins][0]) if np.any(wins) else 0

# Step 9 - four_in_a_row_diagonal_up_right
def four_in_a_row_diagonal_up_right(board : np.ndarray) -> int:
    # Scan every up-right diagonal for four consecutive matching non-zero pieces
    first = board[:-3, 3:]
    
    wins = (
        (first != 0)
        & (first == board[1:-2, 2:-1])
        & (first == board[2:-1, 1:-2])
        & (first == board[3:, :-3])
    )

    return int(first[wins][0]) if np.any(wins) else 0

# Step 10 - check_winner
import numpy as np

def check_winner(board : np.ndarray) -> int:
    """Return 1 or 2 if that player has four in a row, else 0."""
    
    horizontal : int = four_in_a_row_horizontal(board)

    if horizontal != 0:
        return horizontal

    vertical : int = four_in_a_row_vertical(board)

    if vertical != 0:
        return vertical

    down_right : int = four_in_a_row_diagonal_down_right(board)

    if down_right != 0:
        return down_right

    up_right : int = four_in_a_row_diagonal_up_right(board)

    if up_right != 0:
        return up_right
    return 0

# Step 11 - board_is_full
def board_is_full(board : np.ndarray) -> bool:
    # Return True when no column has an empty slot left
    return bool(np.all(board[0, :] != 0))

# Step 12 - is_terminal
def is_terminal(board : np.ndarray) -> tuple[bool, int]:
    # return (done, winner) using check_winner and board_is_full.
    winner : int  = check_winner(board)
    done   : bool = board_is_full(board) or (winner != 0)

    return (done, winner)

# Step 13 - other_player
def other_player(player : int) -> int:
    # return the opponent's player code (1 <-> 2)
    return 2 if player == 1 else 1

# Step 14 - step_env
def step_env(
        board  : np.ndarray, 
        column : int, 
        player : int
    ) -> tuple[np.ndarray, bool, int, int]:
    # Drop piece for player, then return (new_board, done, winner, next_player).
    
    new_board : np.ndarray = drop_piece(board, column, player)

    done, winner = is_terminal(new_board)

    player : int = other_player(player)

    return new_board, done, winner, player

# Step 15 - encode_board
def encode_board(board : np.ndarray, current_player : int) -> np.ndarray:
    """Encode a 6x7 board as a (2, 6, 7) float32 tensor from current_player's view."""
    # Build two binary planes (current player, opponent) and stack them
    otherplayer : int = other_player(current_player)

    encoding : np.ndarray = np.zeros((2, 6, 7), dtype=np.float32)

    encoding[0] = board == current_player
    encoding[1] = board == otherplayer

    return encoding

# Step 16 - board_to_torch_tensor
import torch

def board_to_torch_tensor(board : np.ndarray, current_player : int) -> torch.Tensor:
    # Encode the board and return it as a float32 torch tensor of shape (1, 2, 6, 7).
    return torch.from_numpy(encode_board(board, current_player)).unsqueeze(0)

# Step 17 - init_conv_backbone
import torch.nn as nn

def init_conv_backbone(
        in_channels     : int = 2, 
        hidden_channels : int = 16
    ) -> nn.Sequential:

    # Build a small convolutional backbone preserving the 6x7 spatial shape.
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )

# Step 18 - init_policy_head
import torch
import torch.nn as nn

def init_policy_head(
        hidden_channels : int = 16, 
        num_columns     : int = 7
    ) -> nn.Sequential:
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, num_columns) logits."""
    # Build a small policy head that projects backbone features to column logits
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(hidden_channels * 6 * 7, 64),
        nn.ReLU(),
        nn.Linear(64, num_columns),
    )

# Step 19 - init_value_head
import torch
import torch.nn as nn

def init_value_head(
    hidden_channels : int=16
    ) -> nn.Sequential:
    """Return an nn.Module mapping (B, hidden_channels, 6, 7) -> (B, 1) in (-1, 1)."""
    # Build a value head that collapses backbone features to a single bounded scalar per board.
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(hidden_channels * 6 * 7, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Tanh(),
    )

# Step 20 - build_policy_value_net
import torch
import torch.nn as nn

def build_policy_value_net(
    in_channels     : int = 2, 
    hidden_channels : int = 16, 
    num_columns     : int = 7
    ) -> nn.Module:
    """Compose backbone + policy head + value head into one nn.Module."""
    # Build an nn.Module with backbone, policy_head, value_head attributes
    class AlphaZero(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.backbone    : nn.Sequential = init_conv_backbone(in_channels, hidden_channels)
            self.policy_head : nn.Sequential = init_policy_head(hidden_channels, num_columns)
            self.value_head  : nn.Sequential = init_value_head(hidden_channels)
        
        def forward(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x = self.backbone(x)

            return self.policy_head(x), self.value_head(x)
            
    return AlphaZero()

# Step 21 - policy_value_forward (not yet solved)
# TODO: implement

# Step 22 - action_mask (not yet solved)
# TODO: implement

# Step 23 - masked_policy_logits (not yet solved)
# TODO: implement

# Step 24 - masked_log_softmax (not yet solved)
# TODO: implement

# Step 25 - sample_action_from_policy (not yet solved)
# TODO: implement

# Step 26 - greedy_action_from_policy (not yet solved)
# TODO: implement

# Step 27 - make_mcts_node (not yet solved)
# TODO: implement

# Step 28 - node_q_value (not yet solved)
# TODO: implement

# Step 29 - ucb_score (not yet solved)
# TODO: implement

# Step 30 - select_best_child (not yet solved)
# TODO: implement

# Step 31 - select_leaf (not yet solved)
# TODO: implement

# Step 32 - evaluate_with_network (not yet solved)
# TODO: implement

# Step 33 - expand_node (not yet solved)
# TODO: implement

# Step 34 - backup_value (not yet solved)
# TODO: implement

# Step 35 - run_one_simulation (not yet solved)
# TODO: implement

# Step 36 - run_mcts (not yet solved)
# TODO: implement

# Step 37 - visit_count_policy (not yet solved)
# TODO: implement

# Step 38 - mcts_choose_action (not yet solved)
# TODO: implement

# Step 39 - record_self_play_step (not yet solved)
# TODO: implement

# Step 40 - play_self_play_game (not yet solved)
# TODO: implement

# Step 41 - assign_value_targets (not yet solved)
# TODO: implement

# Step 42 - generate_self_play_batch (not yet solved)
# TODO: implement

# Step 43 - value_loss_mse (not yet solved)
# TODO: implement

# Step 44 - policy_loss_cross_entropy (not yet solved)
# TODO: implement

# Step 45 - l2_regularization_loss (not yet solved)
# TODO: implement

# Step 46 - combined_loss (not yet solved)
# TODO: implement

# Step 47 - encode_batch_states (not yet solved)
# TODO: implement

# Step 48 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 49 - training_step (not yet solved)
# TODO: implement

# Step 50 - training_epoch (not yet solved)
# TODO: implement

# Step 51 - self_play_iteration (not yet solved)
# TODO: implement

# Step 52 - train_loop (not yet solved)
# TODO: implement

# Step 53 - random_policy_action (not yet solved)
# TODO: implement

# Step 54 - greedy_agent_action (not yet solved)
# TODO: implement

# Step 55 - play_one_match (not yet solved)
# TODO: implement

# Step 56 - match_win_rate (not yet solved)
# TODO: implement

# Step 57 - evaluate_against_random (not yet solved)
# TODO: implement

