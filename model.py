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

# Step 21 - policy_value_forward
import torch
import torch.nn as nn

def policy_value_forward(net, encoded_board : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run encoded_board (B,2,6,7) through net and return (logits, value)."""
    # Call the network on the encoded board and return its two outputs
    return net.forward(encoded_board)

# Step 22 - action_mask
import numpy as np

def action_mask(board : np.ndarray) -> np.ndarray:
    legal_moves : np.ndarray = valid_moves(board)

    legal_mask : np.ndarray = np.zeros((7, ), dtype=np.bool)

    legal_mask[legal_moves] = 1

    return legal_mask

# Step 23 - masked_policy_logits
import torch

def masked_policy_logits(
        logits : torch.Tensor, 
        mask   : torch.Tensor,
    ) -> torch.Tensor:
    """
    Set logits at illegal columns to -inf.

    logits: torch.Tensor of shape (..., 7)
    mask:   bool array/tensor of shape (7,), True = legal
    returns: torch.Tensor of same shape as logits
    """
    
    return torch.where(
        torch.as_tensor(mask, dtype=torch.bool),
        logits,
        float('-inf')
    )

# Step 24 - masked_log_softmax
import torch
import torch.nn as nn

def masked_log_softmax(
        logits : torch.Tensor, 
        mask   : torch.Tensor
    ) -> torch.Tensor:
    """
    Log-softmax of logits with illegal columns (mask=False) forced to -inf.
    """

    return nn.functional.log_softmax(
        masked_policy_logits(logits, mask), dim=-1
    )

# Step 25 - sample_action_from_policy
import torch

def sample_action_from_policy(
        logits      : torch.Tensor, 
        mask        : torch.Tensor, 
        temperature : float = 1.0
    ) -> int:
    """
    Sample a legal column from a tempered masked categorical policy.
    """
    # Scale logits
    scaled_masked_logits : torch.Tensor = torch.nn.functional.softmax(masked_policy_logits(
        logits / temperature, mask
    ), dim=0)

    return torch.multinomial(scaled_masked_logits, num_samples=1).item()

# Step 26 - greedy_action_from_policy
import torch

def greedy_action_from_policy(
        logits : torch.Tensor, 
        mask   : torch.Tensor
    ) -> int:
    """
    Return the argmax legal column index from masked policy logits.
    """
    return torch.argmax(
        masked_policy_logits(logits, mask),
        dim=0
    ).item()

# Step 27 - make_mcts_node
def make_mcts_node(
        prior  : float = 0.0, 
        parent : dict | None = None
    ) -> dict:
    # TODO: build a dict with prior, visit_count, value_sum, children, and parent fields.
    return {
        "prior"       : prior,
        "visit_count" : 0,
        "value_sum"   : 0.0,
        "children"    : {},
        "parent"      : parent,
        "is_expanded" : False
    }

# Step 28 - node_q_value
def node_q_value(node : dict) -> float:
    # TODO: return the mean value Q = value_sum / visit_count, or 0.0 if visit_count == 0
    return node['value_sum'] / node['visit_count'] if node['visit_count'] > 0 else 0.0

# Step 29 - ucb_score
import math

def ucb_score(
        parent : dict, 
        child  : dict, 
        c_puct : float = 1.5
    ) -> float:
    # TODO: return Q(child) + c_puct * prior * sqrt(N_parent) / (1 + N_child)
    return node_q_value(child) + c_puct * child['prior'] * math.sqrt(parent['visit_count']) / (1 + child['visit_count'])

# Step 30 - select_best_child
def select_best_child(
        node          : dict, 
        legal_actions : list[int], 
        c_puct        : float = 1.5
    ) -> tuple[int, dict]:
    # TODO: return (action, child) maximizing PUCT among legal children of node.
    best_action, best_child = max(
        (
            (action_id, child)
            for action_id, child in node["children"].items()
            if action_id in legal_actions
        ),
        key=lambda item: ucb_score(node, item[1], c_puct),
    )

    return best_action, best_child

# Step 31 - select_leaf
def select_leaf(
        root   : dict, 
        c_puct : float
    ) -> dict:
    # TODO: walk down the MCTS tree picking the best PUCT child until a non-expanded node is reached
    node : dict = root

    while node.get('is_expanded', False):
        legal_moves : list[int] = list(node['children'].keys())

        _, node = select_best_child(node, legal_moves, c_puct)

    return node

# Step 32 - evaluate_with_network
def evaluate_with_network(
    net,
    state: np.ndarray,
    to_play: int,
) -> tuple[np.ndarray, float]:

    net.eval()

    with torch.no_grad():
        board_encoding = board_to_torch_tensor(state, to_play)
        policy_logits, value = policy_value_forward(net, board_encoding)

        mask = torch.as_tensor(
            action_mask(state),
            dtype=torch.bool,
            device=policy_logits.device,
        )

        log_priors = masked_log_softmax(policy_logits, mask)

        priors = log_priors.exp().squeeze(0).cpu().numpy()
        value = value.item()

        return priors, value

# Step 33 - expand_node
def expand_node(
        node   : dict, 
        priors : np.ndarray,
    ) -> None:
    # TODO: attach a child node for every legal move with the corresponding network prior
    legal_moves : list[int] = valid_moves(node['board'])

    for legal_move in legal_moves:
        # Make the MCTS Node and add it to children with the corresponding prior
        child : dict = make_mcts_node(priors[legal_move], node)

        # Add the board to the node
        child['board'] = drop_piece(
            node['board'].copy(), legal_move, node['to_play']
        )

        child["to_play"] = other_player(node["to_play"])

        node["children"][legal_move] = child

    node['is_expanded'] = True

# Step 34 - backup_value
def backup_value(
        leaf  : dict, 
        value : float
    ) -> None:
    # TODO: walk from leaf up through parents, updating visit_count and value_sum with alternating signs
    if leaf is None:
        return

    leaf['visit_count'] += 1
    leaf['value_sum']   += value

    backup_value(leaf.get('parent', None), -value)

# Step 35 - run_one_simulation
def run_one_simulation(
        root, 
        net, 
        c_puct,
    ) -> None:
    node = select_leaf(root, c_puct)

    done, winner = is_terminal(node["board"])

    if done:
        if winner is None or winner == 0:
            value = 0.0  # Draw
        else:
            value = 1.0 if winner == node["to_play"] else -1.0
    else:
        priors, value = evaluate_with_network(
            net,
            node["board"],
            node["to_play"],
        )
        expand_node(node, priors)

    backup_value(node, value)

# Step 36 - run_mcts
def run_mcts(
        state, 
        to_play, 
        net, 
        num_simulations, 
        c_puct
    ) -> dict:
    # TODO: build a fresh root for (state, to_play) and run num_simulations PUCT simulations
    root : dict = make_mcts_node()
    
    root['board']   = state
    root['to_play'] = to_play

    for _ in range(num_simulations):
        run_one_simulation(root, net, c_puct)
    
    return root

# Step 37 - visit_count_policy
def visit_count_policy(
        root        : dict, 
        temperature : float = 1.0
    ) -> np.ndarray:

    counts = np.zeros(7, dtype=np.float64)
    
    for idx, child in root['children'].items():
        counts[idx] = child['visit_count']

    if counts.sum() == 0:
        return np.full(7, 1.0 / 7, dtype=np.float64)

    if temperature < 1e-3:
        policy = np.zeros(7, dtype=np.float64)
        policy[np.argmax(counts)] = 1.0
        return policy

    scaled = counts / counts.max()
    temp_dist = np.power(scaled, 1.0 / temperature)
    return temp_dist / temp_dist.sum()

# Step 38 - mcts_choose_action
def mcts_choose_action(
        state, 
        to_play, 
        net, 
        num_simulations, 
        c_puct, 
        temperature=1.0
    ) -> tuple[int, np.ndarray]:
    # Run MCTS at the given state and return (action, visit-count policy vector).
    root : dict = run_mcts(state, to_play, net, num_simulations, c_puct)

    policy_distribution : torch.Tensor = torch.from_numpy(visit_count_policy(
        root, temperature
    ))

    mask : torch.Tensor = torch.from_numpy(
        action_mask(state)
    )

    return sample_action_from_policy(
        policy_distribution, mask, temperature
    ), policy_distribution.numpy()

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

