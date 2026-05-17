"""
Tic-Tac-Toe AI using Minimax Algorithm with Alpha-Beta Pruning
==============================================================
Author  : Antigravity AI
Project : CODSOFT Internship - Task 2

HOW MINIMAX WORKS:
  Minimax is a backtracking algorithm used in two-player games.
  The AI (maximizer) tries to MAXIMIZE its score, while the
  Human (minimizer) tries to MINIMIZE the AI's score.
  The algorithm recursively simulates every possible game state
  and picks the move that leads to the best guaranteed outcome.

HOW ALPHA-BETA PRUNING WORKS:
  Alpha-Beta Pruning is an optimization on top of Minimax.
  It keeps track of two values:
    - alpha : best score the maximizer (AI) is guaranteed so far
    - beta  : best score the minimizer (Human) is guaranteed so far
  When beta <= alpha, the current branch can NEVER affect the
  final decision, so we "prune" (skip) it entirely. This can
  dramatically reduce the number of nodes evaluated.
"""

import math

# ---------------------------------------------------------------------------
# BOARD SETUP
# ---------------------------------------------------------------------------

# The board is a flat list of 9 elements (indexes 0-8).
# " " = empty, "X" = Human, "O" = AI
def create_board():
    """Return a fresh, empty 3×3 board."""
    return [" "] * 9


def print_board(board):
    """
    Display the board in a readable grid format, e.g.:

        X | O | X
       ---+---+---
          | O |
       ---+---+---
          |   | X
    """
    print()
    for row in range(3):
        # Each row holds 3 cells: indexes row*3, row*3+1, row*3+2
        cells = board[row * 3 : row * 3 + 3]
        print(f"  {cells[0]} | {cells[1]} | {cells[2]}")
        if row < 2:
            print(" ---+---+---")
    print()


def print_position_guide():
    """
    Show a numbered guide so the user knows which number
    corresponds to which board position (1-9).
    """
    guide = [str(i) for i in range(1, 10)]
    print("\n  Position Guide:")
    for row in range(3):
        cells = guide[row * 3 : row * 3 + 3]
        print(f"  {cells[0]} | {cells[1]} | {cells[2]}")
        if row < 2:
            print(" ---+---+---")
    print()


# ---------------------------------------------------------------------------
# GAME LOGIC HELPERS
# ---------------------------------------------------------------------------

# All 8 possible winning combinations (stored as board index triplets)
WIN_COMBINATIONS = [
    [0, 1, 2],  # top row
    [3, 4, 5],  # middle row
    [6, 7, 8],  # bottom row
    [0, 3, 6],  # left column
    [1, 4, 7],  # middle column
    [2, 5, 8],  # right column
    [0, 4, 8],  # diagonal top-left → bottom-right
    [2, 4, 6],  # diagonal top-right → bottom-left
]


def check_winner(board, player):
    """
    Return True if 'player' ("X" or "O") has won the game
    by matching any of the 8 WIN_COMBINATIONS.
    """
    for combo in WIN_COMBINATIONS:
        if all(board[idx] == player for idx in combo):
            return True
    return False


def is_board_full(board):
    """Return True when there are no empty (" ") spots left."""
    return " " not in board


def get_available_moves(board):
    """Return a list of indexes where the board is still empty."""
    return [i for i, cell in enumerate(board) if cell == " "]


# ---------------------------------------------------------------------------
# MINIMAX ALGORITHM WITH ALPHA-BETA PRUNING
# ---------------------------------------------------------------------------

def minimax(board, depth, is_maximizing, alpha, beta):
    """
    Recursively evaluate every possible future game state.

    Parameters
    ----------
    board          : current game board (list of 9)
    depth          : how many moves deep we currently are
    is_maximizing  : True when it's the AI's turn (maximizer),
                     False when it's the Human's turn (minimizer)
    alpha          : best score the AI can guarantee from this point up
    beta           : best score the Human can guarantee from this point up

    Scoring
    -------
    AI wins   →  +10 - depth  (earlier wins score higher)
    Human wins→  depth - 10   (penalises AI for letting human win fast)
    Draw      →   0
    """

    # --- Base cases: check terminal states ---

    if check_winner(board, "O"):      # AI has won
        return 10 - depth

    if check_winner(board, "X"):      # Human has won
        return depth - 10

    if is_board_full(board):          # Draw
        return 0

    # --- Recursive case ---

    if is_maximizing:
        # AI's turn: try every move and pick the HIGHEST score
        best_score = -math.inf

        for move in get_available_moves(board):
            board[move] = "O"                              # AI plays here
            score = minimax(board, depth + 1, False, alpha, beta)
            board[move] = " "                              # undo the move

            best_score = max(best_score, score)
            alpha = max(alpha, score)                      # update alpha

            # Alpha-Beta Pruning: Human already has a better option elsewhere,
            # so this branch will never be chosen → stop evaluating it.
            if beta <= alpha:
                break

        return best_score

    else:
        # Human's turn: try every move and pick the LOWEST score
        best_score = math.inf

        for move in get_available_moves(board):
            board[move] = "X"                              # Human plays here
            score = minimax(board, depth + 1, True, alpha, beta)
            board[move] = " "                              # undo the move

            best_score = min(best_score, score)
            beta = min(beta, score)                        # update beta

            # Alpha-Beta Pruning: AI already has a better option elsewhere,
            # so this branch will never be chosen → stop evaluating it.
            if beta <= alpha:
                break

        return best_score


def get_best_move(board):
    """
    Iterate over all available moves, run minimax for each,
    and return the index that yields the highest score for the AI.
    """
    best_score = -math.inf
    best_move  = None

    for move in get_available_moves(board):
        board[move] = "O"                                  # try AI move
        # Start minimax from depth=0; next turn is Human's (minimizing)
        score = minimax(board, 0, False, -math.inf, math.inf)
        board[move] = " "                                  # undo

        if score > best_score:
            best_score = score
            best_move  = move

    return best_move


# ---------------------------------------------------------------------------
# HUMAN INPUT
# ---------------------------------------------------------------------------

def get_human_move(board):
    """
    Prompt the human for a position (1-9), validate it, and
    return the 0-based index (0-8) of the chosen cell.
    """
    while True:
        try:
            raw = input("  Your move (1-9): ").strip()
            pos = int(raw)

            if pos < 1 or pos > 9:
                print("  ⚠  Please enter a number between 1 and 9.")
                continue

            index = pos - 1          # convert 1-9 → 0-8

            if board[index] != " ":
                print("  ⚠  That spot is already taken! Choose another.")
                continue

            return index

        except ValueError:
            print("  ⚠  Invalid input. Please enter a number (1-9).")


# ---------------------------------------------------------------------------
# GAME FLOW
# ---------------------------------------------------------------------------

def print_banner():
    """Print a welcome banner at the start of every game."""
    print("\n" + "=" * 40)
    print("      TIC-TAC-TOE  —  AI vs YOU")
    print("=" * 40)
    print("  You = X  |  AI = O")
    print("  Try to beat the unbeatable AI!")
    print("=" * 40)


def play_game():
    """Run a single game from start to finish."""
    board = create_board()

    print_banner()
    print_position_guide()

    # Human always goes first
    current_turn = "human"

    while True:
        print_board(board)

        if current_turn == "human":
            print("  --- Your Turn (X) ---")
            move = get_human_move(board)
            board[move] = "X"

            if check_winner(board, "X"):
                print_board(board)
                print("  🎉  Congratulations! You won!\n")
                break

        else:
            print("  --- AI's Turn (O) ---")
            move = get_best_move(board)
            board[move] = "O"
            print(f"  AI played position {move + 1}.")

            if check_winner(board, "O"):
                print_board(board)
                print("  🤖  AI wins! Better luck next time.\n")
                break

        # Check for draw after every move
        if is_board_full(board):
            print_board(board)
            print("  🤝  It's a draw! Well played.\n")
            break

        # Alternate turns
        current_turn = "ai" if current_turn == "human" else "human"


def main():
    """Entry point: keep playing until the user says no."""
    while True:
        play_game()

        while True:
            again = input("  Play again? (yes/no): ").strip().lower()
            if again in ("yes", "y"):
                break
            elif again in ("no", "n"):
                print("\n  Thanks for playing! Goodbye. 👋\n")
                return
            else:
                print("  ⚠  Please type 'yes' or 'no'.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
