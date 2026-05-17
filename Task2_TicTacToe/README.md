# Task 2 - Tic-Tac-Toe AI

A terminal-based Tic-Tac-Toe game where you play against an unbeatable AI. The AI uses the Minimax algorithm with Alpha-Beta Pruning to always find the best possible move. You play as **X**, the AI plays as **O**.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `math` | Provides `math.inf` for infinity values used in the Minimax algorithm |

Part of Python's standard library — no installation needed.

---

## How to Run

No external packages required. Just run the script directly.

```bash
python tictactoe.py
```

---

## Sample Output

```
==================================================
        TIC-TAC-TOE: Human vs AI
     You are X  |  AI is O
==================================================

   1 | 2 | 3
  -----------
   4 | 5 | 6
  -----------
   7 | 8 | 9

Your move (1-9): 5

   1 | 2 | 3
  -----------
   4 | X | 6
  -----------
   7 | 8 | 9

AI is thinking...

   O | 2 | 3
  -----------
   4 | X | 6
  -----------
   7 | 8 | 9

Your move (1-9): 3

   O | 2 | X
  -----------
   4 | X | 6
  -----------
   7 | 8 | 9

AI is thinking...

   O | 2 | X
  -----------
   4 | X | 6
  -----------
   7 | O | 9

Your move (1-9): 7

   O | 2 | X
  -----------
   4 | X | 6
  -----------
   X | O | 9

AI is thinking...

   O | 2 | X
  -----------
   O | X | 6
  -----------
   X | O | O

==================================================
   AI wins! Better luck next time.
==================================================

Play again? (yes/no): no
Thanks for playing!
```
