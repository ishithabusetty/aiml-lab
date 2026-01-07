import copy
import heapq as pq

class Node:
    def __init__(self, board, h, path):
        self.board = board
        self.h = h
        self.path = path

    def __lt__(self, other):
        return self.h < other.h

def heuristic(board, goal):
    diff = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] != 0 and board[i][j] != goal[i][j]:
                diff += 1
    return diff

def findZero(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return i, j

def calculateSteps(board, goal):
    visited = set()
    q = []

    pq.heappush(q, Node(board, heuristic(board, goal), [board]))
    visited.add(str(board))

    while q:
        node = pq.heappop(q)

        if heuristic(node.board, goal) == 0:
            return node.path

        row, col = findZero(node.board)
        moves = [(0,1), (0,-1), (1,0), (-1,0)]

        for dx, dy in moves:
            new_r, new_c = row + dx, col + dy
            if 0 <= new_r < 3 and 0 <= new_c < 3:
                temp = copy.deepcopy(node.board)
                temp[row][col], temp[new_r][new_c] = temp[new_r][new_c], temp[row][col]

                if str(temp) not in visited:
                    visited.add(str(temp))
                    new_path = node.path + [temp]
                    pq.heappush(q, Node(temp, heuristic(temp, goal), new_path))

    return None

# ---------- USER INPUT ----------
print("Enter initial board (row-wise, use 0 for blank):")
board = [list(map(int, input().split())) for _ in range(3)]

print("\nEnter goal board:")
goal = [list(map(int, input().split())) for _ in range(3)]

solution = calculateSteps(board, goal)

# ---------- OUTPUT ----------
if solution:
    print("\nSolution Path:")
    for step, state in enumerate(solution):
        print(f"Step {step}:")
        for row in state:
            print(row)
        print()
else:
    print("No solution found.")
