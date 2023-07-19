import chess
import chess.engine
import random
import numpy



def random_board(max_depth=20):
    board = chess.Board()
    depth = random.randrange(0, max_depth)

    for _ in range(depth):
        all_moves = list(board.legal_moves)
        random_move = random.choice(all_moves)
        board.push(random_move)
        if board.is_game_over():
            break
    return board

def stockfish(board, depth):
    with chess.engine.SimpleEngine.popen_uci(
            r"C:\Users\Panton\Desktop\stockfish\stockfish_14.1_win_x64_avx2.exe") as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result['score'].white().score()
        return score

squares_index = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}


# example: h3 -> 17
def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]


def split_dims(board):
    # this is the 3d matrix
    board3d = numpy.zeros((12, 8, 8), dtype=numpy.int8)

    # here we add the pieces's view on the matrix
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    return board3d
def data():
    for j in range (30):
        y = []
        x = numpy.zeros((100000, 12, 8, 8), dtype=numpy.int8)
        for i in range(100000):
            print(i)
            board = random_board()
            score = stockfish(board, 10)
            if score == None:
                y.append(0)
                x[i] = numpy.zeros((12, 8, 8))
            else:
                y.append(score)
                x[i] = split_dims(board)

        y = numpy.array(y)
        numpy.savez_compressed('games/EarlyPosition{}'.format(j+1), x)
        numpy.savez_compressed('games/EarlyPosition{}Eval'.format(j+1), y)

data()



