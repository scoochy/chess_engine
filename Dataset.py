import chess
import chess.engine
import random
import numpy


# this function will create our x (board)
def random_board(max_depth=200):
    board = chess.Board()
    depth = random.randrange(0, max_depth)

    for _ in range(depth):
        all_moves = list(board.legal_moves)
        random_move = random.choice(all_moves)
        board.push(random_move)
        if board.is_game_over():
            break

    return board


# this function will create our f(x) (score)
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
    board3d = numpy.zeros((19, 8, 8), dtype=numpy.int8)

    # here we add the pieces's view on the matrix
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    # add attacks and valid moves too
    # so the network knows what is being attacked
    aux = board.turn
    if aux:
        board3d[18] = numpy.ones((8, 8),  dtype=numpy.int)
    else:
        board3d[18] = numpy.zeros((8, 8), dtype=numpy.int)
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = aux
    castling1 = board.has_kingside_castling_rights(chess.WHITE)
    castling2 = board.has_queenside_castling_rights(chess.WHITE)
    castling3 = board.has_kingside_castling_rights(chess.BLACK)
    castling4 = board.has_queenside_castling_rights(chess.BLACK)
    if castling1:
        board3d[14] = numpy.ones((8, 8), dtype=numpy.int)
    if castling2:
        board3d[15] = numpy.ones((8, 8), dtype=numpy.int)
    if castling3:
        board3d[16] = numpy.ones((8, 8), dtype=numpy.int)
    if castling4:
        board3d[17] = numpy.ones((8, 8), dtype=numpy.int)
    return board3d

def data():
    for j in range (10):
        y = []
        x = numpy.zeros((100000, 19, 8, 8), dtype=numpy.int)
        for i in range(100000):
            print(i)
            board = random_board()
            score = stockfish(board, 10)
            if score == None:
                y.append(0)
                x[i] = numpy.zeros((19, 8, 8))
            else:
                y.append(score)
                x[i] = split_dims(board)

        y = numpy.array(y)
        numpy.savez_compressed('games/V3New{}'.format(j), x)
        numpy.savez_compressed('games/V3New{}eval'.format(j), y)

data()