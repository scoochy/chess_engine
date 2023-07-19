import os
import re
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


import chess.engine
from chess.pgn import read_game
import chess
import chess.uci
import chess.pgn
import numpy as np



squares_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

def square_to_index(square):
  letter = chess.square_name(square)
  return 8 - int(letter[1]), squares_index[letter[0]]

def extract_attacks(game):
    if game is None:
        return []
    moves = list(game.mainline_moves())
    if len(moves) < 10:
        return []
    else:
        boardattacks = np.zeros((len(moves)*2, 8, 8), dtype=np.int8)
        board = chess.Board()
        board.turn = chess.WHITE
        for move in board.legal_moves:
            i, j = square_to_index(move.to_square)
            boardattacks[0][i][j] = 1
        board.turn = chess.BLACK
        for move in board.legal_moves:
            i, j = square_to_index(move.to_square)
            boardattacks[1][i][j] = 1
        m = 0
        for h in range(1, len(moves)-1):
            if h % 2 == 1:
                board.turn = chess.WHITE
            if h % 2 == 0:
                board.turn = chess.BLACK
            board.push(moves[m])
            board.turn = chess.WHITE
            for move in board.legal_moves:
                i, j = square_to_index(move.to_square)
                boardattacks[2*h][i][j] = 1
            board.turn = chess.BLACK
            for move in board.legal_moves:
                i, j = square_to_index(move.to_square)
                boardattacks[(2*h)+1][i][j] = 1
            m+=1
        return np.array(boardattacks)

def extract_moves(game):
    # Takes a game from the pgn and creates list of the board state and the next
    # move that was made from that position.  The next move will be our
    # prediction target when we turn this data over to the ConvNN.
    if game is None:
        return []
    positions = list()
    board = chess.Board()
    moves = list(game.mainline_moves())
    for move in moves:
        position, move_code = board.fen(), move.uci()
        positions.append([position, move_code])
        board.push(move)
    return positions


def replace_nums(line):
    # This function cycles through a string which represents one line on the
    # chess board from the FEN notation.  It will then swap out the numbers
    # for an equivalent number of spaces.
    return ''.join(
        [' ' * 8 if h == '8' else ' ' * int(h) if h.isdigit() else '\n' if h == '/' else '' + h for h in line])


def split_fen(fen):
    # Takes the fen string and splits it into its component lines corresponding
    # to lines on the chess board and the game status.
    fen_comps = fen.split(' ', maxsplit=1)
    board = fen_comps[0].split('/')
    status = fen_comps[1]
    board = [replace_nums(line) for line in board]
    return board, status


def list_to_matrix(board_list):
    # Converts a list of strings into a numpy array by first
    # converting each string into a list of its characters.
    pos_list = [list(line) for line in board_list]
    return np.array(pos_list)


def channelize(mat):
    # processes a board into a 8 x 8 x 12 matrix where there is a
    # channel for each type of piece.  1's correspond to white, and
    # -1's correpond to black.
    pc = ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k']
    positions = [np.isin(mat, pc[i]).astype('int') for i in range(12)]

    return np.stack(positions)

def uci_to_coords(uci):
    def conv_alpha_num(alpha):
        num = ord(alpha) - 97
        return num

    # Every UCI is a 4 character code indicated the from and to squares
    fc, fr = uci[0:2]
    tc, tr = uci[2:4]

    return [8 - int(fr), conv_alpha_num(fc)], [8 - int(tr), conv_alpha_num(tc)]


def process_status(status):
    # The last combination of characters in the FEN notation convey some different pieces of information
    # like the player who is to move next, and who can still castle.
    # I have written the code to extract all of the different pieces, but the Agent will only need to know next_to_move.
    splt = status.split(" ")
    next_to_move = splt[0]
    castling = splt[1]
    en_passant = splt[2]
    half_clock = splt[3]
    full_clock = splt[4]
    return next_to_move


def process_game(positions, attacks):
    # Takes a single game from a pgn and produces a dict of dicts which contains
    # the board state, the next player to move, and the what the next move was (the prediction task).
    boards = []
    next_to_move = []
    h = 0
    if len(positions) < 10:
        return [], []
    for position in positions:
        board, status = split_fen(position[0])
        orig, dest = uci_to_coords(position[1])
        arrays = channelize(list_to_matrix(board), )
        attack = np.stack((attacks[2*h], attacks[(2*h)+1]))
        both = np.concatenate((arrays, attack))
        boards.append(both)
        piece_moved = [i for (i, mat) in enumerate(arrays) if
                       (mat[int(orig[0]), int(orig[1])] == 1) | (mat[int(orig[0]), int(orig[1])] == -1)]
        if piece_moved == []:
            piece_moved = -1
        else:
            piece_moved = piece_moved[0]
        next_to_move.append([process_status(status), piece_moved, orig[0], orig[1], dest[0], dest[1]])
        h+=1
    try:
        boards, ntm = np.stack(boards), np.stack(next_to_move)

    except:
        return [], []
    return boards, ntm


def read_and_process(iteration):
    counter = 0
    gm = read_game(pgn)
    attacks = extract_attacks(gm)
    positions = extract_moves(gm)
    boards, next_to_move = process_game(positions, attacks)
    evaluation = stockfish(gm)
    counter += 1
    print(counter)
    # print("".join(["Completed: ", str(iteration),]))
    return boards, next_to_move, evaluation


def wrangle_data_ip(num_games, save_file=False):
    pool = ThreadPool(2)  # Its even shorter than the single threaded version! Well... minus the other function I had to write...
    results = pool.map(read_and_process, range(num_games))  # Runs into a problem which will kill a small percentage of your games.
    pool.close()  # But its totally worth it
    pool.join()  # lol (I'll figure it out eventually...)
    return results


def wrangle_data(num_games=10000, save_file=False):
    # Meta process for data extraction in serial.. See above for parallelized version!
    boards, next_to_move = read_and_process(0)
    for i in range(1, num_games):
        new_boards, new_next_to_move = read_and_process(i)
        boards, next_to_move = np.concatenate((boards, new_boards), axis=0), np.concatenate(
            (next_to_move, new_next_to_move), axis=0)
    if save_file:
        np.savez_compressed('first_{}_games'.format(num_games), results)
    return boards, next_to_move


def ip_results_to_np(results):
    # Splits a list of tuples into two lists.  Also filters out any errors which wrote as []'s.
    boards = [result[0] for result in results if isinstance(result[0], np.ndarray)]
    targets = [result[1] for result in results if isinstance(result[1], np.ndarray)]
    evaluation = [result[2] for result in results if isinstance(result[2], np.ndarray)]
    # Then returns the full lists concatenated together
    return np.concatenate(boards, axis =0), np.concatenate(targets, axis=0), np.concatenate(evaluation, axis=0)

def stockfish(game):
    if game is None:
        return []
    stockfish = []
    moves = list(game.mainline_moves())
    if len(moves) < 10:
        return []
    with chess.engine.SimpleEngine.popen_uci(
        r"C:\Users\Panton\Desktop\stockfish\stockfish_14.1_win_x64_avx2.exe")  as engine: # give correct address of your engine here
        board = chess.Board()
        for move in moves:
            board.push(move)
            evaluation = engine.analyse(board, chess.engine.Limit(depth=10))
            analysis = evaluation["score"]
            score = str(evaluation["score"])
            if '#+' in score:
                stockfish.append(3000-(100*(int(score[2]))))
            elif '#-' in score:
                stockfish.append(-3000+(100*(int(score[2]))))
            else:
                stockfish.append(int(analysis.relative.cp))

    stockfish = np.array(stockfish)
    return stockfish


if __name__ == "__main__":
    with open('games/Lichess.pgn', encoding='latin1') as pgn:
        for i in range (20):
            num_games = 5000
            results = wrangle_data_ip(num_games=num_games, save_file=True)
            boards, targets, evaluation = ip_results_to_np(results)
            np.savez_compressed('games/Lichess{}'.format(i), boards, targets)
            np.savez_compressed('games/Lichess{}eval'.format(i), evaluation)



#def stockfish(positions):
#    stockfish = []
#    while True:
#        game = read_game(pgn)
#        if game is not None:
#            moves = list(game.mainline_moves())
#            engine = chess.engine.SimpleEngine.popen_uci(
#                r"C:\Users\Panton\Desktop\stockfish\stockfish_14.1_win_x64_avx2.exe")  # give correct address of your engine here
#            board = chess.Board()
#            for move in moves:
#                board.push(move)
#                evaluation = engine.analyse(board, chess.engine.Limit(time=0.1))
#                analysis = evaluation["score"]
#                score = str(evaluation["score"])
#                if '#+' in score:
#                    stockfish.append(1)
#                    print('firing1')
#                elif '#-' in score:
#                    stockfish.append(-1)
#                    print('firing2')
#                else:
#                    stockfish.append(float(analysis.relative.cp / 100))
#                    print('firing3')
#       else:
#            break
#    print(stockfish)
#    stockfish = np.stack(stockfish)
#    return stockfish
#analysis = evaluation["score"]
#            score = str(evaluation["score"])
#            if '#+' in score:
#                stockfish.append(100*float(1/(int(score[2:])+1)))
#            elif '#-' in score:
#                stockfish.append(-100*float(1/(int(score[2:])+1)))
#            else:
#                stockfish.append(float(analysis.relative.cp / 100))