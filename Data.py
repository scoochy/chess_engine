import copy

import numpy

import ChessMain
import GameEngine
import chess
import chess.uci
import chess.pgn
import numpy as np



class LoadData:

    x = []
    bp = []
    bR = []
    bN = []
    bK = []
    bQ = []
    bB = []
    wp = []
    wR = []
    wN = []
    wB = []
    wQ = []
    wK = []

    def square_to_index(self, square):
        squares_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        letter = chess.square_name(square)
        return 8 - int(letter[1]), squares_index[letter[0]]

    def extract_attacks(self, moves, board, wking, wqueen, bking, bqueen):
        boardattacks = []
        castling_rights = []
        gs = GameEngine.GameState()
        board1 = copy.deepcopy(board)
        print(len(moves))
        for i in range(len(moves)):
            gs.currentCastlingRights.wks = wking
            gs.currentCastlingRights.wqs = wqueen
            gs.currentCastlingRights.bks = bking
            gs.currentCastlingRights.bqs = bqueen
            gs.WhiteToMove = True
            gs.board = board1
            gs.makeMove(moves[i])
            if gs.currentCastlingRights.wks == False:
                castling_rights.append(numpy.zeros((8, 8),  dtype=numpy.int))
            else:
                castling_rights.append(numpy.ones((8, 8), dtype=numpy.int))
            if gs.currentCastlingRights.wqs == False:
                castling_rights.append(numpy.zeros((8, 8),  dtype=numpy.int))
            else:
                castling_rights.append(numpy.ones((8, 8), dtype=numpy.int))
            if gs.currentCastlingRights.bks == False:
                castling_rights.append(numpy.zeros((8, 8),  dtype=numpy.int))
            else:
                castling_rights.append(numpy.ones((8, 8), dtype=numpy.int))
            if gs.currentCastlingRights.bqs == False:
                castling_rights.append(numpy.zeros((8, 8),  dtype=numpy.int))
            else:
                castling_rights.append(numpy.ones((8, 8), dtype=numpy.int))
            gs.WhiteToMove = not gs.WhiteToMove
            validmoves = gs.getValidMoves()
            boardattacks.append(validmoves)
            gs.WhiteToMove = not gs.WhiteToMove
            validmoves2 = gs.getValidMoves()
            boardattacks.append(validmoves2)
            board1 = copy.deepcopy(board)
            castling_rights.append(numpy.zeros((8,8), dtype=numpy.int))
        gs.board = board
        arraysize = len(boardattacks)
        convertedattacks = self.convert_attacks(boardattacks, arraysize)
        return convertedattacks, castling_rights


    def convert_attacks(self, attacks, arraysize):
        boardarrays = np.zeros((arraysize, 8, 8), dtype=np.int8)
        for i in range(len(attacks)):
            for j in range(len(attacks[i])):
                boardarrays[i][attacks[i][j].endRow][attacks[i][j].endCol] = 1
        return boardarrays




    def newBoards(self, moves, currentboard, wking, wqueen, bking, bqueen):
        newBoards = []
        validarrays = []
        validMoves = copy.deepcopy(moves)
        for i in range(len(validMoves)):
            board = copy.deepcopy(currentboard)
            board[validMoves[i].startRow][validMoves[i].startCol] = "--"
            board[validMoves[i].endRow][validMoves[i].endCol] = validMoves[i].pieceMoved
            if validMoves[i].isPawnPromotion:
                board[validMoves[i].endRow][validMoves[i].endCol] = validMoves[i].pieceMoved[0] + 'Q'
            if validMoves[i].isEnpassantMove:
                board[validMoves[i].endRow][validMoves[i].endCol] = "--"
            if validMoves[i].isCastleMove:
                if validMoves[i].endCol - validMoves[i].startCol == 2:  # kingside castle
                    board[validMoves[i].endRow][validMoves[i].endCol - 1] = board[validMoves[i].endRow][validMoves[i].endCol + 1]
                    board[validMoves[i].endRow][validMoves[i].endCol + 1] = '--'
                else:  # queenside
                    board[validMoves[i].endRow][validMoves[i].endCol + 1] = board[validMoves[i].endRow][validMoves[i].endCol - 2]
                    board[validMoves[i].endRow][validMoves[i].endCol - 2] = '--'
            newBoards.append(board)
        arrays = self.convertBoard(newBoards)
        attacks, castling_rights = self.extract_attacks(moves, currentboard, wking, wqueen, bking, bqueen)
        for i in range(len(arrays)):
            attack = np.stack((attacks[2*i], attacks[2*i +1]))
            castling = np.stack((castling_rights[5*i], castling_rights[5*i+1], castling_rights[5*i+2], castling_rights[5*i+3], castling_rights[5*i+4]))
            "both = np.concatenate((arrays[i], attack, castling))"
            both = arrays[i]
            validarrays.append(both)
        validarrays = np.stack(validarrays)
        return validarrays

    def convertBoard(self, boardpositions):
        self.positions = len(boardpositions)
        for position in range(self.positions):
            board1 = copy.deepcopy(boardpositions[position])
            for row in range(8):
                for column in range(8):
                    if board1[row][column] == "wp":
                        board1[row][column] = 1
                    else:
                        board1[row][column] = 0
            self.wp.append(board1)

        for position in range(self.positions):
            board2 = copy.deepcopy(boardpositions[position])
            for row in range(8):
                for column in range(8):
                    if board2[row][column] == "wR":
                        board2[row][column] = 1
                    else:
                        board2[row][column] = 0
            self.wR.append(board2)

        for position in range(self.positions):
            board3 = copy.deepcopy(boardpositions[position])
            for row in range(8):
                for column in range(8):
                    if board3[row][column] == "wN":
                        board3[row][column] = 1
                    else:
                        board3[row][column] = 0
            self.wN.append(board3)

        for position in range(self.positions):
            board4 = copy.deepcopy(boardpositions[position])
            for row in range(8):
                for column in range(8):
                    if board4[row][column] == "wB":
                        board4[row][column] = 1
                    else:
                        board4[row][column] = 0
            self.wB.append(board4)

        for position in range(self.positions):
            board5 = copy.deepcopy(boardpositions[position])
            for row in range(8):
                for column in range(8):
                    if board5[row][column] == "wQ":
                        board5[row][column] = 1
                    else:
                        board5[row][column] = 0
            self.wQ.append(board5)

        for position in range(self.positions):
            board6 = copy.deepcopy(boardpositions[position])
            for row in range(8):
                for column in range(8):
                    if board6[row][column] == "wK":
                        board6[row][column] = 1
                    else:
                        board6[row][column] = 0
            self.wK.append(board6)

        for position in range(self.positions):
            board7 = copy.deepcopy(boardpositions[position])
            for row in range(8):
                for column in range(8):
                    if board7[row][column] == "bp":
                        board7[row][column] = 1
                    else:
                        board7[row][column] = 0
            self.bp.append(board7)

        for position in range(self.positions):
            board8 = copy.deepcopy(boardpositions[position])
            for row in range(8):
                for column in range(8):
                    if board8[row][column] == "bR":
                        board8[row][column] = 1
                    else:
                        board8[row][column] = 0
            self.bR.append(board8)


        for position in range(self.positions):
            board9 = copy.deepcopy(boardpositions[position])
            for row in range(8):
                for column in range(8):
                    if board9[row][column] == "bN":
                        board9[row][column] = 1
                    else:
                        board9[row][column] = 0
            self.bN.append(board9)

        for position in range(self.positions):
            board10 = copy.deepcopy(boardpositions[position])
            for row in range(8):
                for column in range(8):
                    if board10[row][column] == "bB":
                        board10[row][column] = 1
                    else:
                        board10[row][column] = 0
            self.bB.append(board10)



        for position in range(self.positions):
            board11 = copy.deepcopy(boardpositions[position])
            for row in range(8):
                for column in range(8):
                    if board11[row][column] == "bQ":
                        board11[row][column] = 1
                    else:
                        board11[row][column] = 0
            self.bQ.append(board11)

        for position in range(self.positions):
            board12 = copy.deepcopy(boardpositions[position])
            for row in range(8):
                for column in range(8):
                    if board12[row][column] == "bK":
                        board12[row][column] = 1
                    else:
                        board12[row][column] = 0
            self.bK.append(board12)

        arrays = []
        for position in range(self.positions):
            arrays.append((self.wp[position], self.wR[position], self.wN[position], self.wB[position], self.wQ[position], self.wK[position],
                                 self.bp[position], self.bR[position], self.bN[position], self.bB[position], self.bQ[position], self.bK[position]))

        return np.stack(arrays)







"""    def castleRights(self, castleRights):

        for position in range(len(castleRights) - 1):
            self.WhiteCastlingRights.append((int(castleRights[position].wks), int(castleRights[position].wqs)))
            self.BlackCastlingRights.append((int(castleRights[position].bks), int(castleRights[position].bqs)))
        self.combine()
"""