import random
import chess.pgn
import numpy
import copy
import GameEngine
from chess import pgn
import sys
import tensorflow.keras.models as models
import tensorflow as tf
import keras

#def findRandomMove(validMoves):
    #return validMoves[random.randint(0, len(validMoves) - 1)]


class LoadGames:

    def openGame(self):
        games = []
        AImoves = []
        f = open("games/game2.pgn")
        while True:
            game = chess.pgn.read_game(f)
            if game is not None:
                games.append(game)
                i = (len(games) - 1)
                for move in games[i].mainline_moves():
                    moves = []
                    startRank = (str(move)[0])
                    startFile = (str(move)[1])
                    startSquare = self.getRowsAndCols(startFile, startRank)
                    endRank = (str(move)[2])
                    endFile = (str(move)[3])
                    endSquare = self.getRowsAndCols(endFile, endRank)

                    AImoves.append((startSquare, endSquare))
            else:
                break
        print(len(AImoves))
        return AImoves

    def promotionMoves(self):
        blackPromotion = []
        whitePromotion = []
        f = open("games/repetition.pgn")
        game = chess.pgn.read_game(f)
        for moves in game:
            blackpromote = (str(moves)).split("1=")
            whitepromote = (str(moves)).split("8=")
            if len(blackpromote) > 1:
                for i in range(1, len(blackpromote)):
                    blackPromotion.append(blackpromote[i][0])
            if len(whitepromote) > 1:
                for i in range(1, len(whitepromote)):
                    whitePromotion.append(whitepromote[i][0])

            return blackPromotion, whitePromotion

    def getRowsAndCols(self, r, f):
        ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}
        filesToCols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
        return (ranksToRows[r] , filesToCols[f])

class NeuralNetwork:

    def predict(self, arrays, amount):
        model = tf.keras.models.load_model("EarlyPosition")
        predictions = model.predict(arrays)
        print(predictions)
        return predictions