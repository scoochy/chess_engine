# Main file which will handle input/output of moves and interact with the engine
import copy

import numpy
import pygame as p
import GameEngine
import AI
import Data

p.init()
WIDTH = HEIGHT = 512
DIMENSION = 8
MAX_FPS = 15
SQ_SIZE = HEIGHT // DIMENSION
IMAGES = {}
RANK = 8
FILE = 8
playerOne = True  # If a human is playing white this will be true, ai = false
playerTwo = False  # Same as above but for black

'''
Load the images of the pieces into memory 
'''



def LoadImages():
    pieces = {'bB', 'bK', 'bN', 'bp', 'bQ', 'bR', 'wB', 'wK', 'wN', 'wp', 'wQ', 'wR'}
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load("images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))


def main():
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    gs = GameEngine.GameState()
    validMoves = gs.getValidMoves()
    AImoves = []
    computer = AI.LoadGames()
    AImoves = computer.openGame()
    movenumber1 = 0
    movenumber2 = 1
    moveMade = False
    animate = False
    gameOver = False
    LoadImages()
    running = True
    squareSelected = () #tuple of selected square,  (row, col)
    playerClicks = [] #list of tuples to keep track of piece that will move and destination
    while running:
        humanTurn = (gs.WhiteToMove and playerOne) or (not gs.WhiteToMove and playerTwo)
        if humanTurn:
            for e in p.event.get():
                if e.type == p.QUIT:
                    running = False
                #mouse handler
                elif e.type == p.MOUSEBUTTONDOWN:
                    if not gameOver and humanTurn:
                        location = p.mouse.get_pos()
                        col = location[0]//SQ_SIZE
                        row = location[1]//SQ_SIZE
                        if squareSelected == (row, col): #deselect
                            squareSelected = ()
                            playerClicks = []
                        else:
                            squareSelected = (row, col)
                            playerClicks.append(squareSelected)
                        if len(playerClicks) == 2:
                            move = GameEngine.Move(playerClicks[0], playerClicks[1], gs.board)
                            print(move.getChessNotation())
                            for i in range(len(validMoves)):
                                if move == validMoves[i]:
                                    gs.makeMove(validMoves[i])
                                    moveMade = True
                                    animate = True
                                    squareSelected = ()
                                    playerClicks = []
                            if not moveMade:
                                playerClicks = [squareSelected]
                #key handler
                elif e.type == p.KEYDOWN:
                    if e.key == p.K_z:
                        gs.undoMove()
                        moveMade = True
                        animate = False
                    if e.key == p.K_r: #reset board
                        gs = GameEngine.GameState()
                        validMoves = gs.getValidMoves()
                        squareSelected = ()
                        playerClicks = []
                        moveMade = False
                        animate = False


        #AI move logic
        if not gameOver and not humanTurn and not gs.WhiteToMove:
            if gs.currentCastlingRights.wks == False:
                wkingside = False
            if gs.currentCastlingRights.wqs == False:
                wqueenside = False
            else:
                wkingside, wqueenside = True, True
            if gs.currentCastlingRights.bks == False:
                bkingside = False
            if gs.currentCastlingRights.bqs == False:
                bqueenside = False
            else:
                bkingside, bqueenside = True, True
            processing = Data.LoadData()
            neuralnetwork = AI.NeuralNetwork()
            arrays = processing.newBoards(validMoves, gs.board, wkingside, wqueenside, bkingside, bqueenside)
            predictions = neuralnetwork.predict(arrays, len(validMoves))
            AImove = numpy.argmax(predictions)
            gs.makeMove(validMoves[AImove])
            moveMade = True
            animate = False

        if moveMade:
            if animate:
                pass
                #animateMove(gs.moveLog[-1], screen, gs.board, clock)

            moveMade = False
            animate = False

            validMoves = gs.getValidMoves()


        drawGameState(screen, gs)

        if gs.checkMate:
            if gs.WhiteToMove:
                drawText(screen, 'Black wins by checkmate')
                gs = GameEngine.GameState()
            else:
                drawText(screen, 'White wins by checkmate')
                gs = GameEngine.GameState()



        elif gs.staleMate:
            drawText(screen, 'Draw by stalemate')
            gs = GameEngine.GameState()

        if gs.threefold:
            drawText(screen, 'Draw by repetition')
            gs = GameEngine.GameState()


        clock.tick(MAX_FPS)
        p.display.flip()

"""
#highlight squares at move animation

def highlightSquares(screen, gs, validMoves, squareSelected):
    if squareSelected != ():
        r, c = squareSelected
        if gs.board[r][c][0] == ('w' if gs.WhiteToMove else 'b'): #piece can be moved
            #highlight selected square
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(p.Color('purple'))
            screen.blit(s, (c*SQ_SIZE, r*SQ_SIZE))
            s.fill(p.Color('yellow'))
            for move in validMoves:
                if move.startRow == r and move.startCol == c:
                    screen.blit(s, (SQ_SIZE*move.endCol, SQ_SIZE*move.endRow))

"""

# Draws the board, pieces and anything else according to the current game state
def drawGameState(screen, gs):
    drawBoard(screen)
    #highlightSquares(screen, gs, validMoves, squareSelected)
    drawPieces(screen, gs.board)


def drawBoard(screen):
    global colors
    colors = [p.Color("white"), p.Color(35, 122, 53)]
    for r in range (RANK):
        for c in range (FILE):
            color = colors[((r + c) % 2)]
            p.draw.rect(screen, color, p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def drawPieces(screen, board):
    for r in range (RANK):
        for c in range (FILE):
            piece = board[r][c]
            if piece != "--":
                screen.blit(IMAGES[piece], p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
"""
def animateMove(move, screen, board, clock):
    global colors
    dR = move.endRow - move.startRow
    dC = move.endCol - move.startCol
    framesPerSquare = 5
    frameCount = (abs(dR) + abs(dC)) * framesPerSquare
    for frame in range(frameCount + 1):
        r, c = (move.startRow + dR*frame/frameCount, move.startCol + dC*frame/frameCount)
        drawBoard(screen)
        drawPieces(screen, board)
        #erase piece from end square
        color = colors[(move.endRow + move.endCol) % 2]
        endSquare = p.Rect(move.endCol * SQ_SIZE, move.endRow * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        p.draw.rect(screen, color, endSquare)
        #draw captured piece onto rectangle
        if move.pieceCaptured != '--':
            screen.blit(IMAGES[move.pieceCaptured], endSquare)
        #draw moving piece
        screen.blit(IMAGES[move.pieceMoved], p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
        p.display.flip()
        clock.tick(60)
"""

def drawText(screen, text):
    font = p.font.SysFont('Helvitca', 32, True, False)
    textObject = font.render(text, True, p.Color('Black'))
    textLocation = p.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH/2 - textObject.get_width()/2, HEIGHT/2 - textObject.get_height()/2)
    screen.blit(textObject, textLocation)

if __name__ == "__main__":
    main()
