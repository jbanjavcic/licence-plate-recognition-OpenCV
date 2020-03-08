# Main.py
import cv2
import numpy as np
import os
import DetectChars
import DetectPlates
import PossPlate

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False   #True/False

def main():
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # pokušaj KNN treninga

    if blnKNNTrainingSuccessful == False:                               
        print("\nerror: KNN trening nije uspješan\n")  
        return                                                  
    # end if

    imgOriginalScene  = cv2.imread("5.jpg")             

    if imgOriginalScene is None:                            
        print("\nerror: slika nije pronađena \n\n")  
        os.system("pause")                                  
        return                                            
    # end if

    listOfPossPlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detekcija oznaka
    listOfPossPlates = DetectChars.detectCharsInPlates(listOfPossPlates)        # detekcija karaktera na oznaci
    cv2.imshow("imgOriginalScene", imgOriginalScene)            # prikaz polazne slike

    if len(listOfPossPlates) == 0:              # ako reg. oznake nisu nađene
        print("\noznake nisu pronađene\n")  
    else:                                       #ako jesu             
                # poredajte popis mogućih oznaka po padajućem redoslijedu 
        listOfPossPlates.sort(key = lambda possPlate: len(possPlate.strChars), reverse = True)
                # pretpostavlja se da je oznaka s najviše prepoznatih znakova stvarna (prva ploča razvrstana po dužini niza silaznim redoslijedom)  
        licPlate = listOfPossPlates[0]

                #obrezivanje i prikaz obrezane oznake i praga oznake
        cv2.imshow("imgPlate", licPlate.imgPlate)           
        cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # ako karakteri nisu pronađeni
            print("\nni jedan karakter nije pronađen\n\n")  
            return                                         
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # crveni okvir oko reg. oznake

        print("\nreg. oznaka iščitana sa slike = " + licPlate.strChars + "\n")  
        print("***")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # ispis teksta reg. oznake na slici
        cv2.imshow("imgOriginalScene", imgOriginalScene)                # ponovni prikaz polazne slike
        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # zapis slike sa tekstom u .png formatu

    # end if else

    cv2.waitKey(0)					# čeka dok korisnik ne pritisne tipku za nastavak
    return
# end main

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # vrhovi za pravokutnik (okvir)
                # crta 4 crvene linije
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # centralna pozicija gdje će biti ispisan tekst
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # donje lijevo od područja u koje će tekst biti napisan
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # font teksta
    fltFontScale = float(plateHeight) / 30.0                    # veličina fonta
    intFontThickness = int(round(fltFontScale * 2.5))           # debljina fonta

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)       

    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # vodoravno mjesto područja teksta isto je kao i od oznake

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # ako je reg. oznaka na gornjem dijelu slike
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # tekst ispod reg. oznake
    else:                                                                                       # ako je re.oznaka na donjem dijelu slike
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # tekst iznad reg.oznake
    # end if

    textSizeWidth, textSizeHeight = textSize                #

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          

            # tekst oznake
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_RED, intFontThickness)
# end function

if __name__ == "__main__":
    main()


















