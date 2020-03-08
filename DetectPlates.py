# DetectPlates.py
import cv2
import numpy as np
import math
import Main
import random
import Preprocess
import DetectChars
import PossPlate
import PossChar

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

def detectPlatesInScene(imgOriginalScene):
    listOfPossPlates = []                  
    height, width, numChannels = imgOriginalScene.shape
    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main.showSteps == True: 
        cv2.imshow("0", imgOriginalScene)
    # end if

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)         # za grayscale i detekciju rubova 

    if Main.showSteps == True: 
        cv2.imshow("1a", imgGrayscaleScene)
        cv2.imshow("1b", imgThreshScene)
    # end if

            # funkcija prvo pronalazi sve konture, a zatim uključuje samo obrise koji bi mogli biti karakteri (bez usporedbe s ostalim značajkama)
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)

    if Main.showSteps == True: 
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(
            len(listOfPossibleCharsInScene))) 

        imgContours = np.zeros((height, width, 3), np.uint8)
        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        cv2.imshow("2b", imgContours)
    # end if 

            # od popisa svih mogućih znakova, pronalazi grupe podudaranja - svaka će skupina podudarnih znakova pokušati biti prepoznata kao oznaka
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Main.showSteps == True: 
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene)))  

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("3", imgContours)
    # end if 

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:        # za svaku grupu vezanih karaktera pokušat izdvojiti oznaku
        possPlate = extractPlate(imgOriginalScene, listOfMatchingChars)        

        if possPlate.imgPlate is not None:                    # ako je oznaka nađena dodaje se listi mogućih
            listOfPossPlates.append(possPlate)                  
        # end if
    # end for

    print("\n" + str(len(listOfPossPlates)) + " moguće oznake nađene")  # vrati koliko je mogucih oznaka nadeno

    if Main.showSteps == True: 
        print("\n")
        cv2.imshow("4a", imgContours)

        for i in range(0, len(listOfPossPlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossPlates[i].rrLocationOfPlateInScene)

            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Main.SCALAR_RED, 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Main.SCALAR_RED, 2)

            cv2.imshow("4a", imgContours)

            print("moguća oznaka " + str(i) + ", klikni na sliku i pritisni bilo koju tipku za nastavak")

            cv2.imshow("4b", listOfPossPlates[i].imgPlate)
            cv2.waitKey(0)
        # end for

        print("\ndetekcija dovrsena, klikni na sliku i pritisni tipku za nastavak\n")
        cv2.waitKey(0)
    # end if 
    return listOfPossPlates
# end function

def findPossibleCharsInScene(imgThresh):
    listOfPossChars = []                # povrat vrijednosti
    intCountOfPossChars = 0
    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # pronađi obrise

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                       # za svaki obris
        if Main.showSteps == True: 
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)
        # end if 

        possChar = PossChar.PossChar(contours[i])
                # ako je kontura moguća znak, povećaj broj mogućih karaktera i dodaj na listu mogućih karaktera
        if DetectChars.checkIfPossChar(possChar):                   
            intCountOfPossChars = intCountOfPossChars + 1           
            listOfPossChars.append(possChar)                        
        # end if
    # end for

    if Main.showSteps == True: 
        print("\nstep 2 - len(contours) = " + str(len(contours)))  
        print("step 2 - intCountOfPossChars = " + str(intCountOfPossChars))  
        cv2.imshow("2a", imgContours)
    # end if 

    return listOfPossChars
# end function

def extractPlate(imgOriginal, listOfMatchingChars):
    possPlate = PossPlate.PossPlate()           
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sortira karaktere od lijevo do desno prema x poziciji

            # izracun centra moguce oznake
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

            # izracun visine i sirine
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + 
                    listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - 
                    listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

            # izračunati kut korekcije područja oznake
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            # središnja točka, širina i visina i kut korekcije
    possPlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

            # dobiti matricu rotacije za izračunati kut korekcije
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)
    height, width, numChannels = imgOriginal.shape      # visina i širina polazne slike
    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotacija slike
    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possPlate.imgPlate = imgCropped         # kopira izrezanu sliku oznake u odgovarajuću varijablu za moguće oznake

    return possPlate
# end function












