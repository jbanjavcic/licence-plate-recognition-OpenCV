# DetectChars.py
import os
import cv2
import numpy as np
import math
import random
import Main
import Preprocess
import PossChar

kNearest = cv2.ml.KNearest_create()

        # konstante za checkIfPossChar
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0
MIN_PIXEL_AREA = 80

        # konstante za usporedbu 2 karaktera
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0
MAX_CHANGE_IN_AREA = 0.5
MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2
MAX_ANGLE_BETWEEN_CHARS = 12.0

        # ostale konstante
MIN_NUMBER_OF_MATCHING_CHARS = 3
RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30
MIN_CONTOUR_AREA = 100

def loadKNNDataAndTrainKNN():
        #prazne liste za podatke
    allContoursWithData = []                
    validContoursWithData = []             

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                
    except:                                                                                
        print("error, classifications.txt nije moguce otvoriti\n")  
        os.system("pause")
        return False                                                                       
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 
    except:                                                                                 
        print("error, flattened_images.txt nije moguce otvoriti\n")  
        os.system("pause")
        return False                                                                 
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # pretvorba numpy niza u jednodimenzionalni
    kNearest.setDefaultK(1)                                                            
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # KNN trening

    return True                             # ako je trening prošao, return true
# end function

def detectCharsInPlates(listOfPossPlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossPlates) == 0:          # ako je lista prazna
        return listOfPossPlates             
    # end if

    for possPlate in listOfPossPlates:          # for petlja poss plate
        possPlate.imgGrayscale, possPlate.imgThresh = Preprocess.preprocess(possPlate.imgPlate)     # dohvaćanje grayscale i threshold 

        if Main.showSteps == True: #
            cv2.imshow("5a", possPlate.imgPlate)
            cv2.imshow("5b", possPlate.imgGrayscale)
            cv2.imshow("5c", possPlate.imgThresh)
        # end if 

                # prilagodba veičine slike
        possPlate.imgThresh = cv2.resize(possPlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

                # uklanjanje svih sivih područja
        thresholdValue, possPlate.imgThresh = cv2.threshold(possPlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Main.showSteps == True: # 
            cv2.imshow("5d", possPlate.imgThresh)
        # end if 

                # pronalazi sve konture, a zatim uključuje samo obrise koji bi mogli biti karakteri 
        listOfPossCharsInPlate = findPossCharsInPlate(possPlate.imgGrayscale, possPlate.imgThresh)

        if Main.showSteps == True: 
            height, width, numChannels = possPlate.imgPlate.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]                                       # isprazni listu 

            for possChar in listOfPossCharsInPlate:
                contours.append(possChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
            cv2.imshow("6", imgContours)
        # end if 

                # od popisa svih mogućih znakova, pronalazi grupe podudarnih znakova
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossCharsInPlate)

        if Main.showSteps == True: 
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("7", imgContours)
        # end if 

        if (len(listOfListsOfMatchingCharsInPlate) == 0):			# ako nema grupa podudaranja

            if Main.showSteps == True: 
                print("prondađeno karaktera " + str(
                    intPlateCounter) + " = (none), klikni na sliku i pritisni bilo koju tipku za nastavak...")
                intPlateCounter = intPlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            # end if 

            possPlate.strChars = ""
            continue						# vrati se na početak for petlje
        # end if
                    # unutar liste podudarajućih karaktera, sortira ih od lijeva na desno i uklanja moguća preklapanja
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):               
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)       
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])          
        # end for

        if Main.showSteps == True: 
            imgContours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("8", imgContours)
        # end if 

                # unutar svake moguće oznake pretpostavi da je najduži popis potencijalnih podudaranja značajki stvarni popis karaktera
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

                # kroz sve vektore podudaranja znakova, dobiti indeks onoga s najviše znakova
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

                # najduži popis podudaranja znakova unutar oznake je stvarni popis oznaka
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        if Main.showSteps == True: 
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
            cv2.imshow("9", imgContours)
        # end if 

        possPlate.strChars = recognizeCharsInPlate(possPlate.imgThresh, longestListOfMatchingCharsInPlate)

        if Main.showSteps == True: 
            print("kraraktera pronađeno u oznaci " + str(
                intPlateCounter) + " = " + possPlate.strChars + ", klikni na sliku i pritisni bilo koju tipku za nastavak...")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)
        # end if # 
    # end for loop 

    if Main.showSteps == True:
        print("\ndetekcija karaktera uspješna, klikni na sliku i pritisni bilo koju tipku za nastavak...\n")
        cv2.waitKey(0)
    # end if

    return listOfPossPlates
# end function

def findPossCharsInPlate(imgGrayscale, imgThresh):
    listOfPossChars = []                        
    contours = []
    imgThreshCopy = imgThresh.copy()

            # pronalazi sve obrise 
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                        # za svaki obris
        possChar = PossChar.PossChar(contour)

        if checkIfPossChar(possChar):              # ako je obris mogući znak, dodaj na popis mogućih
            listOfPossChars.append(possChar)     
        # end if
    # end if

    return listOfPossChars
# end function

def checkIfPossChar(possChar):
            # provjera obrisa da se vidi može li to biti znak,
    if (possChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possChar.fltAspectRatio and possChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# end function

def findListOfListsOfMatchingChars(listOfPossChars):
            # svrha funkcije je preurediti jedan veliki popis znakova u popis popisa odgovarajućih znakova
    listOfListsOfMatchingChars = []               

    for possChar in listOfPossChars:                        
        listOfMatchingChars = findListOfMatchingChars(possChar, listOfPossChars)        # nađi sve karaktere u listi koji odgovaraju trenutnom
        listOfMatchingChars.append(possChar)                # adodaj trenutni karakter u listu mogućih 
            
            # ako trenutni mogući popis znakova nije dovoljno dug da bi mogao činiti moguću reg. oznaku, vrati se na početak for petlje i pokušaj sa sljedećim
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     
            continue                          
        # end if

            # ako popis je dovoljno dug, dodaje se listi mogućih karaktera
        listOfListsOfMatchingChars.append(listOfMatchingChars)     
        listOfPossCharsWithCurrentMatchesRemoved = []

             # s velikog popisa ukloni trenutni popis odgovarajućih znakova kako ne bismo dvaput koristili iste, napraviti novi veliki popis za to jer ne želimo mijenjati izvorni
        listOfPossCharsWithCurrentMatchesRemoved = list(set(listOfPossChars) - set(listOfMatchingChars))
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossCharsWithCurrentMatchesRemoved)      

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:       # za svaki popis odgovarajućih znakova koji se pronađu rekurzivnim pozivom
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # dodaj popudarajući karakter u originalnu listu
        # end for

        break    # exit for

    # end for

    return listOfListsOfMatchingChars
# end function

def findListOfMatchingChars(possChar, listOfChars):
            # pronađi sve znakove na velikom popisu koji se podudaraju s jednim mogućim znakom i vrati ih kao popis
    listOfMatchingChars = []                

    for possibleMatchingChar in listOfChars:       
            #ako je znak koji pokušavamo pronaći potpuno isti kao i znak na velikom popisu koji provjeravamo, ne bismo ga trebali dodati ​​u popis jer bi dobili duplikat
        if possibleMatchingChar == possChar:    
            continue                              # ne dodaj i vrati se na početak for petlje
        # end if
                    #provjera jesu li znakovi jednaki
        fltDistanceBetweenChars = distanceBetweenChars(possChar, possibleMatchingChar)
        fltAngleBetweenChars = angleBetweenChars(possChar, possibleMatchingChar)
        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possChar.intBoundingRectArea)) / float(possChar.intBoundingRectArea)
        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possChar.intBoundingRectWidth)) / float(possChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possChar.intBoundingRectHeight)) / float(possChar.intBoundingRectHeight)

                # ako su znakovi jednaki
        if (fltDistanceBetweenChars < (possChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        # dodaj trenutni na listu podudarajućih
        # end if
    # end for

    return listOfMatchingChars      
# end function

def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

# koristi se osnovna trigonometrija za izracun kuta izmedu karaktera
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                             # ne dijelimo sa nulom ako su središnji položaji X jednaki, float dijeljenje sa nulom ruši program
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # izračun kuta
    else:
        fltAngleInRad = 1.5708                        #koristi se kao kut
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # izračun kuta u stupnjevima

    return fltAngleInDeg
# end function

# ako postoje dva znaka koji se preklapaju ili se međusobno zatvaraju kako bi bili prepoznati kao zasebni znakovi, ukloniti unutarnji (manji) znak, 
# # to će spriječiti da dva puta uključimo isti char ako se za isti znak pronađu dvije konture, 
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)            

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:        # ako trenutni i drugi znak nisu isti
                                # i ako imaju centralnu točku na skoro istoj poziciji, imamo preklapanje
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                                #ako je trenutni karakter manji od drugog, a nije uklonjen, ukloni ga
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:        
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:             
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)       
                        # end if
                    else:                                                                       
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:               
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)       
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved
# end function

        #  prepoznavanje karaktera
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""                                                
    height, width = imgThresh.shape
    imgThreshColor = np.zeros((height, width, 3), np.uint8)
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)               # sortiraj od lijeva na desno

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     #  verzija slike praga u boji za crtanje kontura u boji 

                # for petlja za svaki karakter u oznaki
    for currentChar in listOfMatchingChars:             
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), 
                (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))
                
                # zeleni okvir oko karaktera
        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)      

                # izdvajanje karaktera 
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

                # promjena velicine slike, vazno za prepoznavanje
        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))    
                
                #pretvara sliku u 1d numerički niz
        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))    
                
                # pretvara 1d numpy niz int-ova u 1d numpy niz float-ova
        npaROIResized = np.float32(npaROIResized)               

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              # poziv findNearest 

        strCurrentChar = str(chr(int(npaResults[0][0])))            # dohvaca karakter iz rezultata

        strChars = strChars + strCurrentChar                        # dodaje trenutni znak cijelom nizu
    # end for

    if Main.showSteps == True:  
        cv2.imshow("10", imgThreshColor)
    # end if 

    return strChars
# end function

