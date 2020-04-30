import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation
from skimage.transform import probabilistic_hough_line
import math
import cv2 #to do pogrupowan punktow
import operator #do sortowania listy centrow
from skimage.draw import rectangle_perimeter
from skimage import feature, io, color, measure
import sys

def kategorie(lines):
    #wyliczenie wspolczynnikow/katow -  prostych
    wspolczynniki = []
    for line in lines:
        if (line[0][0]-line[1][0]) == 0.0:
            mianownik = 0.00001
        else:
            mianownik = line[0][0]-line[1][0]
        a = (line[0][1]-line[1][1])/mianownik
        a = math.atan(a)
        wspolczynniki.append(a*180/3.1415)

    if len(wspolczynniki) == 0:
        return [],[],False
    #z wielu lini Hough dzieli je na dwie kategorie (bo sa dwie rownolegle i dwie inne rownolegle)
    h_line = min(wspolczynniki)
    v_line = max(wspolczynniki)
    v_lines = []
    h_lines = []
    odchylka_katow = 30 #pisze o ile katy prostych dla jednej linii moga sie odchylic
    for c1,line in enumerate(lines):
        if abs(wspolczynniki[c1]-v_line)<odchylka_katow:
            v_lines.append(line)
        elif abs(wspolczynniki[c1]-h_line)<odchylka_katow:
            h_lines.append(line)
    return h_lines,v_lines,True
#za pomoca determinants liczymy przeciecie dla dwoch linii podanych jako line1 = [(x1,y1) (x2,y2)], line2 = [(x3,y3) (x4,y4)]
def find_intersection(line1, line2):
    # extract points
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    # compute determinant
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    return Px, Py
# liczymy przeciecie dla kazdej lini jednej kategorii z kazda linia drugiej kategorii
def przeciecia(h_lines,v_lines):
    Px = []
    Py = []
    for h_line in h_lines:
        for v_line in v_lines:
            px, py = find_intersection(h_line, v_line)
            Px.append(px)
            Py.append(py)
    return Px,Py

# funkcja do grupowania punktow parametry to punkty i liczba grup
def cluster_points(points, nclusters):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(points, nclusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centers
#Grupowanie
def grupuj(Px,Py):
    P = np.float32(np.column_stack((Px, Py)))
    if len(P)<4:
        return [], False
    else:
        nclusters = 4
        centers = cluster_points(P, nclusters)
        centers = [(int(i[0]),int(i[1])) for i in centers] #chcemy inty bo centra to punkty
        return centers, True

#podzial centrow na A B C D (lewy gora, prawy gora, prawy dol, lewy dol)
def centra(centers):
    lewe = []
    prawe = []
    sorted = centers.copy()
    sorted.sort(key = operator.itemgetter(0)) # sortuje wzgledem X
    lewe.append(sorted[0])
    lewe.append(sorted[1])
    prawe.append(sorted[2])
    prawe.append(sorted[3])
    lewe.sort(key = operator.itemgetter(1)) # sortuje wzgledem Y
    prawe.sort(key = operator.itemgetter(1)) # sortuje wzgledem Y
    A = lewe[0]
    D = lewe[1]
    B = prawe[0]
    C = prawe[1]
    return A,B,C,D


#obliczenia katow nachylenia linii od centrow aby wyznaczyc czy to co badamy to wogole plansza czy jakis srubokret xD
def angles(A,B):
    if abs(A[0] - B[0]) < 5.0:
        mianownik = 0.00001
        a = 89
    else:
        mianownik = A[0] - B[0]
        a = (A[1] - B[1]) / mianownik
        a = math.atan(a)
        a= a * 180 / 3.1415
    return a

def wspolczynniki_ab(A,B):
    if abs(A[0] - B[0]) < 5:
        mianownik = 0.00001
        a = 57.290
    else:
        mianownik = A[0] - B[0]
        a = (A[1] - B[1]) / mianownik
    b = A[1] - a*A[0]
    return a,b

def wartosc_punktu(x,a,b):
    y = a*x + b
    return y

def plansza_wykryto(A,B,C,D):
    #katy
    A_B_angle = angles(A,B)
    D_C_angle = angles(D,C)
    A_D_angle = angles(A,D)
    B_C_angle = angles(B,C)

    #odleglosci miedzy centrami
    A_B = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    D_C = math.sqrt((D[0]-C[0])**2 + (D[1]-C[1])**2)
    A_D = math.sqrt((A[0]-D[0])**2 + (A[1]-D[1])**2)
    B_C = math.sqrt((C[0]-B[0])**2 + (C[1]-B[1])**2)

    if abs(A_B_angle - D_C_angle)<20 and abs(A_D_angle-B_C_angle)<20 and abs(A_B_angle-A_D_angle)>30 and A_B>100:
        return True
    else:
        return False

def pole(ABCD,point):
    x = point[0]
    y = point[1]

    AB_a, AB_b = wspolczynniki_ab(ABCD[0], ABCD[1])
    AB_y = wartosc_punktu(x, AB_a, AB_b)
    AB_angle = angles(ABCD[0],ABCD[1])

    DC_a, DC_b = wspolczynniki_ab(ABCD[3], ABCD[2])
    DC_y = wartosc_punktu(x, DC_a, DC_b)
    DC_angle = angles(ABCD[3], ABCD[2])

    AD_a, AD_b = wspolczynniki_ab(ABCD[0], ABCD[3])
    AD_y = wartosc_punktu(x, AD_a, AD_b)
    AD_angle = angles(ABCD[0],ABCD[3])

    BC_a, BC_b = wspolczynniki_ab(ABCD[1], ABCD[2])
    BC_y = wartosc_punktu(x, BC_a, BC_b)
    BC_angle = angles(ABCD[1],ABCD[2])

    if y<AB_y and y<DC_y:           #pierwszy wiersz
        wiersz=0
        if AD_angle>0:
            if y>AD_y and y>BC_y:
                kolumna = 0
            elif y>BC_y and y<AD_y:
                kolumna = 1
            elif y<AD_y and y<BC_y:
                kolumna = 2
        else:
            if y < AD_y and y < BC_y:
                kolumna = 0
            elif y < BC_y and y > AD_y:
                kolumna = 1
            elif y > AD_y and y > BC_y:
                kolumna = 2
    elif y>AB_y and y<DC_y:         #drugi wiersz
        wiersz=1
        if AD_angle>0:
            if y>AD_y and y>BC_y:
                kolumna = 0
            elif y>BC_y and y<AD_y:
                kolumna = 1
            elif y<AD_y and y<BC_y:
                kolumna = 2
        else:
            if y < AD_y and y < BC_y:
                kolumna = 0
            elif y < BC_y and y > AD_y:
                kolumna = 1
            elif y > AD_y and y > BC_y:
                kolumna = 2
    elif y>AB_y and y>DC_y:         #trzeci wiersz
        wiersz=2
        if AD_angle>0:
            if y>AD_y and y>BC_y:
                kolumna = 0
            elif y>BC_y and y<AD_y:
                kolumna = 1
            elif y<AD_y and y<BC_y:
                kolumna = 2
        else:
            if y < AD_y and y < BC_y:
                kolumna = 0
            elif y < BC_y and y > AD_y:
                kolumna = 1
            elif y > AD_y and y > BC_y:
                kolumna = 2
    return wiersz,kolumna



# load an image
def foto(nazwa):
    fig, ax = plt.subplots(1,2)
    ax[0].axis('off')
    ax[1].axis('off')
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.0)
    im = io.imread(nazwa)
    oryginal = im.copy()
    im = color.rgb2gray(im)
    im = feature.canny(im,sigma=3)
    im = binary_dilation(im)
    contours = measure.find_contours(im, 0.8, fully_connected='high')

    # punkt 0 - stworzenie potrzebnej tablicy z danymi dla kontur
    contours_dane = []
    for c1,contour in enumerate(contours):
        Xmin = np.min(contour[:,1])
        Xmax = np.max(contour[:,1])               #Tu zmienilem x na y
        Ymax = np.max(contour[:,0])
        Ymin = np.min(contour[:,0])
        centerX = Xmin + (Xmax-Xmin)/2
        centerY = Ymin + (Ymax-Ymin)/2
        czy_plansza = False
        czy_dubel = False
        # Expand numpy dimmensions
        c = np.expand_dims(contour.astype(np.float32), 1)
        # Convert it to UMat object
        c = cv2.UMat(c)
        area1 = cv2.contourArea(c)
        area2 = abs(Xmax-Xmin)*abs(Ymax-Ymin)
        area_ratio = area1/area2
        if area_ratio>0.5:
            czy_kolko = 1
        else:
            czy_kolko = 2
        if abs(Xmax-Xmin) < 10:
            czy_dubel=True
        contours_dane.append([Xmin,Xmax,Ymin,Ymax,centerX,centerY,czy_plansza,czy_dubel,czy_kolko]) #1 to kolko 2 to krzyzyk 0 to nic

    # punkt 1 - usuwane dubli
    duble = []
    for ca, A in enumerate(contours):
        A_minx = contours_dane[ca][0]
        A_maxx = contours_dane[ca][1]
        A_miny = contours_dane[ca][2]
        A_maxy = contours_dane[ca][3]
        A_centerx = contours_dane[ca][4]
        A_centery = contours_dane[ca][5]
        for cb,B in enumerate(contours):
            B_minx = contours_dane[cb][0]
            B_maxx = contours_dane[cb][1]
            B_miny = contours_dane[cb][2]
            B_maxy = contours_dane[cb][3]
            B_centerx = contours_dane[cb][4]
            B_centery = contours_dane[cb][5]
            if (ca!=cb and ca not in duble and abs(B_minx-A_minx)<10 and abs(B_centerx-A_centerx)<10 and abs(B_centery-A_centery)<10 and abs(B_miny-A_miny)<10):
                contours_dane[cb][7] = True
                duble.append(cb)

    #punkt 2 podzial na plansze i nie-plansze
    def temp_img(im,kontura):
        temp = np.zeros((len(im),len(im[0])))
        for point in kontura:
            temp[int(point[0])][int(point[1])] = 1.0
        temp = binary_dilation(temp)
        return temp


    ABCD = []
    for c1,contour in enumerate(contours):
        A = [1,1]
        B = [2,2]
        C = [3,3]
        D = [4,4]
        if 1<2:      #contours_dane[c1][7] == False
            temp = temp_img(im,contour)
            mini = np.min(contour[:, 0])
            maxi = np.max(contour[:, 0])
            dlugosc = int(abs(maxi - mini) / 2)
            lines = probabilistic_hough_line(temp, line_length=dlugosc, line_gap=int(dlugosc/5), seed=1)  # 1/2

            for line in lines:
                p0, p1 = line
                ax[0].plot((p0[0], p1[0]), (p0[1], p1[1]))
            # FUNKCJE DO WYKRYCIA PLANSZY I JEJ CENTROW
            h_lines, v_lines, czysalinie = kategorie(lines)
            if czysalinie==True:
                Px, Py = przeciecia(h_lines, v_lines)
                centers, czysapunkty = grupuj(Px, Py)
            else:
                czysapunkty=False
            if czysapunkty == True:
                A, B, C, D = centra(centers)
                czyplansza = plansza_wykryto(A,B,C,D)
                czyplansza = czyplansza and abs(contours_dane[c1][0]-contours_dane[c1][1])>100
                contours_dane[c1][6] = czyplansza
            else:
                contours_dane[c1][6] = False

        ABCD.append([A,B,C,D])

    #  punkt 3 - ustalenie ktore kontury zawieraja sie w ktorych planszach
    zawierajace_sie = [] # <- kazda plansza bedzie miala ta liste wypelniona obiektami ktora na niej sa
    for ca,A in enumerate(contours):
        if contours_dane[ca][6] == True and contours_dane[ca][7] == False: #dla kazdej planszy
            #krawedzie sprawdzanej planszy
            AMinX = contours_dane[ca][0]
            AMaxX = contours_dane[ca][1]
            AMinY = contours_dane[ca][2]
            AMaxY = contours_dane[ca][3]
            elementy = []
            for cb,B in enumerate(contours):
                if cb!=ca and contours_dane[cb][7] == False:   #plansza ma sie nie zawierac sama w sobie i obiekt nie ma byc dublem
                    BcenterX = contours_dane[cb][4]
                    BcenterY = contours_dane[cb][5]
                    BminX = contours_dane[cb][0]
                    BmaxX = contours_dane[cb][1]
                    BminY = contours_dane[cb][2]
                    BmaxY = contours_dane[cb][3]
                    if BcenterX+20>AMinX and BcenterX-20<AMaxX and BcenterY+20>AMinY and BcenterY-20<AMaxY and contours_dane[cb][6] == False and abs(BminX-BmaxX)<abs(AMinX-AMaxX) and abs(BminY-BmaxY)<abs(AMinY-AMaxY) and abs(BminY-BmaxY)>50 :
                        elementy.append(cb)
                    elif BcenterX>AMinX and BcenterX<AMaxX and BcenterY>AMinY and BcenterY<AMaxY and contours_dane[cb][6] == True and abs(BminX-BmaxX)<abs(AMinX-AMaxX) and abs(BminY-BmaxY)<abs(AMinY-AMaxY):
                        contours_dane[cb][6] = False
            zawierajace_sie.append(elementy)
        else:
            zawierajace_sie.append([])


    #punkt 4 - odrzucenie srodkowego kwadracika planszy ktore uznany jest jako plansza xD
    for c1,kontura in enumerate(contours):
        czy_siezawiera = False #flaga ktora mowi czy proponowana plansza sie w czyms zawiera
        if contours_dane[c1][6] == True:
            for tablica in zawierajace_sie:
                if c1 in tablica:
                    czy_siezawiera=True
        if czy_siezawiera == True:
            contours_dane[c1][6] = False


    matrixy = []
    for c1,kontura in enumerate(contours_dane):
        if kontura[6] == True and kontura[7]==False:
            matrix = np.array([[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']])
            for indeks in zawierajace_sie[c1]:
                iks = contours_dane[indeks][4]
                igrek = contours_dane[indeks][5]
                wiersze,kolumny = pole(ABCD[c1],[iks,igrek])
                if contours_dane[indeks][8]==2:
                    matrix[wiersze][kolumny]='X'
                else:
                    matrix[wiersze][kolumny]='O'
            matrixy.append(matrix)


    for c,con in enumerate(contours_dane):
        if con[6] == True and con[7]==False:
            for pkt in ABCD[c]:
                x, y = pkt
                ax[0].scatter(x, y, s=100)
            ax[0].text(ABCD[c][0][0], ABCD[c][0][1], 'A', color='white')
            ax[0].text(ABCD[c][1][0], ABCD[c][1][1], 'B', color='white')
            ax[0].text(ABCD[c][2][0], ABCD[c][2][1], 'C',color='white')
            ax[0].text(ABCD[c][3][0], ABCD[c][3][1], 'D',color='white')


    for kontura in contours_dane:
        if kontura[6]==True and kontura[7]==False:

            start = (int(kontura[2]),int(kontura[0])) #Xmin/Ymin
            end = (int(kontura[3]),int(kontura[1]))  # Xmin/Ymin
            rr, cc = rectangle_perimeter(start,end,shape=im.shape)
            im[rr,cc] = 1.0




    ax[0].imshow(im, interpolation='nearest', cmap=plt.cm.gray)
    ax[1].imshow(oryginal, interpolation='nearest')
    
    tytul = ''
    for x in matrixy:
        y = np.array2string(x)
        tytul = tytul + '\n\n' + y

    fig.suptitle(tytul, fontsize=7)


    plt.show()


foto(sys.argv[1])





