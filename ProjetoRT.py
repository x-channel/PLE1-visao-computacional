#Este programa pega o arquivo pickle e usa ele para identificar os objetos
import cv2
import numpy as np
import math
import pickle
import sys

threshold = 31.0

entrada = open("features.pickle", 'rb')

wc = pickle.load(entrada)

entrada.close()

orb = cv2.ORB_create()
orb.setMaxFeatures(1000)
bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=True)


def npdist(p, q):
    valor = p-q
    return np.sqrt(np.sum(valor))

keys = {}
desc = {}

print(len(wc))

for n in wc:
    print(n)
    print(len(wc[n]))
    lk = []
    lf = []
    for j in wc[n]:
        tmp = cv2.KeyPoint(x = j[0][0], y = j[0][1], _size = j[1],_angle = j[2],
                           _response = j[3], _octave = j[4], _class_id = j[5])
        lk.append(tmp)
        lf.append(j[6])
    lf = np.array(lf)
##    print(type(lf))
##    cv2.imshow(n, lf)
    keys[n] = lk
    desc[n] = lf



for i in desc:
    for j in desc:
        vish = bf.match(desc[i], desc[j])
        vish = sorted(vish, key = lambda x:x.distance)
        print("Problemas entre %s e %s sao: %i"%(i, j, len(vish)))

        if (i != j):
            for k in vish:
                pass
##                print(k.distance)

##for i in wc:
##    desc[i] = []
##    for j in wc[i]:
##        tmp = cv2.KeyPoint(x = j[0][0], y = j[0][1], _size = j[1],_angle = j[2],
##                           _response = j[3], _octave = j[4], _class_id = j[5])
##        desc[i].append(tmp)

def compara (k1, k2, thresh = 10):
    pontos = 0
    for i in k1:
        for j in k2:
            if (i.overlap(i,j) > 0.7):
                pontos += 1
    if (pontos > thresh):
        print(pontos)
        return True
    else:
        return False

def comparad (d1, d2, thresh = 1.5):
    saida = []
    for i in d1:
        pontos = 0
        for j in d2:
            if (npdist(i, j) < thresh):
                pontos += 1
        if (pontos > 0):
            saida.append(i)
    return saida

#Elimina descritores que aparecem entre d1 e d2
def diferenca (d1, d2, mt):
    desc = mt.match(d1, d2)
    saida = []

capitu = cv2.VideoCapture(0)
##capitu = cv2.VideoCapture("Mario.mp4")

while 1:
    im = capitu.read()
    impb = cv2.cvtColor(im[1], cv2.COLOR_BGR2GRAY)
    imc = orb.detect(impb)
    lugares = orb.detectAndCompute(impb, None)
##    print(len(lugares[0]))
    for i in desc:
##        pass
        try:
            if (len(lugares[1]) > 0):
##                print("lugares ", len(lugares[1]))
##                k = bf.knnMatch(lugares[1], desc[i][1], k=2)
                k = bf.match(lugares[1], desc[i])

                pts = 0
                for g in k:
                    if (g.distance < threshold):
                        pts += 1
                
##                print(i, len(k))
                print(i, pts)
                
        except Exception as e:
            tb = sys.exc_info()[2]
            print(e.with_traceback(tb))
##            mats = bf.knnMatch(lugares[0], desc[i])
##            mats = bf.match(lugares[0], desc[i])
##            print(len(mats))
##        pts = desc[i]
##        if (compara(imc, pts, 100)):
##            print(i)
##        try:
##            lcs = orb.compute(im, pts)
##        except:
##            pass
##            print("vish")
        
##    iml = cv2.drawKeypoints(impb, lugares[0], np.array([]), (0, 0, 255),
##                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    gr = cv2.cvtColor(im[1], cv2.COLOR_BGR2GRAY)
    cv2.imshow("cor 0", impb)
    if (cv2.waitKey(1) == 27):
        break


capitu.release()
cv2.destroyAllWindows()


