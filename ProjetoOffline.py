#Importar coisas *Facil
#Esse programa pega as imagens e produz um arquivo pickle contendo os descritores
import cv2
import numpy as np
import math
import pickle


fotos = {"Mario": "Mario.jpg",
         "Kurumada": "CDZ.jpg"}

saida = open("features.pickle", 'wb')

#Parametros
ICOANEIS    = 6
ORBTHRESH   = 1.5
MAXFEATURES = 3000


#Implementar funcoes de transformada geometrica para as imagens *Medio
def dist(p, q):
    valor = 0
    for i in range(len(p)):
        valor += (p[i] - q[i])**2
    valor = math.sqrt(valor)
    return valor

def npdist(p, q):
    valor = p-q
    return np.sqrt(np.sum(valor))

def transla(imagem, tx, ty):
    pass

def redimen(formato, transformacao):
    x = formato[1]
    y = formato[0]
##    pontos = np.array([[0,0],[0,1],[1,0],[1,1]])
    pontos = np.array([[0, 0],
                       [0, x],
                       [y, 0],
                       [y, x]])
    nv = pontos.dot(transformacao)
##    print(nv[0])
##    print(nv[1])
    vy = [nv[0][0],nv[1][0],nv[2][0],nv[3][0]]
    vx = [nv[0][1],nv[1][1],nv[2][1],nv[3][1]]
    return (int(max(vx)-min(vx)),int(max(vy)-min(vy)))

def rotacao(imagem, angulo, pivo, cor = 255):
    formato = np.shape(imagem)
    rot = cv2.getRotationMatrix2D(pivo, angulo, 1)
    tam = redimen(formato, rot)

    
    rot[1][2] -= (formato[0]-tam[1])/2.0 #alterar posicao y
    rot[0][2] -= (formato[1]-tam[0])/2.0 #alterar posicao x
    #tam = (512,512) #sera substituido por uma funcao que calcula um novo tamaho *Medio
    #eh necessario compor a rotacao com uma translacao para centralizar a imagem
    aff = cv2.warpAffine(imagem, rot, tam, borderValue = cor)
    return aff

def escalar(imagem, val, pivo):
    pass

def perspec(imagem, angulo, pivo, cor = 200, distancia = 250):
    formato = np.shape(imagem)
    x = formato[1]
    y = formato[0]

    pt = [(0, 0, 0),
          (x, 0, 0),
          (0, y, 0),
          (x, y, 0)]

    ps = [(0, 0),
          (x, 0),
          (0, y),
          (x, y)]

    pc = [(0, 0),
          (x, 0),
          (0, y),
          (x, y)]
    
    #Gerar um ponto pd acima do ponto pivo
    #Rotacionar esse ponto
    nv = (pivo[0], pivo[1]+math.sin(angulo)*distancia, math.cos(angulo)*distancia)
##    nv = (pivo[0], pivo[1]+math.sin(angulo)*distancia, math.cos(angulo)*distancia)
    
    #Calcular a distancia entre o ponto pd e os pontos que compoem a borda da imagem
    pd = [dist(pt[0], nv),
          dist(pt[1], nv),
          dist(pt[2], nv),
          dist(pt[3], nv),]
    
    #Dividir a distancia entre pd e os pontos que compoe a borda pela distancia de pd ao pivo
    for i in range(4):
        pd[i] = pd[i]/distancia
        
    #A imagem eh dividida em 4 na altura do pivo
    #A altura da imagem sera proporcional ao cosseno do angulo onde cos0 a imagem de saida tem a mesma altura da entrada
    #A largura da imagem sera proporcional ao inverso da distancia 1/x
    #               onde ao dobrar a distancia da visao ate um ponto, ele anda metade do caminho ate o pivo
    for i in range(4):
        ny = pt[i][1]-pivo[1]
        nx = pt[i][0]-pivo[0]

        ny /= pd[i]
        ny *= math.cos(angulo) #Precisa ser reparado, mas nao eh prioridade
        nx /= pd[i]

        ny += pivo[1]
        nx += pivo[0]
        
        ps[i] = (nx, ny)

##    print(pc)
##    print(ps)

    minix = int(min(ps[0][0],ps[1][0],ps[2][0],ps[3][0]))
    maxix = int(max(ps[0][0],ps[1][0],ps[2][0],ps[3][0]))

    miniy = int(min(ps[0][1], ps[2][1]))
    maxiy = int(max(ps[0][1], ps[2][1]))
    
    #Eh calculado um novo tamanho para a imagem que leva em consideracao a nova altura e a largura da base maior
    base = int(max(abs(ps[0][0]-ps[1][0]),abs(ps[2][0]-ps[3][0])))
    altu = int(abs(ps[0][1] - ps[2][1]))

    pc = np.array(pc, dtype = "float32")
    ps = np.array(ps, dtype = "float32")
    
    trs = cv2.getPerspectiveTransform(pc, ps)

    deformada = cv2.warpPerspective(imagem, trs, (x,y), borderValue = cor)

##    print(trs)
    
    return deformada[miniy:maxiy, minix:maxix] #O corte nao esta perfeito
    #return cv2.warpPerspective(imagem, trs, (base,altu))



#Implementar uma funcao para riscar por cima dos objetos identificados. *facil ou medio





#Gerar um conjunto de imagens para simular a visao da camera ao redor dos objetos fotografados em uma icosphere *Facil
#Possivelmente eu tente tirar mais fotos de outros objetos como um mouse para testar o algoritmo em objetos 3d

#Cada anel tem 5 vertices a mais que o anterior
#Cada anel pode ter entre angulo 0 a angulo 90
#A aboboda eh um vertice
def icoLoop (imagem, aneis = 1, cor = 200, distancia = 250):
    formato = np.shape(imagem)
    x = int(formato[1]/2.0)
    y = int(formato[0]/2.0)
    
    imagens = []
    anguloh = math.pi / 2.0 /(aneis+1.0)
    imagens.append(perspec(imagem, 0, (x,y), cor, distancia))
    
    for i in range(aneis+1):
        
        for j in range(i*5):
            angulor = 360/(i*5)
            rotac = rotacao(imagem, angulor*j, (x,y), cor)
            pf = np.shape(rotac)
            px = int(pf[1]/2.0)
            py = int(pf[0]/2.0)
            imagens.append(perspec(rotac, anguloh*i, (x,y), cor, distancia))
    return imagens


#Aplicar a extracao de caracteristicas nas imagens geradas *Medio

#define o surf
#Nao estou conseguindo definir o surf
sofri = cv2.xfeatures2d_SURF(400) #Estou usando o python 3.7.3 com o opencv 4.4.0.42
#Define o sift que nao pode ser usado por nao ser invariante a rotacao
sith  = cv2.xfeatures2d.SIFT_create()

#Teoricamente o ORB eh invariante a rotacao e a scala.
orb = cv2.ORB_create()

def caracteristicas (imagens, descritor):
    ds = []
    dd = []
    for i in imagens:
        novos = descritor.detectAndCompute(i, None)
        ds.append(novos[0])
        dd.append(novos[1])
    return (ds, dd)

#Encontrar quais sao as melhores features para o problema. *Medio
def melhores (feat, thresh = 1.5, maximo = 1000, discreto = True):
    # Uma das propostas do artigo a ideia eh filtrar os M descritores que mais aparecem
    l = [] #lista de features
    p = [] #pontuacao de cada feature

    print("roll: ", len(feat[0]), thresh, maximo)
    roll = 0
    
    for i,j in zip(feat[0], feat[1]):
        pontos = 0.0
        for x,y in zip(feat[0], feat[1]):
            #Verifica se as features i j sao compativeis para adicionar um ponto
            #print(i, j)
            if(npdist(j, y) < thresh and discreto):
                pontos += 1
            elif(not discreto):
                pontos += i.overlap(i,j)
##        print(i, pontos)
        #Adiciona a feature ao vetor p
        p.append((i, j, pontos))

        roll += 1
        if (roll >= 1000):
            print(roll)
            roll = 0
    #Ordena a lista p para descobrir quais sao as features com melhor pontuacao
    p.sort(key=lambda x: x[2])

##    print(p)
    
    for i in range(maximo):
        #Pega os m primeiros objetos da lista para passar para o vetor l
        l.append((p[i][0], p[i][1]))
    
    return l


def listao (rays):
    ds = []
    dd = []
    for i in rays[0]:
        for j in i:
            ds.append(j)
    for i in rays[1]:
        for j in i:
            dd.append(j)
    
    return (ds, dd)


#Produzir um video curto para testar os descritores ou usar a webcam *Facil


#Carregar as imagens *Facil
##mario = cv2.imread("Mario.jpg", cv2.IMREAD_GRAYSCALE)
##kurumada = cv2.imread("CDZ.jpg", cv2.IMREAD_GRAYSCALE)
carac = {}
for i in fotos:
    fotos[i] = cv2.imread(fotos[i], cv2.IMREAD_GRAYSCALE)
    carac[i] = caracteristicas(icoLoop(fotos[i], ICOANEIS, 255), orb)
    carac[i] = listao(carac[i])


#mrot = rotacao(mario, 1, (256,256))

#Escolhe as melhores caracteristicas
for i in fotos:
    carac[i] = melhores(carac[i], ORBTHRESH, MAXFEATURES)

#converte os keypoints em tuplas
vds = {}
for i in fotos:
    vds[i] = []
    for j in carac[i]:
        vetor = (j[0].pt, j[0].size, j[0].angle, j[0].response, j[0].octave, j[0].class_id, j[1])
        vds[i].append(vetor)
        

pickle.dump(vds, saida)
saida.close()

##ks = icoLoop (kurumada, 3)
##ms = icoLoop (mario, 3, 255)
##
##cks = caracteristicas (ks, orb, True)
##cms = caracteristicas (ms, orb, True)
##
##kks = melhoresORB(cks)
##kms = melhoresORB(cms)


##for i in range(len(ks)):
##    formato = np.shape(ks[i])
##    x = formato[1]
##    y = formato[0]
##    if (x == 0 or y == 0):
##        print("vish")
##    else:
##        cv2.imshow(str(i), ks[i])


##fact = 0
##
##while True:
##    mrot = rotacao(mario, fact*0.4, (256,256))
####    mrot = rotacao(kurumada, fact*0.8, (256,390))
##    kper = perspec(kurumada, fact*0.03, (256,390))
##    fact += 1
##    cv2.imshow("mario", mrot)
##    cv2.imshow("kuru", kper)
####    break
##    if (cv2.waitKey(1) == 27):
##        break


##capitu = cv2.VideoCapture(0)
##
##while 1:
##    im = capitu.read()
##    for i in fotos:
##        pts = carac[i]
##        try:
##            lcs = orb.compute(im, pts)
##            print(i)
##        except:
##            pass
##            print("vish")
##        
##        
##    gr = cv2.cvtColor(im[1], cv2.COLOR_BGR2GRAY)
##    cv2.imshow("cor 0", gr)
##    if (cv2.waitKey(1) == 27):
##        break
##
##capitu.release()
##cv2.destroyAllWindows()


#cv2.destroyAllWindows()
