import numpy
import scipy.fftpack as sf
from math import sqrt, pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pyplot
from scipy import interpolate
import pandas as pd 
import copy 
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.signal import savgol_filter

""" Pasos para la creación de la superficie rugosa aleatoria 

1- Se crea una distribución  Gaussiana aleatoria no-correlacionada con una altura rms definida

2-  Se convoluciona esta función con otra función Gaussiana 
    2.1 - Primero se evalúa la función no-correlacionada (Z) mediante la transformación rápida de Fourier 
    2.2 - Segundo, se realiza el mismo procedimiento para la segunda Gussiana (F)
    2.3 - Tercero, se multiplican los dos resultados de las transformadas
    2.4 - Por último, se realiza la transformada inversa del resultado
3- El resultado final es la función correlacionada para el perfil de superficie 

Fuente: 
    Garcia, N., & Stoll, E. (1984). 
    Monte Carlo Calculation for Electromagnetic-Wave Scattering from Random Rough Surfaces.
    Physical Review Letters, 52(20), 1798–1801. doi:10.1103/physrevlett.52.1798

base para el código:
    http://www.mysimlabs.com/surface_generation.html


""" 
"""
data = pd.read_csv("Perturbado.csv")
data2 = pd.read_csv("Equilibrio.csv")
#print(data2)
xp = numpy.array(data["x"].tolist())
yp = numpy.array(data["y"].tolist())
zp = numpy.array(data["z"].tolist())
zi = numpy.array(data2["z"].tolist())
"""
def generator_initial_surface(xp,yp,zp_total,zi_total,XL,YL,ZL,radio,N,clx,cly,nargin):
    
    delta_Z_total = zp_total-zi_total 


    #dimensiones del sistema en unidades escaladas
    #XL = 0.3 # extensión en el eje x (0.3*escala = 30 micras)
    #YL = 0.3 # extensión en el eje y (0.3*escala = 30 micras)
    #ZL = 0.1 # extensión en el eje x (0.1*escala = 10 micras)
    #radio= 0.008 # radio de la partícula (0.008*escala = 0.8 micras)

    rms_srf = ZL*0.2 #la altura rms será un 20% de la altura del sistema
    z_relev = ZL*0.7 #altura donde las partículas son relevantes 

    #selección de partículas relevantes 

    #Se tendrá en cuenta el movimientos de las párticulas que se encuentran más cercanas a la
    # superficie, de tal manera, será la granja del 25% desde el tope hacia abajo de las párticulas iniciales 
    mask = zp_total>=z_relev
    delta_Z_1 = delta_Z_total[mask]
    delta_Z = numpy.ones(len(delta_Z_1))*radio



    # N => Número de puntos en las superficie 
    # rL => longitud de la superficie , tamaño máximo en xX 
    # h => altura rms, altura de la rugosidad rms 
    # clx,cly => longitud de correlacion: distancia típica de pico a pico o de valle a valle en el eje x e y respectivamente
    # nargin=> parámetro para superficie istrópica o anisotrópica 

    def rsgene2D(N,rL,h,clx,cly,nargin):
        x = numpy.linspace(-rL/2,rL/2,N)
        y = numpy.linspace(-rL/2,rL/2,N)
        x1 = numpy.linspace(0,rL,N)
        y1 = numpy.linspace(0,rL,N)
        X,Y = numpy.meshgrid(x,y)
        X1,Y1 = numpy.meshgrid(x1,y1)
        numpy.random.seed(13677)
        Z = h*numpy.random.randn(N,N); # uncorrelated Gaussian random rough surface distribution with rms height h
        # isotropic surface
        if nargin == 4:
            F = numpy.exp(-(abs(X)+abs(Y))/(clx/2)); # Gaussian filter
            f = 2*rL/N/clx*sf.ifft2(sf.fft2(Z)*sf.fft2(F)).real; # correlation of surface including convolution (faltung), inverse Fourier transform and normalizing prefactors
        # non-isotropic surface
        elif nargin == 5:
            F = numpy.exp(-(abs(X)**2/(clx**2/2)+abs(Y)**2/(cly**2/2))); # Gaussian filter
            f = 2*rL/N/sqrt(clx*cly)*sf.ifft2(sf.fft2(Z)*sf.fft2(F)).real; # correlation of surface including convolution (faltung), inverse Fourier transform and suitable prefactors

        return [X1,Y1,f,x1,y1]


    Xa,Ya,za,xa,ya = rsgene2D(N,XL,rms_srf,clx,cly,nargin)


    z_altura = za +0.12



    f2 = interpolate.interp2d(xa,ya,z_altura,kind='cubic', copy=True)

    """Concatenación de X y Y para entrar a la función"""
    x_final1 = numpy.concatenate((xa,xp),axis=0)
    y_final1 = numpy.concatenate((ya,yp),axis=0)


    """ordenamiento pata las gráficas más no para le cálculo """

    new_XY = numpy.column_stack((x_final1,y_final1)) #Se colocan en una misma matríz para que no se pierdan las parejas x,y 
    new_XY_order = new_XY[new_XY[:,0].argsort()] #Se ordena de menor a mayor por los valores de x 


    znew_all_init = f2(new_XY_order[:,0],new_XY_order[:,1]) #matriz con los dos parejas de X y Y, tanto del perfil como con los valores de entrada 


    """AQUI COMIENZA EL CAMBIO DE COMPAAR LA DOS MATRICES , LA Z_NEW_ALL Y LA Z_NEW2

        la idea es, tomar como matriz referencia solo los 64 puntos de entrada, para esto se toman xp y yp como listas 
        a cada pareja de x,y se le calculará el valor de z siguiendo la función de ajuste f2, 
        luego cada valor se guardará en znew_2 (el cual tendrá solo 64 valores y seguirá siendo una lista ), luego se 
        compara si el valor se encuentra en esta lista, si está, entonces se toma su posición en la lista znew_2
        esa misma posición será tomada para el valor a sumar de delta_z, así solo se cambian 64 puntos

        Nota:corroborar si el perfil ajustado se ajusta con el perfil inicial (si se parecen!!)
        """

    """Segundo cambio, implementación de una campana gaussiana como deformación de la superficie 
    #Para los valores de x,y en la superficie, se deben cambiar por los valores que arroja Z
    #1. lo primero es crear las dos matrices de igual tamaño con los mismos valores, es decir el meshgrid con los dos valores, esto se lograría utilizando 
    #sigma como condición, es decir , los valores de x que estén desde -sigma hasta sigma desde el valor X y Y del vector de los puntos 
    #(ya que estos son las posiciones de cada partícula), lo mismo para y  (de los x,y ordenados en el código de rugosidad)
    #2. se crea un meshgrid con los dos valores encontrados de x,y
    #3. Se crean las dos matrices, una con la función de la rugosidad y otra con la función gaussiana 
    #4. se compara la matriz grande con la función pequeña de la rugosidad y luego se cambia por el mismo elemento en la matriz gaussiana 
    """

    znew2 = [] #znew2 son los znew del ajuste 

    for i in range(len(xp)):
        z_valor = f2(numpy.array(xp[i]),numpy.array(yp[i]))
        znew2.append(z_valor[0])


    znew_all_copy_init = copy.deepcopy(znew_all_init)
    #print("esta es la shape de znew2", znew2.shape)
    """función para encontrar los valores de x o y que se encuentran dentro de sigma"""

    def gauss_pos (xye,x_all): #le entra delta_z, el punto x,y de la partícula y todos los x del perfil de rugisodad
        sigmader=xye+radio*2 #desde el centro hasta 2 veces la desviación hacia la derecha
        sigmaizq=xye-radio*2 #desde el centro hasta 2 veces la desviación hacia la izquierda
        x_gauss=[] #vector que contiene los valores que están dentro del rango entre sigma derecha y sigma izquierda
        for elx in x_all: #se reccore todo el vector de entrada 
            if elx>=sigmaizq and elx<=sigmader: #si el valor está dentro de la campana 
                x_gauss.append(elx)
        return x_gauss #se regresa el vector 

    def gauss_func(base,delta_z,n,m):
        sigma= radio*2 #desviación estandar sería el diametro de la y se toma desde la mitad hasta 2 desviaciones estandar
        xgs=numpy.linspace(-sigma*1.75,sigma*1.75,n) #se crea ña base para la gaussiana tanto en x como en y 
        ygs=numpy.linspace(-sigma*1.75,sigma*1.75,m)
        X,Y = numpy.meshgrid(xgs,ygs) #rejilla para la  función 
        Zg = base+delta_z*numpy.exp(-(X**2 + Y**2)/sigma**2) #función gaussiana, base es desde donde comienza a crecer la gaussiana 
                                                #la base sería el valor de z de la partícula en el perfil aleatorio
                                                #delta_z me controla la altura de la campana 
                                                #sigma me controla el ancho, se coloca al cuadrado por si el delta es negativo 
                                                # de esta forma se cambia la concavidad de la campana 
        return Zg 
    #método para cambiar los valores de los puntos dentro de la matríz linealizada 
    contador = 0 #contador para revisar si se estan visitando las 64 partículas de entrada 
    contadorif=0
    for el in tqdm(numpy.nditer(znew_all_init, order='C')): #Se recorre la matriz por filas 
        
        if el in znew2: #si ese valor se encuentra entre las partículas de entrada 
            
            indexa= znew2.index(el) #elemento es el punto a cambiar, se busca su indice para buscar el xp,yp y delta_Z
    
            if delta_Z[indexa] !=0 and abs(delta_Z[indexa])>0.005:
                xg=gauss_pos(xp[indexa],new_XY_order[:,0])  #puntos en x de la poscición de la partícula 
                yg=gauss_pos(yp[indexa],new_XY_order[:,1])#puntos en y de la poscición de la partícula 
        
                Z_gauss = gauss_func(el,delta_Z[indexa],len(xg),len(yg)) #función para crear la campana 
    
                Z_change= f2(xg,yg) #función para crear la matriz de 
                if Z_gauss.shape == Z_change.shape:
                    contadorif+=1
                    #print("Shape de z_change",Z_change.shape)
                    for el2 in numpy.nditer(Z_change, order='C'):
                        if el2 in znew_all_init:
                            a,b = numpy.where(znew_all_init==el2)
                            a0,b0 = numpy.where(Z_change==el2)
                            if znew_all_copy_init[a[0],b[0]] == znew_all_init[a[0],b[0]]:
                                if delta_Z[indexa] < 0: 
                                    if Z_gauss[a0[0],b0[0]] < znew_all_init[a[0],b[0]]:
                                        znew_all_copy_init[a[0],b[0]] = Z_gauss[a0[0],b0[0]]
                                    #else: 
                                    #   znew_all_copy[a[0],b[0]] = znew_all[a0[0],b0[0]]
                                else: 
                                    if Z_gauss[a0[0],b0[0]] > znew_all_init[a[0],b[0]]:
                                        znew_all_copy_init[a[0],b[0]] = Z_gauss[a0[0],b0[0]]
                                    #else: 
                                    #   znew_all_copy[a[0],b[0]] = znew_all[a0[0],b0[0]]

                            #else:
                            """
                            else:

                                if delta_Z[indexa] > 0:
                                    if Z_gauss[a0[0],b0[0]] > znew_all_copy_init[a[0],b[0]]:
                                        if Z_gauss[a0[0],b0[0]] > znew_all_init[a[0],b[0]]:
                                            znew_all_copy_init[a[0],b[0]] = Z_gauss[a0[0],b0[0]]
                                        else:
                                #          znew_all_copy[a[0],b[0]] = znew_all[a0[0],b0[0]]
                                            change_value = (Z_gauss[a0[0],b0[0]] + znew_all_init[a[0],b[0]]) / 2
                                            znew_all_copy_init[a[0],b[0]] = change_value#Z_gauss[a0[0],b0[0]]
                               """ 
                contador+=1






    znew_all_copy_init = savgol_filter(znew_all_copy_init,31,6,mode='nearest',axis=-2)
    znew_all_copy_init = savgol_filter(znew_all_copy_init,31,6,mode='nearest',axis=-1)

    return znew2,znew_all_copy_init,new_XY_order,f2,z_altura,xa