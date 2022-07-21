import numpy
import scipy.fftpack as sf
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pyplot
from scipy import interpolate
import pandas as pd 
import copy 
from tqdm import tqdm

from scipy.signal import savgol_filter

from scipy.signal import savgol_filter

import InitialSurface as ins

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

#lECTURA DE ARCHIVOS 
def data_temporal_one_file(name, path="/."):#"Data/Data2D/DataTemporal/Oscillators/" ):        
   # data = np.loadtxt(path+"conf_E00_T01_H00.dat")
    data = numpy.loadtxt(path+name)
    col_names = [ "RposX","RposY","RposZ","SX","SY","SZ", "index"]
    df = pd.DataFrame(data, columns = col_names) 
   # df=df.groupby(["pm"]).mean()      #Particules mean
    return df

#Estado base configuracion 1
def read_state(name, path):
    F = data_temporal_one_file(name,path)   
    df = F
    R = numpy.array(df[["RposX","RposY","RposZ"]])
    # 2000 pasos de monte carlo, 64 particulas dos cordenadas 
    
    return R

path = "result_3D_2/"
name = "conf_spin_E00_F00_02.dat"
P_base = read_state(name, path)
    
#i = 0
path = "result_3D_2/"
name = "conf_spin_E00_F00_03.dat"
P_pert = read_state(name, path)



xp_total = numpy.array(P_pert[:,0])
yp_total = numpy.array(P_pert[:,1])
zp_total = numpy.array(P_pert[:,2])
zi_total = numpy.array(P_base[:,2])
delta_Z_total = zp_total-zi_total 


#dimensiones del sistema en unidades escaladas
XL = 0.3 # extensión en el eje x (0.3*escala = 30 micras)
YL = 0.3 # extensión en el eje y (0.3*escala = 30 micras)
ZL = 0.1 # extensión en el eje x (0.1*escala = 10 micras)
radio= 0.008 # radio de la partícula (0.008*escala = 0.8 micras)

rms_srf = ZL*0.2 #la altura rms será un 20% de la altura del sistema
z_relev = ZL*0.7 #altura donde las partículas son relevantes 

escala = 100

#selección de partículas relevantes 

#Se tendrá en cuenta el movimientos de las párticulas que se encuentran más cercanas a la
# superficie, de tal manera, será la granja del 25% desde el tope hacia abajo de las párticulas iniciales 
mask = zp_total>=z_relev
xp = xp_total[mask]
yp = yp_total[mask]
zp_t = zp_total[mask]
delta_Z = delta_Z_total[mask]

#mask2 = delta_Z1!=0
#delta_Z2 = delta_Z1[mask2]

#zp_to = zp_t[mask2]

#delta_Z = zp_to - z_relev

print("shape de xp de entrada", xp.shape)
print("shape de zi de entrada", zi_total.shape)
print("esto es shape de delta z:", delta_Z.shape)
print("estoes delta z: \n", delta_Z)

## Parámetros para el generador aleatorio 
N= 400;clx=0.02;cly=0.02;nargin=4

znew1,znew_all_in,new_XY_order_1,f1,z_altura,xa = ins.generator_initial_surface(xp,yp,zp_total,zi_total,XL,YL,ZL,radio,N,clx,cly,nargin)
f2 = interpolate.interp2d(new_XY_order_1[:,0],new_XY_order_1[:,0],znew_all_in,kind='cubic', copy=True)

znew_all =  f2(new_XY_order_1[:,0],new_XY_order_1[:,1])

znew3 = [] #znew2 son los znew del ajuste 
#znew2 = f2(xp,yp)
for i in range(len(xp)):
    z_valor = f2(numpy.array(xp[i]),numpy.array(yp[i]))
    znew3.append(z_valor[0])
print("longitud de znew_all: ", len(znew_all))

znew_all_copy = copy.deepcopy(znew_all)
print("luego del copy")
print("longitud de znew_all: ", len(znew_all))
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
for el in tqdm(numpy.nditer(znew_all, order='C')): #Se recorre la matriz por filas 
    
    if el in znew3: #si ese valor se encuentra entre las partículas de entrada 
        #print("Aqui se encotró una partícula")
        indexa= znew3.index(el) #elemento es el punto a cambiar, se busca su indice para buscar el xp,yp y delta_Z
        #print("esto es delta_z",delta_Z[indexa])
        if delta_Z[indexa] !=0 and abs(delta_Z[indexa])>0.01:
            #print("Aqui se encontró que la particula si tiene un delta alto")
            xg=gauss_pos(xp[indexa],new_XY_order_1[:,0])  #puntos en x de la poscición de la partícula 
            yg=gauss_pos(yp[indexa],new_XY_order_1[:,1])#puntos en y de la poscición de la partícula 
            #print("longitud de xg",len(xg))
            #print("esto es xp en el indexa",xp[indexa])
            #print("estos son todos los x ",new_XY_order[:,1])
            #Xg,Yg = numpy.meshgrid(xg,yg) #meshgrid para las dos matrices 
            Z_gauss = gauss_func(el,delta_Z[indexa],len(xg),len(yg)) #función para crear la campana 
            #znew_aux = znew_all_copy
            
            #print("esto es lo que entra a la función de gauss",el,delta_Z[indexa])
            #print("eso es xg segun gauss",xg)
            #print("esto es z gauss", Z_gauss)
            #print("/n")
            Z_change= f2(xg,yg) #función para crear la matriz de 
            if Z_gauss.shape == Z_change.shape:
                #print("Aqui las matrices son del mismo tamaño")
                contadorif+=1
                #print("Shape de z_change",Z_change.shape)
                for el2 in numpy.nditer(Z_change, order='C'):
                    if el2 in znew_all:
                        a,b = numpy.where(znew_all==el2)
                        a0,b0 = numpy.where(Z_change==el2)
                        if znew_all_copy[a[0],b[0]] == znew_all[a[0],b[0]]:
                            if delta_Z[indexa] < 0: 
                                if Z_gauss[a0[0],b0[0]] < znew_all[a[0],b[0]]:
                                    znew_all_copy[a[0],b[0]] = Z_gauss[a0[0],b0[0]]
                                else: 
                                    znew_all_copy[a[0],b[0]] = znew_all[a0[0],b0[0]] 
                            else: 
                                if Z_gauss[a0[0],b0[0]] > znew_all[a[0],b[0]]:
                                    znew_all_copy[a[0],b[0]] = Z_gauss[a0[0],b0[0]]
                                else: 
                                    znew_all_copy[a[0],b[0]] = znew_all[a0[0],b0[0]] 

                        #else:
                        
                        else:
                        #    if delta_Z[indexa] > 0:
                        #        if Z_gauss[a0[0],b0[0]] > znew_all_copy[a[0],b[0]]:
                        #            if Z_gauss[a0[0],b0[0]] > znew_all[a[0],b[0]]:
                        #                znew_all_copy[a[0],b[0]] = Z_gauss[a0[0],b0[0]]
                              

                         
                            #if Z_gauss[a0[0],b0[0]] > znew_all_copy[a[0],b[0]]:
                                    #znew_all_copy[a[0],b[0]] = Z_gauss[a0[0],b0[0]]
                            if delta_Z[indexa] > 0:
                                if Z_gauss[a0[0],b0[0]] > znew_all_copy[a[0],b[0]]:
                                    if Z_gauss[a0[0],b0[0]] > znew_all[a[0],b[0]]:
                                        znew_all_copy[a[0],b[0]] = Z_gauss[a0[0],b0[0]]
                                    else:
                              #          znew_all_copy[a[0],b[0]] = znew_all[a0[0],b0[0]]
                                        change_value = (Z_gauss[a0[0],b0[0]] + znew_all_copy[a[0],b[0]]) / 2
                                        znew_all_copy[a[0],b[0]] = change_value#Z_gauss[a0[0],b0[0]]
                            else: 
                                if Z_gauss[a0[0],b0[0]] < znew_all[a[0],b[0]]:
                                    change_value = (Z_gauss[a0[0],b0[0]] + znew_all_copy[a[0],b[0]]) / 2
                                    znew_all_copy[a[0],b[0]] = change_value#Z_gauss[a0[0],b0[0]]
                                    
                                  
                                else: 
                                    znew_all_copy[a[0],b[0]] = znew_all_copy[a0[0],b0[0]] + 0.05
           
                                               






                            


            #else:
            #    a,b = numpy.where(znew_all==el)
             #   indexa= znew2.index(el)
                #print("estas son las diferentes")
                #print("esto es xg", xg)
                #print("esto es yg", yg)
                #print("shape de Z_gauss",Z_gauss.shape)
                #print("shape de Z_change",Z_change.shape)
              #  znew_all_copy[a[0],b[0]] =  znew_all_copy[a[0],b[0]] + delta_Z[indexa] 
            contador+=1
    #else: 
     #   a,b = numpy.where(znew_all==el)
      #  znew_all_copy[a[0],b[0]] = znew_all_copy[a[0],b[0]]+0.05

print("esta es la shape de znew_all_copy", znew_all_copy.shape)
print("esto dió el contador, deberían ser 64",contador)
print("partículas cambiadas a gaussiana ",contadorif)
print("longitud de znew_all: ", len(znew_all))
print("longitud de X", len(new_XY_order_1[:,0]))
print("longitud de Y", len(new_XY_order_1[:,1]))
print("longitud de Z", len(znew_all_copy))
#numpy.savetxt("Z_new_all.txt",znew_all_copy)
#numpy.savetxt("X_order.txt",new_XY_order_1[:,0])

#f2 = interpolate.interp2d(xa,ya,z_altura,kind='cubic', copy=True)
#X,Y = numpy.meshgrid(new_XY_order[:,0],new_XY_order[:,1])
#f2_new = interpolate.interp2d(new_XY_order[:,0],new_XY_order[:,0], znew_all_copy,kind='cubic', copy=False)
#xin = numpy.linspace(0,0.3,len(new_XY_order[:,0]))
#yin= numpy.linspace(0,0.3,len(new_XY_order[:,1]))

#znew_all_copy_2 = f2_new(new_XY_order[:,0],new_XY_order[:,1])
#znew_all_copy = f2_new(xin,xin )
#X_new, Y_new = numpy.meshgrid(xin,yin)
#znew_all_copy = interpolate.bisplev(X_new[:,0],Y_new[0,:], f2_new)
#znew_all_copy_2 = interpolate.griddata((new_XY_order[:,0],new_XY_order[:,1]), znew_all_copy, (new_XY_order[:,0],new_XY_order[:,1]), method='cubic')

znew_all_copy = savgol_filter(znew_all_copy,41,6,mode='nearest',axis=-2)
znew_all_copy = savgol_filter(znew_all_copy,41,6,mode='nearest',axis=-1)

"""
fig2 = pyplot.figure()
pyplot.plot(xa, z_altura[0, :], 'ro-', xp, znew2[0, :], 'go', xa, move_z[0, :], 'b-')
pyplot.show()
"""


fig3 = pyplot.figure()
pyplot.plot(new_XY_order_1[:,0]*escala,znew_all[:,0]*escala,'ro-',label='Perfil ajustado' )
pyplot.plot(new_XY_order_1[:,0]*escala,znew_all_copy[:,0]*escala , 'bo-',label='perfil modificado') 

pyplot.plot(  xa*escala, z_altura[:,0]*escala, 'go-', label='perfil original')
pyplot.xlabel("X $  \mu$m", size= 15)
pyplot.ylabel("Z $  \mu$m", size=15)
pyplot.legend(loc ="upper left",fontsize=15) 
pyplot.show()

fig4 = pyplot.figure()
pyplot.plot(new_XY_order_1[:,0]*escala,znew_all[:,10]*escala,'ro-',label='Perfil ajustado' )
pyplot.plot(new_XY_order_1[:,0]*escala,znew_all_copy[:,10]*escala , 'bo-',label='perfil modificado') 
pyplot.plot(  xa*escala, z_altura[:,10]*escala, 'go-', label='perfil original')
pyplot.legend(loc ="upper left",fontsize=15) 
pyplot.xlabel("X$  \mu$m", size=15)
pyplot.ylabel("Z$  \mu$m", size=15)
pyplot.show()

"""Se coloca una partícula en el perfil"""

punto = numpy.random.choice(yp)
xpp,ypp = numpy.where(new_XY_order_1==punto)
print("esto fue lo que se encontró con where en new XY: ", xpp,ypp)


fig6 = pyplot.figure()
pyplot.plot(new_XY_order_1[:,0]*escala,znew_all[:,100]*escala,'r-',label='Perfil inicial' )
pyplot.plot(new_XY_order_1[:,0]*escala,znew_all_copy[:,100]*escala , 'b-',label='perfil modificado') 
""" circle = pyplot.Circle((10,-5),radio*escala, fc='blue',ec="red")
circle1 = pyplot.Circle((20,-7),radio*escala, fc='green',ec="blue")
circle2 = pyplot.Circle((30,-5),radio*escala, fc='red',ec="blue")
pyplot.gca().add_patch(circle)
pyplot.gca().add_patch(circle1)
pyplot.gca().add_patch(circle2) """
xp = numpy.array(P_pert[:,0])[::6]*escala
zp = numpy.array(P_pert[:,2])[::6]*escala
zi = numpy.array(P_base[:,2])[::6]*escala
for i in range(len(xp)):
                
                cir=pyplot.Circle((xp[i],zi[i]),radio*escala/2,fc='red')
                cir2 = pyplot.Circle((xp[i],zp[i]),radio*escala/2,fc='blue')
                pyplot.gca().add_patch(cir2)
                pyplot.gca().add_patch(cir) #Agregar círculo (disco) a la gráfic
#pyplot.axis('scaled') #Ejes de igual tamaño (Caja Cuadrada)
#pyplot.title('grafica de puntos') #Titulo Gráfica
#plt.axis([0.0,10,0.0,10]) #Tamaño Ejes

#pyplot.plot(  xa, z_altura[10,:], 'go-', label='perfil original')
#pyplot.legend(loc ="upper left",fontsize=15) 
pyplot.xlabel("X   $  \mu$m", size=15)
pyplot.ylabel("Z   $  \mu$m", size=15)
pyplot.axis("scaled")
#pyplot.ylim((-15.0,15.0))
pyplot.title("gráfica de corte escogido")
pyplot.show()

"dos graficas de perfil, uno sin modificar, el otro modificado"
fig7,axs = pyplot.subplots(3)
axs[0].plot(new_XY_order_1[:,0]*escala,znew_all[:,250]*escala,'r-',label='Perfil ajustado' )
axs[1].plot(new_XY_order_1[:,0]*escala,znew_all_copy[:,250]*escala , 'b-',label='perfil modificado') 
axs[2].plot(new_XY_order_1[:,0]*escala,znew_all[:,250]*escala,'r-',label='Perfil ajustado' )
axs[2].plot(new_XY_order_1[:,0]*escala,znew_all_copy[:,250]*escala , 'b-',label='perfil modificado')
#axs[0].set_xlabel("X   $  \mu$m", size=15)
axs[0].set_ylabel("Z   $  \mu$m", size=15)
axs[1].set_ylabel("Z   $  \mu$m", size=15)
axs[2].set_ylabel("Z   $  \mu$m", size=15)
axs[2].set_xlabel("X   $  \mu$m", size=15)

pyplot.show()

"dos graficas de perfil, uno sin modificar, el otro modificado"
fig5,axs = pyplot.subplots(2)
axs[0].plot(new_XY_order_1[:,0]*escala,znew_all[:,xpp[0]]*escala,'ro-',label='Perfil ajustado' )
axs[1].plot(new_XY_order_1[:,0]*escala,znew_all_copy[:,xpp[0]]*escala , 'bo-',label='perfil modificado') 
 
axs[0].set_xlabel("X   $  \mu$m", size=15)
axs[0].set_ylabel("Z   $  \mu$m", size=15)
axs[0].axis("scaled")
axs[0].set_ylim((-15.0,15.0))
axs[1].set_xlabel("X   $  \mu$m", size=15)
axs[1].set_ylabel("Z   $  \mu$m", size=15)
axs[1].axis("scaled")
axs[1].set_ylim((-15.0,15.0))
pyplot.show()

"""subplots"""
fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2)

ax1.plot(new_XY_order_1[:,0]*escala,znew_all[:,100]*escala,'ro-',label='Perfil ajustado y = 0' )
ax1.plot(new_XY_order_1[:,0]*escala,znew_all_copy[:,100]*escala , 'bo-',label='perfil modificado y = 0') 
#ax1.plot(  xa*escala, z_altura[:,0]*escala, 'go-', label='perfil original')
ax1.set_xlabel("X $  \mu$m", size=15)
ax1.set_ylabel("Z $  \mu$m", size=15)

ax1.legend(loc ="upper left",fontsize=10)

ax2.plot(new_XY_order_1[:,0]*escala,znew_all[:,200]*escala,'ro-',label='Perfil ajustado y = 30' )
ax2.plot(new_XY_order_1[:,0]*escala,znew_all_copy[:,200]*escala , 'bo-',label='perfil modificado y = 30') 
#ax2.plot(  xa*escala, z_altura[:,10]*escala, 'go-', label='perfil original')
ax2.legend(loc ="upper left",fontsize=10) 
ax2.set_xlabel("X $  \mu$m", size=15)
ax2.set_ylabel("Z $  \mu$m", size=15)

ax3.plot(new_XY_order_1[:,0]*escala,znew_all[:,250]*escala,'ro-',label='Perfil ajustado y = 100' )
ax3.plot(new_XY_order_1[:,0]*escala,znew_all_copy[:,250]*escala , 'bo-',label='perfil modificado y = 100') 
#ax3.plot(  xa, z_altura[10,:], 'go-', label='perfil original')
#pyplot.legend(loc ="upper left",fontsize=15)
ax3.set_xlabel("X $  \mu$m", size=15)
ax3.set_ylabel("Z $  \mu$m", size=15)
ax3.legend(loc ="upper left",fontsize=10) 

ax4.plot(new_XY_order_1[:,0]*escala,znew_all[:,400]*escala,'ro-',label='Perfil ajustado y = 150' )
ax4.plot(new_XY_order_1[:,0]*escala,znew_all_copy[:,400]*escala , 'bo-',label='perfil modificado y = 150')
ax4.set_xlabel("X $  \mu$m", size=15)
ax4.set_ylabel("Z $  \mu$m", size=15)
ax4.legend(loc ="upper left",fontsize=10) 

pyplot.show()

"""Perfil normalizado """
def rms(new_all):
    perfil = []
    for i in range(0,len(new_all[0])):
        base=perfil
        if i ==0:
            base=new_all[:,i]
        comparar = new_all[:,i]
        new_base =[]
  
        for j in range(0,len(comparar)):
            if base[j] >= comparar[j]:
                new_base.append(base[j])
            else:
                new_base.append(comparar[j])
        perfil = new_base
    return perfil 
#print("valor maximo de znew_all", numpy.max(znew_all))
#print("valor maximo de perfil", max(perfil))

"rugosidad RMS"
all_flat = numpy.ravel(znew_all_copy)
all_flat_origin = numpy.ravel(znew_all)
all_flat_inicial = numpy.ravel(z_altura)
RMS_all = numpy.std(all_flat,axis = 0)
RMS_origin = numpy.std(all_flat_origin,axis = 0)
RMS_inicial = numpy.std(all_flat_inicial,axis = 0)
Ra_all = numpy.mean(abs(all_flat),axis=0)
Ra_origin = numpy.mean(abs(all_flat_origin),axis=0)
Ra_inicial = numpy.mean(abs(all_flat_inicial),axis=0)
RMS_porcentual = ((abs(RMS_all-RMS_inicial))/RMS_inicial)*100
print("RMS de la matriz total original ", RMS_origin*escala)
print("RMS de la matriz total inicial ", RMS_inicial*escala)
print("RMS de la matriz total modificada: ", RMS_all*escala)
print("Ra de la matriz total original ", Ra_origin*escala)
print("Ra de la matriz total inicial ", Ra_inicial*escala)
print("Ra de la matriz total modificada: ", Ra_all*escala)
print("el cambio porcentual de la superficie es de {}%".format(RMS_porcentual))

print(" ")

profileOrigin = rms(znew_all)
profile = rms(znew_all_copy)
profileInicial = rms(z_altura)
RMS_perfil_orgin = numpy.std(profileOrigin,axis=0)
print("RMS del perfil original", RMS_perfil_orgin*100)
RMS_perfil = numpy.std(profile,axis=0)
print("RMS del perfil modificado", RMS_perfil*100)
RMS_perfil_inicial = numpy.std(profileInicial,axis=0)
print("RMS del perfil inicial", RMS_perfil_inicial*100)

"""
fig = pyplot.figure()
pyplot.plot(new_XY_order[:,0]*escala,profile*escala,'ro-',label='Perfil normalizado' )
pyplot.plot(new_XY_order[:,0]*escala,znew_all[:,260]*escala,'ro-',label='Perfil ajustado' )
pyplot.show()
"""
"Cambios en la rugosidad por cortes"
RMS_changes = []
cortes= []
for i in range(0,len(znew_all_copy[0]),10):
    perfil1 =znew_all_copy[:,i]
    RMS_perfil1 = numpy.std(perfil1)
    RMS_changes.append(RMS_perfil1)
    cortes.append(i)
mediaRMS = numpy.mean(numpy.array(RMS_changes))
desviacionRMS = numpy.std(numpy.array(RMS_changes))
print("media de la desviación estandar en cortes: ", mediaRMS*escala)
print("desviación de la desviación estandar en cortes: ",desviacionRMS*escala)
RMS_changes = numpy.array(RMS_changes)
cortes = numpy.array(cortes)
fig = pyplot.figure()
pyplot.plot(cortes,RMS_changes*escala,'bo-', label='RMS en cortes de la rugosidad')
pyplot.xlabel("Cortes en Y ", size=15)
pyplot.ylabel("Desviación estandar por corte $  \mu$m", size=15)
pyplot.title("Cambios de la rugosidad en cortes transversales")
pyplot.show()

"""Figuras 3D"""
fig = pyplot.figure()
ax = pyplot.subplot(1,2,2, projection='3d')
ax.set_box_aspect(aspect=(3,3,1)) 
ax.grid(False)

Xmesh,Ymesh = numpy.meshgrid(new_XY_order_1[:,0],new_XY_order_1[:,0])
# Plot surface
znew_all3 =znew_all + 0.12
#print("shape para el trisurf", new_XY_order[:,0].shape, new_XY_order[:,1].shape,znew_all_copy)
#plot = ax.plot_surface(X=Xa, Y=Ya, Z=z_altura, cmap='YlGnBu_r')
#plot = ax.plot_surface(Xmesh,Ymesh,znew_all3, cmap='magma')
plot = ax.plot_surface(Xmesh*escala,Ymesh*escala,znew_all_copy*escala, cmap='magma',vmin = 8,vmax=18)
#plot = ax.plot_surface(Xmesh,Ymesh,znew_all, cmap='YlGnBu_r')
#plot = ax.plot_surface(X=surf[0], Y=surf[1], Z=surf[2], cmap='YlGnBu_r')
ax.set_xlabel("X $  \mu$m")
ax.set_ylabel("Y $  \mu$m")
ax.set_zlabel("Z $  \mu$m")
cs = fig.colorbar(plot, ax=ax, shrink=0.6)
ax.set_title("Superficie deformada")

ax = pyplot.subplot(1,2,1, projection='3d')
ax.set_box_aspect(aspect=(3,3,1)) 
ax.grid(False)

Xmesh,Ymesh = numpy.meshgrid(new_XY_order_1[:,0],new_XY_order_1[:,0])
# Plot surface
znew_all3 =znew_all + 0.12
#print("shape para el trisurf", new_XY_order[:,0].shape, new_XY_order[:,1].shape,znew_all)
#plot = ax.plot_surface(X=Xa, Y=Ya, Z=z_altura, cmap='YlGnBu_r')
#plot = ax.plot_surface(Xmesh,Ymesh,znew_all3, cmap='magma')
plot = ax.plot_surface(Xmesh*escala,Ymesh*escala,znew_all*escala, cmap='magma',vmin = 8,vmax=18)
#plot = ax.plot_surface(Xmesh,Ymesh,znew_all, cmap='YlGnBu_r')
#plot = ax.plot_surface(X=surf[0], Y=surf[1], Z=surf[2], cmap='YlGnBu_r')
ax.set_xlabel("X $  \mu$m")
ax.set_ylabel("Y $  \mu$m")
ax.set_zlabel("Z $  \mu$m")
ax.set_title("Superficie inicial")
cs = fig.colorbar(plot, ax=ax, shrink=0.6)
pyplot.show()

"figura inicial"
fig = pyplot.figure()
ax = pyplot.axes(projection='3d')
ax.set_box_aspect(aspect=(3,3,1)) 
ax.grid(False)

Xmesh,Ymesh = numpy.meshgrid(new_XY_order_1[:,0],new_XY_order_1[:,0])
# Plot surface
#znew_all3 =znew_all + 0.1
#print("shape para el trisurf", new_XY_order[:,0].shape, new_XY_order[:,1].shape,znew_all)
#plot = ax.plot_surface(X=Xa, Y=Ya, Z=z_altura, cmap='YlGnBu_r')
#plot = ax.plot_surface(Xmesh,Ymesh,znew_all3, cmap='magma')
plot = ax.plot_surface(Xmesh*escala,Ymesh*escala,znew_all*escala, cmap='magma',vmin = 8,vmax=18)
#plot = ax.plot_surface(Xmesh,Ymesh,znew_all, cmap='YlGnBu_r')
#plot = ax.plot_surface(X=surf[0], Y=surf[1], Z=surf[2], cmap='YlGnBu_r')
ax.set_xlabel("X $  \mu$m", size=15)
ax.set_ylabel("Y $  \mu$m",size=15)
ax.set_zlabel("Z $  \mu$m",size=15)
#ax.set_title("Superficie inicial")
cs = fig.colorbar(plot, ax=ax, shrink=0.6)
pyplot.show()

"figura deformada"
fig = pyplot.figure()
ax = pyplot.axes(projection='3d')
ax.set_box_aspect(aspect=(3,3,1)) 
ax.grid(False)

Xmesh,Ymesh = numpy.meshgrid(new_XY_order_1[:,0],new_XY_order_1[:,0])
# Plot surface

plot = ax.plot_surface(Xmesh*escala,Ymesh*escala,znew_all_copy*escala, cmap='magma',vmin = 8,vmax=18)

ax.set_xlabel("X $  \mu$m", size=15)
ax.set_ylabel("Y $  \mu$m",size=15)
ax.set_zlabel("Z $  \mu$m",size=15)
#ax.set_title("Superficie inicial")
cs = fig.colorbar(plot, ax=ax, shrink=0.6)
pyplot.show()

z_all_plot = numpy.array(znew_all_copy) #Se guardan los resultados de los máximos de z
numpy.savetxt('randomSuefaceModificada.txt',z_all_plot)

z_all_plot_sh = numpy.array(znew_all) #Se guardan los resultados de los máximos de z
numpy.savetxt('randomSuefaceInicial.txt',z_all_plot_sh)

"""mapa de contornos"""

fig = pyplot.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_box_aspect(aspect=(3,3,1)) 
ax.grid(False)
surf1 = ax.plot_surface(Xmesh*escala,Ymesh*escala,znew_all_copy*escala, cmap = 'magma',
                        linewidth = 0, antialiased = False,vmin = 8,vmax=18)
ax.set_xlabel("X $  \mu$m",size=15)
ax.set_ylabel("Y $  \mu$m",size=15)
ax.set_zlabel("Z $  \mu$m",size=15)

  
ax = fig.add_subplot(1, 2, 2)
cs1 = ax.contourf(Xmesh*escala,Ymesh*escala, znew_all_copy*escala )
fig.colorbar(cs1)
pyplot.show()

"Solo los mapa de contornos"

fig = pyplot.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')

cs2 = ax.contourf(Xmesh*escala,Ymesh*escala, znew_all*escala )
fig.colorbar(cs2)
 
  
ax = fig.add_subplot(1, 2, 2)
cs1 = ax.contourf(Xmesh*escala,Ymesh*escala, znew_all_copy*escala )
fig.colorbar(cs1)
pyplot.show()