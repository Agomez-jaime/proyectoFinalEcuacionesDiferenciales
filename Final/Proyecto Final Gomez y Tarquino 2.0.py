##
import numpy as np
from scipy.optimize import fsolve
#Potencial de membrana
def V(Vm,n,m,h,I):
    gL = 0.3
    EL =-54.4      #magnitudes de conductancias, potenciales de
    gK = 36.0    #equilibrio y capacitancia de la membrana
    EK = -77.0
    gNa = 120.0
    ENa = 50.0
    CM = 1.0
    return (I - gL*(Vm-EL) - gK*(n**4)*(Vm-EK) - gNa*h*(m**3)*(Vm-ENa))/CM #definición de la derivada del voltaje
#Variables de activación (permeabilidades)
#n
def Fn(n,Vm,Temp):
    phi = 3**((Temp-6.3)/10)
    alfan = 0.01*(Vm+55)/(1.0-np.exp(-(Vm+55)/10))    #definición de la derivada de la condición inicial n
    betan = 0.125*np.exp(-(Vm+65)/80)
    return phi*(alfan*(1.0-n)-betan*n)

#m
def Fm(m,Vm,Temp):
    phi = 3**((Temp-6.3)/10)
    alfam = 0.1*(Vm+40)/(1.0-np.exp(-(Vm+40)/10))    #definición de la derivada de la condición inicial m
    betam = 4.0*np.exp(-(Vm+65)/18)
    return phi*(alfam*(1.0-m)-betam*m)

#h
def Fh(h,Vm,Temp):
    phi = 3**((Temp - 6.3) / 10)
    alfah = 0.07*np.exp(-(Vm+65)/20)
    betah = 1.0/(1.0+np.exp(-(Vm+35)/10))   #definición de la derivada de la condición inicial h
    return phi*(alfah*(1.0-h)-betah*h)


#Vector tiempo
t0 = 0.0
tf = 500
H = 0.01
tiempos = np.arange(t0,tf+H,H) #incluye valor final del tiempo

#Método de Runge-Kutta 2: yi= yi-1 + (H/2)(k1+k2), k1 = F(ti-1,yi-1), k2 = F(ti-1 + H, yi-1 + k1*H)
def RungeKutta2(Vm0,n0,m0,h0,tiempos,H,I,Temp):
    Vm = np.zeros(len(tiempos))
    Vm[0] = Vm0
    n =  np.zeros(len(tiempos))  #vectores de los valores de las incógnitas para cada iteración
    n[0] = n0
    m = np.zeros(len(tiempos))
    m[0] = m0
    h = np.zeros(len(tiempos))
    h[0] = h0

    for i in range(1,len(tiempos)):
        k1V = V(Vm[i-1], n[i-1], m[i-1], h[i-1], I[i]) #se define la primera k, con fórmula k1 = F(ti-1,yi-1)
        k1n = Fn(n[i - 1], Vm[i - 1], Temp)
        k1m = Fm(m[i - 1], Vm[i - 1], Temp)
        k1h = Fh(h[i - 1], Vm[i - 1], Temp)

        k2V = V(Vm[i-1] + k1V * H, n[i-1] + k1n * H, m[i-1] + k1m * H,
                h[i-1] + k1h * H, I[i])  #se define la segunda k, con fórmula k2 = F(ti-1 + H, yi-1 + k1*H)
        k2n = Fn(n[i - 1] + H * k1n, Vm[i - 1] + H * k1V, Temp)
        k2m = Fm(m[i - 1] + H * k1m, Vm[i - 1] + H * k1V, Temp)
        k2h = Fh(h[i - 1] + H * k1h, Vm[i - 1] + H * k1V, Temp)

        Vm[i] = Vm[i-1] + (H/2.0)*(k1V + k2V) #actualización de los valores de las incógnitas
        n[i] = n[i-1] + (H/2.0)*(k1n + k2n)
        m[i] = m[i-1] + (H/2.0)*(k1m + k2m)
        h[i] = h[i-1] + (H/2.0)*(k1h + k2h)
    return Vm


#Método de Runge-Kutta 4: yi = yi-1 + (H/6)*(k1+2k2+2k3+k4), k1 = F(ti-1,yi-1), k2 = F(ti-1+0.5H,yi-1+0.5Hk1), k3 = F(ti-1+0.5H,yi-1+0.5Hk2), k4 = F(ti-1+H,yi-1+Hk3)
def RungeKutta4(Vm0, n0, m0, h0, tiempos, H, I, Temp):
    Vm = np.zeros(len(tiempos))
    Vm[0] = Vm0
    n = np.zeros(len(tiempos))     #vectores de los valores de las incógnitas para cada iteración
    n[0] = n0
    m = np.zeros(len(tiempos))
    m[0] = m0
    h = np.zeros(len(tiempos))
    h[0] = h0

    for i in range(1, len(tiempos)):
        k1V = V(Vm[i - 1], n[i - 1], m[i - 1], h[i - 1], I[i])  #definición de la primera k, k1 = F(ti-1,yi-1)
        k1n = Fn(n[i - 1], Vm[i - 1], Temp)
        k1m = Fm(m[i - 1], Vm[i - 1], Temp)
        k1h = Fh(h[i - 1], Vm[i - 1], Temp)

        k2V = V(Vm[i - 1] + H * 0.5 * k1V, n[i - 1] + H * 0.5 * k1n,
                m[i - 1] + H * 0.5 * k1m, h[i - 1] + H * 0.5 * k1h, I[i])  #definición de la segunda k, k2 = F(ti-1+0.5H,yi-1+0.5Hk1)
        k2n = Fn(n[i - 1] + H * 0.5 * k1n, Vm[i - 1] + H * 0.5 * k1V, Temp)
        k2m = Fm(m[i - 1] + H * 0.5 * k1m, Vm[i - 1] + H * 0.5 * k1V, Temp)
        k2h = Fh(h[i - 1] + H * 0.5 * k1h, Vm[i - 1] + H * 0.5 * k1V, Temp)

        k3V = V(Vm[i - 1] + H * 0.5 * k2V, n[i - 1] + H * 0.5 * k2n,
                m[i - 1] + H * 0.5 * k2m, h[i - 1] + H * 0.5 * k2h, I[i])  #definición de la tercera k, k3 = F(ti-1+0.5H,yi-1+0.5Hk2)
        k3n = Fn(n[i - 1] + H * 0.5 * k2n, Vm[i - 1] + H * 0.5 * k2V, Temp)
        k3m = Fm(m[i - 1] + H * 0.5 * k2m, Vm[i - 1] + H * 0.5 * k2V, Temp)
        k3h = Fh(h[i - 1] + H * 0.5 * k2h, Vm[i - 1] + H * 0.5 * k2V, Temp)

        k4V = V(Vm[i - 1] + H * k3V, n[i - 1] + H * k3n,
                m[i - 1] + H * k3m, h[i - 1] + H * k3h, I[i])  #definición de la cuarta k, k4 = F(ti-1+H,yi-1+Hk3)
        k4n = Fn(n[i - 1] + H * k3n, Vm[i - 1] + H * k3V, Temp)
        k4m = Fm(m[i - 1] + H * k3m, Vm[i - 1] + H * k3V, Temp)
        k4h = Fh(h[i - 1] + H * k3h, Vm[i - 1] + H * k3V, Temp)

        Vm[i] = Vm[i - 1] + (H / 6.0) * (k1V + 2.0 * k2V + 2.0 * k3V + k4V) #actualización de las incógnitas
        n[i] = n[i - 1] + (H / 6.0) * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)
        m[i] = m[i - 1] + (H / 6.0) * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
        h[i] = h[i - 1] + (H / 6.0) * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
    return Vm

# Metodo de Euler hacia adelante
# Definicion metodo para y -> yi = y(i-1) + h*F((t-1),y(i-1))
def Eulerfor(Vm0, n0, m0, h0, tiempos, H, I, Temp):
    Vm = np.zeros(len(tiempos))
    Vm[0] = Vm0
    n = np.zeros(len(tiempos))
    n[0] = n0
    m = np.zeros(len(tiempos))     #vectores de los valores de las incógnitas para cada iteración
    m[0] = m0
    h = np.zeros(len(tiempos))
    h[0] = h0

    for ite in range(1, len(tiempos)):
        Vm[ite] = Vm[ite - 1] + H * V(Vm[ite - 1], n[ite - 1], m[ite - 1], h[ite - 1], I[ite])
        n[ite] = n[ite - 1] + H * Fn(n[ite - 1], Vm[ite - 1], Temp)      #actualización de los valores de las
        m[ite] = m[ite - 1] + H * Fm(m[ite - 1], Vm[ite - 1], Temp)     #incógnitas, según yi = y(i-1) + h*F((t-1),y(i-1))
        h[ite] = h[ite - 1] + H * Fh(h[ite - 1], Vm[ite - 1], Temp)
    return Vm

# Metodo de Euler hacia atrás
# Definicion metodo para y -> yi = y(i-1) + h*F(ti,yi)
def backRoot(yt4,nt1,mt1,ht1,Vmt1,H,temp,I):
    return[Vmt1 + H * V(yt4[3],yt4[0],yt4[1],yt4[2],I)-yt4[3],  #función que arma un array
           nt1 + H * Fn(yt4[0],yt4[3],temp)-yt4[0],            #para resolver las incógnitas por el método de euler back
           mt1 + H * Fm(yt4[1],yt4[3],temp)-yt4[1],
           ht1 + H * Fh(yt4[2],yt4[3],temp)-yt4[2]]


def EulerBack(Vm0, n0, m0, h0, tiempos, H, I, Temp):
    Vm = np.zeros(len(tiempos))
    Vm[0] = Vm0
    n = np.zeros(len(tiempos))
    n[0] = n0
    m = np.zeros(len(tiempos))   #vectores de los valores de las incógnitas para cada iteración
    m[0] = m0
    h = np.zeros(len(tiempos))
    h[0] = h0
    for ite in range(1, len(tiempos)):
        #solución del sistema de ecuaciones con scipy optimize
        sol = fsolve(backRoot,np.array([n[ite-1],m[ite-1],h[ite-1],Vm[ite-1]]),(n[ite-1],m[ite-1],h[ite-1],Vm[ite-1],H,Temp,I[ite]))
        n[ite]=sol[0]
        m[ite]= sol[1] #actualización de las incógnitas
        h[ite]=sol[2]
        Vm[ite]=sol[3]
    return Vm

# Metodo de Euler modificado
# Definicion metodo para y -> yi = y(i-1) + h*[(F(ti-1,yi-1)+F(ti,yi)/2]
def ModRoot(yt4,nt1,mt1,ht1,Vmt1,H,temp,I):
    return [Vmt1 + (H/2) * (V(Vmt1,nt1,mt1,ht1,I) + V(yt4[3],yt4[0],yt4[1],yt4[2],I)) - yt4[3],
            nt1 + (H/2) * (Fn(nt1,Vmt1,temp)+Fn(yt4[0],yt4[3],temp)) - yt4[0],   #función que arma un array
            mt1 + (H/2) * (Fm(mt1,Vmt1,temp)+Fm(yt4[1],yt4[3],temp)) - yt4[1],   #para resolver las incógnitas por
            ht1 + (H/2) * (Fh(ht1,Vmt1,temp)+Fh(yt4[2],yt4[3],temp)) - yt4[2]]   #el método de euler modificado

def EulerMod(Vm0, n0, m0, h0, tiempos, H, I, Temp):
    Vm = np.zeros(len(tiempos))
    Vm[0] = Vm0
    n = np.zeros(len(tiempos))
    n[0] = n0
    m = np.zeros(len(tiempos))    #vectores de los valores de las incógnitas para cada iteración
    m[0] = m0
    h = np.zeros(len(tiempos))
    h[0] = h0

    for ite in range(1, len(tiempos)):
        #solución del sistema de ecuaciones con scipy optimize
        sol = fsolve(ModRoot,np.array([n[ite-1],m[ite-1],h[ite-1],Vm[ite-1]]),(n[ite-1],m[ite-1],h[ite-1],Vm[ite-1],H,Temp,I[ite]))
        n[ite]=sol[0]
        m[ite]= sol[1]   #actualización de los valores de las incógnitas
        h[ite]=sol[2]
        Vm[ite]=sol[3]
    return Vm

#Interfaz
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib
matplotlib.use("TkAgg")
import struct as st
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from PIL import ImageTk ,Image

def iniciar_interfaz():
    #configuracion ventana principal
    principal = tk.Tk()
    principal.wm_title("Proyecto Final - Gómez y Tarquino")  # nombre de la ventana
    principal.geometry('900x700')  # tamaño ventana
    principal.config(cursor="star")  # elección de cursor
    titulo = tk.Label(principal,
                      font=('Times New Roman', 20),
                      fg="#4608AA",  # color de fuente
                      text="Modelo matemático de Hodgkin y Huxley del\n "
                           "potencial de acción de la neurona del calamar gigante")
    titulo.place(x=145, y=15)

    def Cerrar():  # función que pregunta salida del GUI
        mensaje = tk.messagebox.askquestion('¿Ya te vas?', '¿Está seguro de que no quieres seguir experimentando?',
                                            icon='warning')
        if mensaje == 'yes':
            principal.destroy()
        else:
            tk.messagebox.showinfo('Retornar', 'De vuelta al juego')

    Style = ttk.Style()
    Style.configure('E1.TButton', font=('Times', 10, 'underline'),
                    foreground='#FC0C01')  # primer tipo de estilo de botón
    Style.map("E1.TButton",
              foreground=[('pressed', 'red'), ('active', 'tomato')],  # colores según el estado del botón
              background=[('pressed', '!disabled', 'red'), ('active', 'tomato')])
    Style.configure('E2.TButton', font=('Times', 10, 'underline'), foreground='limegreen')  # segundo tipo de estilo
    Style.map("E2.TButton",
              foreground=[('pressed', 'darkgreen'), ('active', 'limegreen')],  # colores según el estado del botón
              background=[('pressed', '!disabled', 'darkgreen'), ('active', 'limegreen')])

    # botón enlazado a función de cerrar
    Cerrar = ttk.Button(master=principal, text="X", style="E1.TButton", command=Cerrar).place(x=800, y=15)

    # frames
    # frame principal
    parametros = tk.Frame(master=principal)
    parametros.place(x=570, y=100)  # se coloca el frame en la principal
    parametros.config(bg="#F4D03F", width=300, height=300, relief=tk.GROOVE, bd=8)  # color, relieve tamaño del frame
    # etiquetas para el título del frame y
    # los parámetros a cambiar por el usuario
    lp = tk.Label(parametros,
                  font=('Times New Roman', 17),
                  fg="#4608AA",
                  bg="#F4D03F",
                  text="Parámetros").place(x=7, y=2)
    lpvm0 = tk.Label(parametros,
                     font=('Times New Roman', 14),
                     fg="#4608AA",
                     bg="#F4D03F",
                     text="Potencial de \n membrana inicial"
                     ).place(x=7, y=40)
    lpn0 = tk.Label(parametros,
                    font=('Times New Roman', 14),
                    fg="#4608AA",
                    bg="#F4D03F",
                    text="Probabilidad inicial n").place(x=7, y=90)
    lpm0 = tk.Label(parametros,
                    font=('Times New Roman', 14),
                    fg="#4608AA",
                    bg="#F4D03F",
                    text="Probabilidad inicial m").place(x=7, y=130)
    lph0 = tk.Label(parametros,
                    font=('Times New Roman', 14),
                    fg="#4608AA",
                    bg="#F4D03F",
                    text="Probabilidad inicial h").place(x=7, y=170)
    lpT = tk.Label(parametros,
                   font=('Times New Roman', 14),
                   fg="#4608AA",
                   bg="#F4D03F",
                   text="Temperatura inicial").place(x=7, y=210)
    lpt = tk.Label(parametros,
                   font=('Times New Roman', 14),
                   fg="#4608AA",
                   bg="#F4D03F",
                   text="Tiempo").place(x=7, y=250)
    # widgets de entrada para cambiar los valores de los parámetros y insercción de valores sugeridos para cada uno
    TxtVm0 = tk.StringVar()
    TxtVm0.set("-65")
    evm0 = ttk.Entry(master=parametros, textvariable=TxtVm0, width=15)
    evm0.place(x=180, y=50)

    Txtn0 = tk.StringVar()
    Txtn0.set("0.3")
    en0 = ttk.Entry(master=parametros, textvariable=Txtn0, width=15)
    en0.place(x=180, y=95)

    Txtm0 = tk.StringVar()
    Txtm0.set("0.05")
    em0 = ttk.Entry(master=parametros, textvariable=Txtm0, width=15)
    em0.place(x=180, y=135)

    Txth0 = tk.StringVar()
    Txth0.set("0.65")
    eh0 = ttk.Entry(master=parametros, textvariable=Txth0, width=15)
    eh0.place(x=180, y=175)

    TxtT0 = tk.StringVar()
    TxtT0.set("6.3")
    eT0 = ttk.Entry(master=parametros, textvariable=TxtT0, width=15)
    eT0.place(x=180, y=215)

    Txtt0 = tk.StringVar()
    Txtt0.set("500")
    et0 = ttk.Entry(master=parametros, textvariable=Txtt0, width=15)
    et0.place(x=180, y=245)

    # Botón actualizar
    def Actualizar():
        Vm0 = float(evm0.get())
        n0 = float(en0.get())
        m0 = float(em0.get())  # función que toma los valores insertados por el usuario
        h0 = float(eh0.get())  # en los widget de entrada y los actualiza en las variables
        Temp = float(eT0.get())  # globales, después de convertir el tipo de dato de StringVar a float
        tf = float(et0.get())
        tiempos = np.arange(t0, tf + H, H)
        datos = np.array([])
        if n0 > 1 or n0 < 0:
            tk.messagebox.showinfo(title="Warning box",
                                   message="Error de magnitud: n debe ser una cantidad adimensional entre 0 y 1.",
                                   icon='error')
        if m0 > 1 or m0 < 0:
            tk.messagebox.showinfo(title="Warning box",
                                   message="Error de magnitud: m debe ser una cantidad adimensional entre 0 y 1.",
                                   icon='error')
        if h0 > 1 or h0 < 0:
            tk.messagebox.showinfo(title="Warning box",
                                   message="Error de magnitud: h debe ser una cantidad adimensional entre 0 y 1.",
                                   icon='error')
        global variables
        variables = (Vm0, n0, m0, h0, Temp, tiempos, datos)

    # botón que dispara la función anterior
    actualizarbut = ttk.Button(master=parametros, text="Cargar", style="E2.TButton", command=Actualizar)
    actualizarbut.place(x=200, y=2)

    # Frame corriente
    corriente = tk.Frame(master=principal)
    corriente.place(x=50, y=500)  # posición
    corriente.config(bg="#FF6F56", width=500, height=170, relief=tk.GROOVE, bd=8)  # color, dimensiones, relieve
    # etiquetas para el título de frame y las dos secciones del mismo: fija y variable
    lc = tk.Label(corriente,
                  font=('Times New Roman', 17),
                  fg="#4608AA",
                  bg="#FF6F56",
                  text="Corriente").place(x=7, y=2)
    lcf = tk.Label(corriente,
                   font=('Times New Roman', 16),
                   fg="#4608AA",
                   bg="#FF6F56",
                   text="Corriente fija").place(x=7, y=30)
    lcv = tk.Label(corriente,
                   font=('Times New Roman', 16),
                   fg="#4608AA",
                   bg="#FF6F56",
                   text="Corriente variable").place(x=7, y=60)

    # Valor corriente fija
    f = tk.StringVar()  # Entrada que permite al usuario cambiar la magnitud
    f.set("10")  # e intensidad de la corriente, en caso de ser fija,
    ef = ttk.Entry(master=corriente, textvariable=f, width=5)  # y fijación de un valor predeterminado sugerido
    ef.place(x=220, y=30)
    lfu = tk.Label(corriente,
                   font=('Times New Roman', 12, 'bold'),
                   fg="#4608AA",
                   bg="#FF6F56",
                   text="mA").place(x=260, y=30)

    # Valor 1 corriente variable
    v = tk.StringVar()  # Entrada que permite al usuario cambiar la magnitud
    v.set("10")  # e intensidad del primer intervalo de  la corriente, en caso de ser variable,
    ev = ttk.Entry(master=corriente, textvariable=v, width=5)  # y fijación de un valor predeterminado sugerido
    ev.place(x=220, y=60)
    lvu = tk.Label(corriente,
                   font=('Times New Roman', 12, 'bold'),
                   fg="#4608AA",
                   bg="#FF6F56",
                   text="mA").place(x=260, y=60)

    # Rango tiempos para valor 1
    # Entradas que permiten modificar los valores de tiempo del primer intervalo
    variable_tiempo_1 = tk.StringVar()
    variable_tiempo_1.set("10")
    evt1 = ttk.Entry(master=corriente, textvariable=variable_tiempo_1, width=5)
    evt1.place(x=300, y=60)
    lvt = tk.Label(corriente,
                   font=('Times New Roman', 25, 'bold'),
                   fg="#4608AA",
                   bg="#FF6F56",
                   text="-").place(x=340, y=45)
    variable_tiempo_2 = tk.StringVar()
    variable_tiempo_2.set("50")
    evt2 = ttk.Entry(master=corriente, textvariable=variable_tiempo_2, width=5)
    evt2.place(x=360, y=60)
    lvtu = tk.Label(corriente,
                    font=('Times New Roman', 12, 'bold'),
                    fg="#4608AA",
                    bg="#FF6F56",
                    text="ms").place(x=400, y=60)

    # Valor 2 corriente variable
    v2 = tk.StringVar()
    v2.set("120")
    ev2 = ttk.Entry(master=corriente, textvariable=v2, width=5)
    ev2.place(x=220, y=100)
    lvu2 = tk.Label(corriente,
                    font=('Times New Roman', 12, 'bold'),
                    fg="#4608AA",
                    bg="#FF6F56",
                    text="mA").place(x=260, y=100)

    # Rango tiempos para valor 2
    variable_tiempo_12 = tk.StringVar()  # Entrada que permite al usuario cambiar la magnitud
    variable_tiempo_12.set("100")  # e intensidad del segundo intervalo de  la corriente, en caso de ser variable,
    evt12 = ttk.Entry(master=corriente, textvariable=variable_tiempo_12,
                      width=5)  # y fijación de un valor predeterminado sugerido
    evt12.place(x=300, y=100)
    lvt2 = tk.Label(corriente,
                    font=('Times New Roman', 25, 'bold'),
                    fg="#4608AA",
                    bg="#FF6F56",
                    text="-").place(x=340, y=85)

    variable_tiempo_22 = tk.StringVar()
    variable_tiempo_22.set("255")
    evt22 = ttk.Entry(master=corriente, textvariable=variable_tiempo_22, width=5)
    evt22.place(x=360, y=100)
    lvtu2 = tk.Label(corriente,
                     font=('Times New Roman', 12, 'bold'),
                     fg="#4608AA",
                     bg="#FF6F56",
                     text="ms").place(x=400, y=100)

    # Función que cambia el valor de la corriente fija al colocado por el usuario en la entrada
    def Corriente_fija(tiempos):
        magnitud = int(ef.get())
        I = magnitud * np.ones(np.size(tiempos))
        return I

    # Función que cambia los valores de la corriente variable a los colocados por el usuario
    # en la entrada, de acuerdo a los intervalos de tiempo
    def Corriente_variable(tiempos):
        tf = float(et0.get())
        I = np.zeros(np.size(tiempos))
        magnitud1 = int(ev.get())
        t11 = int(evt1.get())
        t12 = int(evt2.get())
        magnitud2 = int(ev2.get())
        t21 = int(evt12.get())
        t22 = int(evt22.get())
        index = np.where((tiempos >= t11) & (tiempos <= t12))
        I[index] = magnitud1
        index = np.where((tiempos >= t21) & (tiempos <= t22))
        I[index] = magnitud2
        return I

    def revisar():
        tf = float(et0.get())
        t12 = int(evt2.get())
        t22 = int(evt22.get())
        if t12 > tf:
            tk.messagebox.showinfo(title="Error box",
                                   message="El primer intervalo de tiempo termina después del tiempo total de estimulación",
                                   icon='error')
        if t22 > tf:
            tk.messagebox.showinfo(title="Error box",
                                   message="El segundo intervalo de tiempo termina después del tiempo total de estimulación",
                                   icon='error')

    # Función que corre las anteriores funciones para corriente fija o variable
    def valor_corriente(t):
        if opc.get() == 1:
            return Corriente_fija(t)
        if opc.get() == 2:
            return Corriente_variable(t)

    # Radio botones que permiten elegir corriente fija o variable
    opc = tk.IntVar()
    fija = tk.Radiobutton(master=corriente, value=1, variable=opc, bg="#FF6F56")
    fija.place(x=180, y=30)
    variable = tk.Radiobutton(master=corriente, value=2, variable=opc, command= revisar, bg="#FF6F56")
    variable.place(x=180, y=60)
    # Gráfica
    # función que retorna la información necesaria para graficar cada método
    def fun(Vm0, n0, m0, h0, tiempos, H, I, Temp):
        if opcion.get() == 1:
            return (Eulerfor(Vm0, n0, m0, h0, tiempos, H, I, Temp), "Euler For", "red")
        elif opcion.get() == 2:
            return (EulerBack(Vm0, n0, m0, h0, tiempos, H, I, Temp), "Euler Back", "orange")
        elif opcion.get() == 3:
            return (EulerMod(Vm0, n0, m0, h0, tiempos, H, I, Temp), "Euler Mod", "green")
        elif opcion.get() == 4:
            return (RungeKutta2(Vm0, n0, m0, h0, tiempos, H, I, Temp), "RKTT 2", "blue")
        elif opcion.get() == 5:
            return (RungeKutta4(Vm0, n0, m0, h0, tiempos, H, I, Temp), "RKTT 4", "purple")

    # definición de la figura, fuera de las funciones, para que a
    # la elección de cada método se grafique el potencial de la
    # neurona sin mostrar una nueva gráfica
    figure = plt.Figure(figsize=(4.1, 4.1), dpi=100)
    ax = figure.add_subplot(111)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Voltaje de membrana")
    ax.set_title('Modelamiento Impulso nervioso')
    #figure.tight_layout()
    espaciograf = FigureCanvasTkAgg(figure, master=principal)
    espaciograf.get_tk_widget().place(x=140, y=90)
    #funci[on que permite reiniciar la grafica para poder visualizar nuevos datos
    def clean():
        mensaje = tk.messagebox.askquestion('¿Limpiar grafica?', '¿Seguro que quiere perder su trabajo?', icon='warning')
        if mensaje == 'yes':
            principal.destroy()
            iniciar_interfaz()

        else:
            tk.messagebox.showinfo('Retornar', 'Cancelando la limpieza')
    #Boton para limpiar la grafica
    limpiar = ttk.Button(master=principal, text="Limpiar", style="E2.TButton", command=clean).place(x=140, y=90)

    # función que grafica el potencial de la neurona según el método seleccionado
    # además, retorna un array con los datos graficados. Estos se añaden cada vez que se
    # grafica un nuevo método y el array se vacia con cada actualización de los datos
    def grafica():
        Vm0 = variables[0]
        n0 = variables[1]
        m0 = variables[2]
        h0 = variables[3]
        Temp = variables[4]
        tiempos = variables[5]
        datos = variables[6]
        I = valor_corriente(tiempos)
        aux = fun(Vm0, n0, m0, h0, tiempos, H, I, Temp)
        ax.plot(tiempos, aux[0], aux[2], label=aux[1])
        ax.legend()
        espaciograf.draw()
        datos = np.append(datos, aux[0])
        return datos

    # Metodos frame
    # Configuración de frame con los botones de cada método
    metodos = tk.Frame(master=principal)
    metodos.place(x=570, y=420)  # posición
    metodos.config(bg="#F4913F", width=300, height=250, relief=tk.GROOVE, bd=8)  # dimensión, color de fondo, relieve
    lm = tk.Label(metodos,
                  font=('Times New Roman', 17),
                  fg="#4608AA",
                  bg="#F4913F",
                  text="Métodos").place(x=7, y=2)  # Etiqueta del frame

    # Radio botones configurados para cada método, y enlazado a la función que los grafica
    opcion = tk.IntVar()
    Nombre = tk.StringVar()
    EuFor = tk.Radiobutton(master=metodos, text='Euler For', value=1, command=grafica, variable=opcion, bg="#F4913F",
                           font=("Times", 15), fg="#4608AA")
    EuFor.place(x=30, y=30)
    EuBack = tk.Radiobutton(master=metodos, text='Euler Back', value=2, command=grafica, variable=opcion, bg="#F4913F",
                            font=("Times", 15), fg="#4608AA")
    EuBack.place(x=30, y=70)
    EuMod = tk.Radiobutton(master=metodos, text='Euler Mod', value=3, command=grafica, variable=opcion, bg="#F4913F",
                           font=("Times", 15), fg="#4608AA")
    EuMod.place(x=30, y=110)
    RK2 = tk.Radiobutton(master=metodos, text='Runge-Kutta 2', value=4, command=grafica, variable=opcion, bg="#F4913F",
                         font=("Times", 15), fg="#4608AA")
    RK2.place(x=30, y=150)
    RK4 = tk.Radiobutton(master=metodos, text='Runge-Kutta 4', value=5, command=grafica, variable=opcion, bg="#F4913F",
                         font=("Times", 15), fg="#4608AA")
    RK4.place(x=30, y=190)

    # Exportar datos
    # Función que escribe los datos de la gráfica en tipo double en un archivo, de nombre elegido por el usuario
    def Exportar():
        datos = grafica()
        vrk2d = st.pack("d" * len(datos), *datos)
        nombre = str(eexp.get())
        file = open(nombre, "wb")
        file.write(vrk2d)
        file.close()
        tk.messagebox.showinfo(title="Message box",
                               message="Los datos  de la última  actualización \nhan sido exportados con éxito.",
                               icon='info')  # mensaje de exportación exitosa

    # Entrada para cambiar el nombre del archivo con los datos exportados
    archivo = tk.StringVar()
    archivo.set("Nombre del archivo.bin")
    eexp = ttk.Entry(master=principal, textvariable=archivo, width=25)
    eexp.place(x=8, y=15)
    # Botón que activa función de exportar
    Exportar = ttk.Button(master=principal, text="Exportar", style="E2.TButton", command=Exportar).place(x=8, y=50)
    img = ImageTk.PhotoImage(Image.open("calamarcin2.jpg"))
    lab = tk.Label(image=img)
    lab.place(x=20, y=130)
    principal.mainloop()  # Esto permite que el GIU se siga ejecutando

iniciar_interfaz()