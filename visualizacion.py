import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Simulando los datos de notas
df = pd.read_csv('notas_1u.csv')
alumnos = df['Alumno'].tolist()
notas = df['Nota'].tolist()

# --- REPRESENTACIÓN BINARIA ---
def crear_cromosoma_binario():
    cromosoma = []
    for i in range(39):
        examen = random.randint(0, 2)
        genes = [0, 0, 0]
        genes[examen] = 1
        cromosoma.extend(genes)
    return cromosoma

def decodificar_cromosoma_binario(cromosoma):
    asignaciones = {'A': [], 'B': [], 'C': []}
    examenes = ['A', 'B', 'C']
    
    for i in range(39):
        idx = i * 3
        for j in range(3):
            if cromosoma[idx + j] == 1:
                asignaciones[examenes[j]].append(i)
                break
    
    return asignaciones

def calcular_fitness_binario(cromosoma):
    asignaciones = decodificar_cromosoma_binario(cromosoma)
    
    promedios = {}
    for examen in ['A', 'B', 'C']:
        indices = asignaciones[examen]
        notas_examen = [notas[i] for i in indices]
        promedios[examen] = np.mean(notas_examen)
    
    desv_promedios = np.std(list(promedios.values()))
    
    fitness = -desv_promedios
    return fitness

# --- REPRESENTACIÓN REAL ---
def crear_cromosoma_real():
    cromosoma = []
    for i in range(39):
        pesos = [random.random() for _ in range(3)]
        suma = sum(pesos)
        pesos_norm = [p/suma for p in pesos]
        cromosoma.extend(pesos_norm)
    return cromosoma

def decodificar_cromosoma_real(cromosoma):
    asignaciones = {'A': [], 'B': [], 'C': []}
    examenes = ['A', 'B', 'C']
    
    alumnos_disponibles = list(range(39))
    contadores = {'A': 0, 'B': 0, 'C': 0}
    
    while alumnos_disponibles:
        mejor_alumno = None
        mejor_examen = None
        mejor_valor = -1
        
        for alumno in alumnos_disponibles:
            idx = alumno * 3
            for i, examen in enumerate(examenes):
                if contadores[examen] < 13:
                    valor = cromosoma[idx + i]
                    if valor > mejor_valor:
                        mejor_valor = valor
                        mejor_alumno = alumno
                        mejor_examen = examen
        
        if mejor_alumno is not None:
            asignaciones[mejor_examen].append(mejor_alumno)
            contadores[mejor_examen] += 1
            alumnos_disponibles.remove(mejor_alumno)
    
    return asignaciones

def calcular_fitness_real(cromosoma):
    asignaciones = decodificar_cromosoma_real(cromosoma)
    
    promedios = {}
    for examen in ['A', 'B', 'C']:
        indices = asignaciones[examen]
        notas_examen = [notas[i] for i in indices]
        promedios[examen] = np.mean(notas_examen)
    
    desv_promedios = np.std(list(promedios.values()))
    fitness = -desv_promedios
    return fitness

# --- REPRESENTACIÓN PERMUTACIONAL ---
def crear_cromosoma_permutacional():
    indices = list(range(39))
    random.shuffle(indices)
    return indices

def decodificar_cromosoma_permutacional(cromosoma):
    asignaciones = {
        'A': cromosoma[0:13],
        'B': cromosoma[13:26],
        'C': cromosoma[26:39]
    }
    return asignaciones

def calcular_fitness_permutacional(cromosoma):
    asignaciones = decodificar_cromosoma_permutacional(cromosoma)
    
    promedios = {}
    for examen in ['A', 'B', 'C']:
        indices = asignaciones[examen]
        notas_examen = [notas[i] for i in indices]
        promedios[examen] = np.mean(notas_examen)
    
    desv_promedios = np.std(list(promedios.values()))
    fitness = -desv_promedios
    return fitness

# Cálculo de fitness para las tres representaciones
generaciones = list(range(50))  # Simulamos 50 generaciones
fitness_binaria = np.random.uniform(-1.5, -1, size=50)  # Datos de fitness por generación (ejemplo)
fitness_real = np.random.uniform(-1.5, -1, size=50)  # Datos de fitness por generación (ejemplo)
fitness_permutacional = np.random.uniform(-1.5, -1, size=50)  # Datos de fitness por generación (ejemplo)
notas = pd.read_csv('notas_1u.csv')['Nota'].tolist()  # Lectura de notas

# Funciones de visualización separadas para los histogramas
def histograma_examen_A():
    examen_A = notas[:13]
    plt.figure()
    bins = 5  # Ajuste para un número adecuado de bins
    plt.hist(examen_A, bins=bins, alpha=0.7, label='Examen A', color='blue', edgecolor='black')
    plt.title('Histograma de Notas - Examen A')
    plt.xlabel('Nota')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.show()

def histograma_examen_B():
    examen_B = notas[13:26]
    plt.figure()
    bins = 5  # Ajuste para un número adecuado de bins
    plt.hist(examen_B, bins=bins, alpha=0.7, label='Examen B', color='green', edgecolor='black')
    plt.title('Histograma de Notas - Examen B')
    plt.xlabel('Nota')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.show()

def histograma_examen_C():
    examen_C = notas[26:]
    plt.figure()
    bins = 5  # Ajuste para un número adecuado de bins
    plt.hist(examen_C, bins=bins, alpha=0.7, label='Examen C', color='red', edgecolor='black')
    plt.title('Histograma de Notas - Examen C')
    plt.xlabel('Nota')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.show()

# Función para graficar otros gráficos como el fitness por generación y la comparación de distribuciones
def graficos_comparativos():
    # --- Gráfico 1: Evolución del fitness por generación ---
    plt.figure()
    plt.plot(generaciones, fitness_binaria, marker='o', label='Fitness Binaria', color='blue')
    plt.plot(generaciones, fitness_real, marker='o', label='Fitness Real', color='green')
    plt.plot(generaciones, fitness_permutacional, marker='o', label='Fitness Permutacional', color='red')
    plt.title('Evolución del Fitness por Generación')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Gráfico 2: Comparación de distribuciones de las tres representaciones ---
    rep_binaria = np.random.uniform(0, 20, 39)
    rep_real = np.random.uniform(0, 20, 39)
    rep_permutacional = np.random.uniform(0, 20, 39)

    plt.figure()
    sns.kdeplot(rep_binaria, label='Representación Binaria', color='blue', shade=True)
    sns.kdeplot(rep_real, label='Representación Real', color='green', shade=True)
    sns.kdeplot(rep_permutacional, label='Representación Permutacional', color='red', shade=True)
    plt.title('Comparación de Distribuciones de las 3 Representaciones')
    plt.xlabel('Nota')
    plt.ylabel('Densidad')
    plt.legend()
    plt.show()

# Llamada para generar los gráficos de fitness y distribuciones
graficos_comparativos()

# Llamada para generar los histogramas uno por uno
histograma_examen_A()
histograma_examen_B()
histograma_examen_C()
