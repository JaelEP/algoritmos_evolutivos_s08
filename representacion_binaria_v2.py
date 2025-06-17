import random
import numpy as np
import pandas as pd

# Cargar datos de notas (asegurarse de que el archivo 'notas_1u.csv' esté en el directorio)
df = pd.read_csv('notas_1u.csv')
alumnos = df['Alumno'].tolist()
notas = df['Nota'].tolist()

# --- CREACIÓN DEL CROMOSOMA ---
def crear_cromosoma():
    cromosoma = []
    for i in range(39):
        examen = random.randint(0, 3)  # Cambiamos de 3 a 4 exámenes
        genes = [0, 0, 0, 0]  # Ahora 4 exámenes
        genes[examen] = 1
        cromosoma.extend(genes)
    return cromosoma

# --- DECODIFICACIÓN DEL CROMOSOMA ---
def decodificar_cromosoma(cromosoma):
    asignaciones = {'A': [], 'B': [], 'C': [], 'D': []}  # Ahora 4 exámenes
    examenes = ['A', 'B', 'C', 'D']  # 4 exámenes
    
    for i in range(39):
        idx = i * 4  # Cada alumno ahora tiene 4 bits
        for j in range(4):
            if cromosoma[idx + j] == 1:
                asignaciones[examenes[j]].append(i)
                break
    
    return asignaciones

# --- CÁLCULO DEL FITNESS ---
def calcular_fitness(cromosoma):
    asignaciones = decodificar_cromosoma(cromosoma)
    
    if any(len(asignaciones[ex]) != 10 for ex in ['A', 'B', 'C', 'D']):  # Aseguramos que cada examen tenga 10 alumnos
        return -1000  # Penalización si no hay 10 alumnos por examen
    
    promedios = {}
    varianzas = {}
    diversidades = []
    
    for examen in ['A', 'B', 'C', 'D']:
        indices = asignaciones[examen]
        notas_examen = [notas[i] for i in indices]
        promedios[examen] = np.mean(notas_examen)
        varianzas[examen] = np.var(notas_examen)
        diversidades.append(np.std(notas_examen))  # Medimos dispersión interna
        
    # Penalización por varianza alta
    varianza_media = np.mean(list(varianzas.values()))
    penalizacion_varianza = 0.1 * varianza_media  # Penalty factor for variance
    
    # Premiar diversidad (mejor dispersión de notas)
    diversidad_media = np.mean(diversidades)
    premio_diversidad = 0.2 * diversidad_media  # Diversity reward factor
    
    # Calculamos el fitness total
    desviacion = np.std(list(promedios.values()))
    
    fitness = -desviacion - penalizacion_varianza + premio_diversidad
    return fitness

# --- FUNCIONES DE MUTACIÓN Y CRUCE ---
def mutacion(cromosoma):
    cromosoma_mutado = cromosoma.copy()
    
    alumno1 = random.randint(0, 38)
    alumno2 = random.randint(0, 38)
    
    idx1 = alumno1 * 4  # Cambiado a 4 bits por alumno
    idx2 = alumno2 * 4  # Cambiado a 4 bits por alumno
    
    examen1 = [i for i in range(4) if cromosoma_mutado[idx1 + i] == 1][0]
    examen2 = [i for i in range(4) if cromosoma_mutado[idx2 + i] == 1][0]
    
    if examen1 != examen2:
        cromosoma_mutado[idx1:idx1+4] = [0, 0, 0, 0]
        cromosoma_mutado[idx1 + examen2] = 1
        
        cromosoma_mutado[idx2:idx2+4] = [0, 0, 0, 0]
        cromosoma_mutado[idx2 + examen1] = 1
    
    return cromosoma_mutado

# --- ALGORITMO GENÉTICO ---
def algoritmo_genetico(generaciones=100, tam_poblacion=50):
    poblacion = [crear_cromosoma() for _ in range(tam_poblacion)]
    
    for gen in range(generaciones):
        fitness_scores = [(crom, calcular_fitness(crom)) for crom in poblacion]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        nueva_poblacion = []
        
        elite = int(tam_poblacion * 0.2)
        for i in range(elite):
            nueva_poblacion.append(fitness_scores[i][0])
        
        while len(nueva_poblacion) < tam_poblacion:
            padre = random.choice(poblacion[:tam_poblacion//2])
            hijo = mutacion(padre)
            nueva_poblacion.append(hijo)
        
        poblacion = nueva_poblacion
        
        if gen % 20 == 0:
            mejor_fitness = fitness_scores[0][1]
            print(f"Generación {gen}: Mejor fitness = {mejor_fitness:.4f}")
    
    mejor_cromosoma = fitness_scores[0][0]
    return mejor_cromosoma

# --- PROCESO DE ASIGNACIÓN Y RESULTADOS ---
print("REPRESENTACIÓN BINARIA")
print("Problema: Distribuir 39 alumnos en 4 exámenes (A, B, C, D) de forma equitativa")
print("Cromosoma: 156 bits (39 alumnos × 4 bits cada uno)")
print("Gen: [0,1,0,0] significa alumno asignado a examen B\n")

mejor_solucion = algoritmo_genetico()
asignaciones_finales = decodificar_cromosoma(mejor_solucion)

print("\nDistribución final:")
for examen in ['A', 'B', 'C', 'D']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedio = np.mean(notas_examen)
    print(f"Examen {examen}: {len(indices)} alumnos, promedio = {promedio:.2f}")
    print(f"  Alumnos: {[alumnos[i] for i in indices[:5]]}... (mostrando primeros 5)")

print("\nVerificación de equilibrio:")
promedios = []
for examen in ['A', 'B', 'C', 'D']:
    indices = asignaciones_finales[examen]
    notas_examen = [notas[i] for i in indices]
    promedios.append(np.mean(notas_examen))
print(f"Desviación estándar entre promedios: {np.std(promedios):.4f}")
