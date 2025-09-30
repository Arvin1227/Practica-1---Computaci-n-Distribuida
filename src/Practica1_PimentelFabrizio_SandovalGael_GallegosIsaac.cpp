/**
 * @file Practica1_PimentelFabrizio_SandovalGael_GallegosIsaac.cpp
 * @brief Implementación del algoritmo de Dijkstra distribuido con MPI para 
 *        calcular rutas con retraso mínimo en una red simulada.
 * 
 * Este programa simula una red de comunicación donde cada nodo puede calcular
 * la ruta de menor retraso desde un nodo origen (nodo 0) hacia todos los demás
 * nodos. Los retrasos de comunicación se simulan mediante números aleatorios
 * y son asimétricos (el retraso de A->B puede diferir de B->A).
 * 
 * @authors Fabrizio Pimentel Casillas
 *          Gerardo Gael Sandoval Sandoval
 *          Arvin Isaac Marín Gallegos
 * 
 * Parámetros:
 *   densidad: Probabilidad de conexión entre nodos (0.0-1.0, default: 0.5)
 * 
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <ctime>
#include <algorithm>

/**
 * @def INF
 * @brief Valor que representa distancia infinita (nodo inalcanzable).
 * 
 * Se utiliza para inicializar las distancias en el algoritmo de Dijkstra
 * y para verificar si un nodo es alcanzable desde el origen.
 */
constexpr int INF = 1000000000;

/**
 * @brief Punto de entrada principal del programa.
 * 
 * Ejecuta el algoritmo de Dijkstra distribuido en tres fases principales:
 * 1. Generación de un grafo aleatorio (solo nodo 0)
 * 2. Simulación de retrasos de comunicación asimétricos
 * 3. Cálculo de rutas de menor retraso y presentación de resultados
 * 
 * @param argc Número de argumentos de línea de comandos
 * @param argv Vector de argumentos: [0]=nombre_programa, [1]=densidad (opcional)
 * @return int Código de salida (0 = éxito)
 * 
 * @note Cada proceso MPI representa un nodo en la red
 * @note El grafo generado es no dirigido pero los retrasos son asimétricos
 */
int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int id_proceso, total_procesos;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proceso); 
    MPI_Comm_size(MPI_COMM_WORLD, &total_procesos); 

    double densidad = 0.5;
    if (argc >= 2) densidad = atof(argv[1]);

    // ========================================================================
    // FASE 1: GENERACIÓN DEL GRAFO ALEATORIO
    // ========================================================================
    
    /**
     * @brief Generador de números aleatorios para cada proceso.
     * 
     * El proceso 0 usa una semilla basada en tiempo para aleatoriedad real.
     * Los demás procesos usan semillas deterministas para reproducibilidad
     * en la generación de retrasos.
     */
    std::mt19937 generador(
        id_proceso == 0 
            ? static_cast<unsigned int>(time(nullptr)) 
            : 1234u + id_proceso
    );
    
    std::uniform_real_distribution<> probabilidad(0.0, 1.0);  
    std::uniform_int_distribution<> peso_arista(1, 1000);    

    /**
     * @brief Matriz de adyacencia del grafo en formato lineal.
     * 
     * Almacena el grafo completo donde:
     * - matriz_adyacencia[i * total_procesos + j] = peso de arista i->j
     * - valor 0 indica ausencia de arista
     * - valor > 0 indica peso de la arista
     * 
     * @note Tamaño: total_procesos x total_procesos
     * @note El grafo es no dirigido: peso(i,j) = peso(j,i)
     */
    std::vector<int> matriz_adyacencia(total_procesos * total_procesos, 0);

    if (id_proceso == 0) {
        
        for (int i = 0; i < total_procesos; ++i) {
            for (int j = i + 1; j < total_procesos; ++j) {
                if (probabilidad(generador) <= densidad) {
                    int peso = peso_arista(generador);
                    matriz_adyacencia[i * total_procesos + j] = peso;
                    matriz_adyacencia[j * total_procesos + i] = peso; 
                }
            }
        }

        std::cout << "=== GRAFO GENERADO ===\n";
        for (int i = 0; i < total_procesos; ++i) {
            for (int j = i + 1; j < total_procesos; ++j) {
                if (matriz_adyacencia[i * total_procesos + j] > 0) {
                    std::cout << "Nodo " << i << " <--(" 
                              << matriz_adyacencia[i * total_procesos + j] 
                              << ")--> Nodo " << j << "\n";
                }
            }
        }
        std::cout << std::endl;
    }


    MPI_Bcast(matriz_adyacencia.data(), 
              total_procesos * total_procesos, 
              MPI_INT, 
              0, 
              MPI_COMM_WORLD);

    // ========================================================================
    // FASE 2: SIMULACIÓN DE RETRASOS DE COMUNICACIÓN ASIMÉTRICOS
    // ========================================================================
    
    /**
     * @brief Vector de retrasos locales para este proceso.
     * 
     * retrasos_locales[j] almacena el retraso que el nodo j experimenta
     * al recibir mensajes desde este proceso (id_proceso).
     * 
     * Protocolo según lineamientos:
     * 1. El nodo emisor envía un mensaje
     * 2. El nodo receptor genera un número aleatorio entre 1-1000
     * 3. El receptor reporta este valor al emisor
     * 4. El emisor guarda este valor como el retraso para comunicarse con el receptor
     * 
     * @note Los retrasos son asimétricos: retraso(i->j) ≠ retraso(j->i)
     */
    std::vector<int> retrasos_locales(total_procesos, 0);
    std::uniform_int_distribution<> retraso_aleatorio(1, 1000);  ///< Generador de retrasos
    MPI_Status estado;

    for (int vecino = 0; vecino < total_procesos; ++vecino) {

        if (vecino == id_proceso) {
            retrasos_locales[vecino] = 0;
            continue;
        }

        if (matriz_adyacencia[id_proceso * total_procesos + vecino] > 0) {
        
            int retraso_que_genero = retraso_aleatorio(generador);  
            int retraso_recibido = 0;  

            MPI_Sendrecv(
                &retraso_que_genero, 1, MPI_INT, vecino, 200,  
                &retraso_recibido, 1, MPI_INT, vecino, 200,    
                MPI_COMM_WORLD, &estado
            );

            retrasos_locales[vecino] = retraso_recibido;
        }
    }

    /**
     * @brief Matriz global de todos los retrasos.
     * 
     * retrasos_todos[i * total_procesos + j] contiene el retraso de i hacia j.
     * Se construye recopilando los vectores retrasos_locales de todos los procesos.
     * 
     * @note Necesaria para que todos los nodos ejecuten Dijkstra con información completa
     */
    std::vector<int> retrasos_todos(total_procesos * total_procesos, 0);
    
    MPI_Allgather(
        retrasos_locales.data(), total_procesos, MPI_INT,
        retrasos_todos.data(), total_procesos, MPI_INT,
        MPI_COMM_WORLD
    );

    // ========================================================================
    // FASE 3: ALGORITMO DE DIJKSTRA DISTRIBUIDO
    // ========================================================================
    
    const int nodo_origen = 0;  ///< Nodo desde el cual se calculan todas las rutas
    
    /**
     * @brief Vector de distancias mínimas desde el origen.
     * distancia[i] = retraso mínimo para llegar del nodo_origen al nodo i
     */
    std::vector<int> distancia(total_procesos, INF);
    
    /**
     * @brief Vector para reconstruir caminos.
     * anterior[i] = nodo previo a i en el camino óptimo desde el origen
     */
    std::vector<int> anterior(total_procesos, -1);
    
    /**
     * @brief Marca los nodos ya procesados por Dijkstra.
     * visitado[i] = true si ya se determinó la distancia mínima al nodo i
     */
    std::vector<bool> visitado(total_procesos, false);

    distancia[nodo_origen] = 0;

    /**
     * @struct NodoMinimo
     * @brief Estructura para operación MPI_MINLOC.
     * 
     * Permite encontrar el nodo no visitado con distancia mínima
     * de forma distribuida entre todos los procesos.
     */
    struct { 
        int valor;   
        int indice;  
    } minimo_local, minimo_global;

    for (int iter = 0; iter < total_procesos; ++iter) {
      
        minimo_local.valor = INF;
        minimo_local.indice = -1;

        for (int i = 0; i < total_procesos; ++i) {
            if (!visitado[i] && distancia[i] < minimo_local.valor) {
                minimo_local.valor = distancia[i];
                minimo_local.indice = i;
            }
        }

        if (minimo_local.indice < 0) {
            minimo_local.valor = INF;
            minimo_local.indice = total_procesos;
        }

        MPI_Allreduce(
            &minimo_local, 
            &minimo_global, 
            1, 
            MPI_2INT, 
            MPI_MINLOC, 
            MPI_COMM_WORLD
        );
        if (minimo_global.valor >= INF || minimo_global.indice >= total_procesos) {
            break;
        }

        int u = minimo_global.indice;  
        visitado[u] = true;

        for (int v = 0; v < total_procesos; ++v) {
            if (visitado[v]) continue;  

            int peso = matriz_adyacencia[u * total_procesos + v];
            if (peso == 0) continue;  

            int retraso = retrasos_todos[u * total_procesos + v];
            long long costo_arista = static_cast<long long>(peso) + 
                                     static_cast<long long>(retraso);
            
            long long distancia_alternativa = static_cast<long long>(distancia[u]) + 
                                              costo_arista;

  
            if (distancia_alternativa < distancia[v]) {
                distancia[v] = static_cast<int>(distancia_alternativa);
                anterior[v] = u;  
            }
        }
    }

    // ========================================================================
    // FASE 4: PRESENTACIÓN DE RESULTADOS
    // ========================================================================
    
    if (id_proceso == 0) {
        std::cout << "\n=== RUTAS DESDE NODO " << nodo_origen << " ===\n";
        
        for (int destino = 0; destino < total_procesos; ++destino) {

            if (destino == nodo_origen) {
                std::cout << "Nodo " << destino 
                          << ": [" << destino << "], retraso total = 0\n";
                continue;
            }

            if (distancia[destino] >= INF) {
                std::cout << "Nodo " << destino << ": NO ALCANZABLE\n";
                continue;
            }

            std::vector<int> camino;
            int actual = destino;
            while (actual != -1) {
                camino.push_back(actual);
                if (actual == nodo_origen) break;
                actual = anterior[actual];
            }
            std::reverse(camino.begin(), camino.end());

            std::cout << "Nodo " << destino << ": [";
            for (size_t i = 0; i < camino.size(); ++i) {
                std::cout << camino[i];
                if (i + 1 < camino.size()) std::cout << " -> ";
            }
            std::cout << "], retraso total = ";

            std::vector<int> retrasos_individuales;
            int retraso_acumulado = 0;
            
            for (size_t i = 0; i < camino.size() - 1; ++i) {
                int u = camino[i];
                int v = camino[i + 1];
                
                int peso = matriz_adyacencia[u * total_procesos + v];
                int retraso = retrasos_todos[u * total_procesos + v];
                int costo_salto = peso + retraso;
                
                retrasos_individuales.push_back(costo_salto);
                retraso_acumulado += costo_salto;
            }

            for (size_t i = 0; i < retrasos_individuales.size(); ++i) {
                std::cout << retrasos_individuales[i];
                if (i + 1 < retrasos_individuales.size()) std::cout << " + ";
            }
            std::cout << " = " << retraso_acumulado << "\n";
        }
    }

    // Finalización del entorno MPI
    MPI_Finalize();
    return 0;
}