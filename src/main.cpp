#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <cstdlib>
#include <ctime>

/**
 * @class Practica1_PimentelFabrizio_SandovalGael_GallegosIsaac.cpp
 * @brief pues se usa el algoritmo de dijikstra en open mpi para hacer una
 * cuasi simulacion de una red real con sus retrasos, se imprime su camino
 *
 * autores: Fabrizio Pimentel Casillas
 * Esta clase almacena todo lo referente a esto e imprime
 * el camino generado para la comunicacion con el retraso minimo mas su retraso
 */
constexpr int INF = 1000000000; //si no es alcanzable

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int id_proceso, total_procesos;
    MPI_Comm_rank(MPI_COMM_WORLD, &id_proceso);
    MPI_Comm_size(MPI_COMM_WORLD, &total_procesos);

    double densidad = 0.5;
    if (argc >= 2) densidad = atof(argv[1]);

    // generador de num random pa crear el grafo
    std::mt19937 generador(id_proceso == 0 ? static_cast<unsigned int>(time(nullptr)) : 1234u + id_proceso);
    std::uniform_real_distribution<> probabilidad(0.0, 1.0);
    std::uniform_int_distribution<> peso_arista(1, 1000);

    // Matriz de adyacencia  0 = no arista
    std::vector<int> matriz_adyacencia(total_procesos * total_procesos, 0);

    if (id_proceso == 0) {
        for (int i = 0; i < total_procesos; ++i) {
            for (int j = i + 1; j < total_procesos; ++j) {
                if (probabilidad(generador) <= densidad) {
                    int peso = peso_arista(generador);
                    matriz_adyacencia[i*total_procesos + j] = peso;
                    matriz_adyacencia[j*total_procesos + i] = peso;
                }
            }
        }

        for (int i = 0; i < total_procesos; ++i) {
            for (int j = i+1; j < total_procesos; ++j) {
                if (matriz_adyacencia[i*total_procesos + j] > 0)
                    std::cout << i << " --(" << matriz_adyacencia[i*total_procesos + j] << ")--> " << j << "\n";
            }
        }
        std::cout << std::endl;
    }

    // la matriz se comparte con nodos, ven a cuales etan conectados
    MPI_Bcast(matriz_adyacencia.data(), total_procesos * total_procesos, MPI_INT, 0, MPI_COMM_WORLD);

    // cada nodo "pingea" a sus vecinos para medir retrasos
    std::vector<int> retrasos_locales(total_procesos, 0);
    MPI_Status estado;
    for (int vecino = 0; vecino < total_procesos; ++vecino) {
        if (vecino == id_proceso) { 
            retrasos_locales[vecino] = 0; 
            continue; 
        }
        if (matriz_adyacencia[id_proceso*total_procesos + vecino] > 0) {
            int valor_envio = 1;
            int valor_recepcion = 0;
            double tiempo_inicio = MPI_Wtime();
            MPI_Sendrecv(&valor_envio, 1, MPI_INT, vecino, 100, &valor_recepcion, 1, MPI_INT, vecino, 100, MPI_COMM_WORLD, &estado);
            double tiempo_fin = MPI_Wtime();
            int retraso_ms = static_cast<int>((tiempo_fin - tiempo_inicio) * 1000.0);
            if (retraso_ms < 1) retraso_ms = 1;
            retrasos_locales[vecino] = retraso_ms;
        } else {
            retrasos_locales[vecino] = 0;
        }
    }

    // compartir todos los retrasos con todos los nodos
    std::vector<int> retrasos_todos(total_procesos * total_procesos, 0);
    MPI_Allgather(retrasos_locales.data(), total_procesos, MPI_INT, retrasos_todos.data(), total_procesos, MPI_INT, MPI_COMM_WORLD);

    //donde se guarda el dijikstra
    int nodo_origen = 0;
    std::vector<int> distancia(total_procesos, INF);
    std::vector<int> anterior(total_procesos, -1);
    std::vector<char> visitado(total_procesos, 0);

    distancia[nodo_origen] = 0;

    struct { int valor; int indice; } minimo_local, minimo_global;// para ir guardando nuestro camino jeje

    //aqui se encuentran las distancias minimas
    for (int iter = 0; iter < total_procesos; ++iter) {
        minimo_local.valor = INF;
        minimo_local.indice = -1;
        for (int i = 0; i < total_procesos; ++i) {
            if (!visitado[i] && distancia[i] < minimo_local.valor) {
                minimo_local.valor = distancia[i];
                minimo_local.indice = i;
            }
        }

        if (minimo_local.indice < 0) minimo_local.indice = total_procesos;

        MPI_Allreduce(&minimo_local, &minimo_global, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        if (minimo_global.valor >= INF) break;

        int u = minimo_global.indice;
        if (u >= total_procesos) break;

        visitado[u] = 1;

        // las aristas que ya tenemos con retraso simulado
        for (int v = 0; v < total_procesos; ++v) {
            int peso = matriz_adyacencia[u*total_procesos + v];
            if (peso > 0 && !visitado[v]) {
                int retraso_comunicacion = retrasos_todos[u*total_procesos + v];
                long long combinado = static_cast<long long>(peso) + static_cast<long long>(retraso_comunicacion);
                long long alternativa = static_cast<long long>(distancia[u]) + combinado;
                if (alternativa < distancia[v]) {
                    distancia[v] = static_cast<int>(alternativa);
                    anterior[v] = u;
                }
            }
        }
    }

    // impresion final del camino, primero hay manejo de error y ya despues hay un bucle para la impresion
    if (id_proceso == 0) {
        for (int nodo_destino = 0; nodo_destino < total_procesos; ++nodo_destino) {
            if (distancia[nodo_destino] >= INF) {
                std::cout << "camino[" << nodo_origen << "," << nodo_destino << "]: no alcanzable\n";
                continue;
            }

            // camino en orden inverso
            std::vector<int> camino;
            int actual = nodo_destino;
            while (actual != -1) {
                camino.push_back(actual);
                if (actual == nodo_origen) break;
                actual = anterior[actual];
            }
            std::reverse(camino.begin(), camino.end());

            std::cout << "camino[";
            for (size_t i = 0; i < camino.size(); ++i) {
                std::cout << camino[i];
                if (i + 1 < camino.size()) std::cout << ",";
            }
            std::cout << "], retraso ";

            int retraso_total = 0;
            for (size_t i = 0; i < camino.size() - 1; ++i) {
                int u = camino[i];
                int v = camino[i+1];
                int peso = matriz_adyacencia[u * total_procesos + v]; // peso entre u y v
                retraso_total += peso;
                std::cout << peso;
                if (i + 1 < camino.size() - 1) std::cout << "+";
            }
            std::cout << "=" << retraso_total << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
