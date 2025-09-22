#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <memory>

constexpr int INF = 1000000000;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    double density = 0.5;
    if (argc >= 2) density = atof(argv[1]);

    // Generador de números aleatorios
    std::mt19937 rng(rank == 0 ? time(nullptr) : 1234 + rank);
    std::uniform_real_distribution<> prob(0.0, 1.0);
    std::uniform_int_distribution<> peso(1, 1000);

    // Matriz de adyacencia
    std::vector<int> adj(nprocs * nprocs, 0);

    if (rank == 0) {
        for (int i = 0; i < nprocs; i++) {
            for (int j = i + 1; j < nprocs; j++) {
                if (prob(rng) <= density) {
                    int w = peso(rng);
                    adj[i*nprocs + j] = w;
                    adj[j*nprocs + i] = w;
                }
            }
        }

        std::cout << "Grafo generado:\n";
        for (int i = 0; i < nprocs; i++) {
            for (int j = i+1; j < nprocs; j++) {
                if (adj[i*nprocs + j] > 0)
                    std::cout << i << " --(" << adj[i*nprocs + j] << ")--> " << j << "\n";
            }
        }
        std::cout << std::endl;
    }

    MPI_Bcast(adj.data(), nprocs * nprocs, MPI_INT, 0, MPI_COMM_WORLD);


    std::vector<int> dist(nprocs, INF);
    std::vector<int> next(nprocs, -1);

    dist[rank] = 0;
    next[rank] = rank;


    std::vector<int> neigh;
    std::vector<int> neigh_w;
    for (int j = 0; j < nprocs; j++) {
        if (adj[rank*nprocs + j] > 0) {
            neigh.push_back(j);
            neigh_w.push_back(adj[rank*nprocs + j]);
        }
    }

    std::vector<int> recvbuf(nprocs, INF);

    int global_changed = 1;
    int iter = 0;

    while (true) {
        iter++;
        int local_changed = 0;

        for (size_t k = 0; k < neigh.size(); k++) {
            int v = neigh[k];
            int w_uv = neigh_w[k];

            MPI_Status status;
            MPI_Sendrecv(dist.data(), nprocs, MPI_INT, v, 0,
                         recvbuf.data(), nprocs, MPI_INT, v, 0,
                         MPI_COMM_WORLD, &status);

            for (int d = 0; d < nprocs; d++) {
                if (recvbuf[d] >= INF) continue;
                int alt = w_uv + recvbuf[d];
                if (alt < dist[d]) {
                    dist[d] = alt;
                    next[d] = v;
                    local_changed = 1;
                }
            }
        }

        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (!global_changed) break;
        if (iter > nprocs + 50) break;
    }

 
    std::vector<int> all_next;
    if (rank == 0) all_next.resize(nprocs * nprocs);
    MPI_Gather(next.data(), nprocs, MPI_INT, all_next.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Convergencia alcanzada en " << iter << " iteraciones.\n";
        std::cout << "Rutas desde nodo 0:\n";

        for (int dest = 0; dest < nprocs; dest++) {
            if (dest == 0) {
                std::cout << " 0 -> 0: retraso = 0, ruta = [0]\n";
                continue;
            }
            if (dist[dest] >= INF) {
                std::cout << " 0 -> " << dest << ": no alcanzable\n";
                continue;
            }

            std::cout << " 0 -> " << dest << ": retraso = " << dist[dest] << ", ruta = [";
            int cur = 0;
            std::cout << cur;
            int safety = 0;
            while (cur != dest && safety < nprocs+5) {
                int nh = all_next[cur*nprocs + dest];
                if (nh < 0) break;
                std::cout << " -> " << nh;
                cur = nh;
                safety++;
            }
            std::cout << "]\n";
        }
    }

    MPI_Finalize();
    return 0;
}
