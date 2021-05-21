g++ -std=c++11 -march=native -fopenmp -Ofast node2vec-rnce.cpp -o node2vec-rnce
g++ -std=c++11 -march=native -fopenmp -Ofast verse-rnce-sgd.cpp -o verse-rnce-sgd
g++ -std=c++11 -march=native -fopenmp -Ofast verse-rnce-mbsgd.cpp -o verse-rnce-mbsgd
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result base-rnce.cpp -o base-rnce -lgsl -lm -lgslcblas
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result rns.cpp -o rns -lgsl -lm -lgslcblas
