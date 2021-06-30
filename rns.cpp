// Author: Hao Xiong, 498967825@qq.com
// The architecture of the code follows https://github.com/tangjianpku/LINE and https://github.com/tmikolov/word2vec
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include <vector>

#define MAX_STRING 100
#define SIGMOID_BOUND 6
#define EXP_BOUND 6

using namespace std;

const int neg_table_size = 1e8;
const int sigmoid_table_size = 1000;

char network_file[100] = "tmpfile/net.txt"; 
char embedding_file[100] = "tmpfile/emb_u";
char v_embedding_file[100] = "tmpfile/emb_v";

// Basic Parameters
double NEG_SAMPLING_POWER = 0.75;
int num_threads = 8, dim = 128, num_negative = 5;
int tp = 1; 
/* 0 for undirected (a.k.a. LINE-1st, type-I, NCE-x)
 * 1 for directed (a.k.a. LINE-2nd, type-II, NCE-h)
 */
int rns = 0; 
/* 0 for not use robust negative sampling
 * 1 for only use embedding norm penalty
 * 2 for only use an adaptive negative sampler
 * 3 for use both embedding penalty and the adaptive negative sampler
 */
int *neg_table;
float *degree; // degree for the sum-net
int *adj; // Adajacent Matrix
long long num_memblock = 0;
long long total_samples = 300, current_sample_count = 0;
long long num_edges = 0;
float init_rho = 0.025, rho;
float *emb_vertex, *emb_context;
float *sigmoid_table;
int *edge_source_id, *edge_target_id;
double *edge_weight; 
long long *alias;
double *prob;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

// params for RNS
double lam = 0;

/* Read degree */
void ReadDegree()
{
//    printf("Reading degree...\n");
    degree = (float *)malloc(num_memblock * sizeof(float));
    for (int i = 0; i < num_memblock; i++) degree[i] = 0;

    FILE *fin;
    float weight;
    int vid1, vid2;

    fin = fopen(network_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: network file not found!\n");
        exit(1);
    }

    for (int k = 0; k != num_edges; k++)
    {
        fscanf(fin, "%d %d %f\n", &vid1, &vid2, &weight);
		degree[vid1] += weight;
		degree[vid2] += weight;
    }

    fclose(fin);
}

void ReadAdj()
{
    adj = (int *)malloc((long long)num_memblock * (long long)num_memblock * (long long)sizeof(int));
    if (adj == NULL) {
        cout << "Error: memory allocation of the adjacency matrix failed!" << endl;
    }
    for (long long c=0; c!= (long long)num_memblock * (long long)num_memblock; c++)
        adj[c] = 0;

    FILE *fin;
    int vid1, vid2;
    float weight;

    fin = fopen(network_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: network file not found!\n");
        exit(1);
    }

    for (int k = 0; k != num_edges; k++)
    {
        fscanf(fin, "%d %d %f\n", &vid1, &vid2, &weight);
        adj[(long long)vid2 + num_memblock * (long long)vid1] = 1;
        // add self-loop
        adj[(long long)vid2 + num_memblock * (long long)vid2] = 1;
    }

    fclose(fin);
}

/* Read sum-net from the training file */
void ReadData()
{
    FILE *fin;
    char str[300];
    double weight;
    int vid1, vid2;
	int max_vid = 0;

    fin = fopen(network_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: network file not found!\n");
        exit(1);
    }
    num_edges = 0;
    while (fgets(str, sizeof(str), fin)) num_edges++;
    fclose(fin);
    printf("Number of edges: %lld          \n", num_edges);

    edge_source_id = (int *)malloc(num_edges * sizeof(int));
    edge_target_id = (int *)malloc(num_edges * sizeof(int));
    edge_weight = (double *)malloc(num_edges * sizeof(double));
    if (edge_source_id == NULL || edge_target_id == NULL || edge_weight == NULL) {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    
    // read num_vertices and edges
    fin = fopen(network_file, "rb");

    for (int k = 0; k != num_edges; k++)
    {
        fscanf(fin, "%d %d %lf\n", &vid1, &vid2, &weight);
        
        edge_source_id[k] = vid1;
        edge_target_id[k] = vid2;
        edge_weight[k] = weight;

		max_vid = max_vid > vid1 ? max_vid: vid1;
		max_vid = max_vid > vid2 ? max_vid: vid2;

    }
	num_memblock = max_vid + 1;
    printf("num memblock: %lld\n", num_memblock);

    fclose(fin);

    ReadDegree();
}

void InitLambda()
{
	double d, k, l, m, n;
	n = (double)num_memblock;
	d = 2. * num_edges / n;
    // nce-x
	if (tp == 0) l = log((double)dim) / (log(2) * 6) + 2;
    // nce-h
	else l = log((double)dim) / (log(2) * 6) + 1.9;
	m = log(1 + d / 2) / log(1 + n);
	k = num_negative;
	if (tp == 0){
		lam = (m / (1. + exp(l * l * m)) + k * sqrt(1. - m * m) / (1. + exp(l * l * sqrt(1. - m * m))))/ ((k + 2) / (k + 1) + (k * (n - 2)) / ((k + 1) * (n - d - 1)));
	} else {
		lam = (m / (1. + exp(l * l * m)) + k * sqrt(1. - m * m) / (1. + exp(l * l * sqrt(1. - m * m))))/ 2;
	}
	printf("Init params.. n: %.2lf, d: %.2lf, l: %.2lf, m: %.2lf, k: %.2lf lambda: %.6lf\n", n, d, l, m, k, lam);
}

void InitAliasTable()
{
    printf("Initializing Alias Table...\n");
    alias = (long long *)malloc(num_edges*sizeof(long long));
    prob = (double *)malloc(num_edges*sizeof(double));
    if (alias == NULL || prob == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double *norm_prob = (double*)malloc(num_edges*sizeof(double));
    long long *large_block = (long long*)malloc(num_edges*sizeof(long long));
    long long *small_block = (long long*)malloc(num_edges*sizeof(long long));
    if (norm_prob == NULL || large_block == NULL || small_block == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }

    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;

    for (long long k = 0; k != num_edges; k++) sum += edge_weight[k];
    for (long long k = 0; k != num_edges; k++) 
        norm_prob[k] = edge_weight[k] * num_edges / sum;
    
    // init alias map
    for (long long k = num_edges - 1; k >= 0; k--)
    {
        if (norm_prob[k]<1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }

    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        prob[cur_small_block] = norm_prob[cur_small_block];
        alias[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }

    while (num_large_block) prob[large_block[--num_large_block]] = 1;
    while (num_small_block) prob[small_block[--num_small_block]] = 1;

    free(norm_prob);
    free(small_block);
    free(large_block);
}

// Single-Net edge sampling
long long SampleAnEdge(double rand_value1, double rand_value2)
{
    long long k = (long long)num_edges * rand_value1;
    // k and alias[k] is index
    return rand_value2 < prob[k] ? k : alias[k];
}

/* Initialize the vertex embedding and the context embedding */
void InitVector()
{
    printf("Initializing Vectors...\n");

    long long a, b;

    a = posix_memalign((void **)&emb_vertex, dim, (long long)num_memblock * dim * sizeof(float));
    if (emb_vertex == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
    for (b = 0; b < dim; b++) for (a = 0; a < num_memblock; a++)
        emb_vertex[a * dim + b] = (rand() / (float)RAND_MAX - 0.5) / dim;

    if (tp == 1) // directed graph
    {
        a = posix_memalign((void **)&emb_context, dim, (long long)num_memblock * dim * sizeof(float));
        if (emb_context == NULL) { printf("Error: memory allocation failed\n"); exit(1); }
        for (b = 0; b < dim; b++) for (a = 0; a < num_memblock; a++)
            emb_context[a * dim + b] = 0;
    }

}

/* Sample negative vertex samples according to vertex degrees */
void InitNegTable()
{
    printf("Initializing Neg Table...\n");

    double sum=0, cur_sum = 0, por = 0;
    int vid = 0;
    neg_table = (int *)malloc(neg_table_size * sizeof(int));
    if (neg_table == NULL) {
        printf("neg table malloc error.\n");
        exit(1);
    }

    for (int k = 0; k != num_memblock; k++)
        sum += pow(degree[k], NEG_SAMPLING_POWER);
    for (int k = 0; k != neg_table_size; k++)
    {
        if ((double)(k + 1) / neg_table_size > por)
        {
            cur_sum += pow(degree[vid], NEG_SAMPLING_POWER);
            por = cur_sum / sum;
            vid ++;
        }
        neg_table[k] = vid - 1;
    }
}

/* Fastly compute sigmoid function */
void InitSigmoidTable()
{
    float x;
    sigmoid_table = (float *)malloc((sigmoid_table_size) * sizeof(float));
    for (int k = 0; k != sigmoid_table_size; k++)
    {
        x = 2.0 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
        sigmoid_table[k] = 1 / (1 + exp(-x));
    }
}

float FastSigmoid(float x)
{
    if (x > SIGMOID_BOUND) return 1;
    else if (x < -SIGMOID_BOUND) return 0;
    int k = (x + SIGMOID_BOUND) * sigmoid_table_size / SIGMOID_BOUND / 2;
    return sigmoid_table[k];
}

/* Fastly generate a random integer */
int fastRandInt(unsigned long long &seed)
{
    seed = seed * 25214903917 + 11;
    return (seed >> 16) % neg_table_size;
}

/* Fastly generate a random double */
double fastRand(unsigned long long &seed)
{
    seed = seed * 25214903917 + 11;
    return ((seed >> 16) % neg_table_size) / (double)neg_table_size;
}

/* Update embeddings */
void Update(float *vec_u, float *vec_v, float *vec_error, int label)
{
    float x = 0, g, score;
    float v_norm = 0;
    for (int c = 0; c != dim; c++) x += vec_u[c] * vec_v[c];
    score = FastSigmoid(x);

    g = (label - score) * rho;
    
    bool norm_penalty = (rns == 1 || rns == 3) && !label;
    if (norm_penalty)
        for (int c = 0; c != dim; c++) v_norm += vec_v[c] * vec_v[c];
    v_norm = sqrt(v_norm);

    if (vec_u == vec_v) {
        g = g * 2;
        for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
		// Robust NS
		if (norm_penalty && v_norm > 0)
			for (int c = 0; c != dim; c++) vec_error[c] -= rho * vec_v[c] * lam / (v_norm * (num_negative + 1));
    } else {
        for (int c = 0; c != dim; c++) vec_error[c] += g * vec_v[c];
		// Robust NS
		if (norm_penalty && v_norm > 0) 
			for (int c = 0; c != dim; c++) vec_v[c] -= rho * vec_v[c] * lam / (v_norm * (num_negative + 1));
        for (int c = 0; c != dim; c++) vec_v[c] += g * vec_u[c];
    }
}

void *TrainLINEThread(void *id)
{
    long long u, v, lu, lv, target=0, label=0;
    long long count = 0, last_count = 0, curedge;
    unsigned long long seed = (long long)id;
    int d = 0;
    float u_norm = 0;

    float *vec_error_u = (float *)calloc(dim, sizeof(float));
    if (vec_error_u == NULL) 
    {
        printf("Memory alloc error\n");
        exit(1);
    }

    while (1)
    {
        if (count > total_samples / num_threads + 2) break;

        if (count - last_count > 10000)
        {
            current_sample_count += count - last_count;
            last_count = count;
            printf("%cRho: %f  Progress: %.3lf%%", 13, rho, (float)current_sample_count / (float)(total_samples + 1) * 100);
            fflush(stdout);
            rho = init_rho * (1 - current_sample_count / (float)(total_samples + 1));
            if (rho < init_rho * 0.0001) rho = init_rho * 0.0001;
        }

        curedge = SampleAnEdge(gsl_rng_uniform(gsl_r), gsl_rng_uniform(gsl_r));
        u = edge_source_id[curedge];
        v = edge_target_id[curedge];

        lu = u * dim;

        // Negtive Sampling
        d = 0;
		for (int c=0; c<dim; c++) vec_error_u[c] = 0.;
        while (d < num_negative + 1)
        {
            if (d == 0) {
                target = v;
                label = 1;
            } else {
                target = neg_table[fastRandInt(seed)];
                label = 0;
                // adaptive negative sampler
                while ((rns == 2 || rns == 3) 
                        && adj[(long long)u * num_memblock + target]) {
                    target = neg_table[fastRandInt(seed)];
                }
            }
            lv = target * dim;
            if (tp) Update(&emb_vertex[lu], &emb_context[lv], vec_error_u, label);
            else Update(&emb_vertex[lu], &emb_vertex[lv], vec_error_u, label);
            d++;
        }
        // Update emb_vertex[lu]
        if (rns == 1 || rns == 3){
            u_norm = 0;
            for (int c = 0; c != dim; c++) u_norm += emb_vertex[c + lu] * emb_vertex[c + lu];
            u_norm = sqrt(u_norm);
		    for (int c = 0; c != dim; c++) vec_error_u[c] -= rho * lam * emb_vertex[c + lu] / u_norm;
        }
        for (int c = 0; c != dim; c++) emb_vertex[c + lu] += vec_error_u[c];
        count++;
		//cout << emb_vertex[lu] << endl;
    }
    free(vec_error_u);
    pthread_exit(NULL);
}

void Output()
{
    printf("Writiing Embedding...\n");
    
    FILE *fo;

    // writing embedding file
    fo = fopen(embedding_file, "wb");
    fprintf(fo, "%lld %d\n", num_memblock, dim);
    for (int a = 0; a < num_memblock; a++)
        if (degree[a] >= 1) {
            fprintf(fo, "%d ", a);
            for (int b = 0; b < dim; b++) fprintf(fo, "%f ", emb_vertex[a * dim + b]);                
            fprintf(fo, "\n");
        }
    fclose(fo);
   
    if (tp == 1) { // directed graph
        // writing hidden embedding file
        fo = fopen(v_embedding_file, "wb");
        fprintf(fo, "%lld %d\n", num_memblock, dim);
        for (int a = 0; a < num_memblock; a++)
            if (degree[a] >= 1) {
                fprintf(fo, "%d ", a);
                for (int b = 0; b < dim; b++) fprintf(fo, "%f ", emb_context[a * dim + b]);
                fprintf(fo, "\n");
            }
        fclose(fo);
    }
}

void PrintParameters()
{
    printf("--------------------------------\n");
    printf("File Settings:  \n");
    printf("Net:\t%s \n", network_file);
    printf("Emb-u:\t%s \n", embedding_file);
    if (tp == 1) printf("Emb-v:\t%s \n", v_embedding_file);
    printf("--------------------------------\n");
    printf("Samples: %lldM\n", total_samples / 1000000);
    printf("Negative: %d\n", num_negative);
    printf("Dimension: %d\n", dim);
    printf("Initial rho: %lf\n", init_rho);
    printf("RNS level: %d\n", rns);
    if (rns == 2 || rns == 3)
        printf("WARNING: the process may be blocked if the network is a dense one or there is some nodes whose connections are very dense.\n");
    if (tp == 0) printf("NCE-x (LINE-1st) Modeling...\n");
    else if (tp == 1) printf("NCE-h (LINE-2nd) Modeling...\n");
    printf("--------------------------------\n");
}

void TrainLINE() 
{
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

    PrintParameters();

    ReadData();
    InitAliasTable();
    InitVector();
    InitNegTable();
    InitSigmoidTable();

	if (rns == 1 || rns == 3) InitLambda();
	if (rns == 2 || rns == 3) ReadAdj();

    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);

    clock_t start = clock();
    printf("--------------------------------\n");
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainLINEThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf s\n", (double)(finish - start) / CLOCKS_PER_SEC / num_threads);

    Output();
}

int ArgPos(char *str, int argc, char **argv)
{
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) 
{
    int i;
    printf("Parameters: \n");
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) num_negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) total_samples = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-rho", argc, argv)) > 0) init_rho = atof(argv[i + 1]);

    if ((i = ArgPos((char *)"-type", argc, argv)) > 0) tp = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-dim", argc, argv)) > 0) dim = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-input", argc, argv)) > 0) strcpy(network_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-thread", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);

    if ((i = ArgPos((char *)"-emb-u", argc, argv)) > 0) strcpy(embedding_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-emb-v", argc, argv)) > 0) strcpy(v_embedding_file, argv[i + 1]);

    if ((i = ArgPos((char *)"-rns", argc, argv)) > 0) rns = atoi(argv[i + 1]);


    total_samples *= 1000000;
    rho = init_rho;
    TrainLINE();
    return 0;
}
