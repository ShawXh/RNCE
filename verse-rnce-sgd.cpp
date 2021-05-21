#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <omp.h>

using namespace std;

#if defined(__AVX2__) || \
defined(__FMA__)
#define VECTORIZE 1
#define AVX_LOOP _Pragma("omp simd")
#else
#define AVX_LOOP
#endif

#ifndef UINT64_C
#define UINT64_C(c) (c##ULL)
#endif

#define SIGMOID_BOUND 6.0
#define DEFAULT_ALIGN 128

// By Hao
float *exp_table;
// params for RNCE
float beta = 0.;
float gam = 1.; // the parameter of Gaussian/Laplacian kernel
int reg = 0;
/*
 * 0 for no distance function
 * 1 for wasser-1
 * 2 for wasser-2
 * 3 for wasser-3 (* emprically best)
 * 4 for gaussian
 * 5 for laplacian
 */
float ub_w = 0.; // the upper bound of lambda_dis * beta, better set ub_w = 0.1 when use wasser-1 and laplacian
const float EXP_BOUND = 6;
// 

typedef unsigned long long ull;

bool silent = false;
int n_threads = 8;
float global_lr = 0.0025f;
int n_epochs = 100000;
int n_hidden = 128;
int n_samples = 5;
float ppralpha = 0.85f;

ull total_steps;
ull step = 0;

ull nv = 0, ne = 0;
int *offsets;
int *edges;

float *w0;

const int sigmoid_table_size = 1024;
float *sigmoid_table;
const float SIGMOID_RESOLUTION = sigmoid_table_size / (SIGMOID_BOUND * 2.0f);

uint64_t rng_seed[2];

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

// http://xoroshiro.di.unimi.it/#shootout
uint64_t lrand() {
  const uint64_t s0 = rng_seed[0];
  uint64_t s1 = rng_seed[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  rng_seed[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
  rng_seed[1] = rotl(s1, 36);                   // c
  return result;
}

static inline double drand() {
  const union un {
    uint64_t i;
    double d;
  } a = {UINT64_C(0x3FF) << 52 | lrand() >> 12};
  return a.d - 1.0;
}

inline void *aligned_malloc(
  size_t size,
  size_t align) {
#ifndef _MSC_VER
void *result;
if (posix_memalign(&result, align, size)) result = 0;
#else
void *result = _aligned_malloc(size, align);
#endif
return result;
}

inline void aligned_free(void *ptr) {
#ifdef _MSC_VER
_aligned_free(ptr);
#else
free(ptr);
#endif
}

void init_sigmoid_table() {
  float x;
  sigmoid_table = static_cast<float *>(
    aligned_malloc((sigmoid_table_size + 1) * sizeof(float), DEFAULT_ALIGN));
  for (int k = 0; k != sigmoid_table_size; k++) {
    x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
    sigmoid_table[k] = 1 / (1 + exp(-x));
  }
}

float FastSigmoid(float x) {
  if (x > SIGMOID_BOUND)
    return 1;
  else if (x < -SIGMOID_BOUND)
    return 0;
  int k = (x + SIGMOID_BOUND) * SIGMOID_RESOLUTION;
  return sigmoid_table[k];
}

void InitExpTable()
{
    float x;
    exp_table = (float *)malloc((sigmoid_table_size) * sizeof(float));
    for (int k = 0; k != sigmoid_table_size; k++)
    {
        x = 2.0 * EXP_BOUND * k / sigmoid_table_size - EXP_BOUND;
        exp_table[k] = exp(x);
    }
}

float FastExp(float x)
{
    if (x > EXP_BOUND) return exp_table[sigmoid_table_size - 1];
    else if (x < -EXP_BOUND) return exp_table[0];
    int k = (x + EXP_BOUND) * sigmoid_table_size / EXP_BOUND / 2;
    return exp_table[k];
}

inline int irand(int min, int max) { return lrand() % (max - min) + min; }

inline int irand(int max) { return lrand() % max; }

inline int sample_neighbor(int node) {
  if (offsets[node] == offsets[node + 1])
    return -1;
  return edges[irand(offsets[node], offsets[node + 1])];
}

inline int sample_rw(int node) {
  int n2 = node;
  while (drand() < ppralpha) {
    int neighbor = sample_neighbor(n2);
    if (neighbor == -1)
      return n2;
    n2 = neighbor;
  }
  return n2;
}

int ArgPos(char *str, int argc, char **argv) {
  for (int a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        cout << "Argument missing for " << str << endl;
        exit(1);
      }
      return a;
    }
  return -1;
}

// Regularizer
inline void UpdateReg(float *vec_u, float *vec_v, float *vec_error, float sigmoid_score)
{
	float lam_dis, dis, dis2, g;
	lam_dis = 0;
	dis2 = 0;
	
	if (!reg) return;

	switch(reg) {
		case 1:
			// Wasserstein-1
			for (int c = 0; c != n_hidden; c++) dis2 += (vec_u[c] - vec_v[c]) * (vec_u[c] - vec_v[c]);
			lam_dis = 1. / (sqrt(dis2) + 1e-8);
			break;
		case 2:
			// Wasserstein-2
			lam_dis = 1.; // should be 2 but 1 will not hurt
			break;
		case 3:
			// Wasserstein-3
			for (int c = 0; c != n_hidden; c++) dis2 += (vec_u[c] - vec_v[c]) * (vec_u[c] - vec_v[c]);
			lam_dis = 3 * sqrt(dis2);
			break;
		case 4:
			// Gaussian
			for (int c = 0; c != n_hidden; c++) dis2 += (vec_u[c] - vec_v[c]) * (vec_u[c] - vec_v[c]);
			lam_dis = gam * FastExp(-gam * dis2) * 2;
			break;
		case 5:
			// Laplacian
			for (int c = 0; c != n_hidden; c++) dis2 += (vec_u[c] - vec_v[c]) * (vec_u[c] - vec_v[c]);
			dis = sqrt(dis2);
			lam_dis = gam * FastExp(-gam * dis) / (dis + 1e-8);
			break;
		default:
			break;
	}

	lam_dis *= beta;
	if (ub_w > 1e-6 && lam_dis > ub_w) lam_dis = ub_w;

	for (int c = 0; c != n_hidden; c++) {
	    g = lam_dis * (vec_u[c] - vec_v[c]) * global_lr;
	    vec_error[c] += -g;
	    vec_v[c] += g;
	}
}

/* Update embeddings */
inline void my_update(float *vec_u, float *vec_v, float *vec_error, int label, const float bias)
{
    float x, g, score;
	x = -bias;
    for (int c = 0; c != n_hidden; c++) x += vec_u[c] * vec_v[c];
    score = FastSigmoid(x);

    g = (label - score) * global_lr;
    
    if (vec_u == vec_v) {
        g = g * 2;
        for (int c = 0; c != n_hidden; c++) vec_error[c] += g * vec_v[c];
    } else {
        for (int c = 0; c != n_hidden; c++) vec_error[c] += g * vec_v[c];
		// Regularized NCE
		if (reg && label)
			UpdateReg(vec_u, vec_v, vec_error, score);
        for (int c = 0; c != n_hidden; c++) vec_v[c] += g * vec_u[c];
    }
}

inline void update(float *w_s, float *w_t, int label, const float bias) {
  float score = -bias;
AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    score += w_s[c] * w_t[c];
  score = (label - FastSigmoid(score)) * global_lr;
AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_t[c] += score * w_s[c];
AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_s[c] += score * w_t[c];
}

void Train() {
#pragma omp parallel num_threads(n_threads)
  {

	// Hao
    float *vec_error_u = (float *)calloc(n_hidden, sizeof(float));
    if (vec_error_u == NULL) 
    {
        printf("Memory alloc error\n");
        exit(1);
    }
	// 

    const float nce_bias = log(nv);
    const float nce_bias_neg = log(nv / float(n_samples));
    int tid = omp_get_thread_num();
    ull last_ncount = 0;
    ull ncount = 0;
    float lr = global_lr;
#pragma omp barrier
    while (1) {
      if (ncount - last_ncount > 10000) {
        ull diff = ncount - last_ncount;
#pragma omp atomic
        step += diff;
        if (step > total_steps)
          break;
        if (tid == 0)
          if (!silent)
            cout << fixed << "\r Progress " << std::setprecision(2)
                 << step / (float)(total_steps + 1) * 100 << "%";
        last_ncount = ncount;
      }
	  
	  // Hao
	  for (int c=0; c<n_hidden; c++) vec_error_u[c] = 0.;
	  //

      size_t n1 = irand(nv);
      size_t n2 = sample_rw(n1);
      // update(&w0[n1 * n_hidden], &w0[n2 * n_hidden], 1, nce_bias);
      my_update(&w0[n1 * n_hidden], &w0[n2 * n_hidden], vec_error_u, 1, nce_bias);
      for (int c = 0; c != n_hidden; c++) w0[c + n1 * n_hidden] += vec_error_u[c];
	  //
      for (int i = 0; i < n_samples; i++) {
        size_t neg = irand(nv);
        update(&w0[n1 * n_hidden], &w0[neg * n_hidden], 0, nce_bias_neg);
      }
      ncount++;
    }
  }
}

void Output(string embedding_file)
{
    printf("Writiing Embedding...\n");
    
    FILE *fo;
	char embf[100];
	strcpy(embf, embedding_file.c_str());

    // writing embedding file
    fo = fopen(embf, "wb");
    fprintf(fo, "%lld %d\n", nv, n_hidden);
    for (int a = 0; a < nv; a++) {
        fprintf(fo, "%d ", a);
        for (int b = 0; b < n_hidden; b++) fprintf(fo, "%f ", w0[a * n_hidden + b]);                
        fprintf(fo, "\n");
    }
    fclose(fo);
}

int main(int argc, char **argv) {
  int a = 0;
  string network_file, embedding_file;
  ull x = time(nullptr);
  for (int i = 0; i < 2; i++) {
    ull z = x += UINT64_C(0x9E3779B97F4A7C15);
    z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
    rng_seed[i] = z ^ z >> 31;
  }
  init_sigmoid_table();
  if ((a = ArgPos(const_cast<char *>("-input"), argc, argv)) > 0)
    network_file = argv[a + 1];
  else {
    cout << "Input file not given! Aborting now.." << endl;
    getchar();
    return 0;
  }
  if ((a = ArgPos(const_cast<char *>("-output"), argc, argv)) > 0)
    embedding_file = argv[a + 1];
  else {
    cout << "Output file not given! Aborting now.." << endl;
    getchar();
    return 0;
  }
  if ((a = ArgPos(const_cast<char *>("-dim"), argc, argv)) > 0)
    n_hidden = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-reg"), argc, argv)) > 0)
    reg = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-beta"), argc, argv)) > 0)
    beta = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-gamma"), argc, argv)) > 0)
    gam = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-ub-w"), argc, argv)) > 0)
    ub_w = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-silent"), argc, argv)) > 0)
    silent = true;
  if ((a = ArgPos(const_cast<char *>("-nsamples"), argc, argv)) > 0)
    n_samples = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-threads"), argc, argv)) > 0)
    n_threads = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-steps"), argc, argv)) > 0)
    n_epochs = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-lr"), argc, argv)) > 0)
    global_lr = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-alpha"), argc, argv)) > 0)
    ppralpha = atof(argv[a + 1]);
  ifstream embFile(network_file, ios::in | ios::binary);
  if (embFile.is_open()) {
    char header[] = "----";
    embFile.seekg(0, ios::beg);
    embFile.read(header, 4);
    if (strcmp(header, "XGFS") != 0) {
      cout << "Invalid header!: " << header << endl;
      return 1;
    }
    embFile.read(reinterpret_cast<char *>(&nv), sizeof(long long));
    embFile.read(reinterpret_cast<char *>(&ne), sizeof(long long));
    offsets = static_cast<int *>(aligned_malloc((nv + 1) * sizeof(int32_t), DEFAULT_ALIGN));
  edges = static_cast<int *>(aligned_malloc(ne * sizeof(int32_t), DEFAULT_ALIGN));
    embFile.read(reinterpret_cast<char *>(offsets), nv * sizeof(int32_t));
    offsets[nv] = (int)ne;
    embFile.read(reinterpret_cast<char *>(edges), sizeof(int32_t) * ne);
    cout << "nv: " << nv << ", ne: " << ne << endl;
    embFile.close();
  } else {
    return 0;
  }
  w0 = static_cast<float *>(aligned_malloc(nv * n_hidden * sizeof(float), DEFAULT_ALIGN));
  for (size_t i = 0; i < nv * n_hidden; i++)
    w0[i] = drand() - 0.5;
  total_steps = n_epochs * (long long)nv;
  cout << "Total steps (mil): " << total_steps / 1000000. << endl;


  // Print RNCE related parameters
  cout << "\n^^ RNCE buff ^^" << endl;
  cout << "Regularized dist. func.: " << reg << endl;
  cout << "Beta: " << beta << endl;
  cout << "Upper bound of w: " << ub_w << endl;
  cout << "Gamma(Laplacian/Gaussian kernel): " << gam << endl;
  cout << endl;
  //
  //


  chrono::steady_clock::time_point begin = chrono::steady_clock::now();
  Train();
  chrono::steady_clock::time_point end = chrono::steady_clock::now();

  cout << endl
       << "Calculations took "
       << chrono::duration_cast<std::chrono::duration<float>>(end - begin)
              .count()
       << " s to run" << endl;
  for (size_t i = 0; i < nv * n_hidden; i++)
    if (w0[0] != w0[0]) {
      cout << endl << "NaN! Not saving the result.." << endl;
      return 1;
    }
  /*
  ofstream output(embedding_file, std::ios::binary);
  output.write(reinterpret_cast<char *>(w0), sizeof(float) * n_hidden * nv);
  output.close();
  */
  // Hao
  Output(embedding_file);
  //
}

