#include <omp.h>
#include <stdio.h>

float dotprod(float *a, float *b, size_t n)
{
	int i, tid = 0;
	float sum = 0;
#pragma omp parallel for reduction(+ : sum)
	for (i = 0; i < (int)n; ++i) {
		tid = omp_get_thread_num();
		sum += a[i] * b[i];
		printf("tid = %d i = %d\n", tid, i);
	}

	return sum;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[])
{
	const size_t N = 100;
	int i          = 0;
	float sum      = 0;
	float a[N], b[N];

	omp_set_dynamic(0);
	omp_set_num_threads(16);

	for (i = 0; i < (int)N; ++i) {
		a[i] = b[i] = i;
	}

	sum = dotprod(a, b, N);

	printf("sum = %f\n", sum);

	return 0;
}
