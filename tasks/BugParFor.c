#include <omp.h>
#include <stdio.h>

int main()
{
	omp_set_dynamic(0);
	omp_set_num_threads(16);

	const size_t N     = 100;
	const size_t chunk = 5;

	int i, tid;
	float a[N], b[N], c[N];

#pragma omp parallel for
	for (i = 0; i < (int)N; ++i) {
		a[i] = b[i] = (float)i;
	}

#pragma omp parallel for shared(a, b, c, chunk) private(i, tid) schedule(static, chunk)
	for (i = 0; i < (int)N; ++i) {
		tid  = omp_get_thread_num();
		c[i] = a[i] + b[i];
		printf("tid = %d, c[%d] = %f\n", tid, i, c[i]);
	}

	return 0;
}
