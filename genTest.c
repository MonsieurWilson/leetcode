#include <stdio.h>
#include <unistd.h>

int main(int argc, const char **argv) {
    printf("Input the N value:\n");
    int N;
    scanf("%d", &N);
    const char *fname = "input";
    FILE *fp = fopen(fname, "w");
    fprintf(fp, "%d\n", N);
    int i;
    for (i = 0; i < N; ++i) {
        fprintf(fp, "%d ", i);
    }
    fprintf(fp, "\n");
    fclose(fp);
    return 0;
}
