#include <iostream>
#include <fstream>
#include <string>

void writeByteData(unsigned char *bytes, size_t byteLength, char *tgtFilePath, int *status) {
    FILE *pFile = fopen(tgtFilePath, "wb");
    if (pFile == NULL) {
        printf("Failed to open input file. 3\n");
        *status = 0;
        return;
    }

    fwrite(bytes, 1, byteLength, pFile); //write outSize bytes
    fclose(pFile);
    *status = 0;
}

int main(int argc, char* argv[]) {
    std::ifstream input(argv[1]);
    char str[256];
    double v;
    int length = 0;
    if (input) {
        while (input) {
            input.getline(str, 256);
            if (str[0] != '\0') {
                length++;
            }
        }
    }
    input.clear();
    input.seekg(0);

    double* data = (double*) malloc(sizeof(double) * length);
    int cnt = 0;

    while (input) {
        input.getline(str, 256);
        if (str[0] != '\0') {
            v = std::stod(str);
            data[cnt] = v;
            cnt++;
        }
    }

    writeByteData((unsigned char*)data, sizeof(double) * length, argv[2], &cnt);
    input.close();
    free(data);
    return 0;
}