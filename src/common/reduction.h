#define PLAIN_LENGTH 52
#define CHARSET_LENGTH 62

void reduceV1(unsigned long int columnIndex, const char* hash, char* plain);
void reduceV2(unsigned long int columnIndex, const char* hash, char* plain);
void reduceV3(unsigned long int columnIndex, const char* hash, char* plain);
void reduceV4(unsigned long int columnIndex, const char* hash, char* plain);
void reduceV5(unsigned long int columnIndex, const char* hash, char* plain);
//void reduceRainbowCrackalack(unsigned long int columnIndex, const char* hash, char* plain);