#ifndef GPU_CRACK_FILTRATION2_CUH
#define GPU_CRACK_FILTRATION2_CUH

#include <unordered_map>
#include <string>

long *
filter2(char *start_path, char *end_path, const char *start_out_path, const char *end_out_path, char *path,
        unsigned long long mtMax);

#endif //GPU_CRACK_FILTRATION2_CUH
