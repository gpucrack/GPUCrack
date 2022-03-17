#include "tableCoverageTest.h"

unsigned char charset[CHARSET_LENGTH] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F',
                                         'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                                         'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                                         'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

void swap(char *v1, char *v2, long size) {
    char * buffer = malloc(size);

    memcpy(buffer, v1, size);
    memcpy(v1, v2, size);
    memcpy(v2, buffer, size);

    free(buffer);
}

void q_sort(char *v, long size, long left, long right) {
    char *vt, *v3;
    long i, last, mid = (left + right) / 2;
    if (left >= right) {
        return;
    }

    // v left value
    char *vl = (v + (left*size));
    // v right value
    char *vr = (v + (mid*size));

    swap(vl, vr, size);

    last = left;
    for (i = left + 1; i <= right; i++) {

        // vl and vt will have the starting address
        // of the elements which will be passed to
        // comp function.
        vt = (v + i*size);
        if (memcmp(vl, vt, size) > 0) {
            ++last;
            v3 = (v + (last*size));
            swap(vt, v3, size);
        }
    }
    v3 = (v + (last*size));
    swap(vl, v3, size);
    q_sort(v, size, left, last - 1);
    q_sort(v, size, last + 1, right);
}

unsigned long long power(unsigned long long x, unsigned long long y) {
    if (y == 0) return 0;
    unsigned long long res = 1;
    for(unsigned long long i = 0; i < y; i++) {
        res *= x;
    }
    return res;
}

// Number of possible passwords given its domain (length and number of characters used).
unsigned long long compute_N(unsigned char pwd_length) {
    return power((unsigned long long) CHARSET_LENGTH, (unsigned long long) pwd_length);
}

// Compute qi using the approximate formula (alpha must be > 0.9)
unsigned long long compute_qi(unsigned long m, unsigned int t, unsigned long long N, unsigned long i) {
    return 1 - (m / N) - (i * (i - 1))/(t * (t + 1));
}

unsigned long long compute_pk(unsigned long m, unsigned long long N, unsigned long k) {
    return (m / N) * power((1 - (m / N)), k-1);
}

// l: number of tables
unsigned long long compute_atk_time(unsigned long m, unsigned char l, unsigned int t, unsigned long long N) {

    unsigned long long left_part = 0;

    for(unsigned long k = 1; k < (l * t) + 1; k++) {
        unsigned int c = t - ((k - 1) / l);

        // Compute the sum on the left parenthesis with qi
        unsigned long long left_qsum = 0;
        for(unsigned int i = c; i < t + 1; i++) {
            left_qsum += compute_qi(m, t, N, i) * i;
        }

        left_part += compute_pk(m, N, k) * ((((t - c) * (t - c + 1))/2) + left_qsum) * l;
    }

    // Compute the sum on the right parenthesis with qi
    unsigned long long right_qsum = 0;
    for(unsigned int i = 1; i < t + 1; i++) {
        right_qsum += compute_qi(m, t, N, i) * i;
    }

    unsigned long long right_part = (power(1 - (m / N), l * t)) * (((t * (t - 1)) / 2) + right_qsum) * l;

    return left_part + right_part;
}

unsigned long search_endpoint(char **endpoints, char *plain_text, unsigned long mt, int pwd_length) {
    unsigned long lower = 0;
    unsigned long upper = mt - 1;
    unsigned long step = sizeof(char) * pwd_length;

    while (lower <= upper) {
        unsigned long mid = 1 + (lower + (upper - lower) / 2);
        if (upper == 0) {
            mid = 0;
        } else if (lower == mt-1) {
            mid = mt - 1;
        }
        int compare = memcmp(&(*endpoints)[mid*step], plain_text, pwd_length);
        // Match found
        if (compare == 0) {
            return mid;
        } else if (compare < 0) {
            if (mid == 0)
                break;
            lower = mid + 1;
        }
        else if (lower == upper) {
            return -1; // not found
        } else {
            if (mid == 0)
                break;
            upper = mid - 1;
        }

    }

    return -1; // not found
}

void char_to_password(char text[], Password *password, int pwd_length) {
    for (int i = 0; i < pwd_length; i++) {
        password->bytes[i] = text[i];
    }
}

void password_to_char(Password *password, char text[], int pwd_length) {
    for (int i = 0; i < pwd_length; i++) {
        text[i] = password->bytes[i];
    }
}

void char_to_digest(char text[], Digest *digest, int len) {
    for (int i = 0; i < len; i++) {
        char hex[2];
        hex[0] = text[i * 2];
        hex[1] = text[(i * 2) + 1];
        uint32_t num = (int) strtol(hex, NULL, 16);
        digest->bytes[i] = num;
    }
}

void display_digest(Digest *digest) {
    for (unsigned char i = 0; i < HASH_LENGTH; i++) {
        printf("%02X", (unsigned char) digest->bytes[i]);
    }
}

void reduce_digest(char *char_digest, unsigned int index, char *char_plain, int pwd_length) {
    Digest *digest = (Digest *) malloc(sizeof(Digest));
    char_to_digest(char_digest, digest, pwd_length);

    Password *plain_text = (Password *) malloc(sizeof(Password));
    char_to_password("abcdefg", plain_text, pwd_length);

    for(int i=0; i<pwd_length; i++){
        (*plain_text).bytes[i] = charset[((*digest).bytes[i] + index) % CHARSET_LENGTH];
    }

    password_to_char(plain_text, char_plain, pwd_length);

    free(digest);
    free(plain_text);
}

void ntlm(char *key, char *hash, int pwd_length) {
    int i, j;
    int length = pwd_length;
    unsigned int nt_buffer[16], a, b, c, d, sqrt2, sqrt3, n, output[4];
    static char hex_format[33];
    char itoa16[16] = "0123456789abcdef";

    memset(nt_buffer, 0, 16 * sizeof(unsigned int));
    //The length of key need to be <= 27
    for (i = 0; i < length / 2; i++)
        nt_buffer[i] = key[2 * i] | (key[2 * i + 1] << 16);

    //padding
    if (length % 2 == 1)
        nt_buffer[i] = key[length - 1] | 0x800000;
    else
        nt_buffer[i] = 0x80;
    //put the length
    nt_buffer[14] = length << 4;

    output[0] = a = 0x67452301;
    output[1] = b = 0xefcdab89;
    output[2] = c = 0x98badcfe;
    output[3] = d = 0x10325476;
    sqrt2 = 0x5a827999;
    sqrt3 = 0x6ed9eba1;

    /* Round 1 */
    a += (d ^ (b & (c ^ d))) + nt_buffer[0];
    a = (a << 3) | (a >> 29);
    d += (c ^ (a & (b ^ c))) + nt_buffer[1];
    d = (d << 7) | (d >> 25);
    c += (b ^ (d & (a ^ b))) + nt_buffer[2];
    c = (c << 11) | (c >> 21);
    b += (a ^ (c & (d ^ a))) + nt_buffer[3];
    b = (b << 19) | (b >> 13);

    a += (d ^ (b & (c ^ d))) + nt_buffer[4];
    a = (a << 3) | (a >> 29);
    d += (c ^ (a & (b ^ c))) + nt_buffer[5];
    d = (d << 7) | (d >> 25);
    c += (b ^ (d & (a ^ b))) + nt_buffer[6];
    c = (c << 11) | (c >> 21);
    b += (a ^ (c & (d ^ a))) + nt_buffer[7];
    b = (b << 19) | (b >> 13);

    a += (d ^ (b & (c ^ d))) + nt_buffer[8];
    a = (a << 3) | (a >> 29);
    d += (c ^ (a & (b ^ c))) + nt_buffer[9];
    d = (d << 7) | (d >> 25);
    c += (b ^ (d & (a ^ b))) + nt_buffer[10];
    c = (c << 11) | (c >> 21);
    b += (a ^ (c & (d ^ a))) + nt_buffer[11];
    b = (b << 19) | (b >> 13);

    a += (d ^ (b & (c ^ d))) + nt_buffer[12];
    a = (a << 3) | (a >> 29);
    d += (c ^ (a & (b ^ c))) + nt_buffer[13];
    d = (d << 7) | (d >> 25);
    c += (b ^ (d & (a ^ b))) + nt_buffer[14];
    c = (c << 11) | (c >> 21);
    b += (a ^ (c & (d ^ a))) + nt_buffer[15];
    b = (b << 19) | (b >> 13);

    /* Round 2 */
    a += ((b & (c | d)) | (c & d)) + nt_buffer[0] + sqrt2;
    a = (a << 3) | (a >> 29);
    d += ((a & (b | c)) | (b & c)) + nt_buffer[4] + sqrt2;
    d = (d << 5) | (d >> 27);
    c += ((d & (a | b)) | (a & b)) + nt_buffer[8] + sqrt2;
    c = (c << 9) | (c >> 23);
    b += ((c & (d | a)) | (d & a)) + nt_buffer[12] + sqrt2;
    b = (b << 13) | (b >> 19);

    a += ((b & (c | d)) | (c & d)) + nt_buffer[1] + sqrt2;
    a = (a << 3) | (a >> 29);
    d += ((a & (b | c)) | (b & c)) + nt_buffer[5] + sqrt2;
    d = (d << 5) | (d >> 27);
    c += ((d & (a | b)) | (a & b)) + nt_buffer[9] + sqrt2;
    c = (c << 9) | (c >> 23);
    b += ((c & (d | a)) | (d & a)) + nt_buffer[13] + sqrt2;
    b = (b << 13) | (b >> 19);

    a += ((b & (c | d)) | (c & d)) + nt_buffer[2] + sqrt2;
    a = (a << 3) | (a >> 29);
    d += ((a & (b | c)) | (b & c)) + nt_buffer[6] + sqrt2;
    d = (d << 5) | (d >> 27);
    c += ((d & (a | b)) | (a & b)) + nt_buffer[10] + sqrt2;
    c = (c << 9) | (c >> 23);
    b += ((c & (d | a)) | (d & a)) + nt_buffer[14] + sqrt2;
    b = (b << 13) | (b >> 19);

    a += ((b & (c | d)) | (c & d)) + nt_buffer[3] + sqrt2;
    a = (a << 3) | (a >> 29);
    d += ((a & (b | c)) | (b & c)) + nt_buffer[7] + sqrt2;
    d = (d << 5) | (d >> 27);
    c += ((d & (a | b)) | (a & b)) + nt_buffer[11] + sqrt2;
    c = (c << 9) | (c >> 23);
    b += ((c & (d | a)) | (d & a)) + nt_buffer[15] + sqrt2;
    b = (b << 13) | (b >> 19);

    /* Round 3 */
    a += (d ^ c ^ b) + nt_buffer[0] + sqrt3;
    a = (a << 3) | (a >> 29);
    d += (c ^ b ^ a) + nt_buffer[8] + sqrt3;
    d = (d << 9) | (d >> 23);
    c += (b ^ a ^ d) + nt_buffer[4] + sqrt3;
    c = (c << 11) | (c >> 21);
    b += (a ^ d ^ c) + nt_buffer[12] + sqrt3;
    b = (b << 15) | (b >> 17);

    a += (d ^ c ^ b) + nt_buffer[2] + sqrt3;
    a = (a << 3) | (a >> 29);
    d += (c ^ b ^ a) + nt_buffer[10] + sqrt3;
    d = (d << 9) | (d >> 23);
    c += (b ^ a ^ d) + nt_buffer[6] + sqrt3;
    c = (c << 11) | (c >> 21);
    b += (a ^ d ^ c) + nt_buffer[14] + sqrt3;
    b = (b << 15) | (b >> 17);

    a += (d ^ c ^ b) + nt_buffer[1] + sqrt3;
    a = (a << 3) | (a >> 29);
    d += (c ^ b ^ a) + nt_buffer[9] + sqrt3;
    d = (d << 9) | (d >> 23);
    c += (b ^ a ^ d) + nt_buffer[5] + sqrt3;
    c = (c << 11) | (c >> 21);
    b += (a ^ d ^ c) + nt_buffer[13] + sqrt3;
    b = (b << 15) | (b >> 17);

    a += (d ^ c ^ b) + nt_buffer[3] + sqrt3;
    a = (a << 3) | (a >> 29);
    d += (c ^ b ^ a) + nt_buffer[11] + sqrt3;
    d = (d << 9) | (d >> 23);
    c += (b ^ a ^ d) + nt_buffer[7] + sqrt3;
    c = (c << 11) | (c >> 21);
    b += (a ^ d ^ c) + nt_buffer[15] + sqrt3;
    b = (b << 15) | (b >> 17);

    output[0] += a;
    output[1] += b;
    output[2] += c;
    output[3] += d;
    //Iterate the integer
    for (i = 0; i < 4; i++)
        for (j = 0, n = output[i]; j < 4; j++) {
            unsigned int convert = n % 256;
            hex_format[i * 8 + j * 2 + 1] = itoa16[convert % 16];
            convert = convert / 16;
            hex_format[i * 8 + j * 2 + 0] = itoa16[convert % 16];
            n = n / 256;
        }
    //null terminate the string
    hex_format[33] = 0;
    strcpy(hash, hex_format);
}

int remove_duplicates(char *passwords, int pwd_length, int n) {
    if (n==0 || n==1)
        return n;

    // To store index of next unique element
    int j = 0;

    for (int i=0; i < n-1; i++)
        if (memcmp(passwords + i*pwd_length, passwords + i*pwd_length + pwd_length, pwd_length) != 0) {
            memcpy(passwords + (j++ *pwd_length), passwords + i*pwd_length, pwd_length);
        }
    memcpy(passwords + (j++ *pwd_length), passwords + (n-1) * pwd_length, pwd_length);

    return j;
}

int count_duplicates(char *passwords, int pwd_length, int n) {
    int res = 0;
    for (int i = 0; i < n-1; i++) {
        char *buf1 = passwords + i * pwd_length;
        char *buf2 = passwords + (i + 1) * pwd_length;
        if (memcmp(buf1, buf2, pwd_length) == 0) {
            res++;
        }
    }
    return res;
}

long remove_duplicates_notworking(char *passwords, int pwd_length, long pwd_nb) {
    long index = 1;
    for (long i = 1; i < pwd_nb + 1; i++) {
        char *prev = passwords + i * pwd_length - pwd_length;
        char *actual = passwords + i * pwd_length;

        if (memcmp(prev, actual, pwd_length) == 0) {
            char * indexed = passwords + index * pwd_length;
            memcpy(indexed, actual, pwd_length);
            index++;
        }
    }
    return index-1; // -1 because we initialized it at 1
}

/**
 * Computes the percentage of passwords covered by the given table.
 * Precisely, returns the ratio of numbers of different passwords that appear in the chains of the tables over the
 * number of passwords that can be written using the same number of characters.
 *
 * Does so by recreating every chain from the start points file, and by storing every single password in an array.
 * Then, duplicates are removed from this array, and the number of passwords remaining is divided by the number of
 * passwords that can be written.
 *
 * @param argv[1] the path to the start points file.
 * @return the percentage of coverage of the table
 */
int main(int argc, char *argv[]) {

    // Wrong usage
    if (argc != 2) {
        printf("Error: incorrect number of arguments given.\n"
               "Usage: 'chainCoverageTest startpath', where startpath is the path to the start points file.\n\n");
    }

    const char *start_path = argv[1];

    FILE *fp;
    fp = fopen(start_path, "rb");

    if (fp == NULL) {
        printf("Error: could not open start points file.\n\n");
        exit(1);
    }

    char buff[255];

    printf("=== TABLE DATA ===\n");

    // Retrieve mt
    unsigned long long mt;
    fscanf(fp, "%s", buff);
    sscanf(buff, "%lld", &mt);
    printf("Number of start points (mt): %llu\n", mt);
    fgets(buff, 255, (FILE *) fp);

    // Retrieve the password length
    unsigned int pwd_length;
    fscanf(fp, "%s", buff);
    sscanf(buff, "%d", &pwd_length);
    printf("Password length: %d\n", pwd_length);
    fgets(buff, 255, (FILE *) fp);

    // Retrieve t
    unsigned long long t;
    fgets(buff, 255, (FILE *) fp);
    sscanf(buff, "%lld", &t);
    printf("Chain length (t): %llu\n\n", t);

    // Array containing every single password appearing in the table
    char *passwords = malloc(sizeof(char) * pwd_length * mt * t);

    for (unsigned long long i = 0; i < sizeof(char) * mt; i++) {
        fread(buff, pwd_length, 1, (FILE *) fp);

        // Write the ith start point
        for (unsigned int j = 0; j < pwd_length; j++) {
            passwords[i * t * pwd_length + j] = buff[j];
        }

        char column_plain_text[pwd_length];
        char column_digest[HASH_LENGTH * 2];

        // Copy the current start point and generate its chain
        for (unsigned int j = 0; j < pwd_length; j++) {
            column_plain_text[j] = passwords[i * t * pwd_length + j];
        }

        // Build the ith chain
        for (unsigned int k = 0; k < t; k++) {
            ntlm(column_plain_text, column_digest, pwd_length);
            reduce_digest(column_digest, k, column_plain_text, pwd_length);

            // Save the password in the array
            for (unsigned int j = 0; j < pwd_length; j++) {
                passwords[i * t * pwd_length + (k + 1) * pwd_length + j] = column_plain_text[j];
            }
        }
    }

    q_sort(passwords, pwd_length, 0, (t * mt)-1);

    long nb_duplicates = count_duplicates(passwords, (int) pwd_length, (long) (mt * t));
    unsigned long nb_unique_pwd = mt * t - nb_duplicates;
    printf("This table contains %lu unique passwords.\n", nb_unique_pwd);
    long N = power(CHARSET_LENGTH, pwd_length);
    printf("The domain of %d characters passwords covers %lu passwords in total.\n", pwd_length, N);
    double ratio = (double) nb_unique_pwd / (double) N;
    printf("========> Coverage: %.4f (%.4f %%) <========\n", ratio, ratio * 100);

    // Close the start points file
    fclose(fp);
    exit(0);
}