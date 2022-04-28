#include "online.h"

unsigned long long power(unsigned long long x, unsigned long long y) {
    if (y == 0) return 0;
    unsigned long long res = 1;
    for (unsigned long long i = 0; i < y; i++) {
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
    return 1 - (m / N) - (i * (i - 1)) / (t * (t + 1));
}

unsigned long long compute_pk(unsigned long m, unsigned long long N, unsigned long k) {
    return (m / N) * power((1 - (m / N)), k - 1);
}

// l: number of tables
unsigned long long compute_atk_time(unsigned long m, unsigned char l, unsigned int t, unsigned long long N) {

    unsigned long long left_part = 0;

    for (unsigned long k = 1; k < (l * t) + 1; k++) {
        unsigned int c = t - ((k - 1) / l);

        // Compute the sum on the left parenthesis with qi
        unsigned long long left_qsum = 0;
        for (unsigned int i = c; i < t + 1; i++) {
            left_qsum += compute_qi(m, t, N, i) * i;
        }

        left_part += compute_pk(m, N, k) * ((((t - c) * (t - c + 1)) / 2) + left_qsum) * l;
    }

    // Compute the sum on the right parenthesis with qi
    unsigned long long right_qsum = 0;
    for (unsigned int i = 1; i < t + 1; i++) {
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
        } else if (lower == mt - 1) {
            mid = mt - 1;
        }
        int compare = memcmp(&(*endpoints)[mid * step], plain_text, pwd_length);
        // Match found
        if (compare == 0) {
            return mid;
        } else if (compare < 0) {
            if (mid == 0)
                break;
            lower = mid + 1;
        } else if (lower == upper) {
            int compare2 = memcmp(&(*endpoints)[(mid - 1) * step], plain_text, pwd_length);
            if (compare2 == 0) return lower;
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
    char_to_digest(char_digest, digest, HASH_LENGTH);

    Password *plain_text = (Password *) malloc(sizeof(Password));
    char_to_password("abcdefg", plain_text, pwd_length);

    unsigned long long temp = 0;
    temp = (unsigned long long) ((*digest).i[0] + (*digest).i[1] + (*digest).i[2] + (*digest).i[3] + index) %
           (unsigned long long) (power(CHARSET_LENGTH, pwd_length));

    for (int i = pwd_length - 1; i >= 0; i--) {
        unsigned char reste = charset[(unsigned long long) ((unsigned long long) temp %
                                                            (unsigned long long) CHARSET_LENGTH)];
        temp = (unsigned long long) ((unsigned long long) temp / (unsigned long long) CHARSET_LENGTH);
        (*plain_text).bytes[i] = reste;
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

void online_from_files(char *path, unsigned char *digest, char *password, int pwd_length, int nbTable) {
    int t;
    unsigned long mt[nbTable];
    unsigned long mtTotal = 0;
    char buff[255];

    // Read the tables' headers
    for (int table = 0; table < nbTable; table++) {
        char tableChar[3];
        sprintf(tableChar, "%d", table);

        char tableStartNPath[255];
        strcpy(tableStartNPath, path);
        strcat(tableStartNPath, "_start_");
        strcat(tableStartNPath, tableChar);
        strcat(tableStartNPath, ".bin");

        FILE *fpStartN;
        fpStartN = fopen(tableStartNPath, "rb");

        // Retrieve the table's number of end points (mt)
        unsigned long mtTable;
        fscanf(fpStartN, "%s", buff);
        sscanf(buff, "%ld", &mtTable);
        //printf("Table %d - mt = %lu\n", mt);
        fgets(buff, 255, (FILE *) fpStartN);
        fgets(buff, 255, (FILE *) fpStartN);

        // Retrieve the table's chain length (t)
        int tTable;
        fgets(buff, 255, (FILE *) fpStartN);
        sscanf(buff, "%d", &tTable);

        fclose(fpStartN);

        // t changed between the tables
        if (table > 0 && t != tTable) {
            printf("Error: the chain length is not the same in table number %d and %d.\n", table-1, table);
            exit(1);
        }

        mt[table] = mtTable;
        mtTotal += mtTable;
        t = tTable;
    }

    printf("Total number of end points (mtTotal): %lu\n", mtTotal);
    printf("Chain length (t): %d\n\n", t);

    char **startpoints = malloc(sizeof(char) * pwd_length * mtTotal);
    char **endpoints = malloc(sizeof(char) * pwd_length * mtTotal);
    char buffStart[255];
    char buffEnd[255];

    // Fill the start points and end points arrays
    for (int table = 0; table < nbTable; table++) {
        startpoints[table] = malloc(sizeof(char) * pwd_length * mt[table]);
        endpoints[table] = malloc(sizeof(char) * pwd_length * mt[table]);

        char tableChar[3];
        sprintf(tableChar, "%d", table);

        char tableStartNPath[255];
        strcpy(tableStartNPath, path);
        strcat(tableStartNPath, "_start_");
        strcat(tableStartNPath, tableChar);
        strcat(tableStartNPath, ".bin");

        char tableEndNPath[255];
        strcpy(tableEndNPath, path);
        strcat(tableEndNPath, "_end_");
        strcat(tableEndNPath, tableChar);
        strcat(tableEndNPath, ".bin");

        FILE *fpStartN;
        fpStartN = fopen(tableStartNPath, "rb");
        fscanf(fpStartN, "%s", buffStart);
        fgets(buffStart, 255, (FILE *) fpStartN); // skip
        fgets(buffStart, 255, (FILE *) fpStartN); // the
        fgets(buffStart, 255, (FILE *) fpStartN); // header

        FILE *fpEndN;
        fpEndN = fopen(tableEndNPath, "rb");
        fscanf(fpEndN, "%s", buffEnd);
        fgets(buffEnd, 255, (FILE *) fpEndN); // skip
        fgets(buffEnd, 255, (FILE *) fpEndN); // the
        fgets(buffEnd, 255, (FILE *) fpEndN); // header

        for (unsigned long i = 0; i < mt[table] * pwd_length; i = i + sizeof(char) * pwd_length) {
            fread(buffStart, pwd_length, 1, (FILE *) fpStartN);
            for (unsigned long j = i; j < i + pwd_length; j++) {
                startpoints[table][j] = buffStart[j - i];
            }
            fread(buffEnd, pwd_length, 1, (FILE *) fpEndN);
            for (unsigned long j = i; j < i + pwd_length; j++) {
                endpoints[table][j] = buffEnd[j - i];
            }
        }

        fclose(fpStartN);
        fclose(fpEndN);
    }

    // Perform the attack
    for (long i = t - 1; i >= 0; i--) {
        char column_plain_text[pwd_length];
        unsigned char column_digest[HASH_LENGTH * 2];
        strncpy(column_digest, digest, sizeof(unsigned char) * HASH_LENGTH * 2);

        // Get the reduction corresponding to the current column
        for (unsigned long k = i; k < t - 1; k++) {
            reduce_digest(column_digest, k, column_plain_text, pwd_length);
            ntlm(column_plain_text, column_digest, pwd_length);
        }
        reduce_digest(column_digest, t - 1, column_plain_text, pwd_length);

        //printf("Trying to find %.*s in endpoints...\n", pwd_length, column_plain_text);
        long found = -1;
        int table = -1;
        do {
            table++;
            found = search_endpoint(&(endpoints[table]), column_plain_text, mt[table], pwd_length);

            if (found == -1) {
                continue;
            }

            printf("Match found in chain number %ld of table %d...", found, table);

            // We found a matching endpoint: reconstruct the chain
            char chain_plain_text[pwd_length];
            unsigned char chain_digest[HASH_LENGTH];

            // Copy the corresponding start point into chain_plain_text
            for (long l = 0; l < pwd_length; l++) {
                chain_plain_text[l] = startpoints[table][(found * pwd_length) + l];
            }

            // Reconstruct the chain from the beginning
            for (unsigned long k = 0; k < i; k++) {
                ntlm(chain_plain_text, chain_digest, pwd_length);
                reduce_digest(chain_digest, k, chain_plain_text, pwd_length);
            }
            ntlm(chain_plain_text, chain_digest, pwd_length);

            //printf("Comparing '%s' and '%s' for false alert check.\n", chain_digest, digest);

            // Check if the computed hash is the one we're looking for
            if (memcmp(chain_digest, digest, HASH_LENGTH) == 0) {
                printf(" Password cracked! (column=%ld)\n", i);
                memcpy(password, chain_plain_text, pwd_length);
                return;
            }
            printf(" False alert. (column=%ld)\n", i);
        } while (table < nbTable - 1);
    }

    strcpy(password, ""); // password was not found
}

int online_from_files_coverage(char *start_path, char *end_path, int pwd_length, int nb_cover) {
    FILE *fp;
    fp = fopen(start_path, "rb");

    if (fp == NULL)(exit(1));

    char buff[255];
    // Retrieve the number of points
    fscanf(fp, "%s", buff);
    unsigned long mt;
    sscanf(buff, "%ld", &mt);
    printf("Number of points (mt): %lu\n", mt);
    fgets(buff, 255, (FILE *) fp);

    // Retrieve the password length
    fgets(buff, 255, (FILE *) fp);
    printf("Password length: %d\n", pwd_length);

    // Retrieve the chain length (t)
    int t;
    fgets(buff, 255, (FILE *) fp);
    sscanf(buff, "%d", &t);
    printf("Chain length (t): %d\n\n", t);

    char *startpoints = malloc(sizeof(char) * pwd_length * mt);

    long limit = (long) mt * (long) pwd_length;

    char buff_sp[pwd_length];

    for (long i = 0; i < limit; i = i + sizeof(char) * pwd_length) {
        fread(buff_sp, pwd_length, 1, (FILE *) fp);
        for (long j = i; j < i + pwd_length; j++) {
            startpoints[j] = buff_sp[j - i];
        }
    }

    // Close the start file
    fclose(fp);

    FILE *fp2;
    char buff2[255];
    fp2 = fopen(end_path, "rb");
    fgets(buff2, 255, (FILE *) fp2);
    fgets(buff2, 255, (FILE *) fp2);
    fgets(buff2, 255, (FILE *) fp2);

    char *endpoints = malloc(sizeof(char) * pwd_length * mt);

    char buff_ep[pwd_length];

    for (long i = 0; i < limit; i = i + sizeof(char) * pwd_length) {
        fread(buff_ep, pwd_length, 1, (FILE *) fp);

        for (long j = i; j < i + pwd_length; j++) {
            endpoints[j] = buff_ep[j - i];
        }
    }

    // Close the end file
    fclose(fp2);

    int numberFound = 0;

    srand(time(NULL));

    unsigned long long nb_hashes = 0;

    for (int p = 0; p < nb_cover; p++) {

        char password[pwd_length];
        unsigned char digest[HASH_LENGTH * 2];

        long counter = p;
        // Generate one password
        for (int n = 0; n < pwd_length; n++) {
            //password[n] = charset[rand() % CHARSET_LENGTH];
            password[n] = charset[counter % CHARSET_LENGTH];
            counter /= CHARSET_LENGTH;
        }

        if ((p % 100) == 0) {
            printf("%d: ", p);
            for (int q = 0; q < pwd_length; q++) {
                printf("%c", password[q]);
            }
            printf("\n");
        }

        //printf("%d / %d (%d found) - Trying with: %.*s", p+1, nb_cover, numberFound, pwd_length, password);

        ntlm(password, digest, pwd_length);

        //printf(" (%s)...", digest);

        for (long i = t - 1; i >= 0; i--) {

            char column_plain_text[pwd_length];
            unsigned char column_digest[HASH_LENGTH * 2];
            strncpy(column_digest, digest, sizeof(unsigned char) * HASH_LENGTH * 2);

            /* DEBUG
            if(p == 100) {
                for(int q=0; q<HASH_LENGTH*2; q++){
                    printf("%c", column_digest[q]);
                }
                printf(" === ");

                for(int q=0; q<HASH_LENGTH*2; q++){
                    printf("%c", digest[q]);
                }
                printf("\n");
            }
             */

            // get the reduction corresponding to the current column
            for (unsigned long k = i; k < t - 1; k++) {
                /* DEBUG
                if(p == 100) {
                    for(int q=0; q<HASH_LENGTH*2; q++){
                        printf("%c", column_digest[q]);
                    }
                    printf(" --> ");
                }
                 */
                reduce_digest(column_digest, k, column_plain_text, pwd_length);
                /* DEBUG
                if(p == 100) {
                    for(int q=0; q<pwd_length; q++){
                        printf("%c", column_plain_text[q]);
                    }
                    printf(" --> ");
                }
                 */
                ntlm(column_plain_text, column_digest, pwd_length);
                /* DEBUG
                if(p == 100) {
                    for(int q=0; q<HASH_LENGTH*2; q++){
                        printf("%c", column_digest[q]);
                    }
                    printf("\n");
                }
                 */
                nb_hashes++;
            }
            reduce_digest(column_digest, t - 1, column_plain_text, pwd_length);


            long found = search_endpoint(&endpoints, column_plain_text, mt, pwd_length);

            if (found == -1) {
                continue;
            }

            // we found a matching endpoint, reconstruct the chain
            char chain_plain_text[pwd_length];
            unsigned char chain_digest[HASH_LENGTH * 2];

            // Copy the startpoint into chain_plain_text
            for (long l = 0; l < pwd_length; l++) {
                chain_plain_text[l] = startpoints[(found * pwd_length) + l];
            }

            for (unsigned long k = 0; k < i; k++) {
                ntlm(chain_plain_text, chain_digest, pwd_length);
                nb_hashes++;
                reduce_digest(chain_digest, k, chain_plain_text, pwd_length);
            }
            ntlm(chain_plain_text, chain_digest, pwd_length);
            nb_hashes++;

            if (memcmp(chain_digest, digest, HASH_LENGTH) == 0) {
                strcpy(password, chain_plain_text);
                numberFound++;
                //printf(" Found!");
                break;
            }
        }
        //printf("\n");
    }

    printf("\n%llu cryptographic operations were done.\n", nb_hashes);
    printf("In theory, it should have been %llu.\n\n", nb_cover * compute_atk_time(mt, 1, t, compute_N(pwd_length)));
    return numberFound;
}

int checkArgs(int argc) {
    // Normal number of arguments
    if (argc == 4) {
        return 0;
    }

    // Abnormal number of arguments
    if (argc < 4) {
        printf("Error: not enough arguments were given.\n");
    } else if (argc > 4) {
        printf("Error: too many arguments were given.\n");
    }
    printf("Usage: 'online path -p password', where:"
           "\n   - path is the path to the table (without '_start_N.bin')."
           "\n   - password is the plain text password you're looking to crack. The program will thus hash it first, then try to crack it."
           "\nOther usage: 'online path -h hash', where hash is the NTLM hash you're looking to crack."
           "\nOther usage: 'online path -c N', where N is the number of exhaustively generated passwords you're looking to crack."
           "\n\n");
    exit(1);
}

int checkTables(char *path, int *nbTable, int *pwdLength) {
    int resNbTables = 0;
    int resPwdLength = 0;

    char tableStart0Path[200];
    strcpy(tableStart0Path, path);
    strcat(tableStart0Path, "_start_0.bin");

    char tableEnd0Path[255];
    strcpy(tableEnd0Path, path);
    strcat(tableEnd0Path, "_end_0.bin");

    FILE *fpStart0;
    fpStart0 = fopen(tableStart0Path, "rb");

    FILE *fpEnd0;
    fpEnd0 = fopen(tableEnd0Path, "rb");

    if (fpStart0 == NULL) {
        printf("Error: no start points file found when trying to read '%s'.\n", tableStart0Path);
        exit(1);
    } else if (fpEnd0 == NULL) {
        printf("Error: no end points file found when trying to read '%s'.\n", tableEnd0Path);
        exit(1);
    }

    // Retrieve the passwords' length in the header of the table
    char buff[255];
    fscanf(fpStart0, "%s", buff);
    fgets(buff, 255, (FILE *) fpStart0);
    fgets(buff, 255, (FILE *) fpStart0);
    sscanf(buff, "%d", &resPwdLength);
    fclose(fpStart0);
    fclose(fpEnd0);

    int startOk = 0;
    int endOk = 0;

    // Count the number of tables
    do {
        resNbTables++;

        char nbTablesChar[3];
        sprintf(nbTablesChar, "%d", resNbTables);

        char tableStartNPath[255];
        strcpy(tableStartNPath, path);
        strcat(tableStartNPath, "_start_");
        strcat(tableStartNPath, nbTablesChar);
        strcat(tableStartNPath, ".bin");

        char tableEndNPath[255];
        strcpy(tableEndNPath, path);
        strcat(tableEndNPath, "_end_");
        strcat(tableEndNPath, nbTablesChar);
        strcat(tableEndNPath, ".bin");

        FILE *fpStartN;
        fpStartN = fopen(tableStartNPath, "rb");
        FILE *fpEndN;
        fpEndN = fopen(tableEndNPath, "rb");

        startOk = fpStartN != NULL;
        endOk = fpEndN != NULL;

        if (startOk && endOk) {
            fclose(fpStartN);
            fclose(fpEndN);
        }

        if (startOk && !endOk) {
            printf("Error: start points file found but no corresponding end points file found when trying to read '%s'.\n",
                   tableEndNPath);
            exit(1);
        } else if (!startOk && endOk) {
            printf("Error: end points file found but no corresponding start points file found when trying to read '%s'.\n",
                   tableStartNPath);
            exit(1);
        }
    } while (startOk && endOk);

    *nbTable = resNbTables;
    *pwdLength = resPwdLength;

    return 0;
}

int main(int argc, char *argv[]) {
    printf("GPUCrack v0.1.4\n"
           "<https://github.com/gpucrack/GPUCrack/>\n\n");

    int tableNb = 0;
    int pwdLength = 0;
    checkArgs(argc);
    checkTables(argv[1], &tableNb, &pwdLength);

    // User typed 'online table -p password'
    if (strcmp(argv[2], "-p") == 0) {
        const char *password = argv[3]; // the password we will be looking to crack, after it's hashed
        unsigned char digest[HASH_LENGTH * 2]; // the hashed password
        char found[pwdLength];

        ntlm(password, digest, pwdLength);

        printf("Looking for password '%.*s', hashed as %s.\n", pwdLength, password, digest);
        printf("Starting attack...\n");

        online_from_files(argv[1], digest, found, pwdLength, tableNb);

        if (!strcmp(found, "")) {
            printf("No password found for the given hash.\n");
        } else {
            printf("Password '%.*s' found for the given hash!\n", pwdLength, found);
        }
        exit(0);
    }

        // User typed 'online table -h hash'
    else if (strcmp(argv[2], "-h") == 0) {
        char *digest = argv[3]; // the hashed password
        char found[pwdLength];

        // Convert hash to lowercase
        for (int i = 0; digest[i]; i++) {
            digest[i] = tolower(digest[i]);
        }

        printf("Looking to crack the ntlm hash '%s'.\n", digest);
        printf("Starting attack...\n");

        online_from_files(argv[1], digest, found, pwdLength, tableNb);

        if (!strcmp(found, "")) {
            printf("No password found for the given hash.\n");
        } else {
            printf("Password '%.*s' found for the given hash!\n", pwdLength, found);
        }
        exit(0);
    }

        // User typed 'online -c N'
    else if (strcmp(argv[2], "-c") == 0) {
        int nb_cover = atoi(argv[3]);

        printf("Looking to crack %d passwords.\n", nb_cover);
        printf("Starting the attacks...\n");

        //int foundNumber = online_from_files_coverage(argv[1], pwd_length, nb_cover);

        //printf("%d out of %d passwords were cracked successfully.\n", foundNumber, nb_cover);
        //printf("Success rate: %.2f %%\n", ((double) foundNumber / nb_cover) * 100);
    }
}