#include "online.h"

unsigned long search_endpoint(char **endpoints, char *plainText, unsigned long mt, int pwdLength) {
    unsigned long lower = 0;
    unsigned long upper = mt - 1;
    unsigned long step = sizeof(char) * pwdLength;

    while (lower <= upper) {
        unsigned long mid = 1 + (lower + (upper - lower) / 2);
        if (upper == 0) {
            mid = 0;
        } else if (lower == mt - 1) {
            mid = mt - 1;
        }
        int compare = memcmp(&(*endpoints)[mid * step], plainText, pwdLength);
        // Match found
        if (compare == 0) {
            return mid;
        } else if (compare < 0) {
            if (mid == 0)
                break;
            lower = mid + 1;
        } else if (lower == upper) {
            int compare2 = memcmp(&(*endpoints)[(mid - 1) * step], plainText, pwdLength);
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

void char_to_password(char text[], Password *password, int pwdLength) {
    for (int i = 0; i < pwdLength; i++) {
        password->bytes[i] = text[i];
    }
}

void password_to_char(Password *password, char text[], int pwdLength) {
    for (int i = 0; i < pwdLength; i++) {
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

void reduce_digest(char *charDigest, unsigned int index, char *charPlain, int pwdLength) {
    Digest *digest = (Digest *) malloc(sizeof(Digest));
    char_to_digest(charDigest, digest, HASH_LENGTH);

    Password *plainText = (Password *) malloc(sizeof(Password));
    char_to_password("abcdefg", plainText, pwdLength);

    unsigned long long temp = 0;
    temp = (unsigned long long) ((unsigned long long)(*digest).i[0] + (unsigned long long)(*digest).i[1] +
            (unsigned long long)(*digest).i[2] + (unsigned long long)(*digest).i[3] + (unsigned long long)index) %
           (unsigned long long) (pow((double) CHARSET_LENGTH, (double) pwdLength));

    for (int i = pwdLength - 1; i >= 0; i--) {
        unsigned char reste = charset[(unsigned long long) ((unsigned long long) temp %
                                                            (unsigned long long) CHARSET_LENGTH)];
        temp = (unsigned long long) ((unsigned long long) temp / (unsigned long long) CHARSET_LENGTH);
        (*plainText).bytes[i] = reste;
    }

    password_to_char(plainText, charPlain, pwdLength);

    free(digest);
    free(plainText);
}

void ntlm(char *key, char *hash, int pwdLength) {
    int i, j;
    int length = pwdLength;
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

void online_from_files(char *path, char *digest, char *password, int pwdLength, int nbTable, int debug) {
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

    if (debug) {
        printf("Total number of end points (mtTotal): %lu\n", mtTotal);
        printf("Chain length (t): %d\n\n", t);
    }

    char **startpoints = malloc(sizeof(char) * pwdLength * mtTotal);
    char **endpoints = malloc(sizeof(char) * pwdLength * mtTotal);
    char buffStart[255];
    char buffEnd[255];

    // Fill the start points and end points arrays
    for (int table = 0; table < nbTable; table++) {
        startpoints[table] = malloc(sizeof(char) * pwdLength * mt[table]);
        endpoints[table] = malloc(sizeof(char) * pwdLength * mt[table]);

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

        for (unsigned long i = 0; i < mt[table] * pwdLength; i = i + sizeof(char) * pwdLength) {
            fread(buffStart, pwdLength, 1, (FILE *) fpStartN);
            for (unsigned long j = i; j < i + pwdLength; j++) {
                startpoints[table][j] = buffStart[j - i];
            }
            fread(buffEnd, pwdLength, 1, (FILE *) fpEndN);
            for (unsigned long j = i; j < i + pwdLength; j++) {
                endpoints[table][j] = buffEnd[j - i];
            }
        }

        fclose(fpStartN);
        fclose(fpEndN);
    }

    // Perform the attack
    for (long i = t - 1; i >= 0; i--) {
        char column_plainText[pwdLength];
        unsigned char column_digest[HASH_LENGTH * 2];
        strncpy(column_digest, digest, sizeof(unsigned char) * HASH_LENGTH * 2);

        // Get the reduction corresponding to the current column
        for (unsigned long k = i; k < t - 1; k++) {
            reduce_digest(column_digest, k, column_plainText, pwdLength);
            ntlm(column_plainText, column_digest, pwdLength);
        }
        reduce_digest(column_digest, t - 1, column_plainText, pwdLength);

        //printf("Trying to find %.*s in endpoints...\n", pwdLength, column_plainText);
        long found = -1;
        int table = -1;
        do {
            table++;
            found = search_endpoint(&(endpoints[table]), column_plainText, mt[table], pwdLength);

            if (found == -1) {
                continue;
            }

            if (debug) printf("Match found in chain number %ld of table %d...", found, table);

            // We found a matching endpoint: reconstruct the chain
            char chain_plainText[pwdLength];
            unsigned char chain_digest[HASH_LENGTH];

            // Copy the corresponding start point into chain_plainText
            for (long l = 0; l < pwdLength; l++) {
                chain_plainText[l] = startpoints[table][(found * pwdLength) + l];
            }

            // Reconstruct the chain from the beginning
            for (unsigned long k = 0; k < i; k++) {
                ntlm(chain_plainText, chain_digest, pwdLength);
                reduce_digest(chain_digest, k, chain_plainText, pwdLength);
            }
            ntlm(chain_plainText, chain_digest, pwdLength);

            //printf("Comparing '%s' and '%s' for false alert check.\n", chain_digest, digest);

            // Check if the computed hash is the one we're looking for
            if (memcmp(chain_digest, digest, HASH_LENGTH) == 0) {
                if (debug) printf(" Password cracked! (column=%ld)\n", i);
                memcpy(password, chain_plainText, pwdLength);
                return;
            }
            if (debug) printf(" False alert. (column=%ld)\n", i);
        } while (table < nbTable - 1);
    }

    strcpy(password, ""); // password was not found
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
    printf("GPUCrack v0.1.4\n\n");

    // Default values
    int mode = 0; // 0: bad arguments, 1: password cracking, 2: single hash cracking, 3: multiple hashes cracking, 4: coverage test
    char *path; // path to the table files (without '_start_N.bin') (-t)
    char *pwd; // password to crack (-p)
    char *hash; // hash to crack (-h)
    int nbCoverage; // number of passwords to crack for coverage (-c)
    char *hashFile; // file containing the hashes to crack (-m)
    int debug = 0; // debug mode (-d)

    // Parse arguments
    int opt;
    while((opt = getopt(argc, argv, "t:p:h:c:m:d")) != -1)
    {
        switch(opt)
        {
            case 't':
                path = optarg;
                break;
            case 'p':
                pwd = optarg;
                mode = 1;
                break;
            case 'h':
                hash = optarg;
                mode = 2;
                break;
            case 'm':
                hashFile = optarg;
                mode = 3;
                break;
            case 'c':
                nbCoverage = atoi(optarg);
                mode = 4;
                break;
            case 'd':
                debug = 1;
                break;
            case '?':
            default:
                printf ("Usage: %s -t <path> [-p <password>] [-h <hash>] [-c <nbCoverage>] [-m <hashFile>] [-d].\n"
                        "    <path> is the path to the table (without '_start_N.bin')\n"
                        "    -p : plaintext cracking. <password> is the plain-text password you're looking to crack.\n"
                        "    -h : single hash cracking. <hash> is the hash you're looking to crack.\n"
                        "    -c : coverage test. <nbCoverage> is the number of passwords you want to crack to test the table's coverage.\n"
                        "    -m : multiple hashes. <hashFile> is the text file containing all the hashes you're looking to crack.\n"
                        "    -d : debug mode. Prints false alerts and more details about the cracking process.\n"
                        , argv[0]);
                exit(1);
        }
    }

    // Extra arguments (ignored)
    if (optind < argc){
        printf ("%d extra command-line arguments were given and will be ignored:\n", argc - optind);
        for (; optind < argc; optind++) {
            printf ("    '%s'\n", argv[optind]);
        }
        printf("\n");
    }

    // Check the tables and retrieve their parameters
    int tableNb = 0;
    int pwdLength = 0;
    checkTables(path, &tableNb, &pwdLength);

    // Initialize variables
    char found[pwdLength];


    switch(mode) {
        // Password cracking
        case 1:
            {
                char digest[HASH_LENGTH * 2];
                ntlm(pwd, digest, pwdLength);
                printf("Looking for password '%.*s', hashed as %s.\n", pwdLength, pwd, digest);
                online_from_files(path, digest, found, pwdLength, tableNb, debug);
                break;
            }
        // Single hash cracking
        case 2:
            // Convert hash to lowercase
            for (int i = 0; hash[i]; i++) {
                hash[i] = tolower(hash[i]);
            }
            printf("Looking for hash '%s'.\n", hash);
            online_from_files(path, hash, found, pwdLength, tableNb, debug);
            break;

        // Multiple hashes cracking
        case 3:
            {
            // Read hashes from file
            FILE *fhashFile = fopen(hashFile, "r");
            if (fhashFile == NULL) {
                printf("Error: cannot open file '%s'\n", hashFile);
                exit(1);
            }
            char line[HASH_LENGTH * 2 + 1];
            while (fgets(line, sizeof(line), fhashFile) != NULL) {
                char foundLocal[pwdLength];
                // Remove newline character
                line[strcspn(line, "\n")] = 0;
                // Convert hash to lowercase
                for (int i = 0; line[i]; i++) {
                    line[i] = tolower(line[i]);
                }
                // If the hash is not empty
                if (line[0] != '\0') {
                    printf("Looking for hash '%s'.\n", line);
                    online_from_files(path, line, foundLocal, pwdLength, tableNb, debug);
                    if (!strcmp(foundLocal, "")) {
                        printf("No password found for the hash '%s'.\n", line);
                    } else {
                        printf("Password '%.*s' found!\n", pwdLength, foundLocal);
                    }
                }
            }

                exit(0);
            }

        // Coverage test
        case 4:
        {
            int foundNumber = 0;
            // Initialize random seed
            srand(time(0));
            // nbCoverage loops
            for (int i = 0; i < nbCoverage; i++) {

                // Generate a random password using the charset
                char pwdLocal[pwdLength];
                for (int j = 0; j < pwdLength; j++) {
                    pwdLocal[j] = charset[rand() % CHARSET_LENGTH];
                }

                pwdLocal[pwdLength] = '\0';
                char digest[HASH_LENGTH * 2];
                ntlm(pwdLocal, digest, pwdLength);

                // Check if the password is in the table
                printf("%d - Looking for password '%.*s', hashed as %s.", i, pwdLength, pwdLocal, digest);
                online_from_files(path, digest, found, pwdLength, tableNb, debug);
                if (!strcmp(found, "")) {
                    printf(" Not found ");
                }
                else {
                    printf(" Found ");
                    foundNumber++;
                }
                // Print current success rate and percentage
                printf("%d/%d (%d%%)\n", foundNumber, i + 1, (foundNumber * 100) / (i + 1));
            }

            printf("\n%d out of %d passwords were cracked successfully.\n", foundNumber, nbCoverage);
            printf("Success rate: %.2f %%\n", ((double) foundNumber / nbCoverage) * 100);
            exit(0);
        }

        case 0:
        default:
            printf("Error: no instruction was given.\n");
            printf ("Usage: %s -t <path> [-p <password>] [-h <hash>] [-c <nbCoverage>] [-m <hashFile>] [-d].\n"
                    "    <path> is the path to the table (without '_start_N.bin')\n"
                    "    -p : plaintext cracking. <password> is the plain-text password you're looking to crack.\n"
                    "    -h : single hash cracking. <hash> is the hash you're looking to crack.\n"
                    "    -c : coverage test. <nbCoverage> is the number of passwords you want to crack to test the table's coverage.\n"
                    "    -m : multiple hashes. <hashFile> is the text file containing all the hashes you're looking to crack.\n"
                    "    -d : debug mode. Prints false alerts and more details about the cracking process.\n"
                    , argv[0]);
            exit(1);
    }

    if (!strcmp(found, "")) {
        printf("No password found for the given hash.\n");
    } else {
        printf("Password '%.*s' found for the given hash!\n", pwdLength, found);
    }
    exit(0);
}