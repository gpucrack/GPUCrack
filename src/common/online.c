#include "online.h"

unsigned char charset[CHARSET_LENGTH] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
                                         'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                                         'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                                         'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                                         's', 't',
                                         'u', 'v', 'w', 'x', 'y', 'z'};

void print_hash(const unsigned char *digest) {
    for (int i = 0; i < HASH_LENGTH; i++) {
        printf("%02x", digest[i]);
    }
}

int search_endpoint(char **endpoints, char *plain_text, int mt, int pwd_length) {
    for (int i = 0; i < mt; i = i + sizeof(char) * pwd_length) {
        if (memcmp(endpoints[i], plain_text, pwd_length) == 0) {
            return i/pwd_length;
        }
    }

    return -1;


    /*for (int i = 0; i < mt; i = i + sizeof(char) * pwd_length) {
        char res = 0;
        for (int j = i; (j < i + pwd_length && res != -1); j++) {
            if ((*endpoints)[j] != plain_text[j - i]) {
                res = -1;
            }
        }
        if (res==0) {
            return i;
        }
    }

    return -1;*/
}

/*
 *     for (long i = 0; i < mt*pwd_length; i=i+pwd_length) {
        if(i < 50) {
            printf("aaa");
            printf("Comparing starting from %c :::: %s", endpoints[i], plain_text);
        }
        if (memcmp(endpoints[i], plain_text, pwd_length) == 0) {
            // printf("Match found when comparing %s and %s (row %d).\n", endpoints[i], plain_text, i);
            return i/pwd_length;
        }
    }
    return -1;
 */

void char_to_password(char text[], Password *password) {
    for (int i = 0; i < PASSWORD_LENGTH; i++) {
        password->bytes[i] = text[i];
    }
}

void password_to_char(Password *password, char text[]) {
    for (int i = 0; i < PASSWORD_LENGTH; i++) {
        text[i] = password->bytes[i];
    }
}

void char_to_digest(char text[], Digest *digest) {
    for (int i = 0; i < HASH_LENGTH; i++) {
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

void display_password(Password *pwd) {
    for (unsigned char i = 0; i < PASSWORD_LENGTH; i++) {
        printf("%c", (unsigned char) pwd->bytes[i]);
    }
}

void reduce_digest(char *char_digest, unsigned int index, char *char_plain, int pwd_length) {
    for (int i = 0; i < pwd_length - 1; i++) {
        char_plain[i] = charset[(char_digest[i] + index) % CHARSET_LENGTH];
    }
}

void reduce_digest2(char *char_digest, unsigned int index, char *char_plain, int pwd_length) {
    Digest *digest = (Digest *) malloc(sizeof(Digest));
    // printf("\nDEBUT REDUCTION : %s", char_digest);
    //print_hash(char_digest);
    char_to_digest(char_digest, digest);

    // printf("\nDigest : ");
    // display_digest(digest);

    Password *plain_text = (Password *) malloc(sizeof(Password));
    char_to_password("abcdefg", plain_text);

    // printf("   ---   Password : ");
    // display_password(plain_text);

    (*plain_text).i[0] =
            charset[((*digest).bytes[0] + index) % CHARSET_LENGTH] |
            (charset[((*digest).bytes[1] + index) % CHARSET_LENGTH] << 8) |
            (charset[((*digest).bytes[2] + index) % CHARSET_LENGTH] << 16) |
            (charset[((*digest).bytes[3] + index) % CHARSET_LENGTH] << 24);
    (*plain_text).i[1] =
            charset[((*digest).bytes[4] + index) % CHARSET_LENGTH] |
            (charset[((*digest).bytes[5] + index) % CHARSET_LENGTH] << 8) |
            (charset[((*digest).bytes[6] + index) % CHARSET_LENGTH] << 16) |
            (charset[((*digest).bytes[7] + index) % CHARSET_LENGTH] << 24);
    password_to_char(plain_text, char_plain);

    // printf("\n");
    // printf(" %s a été réduit en '%s'\n", char_digest, char_plain);
}

void ntlm(char *key, char *hash) {

    // printf("Key: %s", key);
    int i, j;
    int length = strlen(key);
    unsigned int nt_buffer[16],
            a, b, c, d, sqrt2, sqrt3, n,
            output[4];
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
    // printf("   ---   Hash: %s\n", hash);
}

void online_from_files(char *start_path, char *end_path, unsigned char *digest, char *password) {
    FILE *fp;
    char buff[255];

    fp = fopen(start_path, "r");

    // Retrieve the number of points
    fscanf(fp, "%s", buff);
    int mt;
    sscanf(buff, "%d", &mt);
    printf("Number of points: %d\n", mt);
    fgets(buff, 255, (FILE *) fp);

    // Retrieve the password length
    int pwd_length;
    fgets(buff, 255, (FILE *) fp);
    sscanf(buff, "%d", &pwd_length);
    printf("Password length: %d\n", pwd_length);

    // Retrieve the chain length (t)
    int t;
    fgets(buff, 255, (FILE *) fp);
    sscanf(buff, "%d", &t);
    printf("Chain length (t): %d\n\n", t);

    char *startpoints = malloc(sizeof(char) * pwd_length * mt);

    for (int i = 0; i < mt; i = i + sizeof(char) * pwd_length) {
        fgets(buff, 255, (FILE *) fp);
        for (int j = i; j < i + pwd_length; j++) {
            startpoints[j] = buff[j - i];
        }
    }
/*    for (int i = 0; i < mt; i++) {
        fgets(buff, 255, (FILE *) fp);
        startpoints[i] = strdup(buff);
        startpoints[i][strcspn(startpoints[i], "\n")] = '\0'; // remove line break
    }*/

    // Close the start file
    fclose(fp);

    FILE *fp2;
    char buff2[255];
    fp2 = fopen(end_path, "r");
    fgets(buff2, 255, (FILE *) fp2);
    fgets(buff2, 255, (FILE *) fp2);
    fgets(buff2, 255, (FILE *) fp2);

    char *endpoints = malloc(sizeof(char) * pwd_length * mt);

    for (int i = 0; i < mt; i = i + sizeof(char) * pwd_length) {
        fgets(buff2, 255, (FILE *) fp);
        for (int j = i; j < i + pwd_length; j++) {
            endpoints[j] = buff2[j - i];
        }
    }

    // Close the end file
    fclose(fp2);

    printf("0.00 %%");

    for (long i = t - 1; i >= 0; i--) {

        printf("\r%ld", i);

        char column_plain_text[pwd_length + 1];
        unsigned char column_digest[HASH_LENGTH];
        strcpy(column_digest, digest);

        // printf("\nstrcpy : digest: %s\n", digest);
        // printf("strcpy : column_digest: %s\n", column_digest);

        // printf("\nWe suppose that the digest '%s' is in row %lu\n", digest, i);

        // get the reduction corresponding to the current column
        for (unsigned long k = i; k < t - 1; k++) {
            reduce_digest2(column_digest, k, column_plain_text, pwd_length);
            ntlm(column_plain_text, column_digest);
            // printf("k=%d   -   password: '%s'   -   hash: '%s'\n", k, column_plain_text, column_digest);
        }
        reduce_digest2(column_digest, t - 1, column_plain_text, pwd_length);
        // printf("k=%d   -   password: '%s'   -   hash: '%s'\n", t - 1, column_plain_text, column_digest);

        // printf("Trying to find %s in endpoints...\n", column_plain_text);
        int found = search_endpoint(&endpoints, column_plain_text, mt, pwd_length);

        if (found == -1) {
            continue;
        }

        printf("Match found in chain number %d.\n", found+1);

        // we found a matching endpoint, reconstruct the chain
        char chain_plain_text[pwd_length + 1];
        unsigned char chain_digest[HASH_LENGTH];

        // Copy the startpoint into chain_plain_text
        for (int l = found; l < found + pwd_length ; l++) {
            chain_plain_text[l - found] = startpoints[found*pwd_length + l];
        }

        for (unsigned long k = 0; k < i; k++) {
            ntlm(chain_plain_text, chain_digest);
            reduce_digest2(chain_digest, k, chain_plain_text, pwd_length);
        }
        ntlm(chain_plain_text, chain_digest);

        printf("FALSE ALERT ???????? C'EST : %s et %s\n", chain_digest, digest);

        if (!memcmp(chain_digest, digest, HASH_LENGTH)) {
            strcpy(password, chain_plain_text);
            return;
        }
        printf("   ---   False alert %ld.\n", i);
    }

    strcpy(password, "");
}

/*
    Example showing how we create a rainbow table given its start and end point files.
*/
int main(int argc, char *argv[]) {

    if (argc < 5) {
        printf("Error: not enough arguments given.\nUsage: 'online startpath endpath -p password', where:"
               "\n   - startpath is the path to the start points file."
               "\n   - endpath is the path to the end points file."
               "\n   - password is the plain text password you're looking to crack. The program will thus hash it first, then try to crack it."
               "\nOther usage: 'online startpath endpath -h hash', where hash is the NTLM hash you're looking to crack.\n\n");
        exit(1);
    }

    if (argc > 5) {
        printf("Error: too many arguments given.\nUsage: 'online startpath endpath -p password', where:"
               "\n   - startpath is the path to the start points file."
               "\n   - endpath is the path to the end points file."
               "\n   - password is the plain text password you're looking to crack. The program will thus hash it first, then try to crack it."
               "\nOther usage: 'online startpath endpath -h hash', where hash is the NTLM hash you're looking to crack.\n\n");
        exit(1);
    }

    printf("GPUCrack v0.1.0\n"
           "<https://github.com/gpucrack/GPUCrack/>\n\n");

    const char *start_path = argv[1];
    const char *end_path = argv[2];

    if (strcmp(argv[3], "-p") == 0) {
        // A plain text password was given.
        // the password we will be looking to crack, after it's hashed
        const char *password = argv[4];
        // `digest` now contains the hashed password
        unsigned char digest[HASH_LENGTH];
        ntlm(password, digest);

        printf("Looking for password '%s', hashed as %s", password, digest);
        printf(".\nStarting attack...\n");

        // try to crack the password
        char found[PASSWORD_LENGTH + 1];
        online_from_files(start_path, end_path, digest, found);

        // if `found` is not empty, then we successfully cracked the password
        if (!strcmp(found, "")) {
            printf("No password found for the given hash.\n");
        } else {
            printf("Password '%s' found for the given hash!\n", found);
        }
        exit(0);
    } else if (strcmp(argv[3], "-h") == 0) {
        // the password we will be looking to crack, after it's hashed

        // `digest` now contains the hashed password
        char *digest = argv[4];
        for (int i = 0; digest[i]; i++) {
            digest[i] = tolower(digest[i]);
        }

        printf("Looking to crack the ntlm hash '%s'", digest);
        printf(".\nStarting attack...\n");

        // try to crack the password
        char found[PASSWORD_LENGTH + 1];
        online_from_files(start_path, end_path, digest, found);

        // if `found` is not empty, then we successfully cracked the password
        if (!strcmp(found, "")) {
            printf("No password found for the given hash.\n");
        } else {
            printf("Password '%s' found for the given hash!\n", found);
        }
        exit(0);
    }
}
