#include "online.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *ntlm(char *key) {
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
    return hex_format;
}

void fileToArray(char *path, Password **points, int mt) {
    // Create a text string, which is used to output the text file
    string myText;

    // Read from the text file
    ifstream MyReadFile(path);

    // Use a while loop together with the getline() function to read the file line by line
    for (int j = 0; j < mt; j++) {
        char *tmp = (char *) malloc(sizeof(char) * 7);
        getline(MyReadFile, myText);
        strcpy(tmp, myText.c_str());
        for (int i = 0; i < 7; i++) {
            (*points)[j].bytes[i] = tmp[i];
        }
    }

    // Close the file
    MyReadFile.close();
}

int findPwdInEndpoints(Password password, Password *endPoints, int mt) {
    for (int i = 0; i < mt; i++) {
        if (pwdcmp(password, endPoints[i])) return i;
    }
    return -1;
}

void online(Password *startPoints, Password *endPoints, Digest digest, Password password) {
    // DISCLAIMER : here, mt and t are hard-coded for the moment.
    int mt = 1'000'000'000; // the number of end points
    int t = 1'000; // the length of a chain

    // Iterate over the columns of the table, starting from the end points toward the start points.
    for (unsigned long i = t - 1; i > 0; i--) {
        Password currentPwd;
        Digest currentDigest;
        memcpy(currentDigest, digest, HASH_LENGTH);

        // Get the reduction corresponding to the current column
        for (unsigned long k = i; k < t - 2; k++) {
            reduceDigest(k, currentDigest, currentPwd);
            currentDigest = ntlm(currentPwd);
        }
        reduceDigest(t - 2, currentDigest, currentPwd);

        // Search for the reduction in the end points...
        int found = findPwdInEndpoints(currentPwd, endPoints);

        // If the currentPwd matches one of the end points, then reconstruct the chain
        if (found != -1) {
            printf("Found a match! Reconstructing the chain...");
            Password chain_plain_text;
            Digest chain_digest;
            strcpy(chain_plain_text, startPoints);

            for (unsigned long k = 0; k < i; k++) {
                chain_digest = ntlm(chain_plain_text);
                reduceDigest(k, chain_digest, chain_plain_text);
            }
            chain_digest = ntlm(chain_plain_text);

            // Verify whether it was a false alarm or not
            if (!memcmp(chain_digest, digest, HASH_LENGTH)) {
                strcpy(password, chain_plain_text);
                return;
            } else printf(" False alarm.\n");
        }
    }

    // No match found
    password[0] = '\0';
}

int main(int argc, char *argv[]) {

    // Check if the arguments were given correctly
    if (argc != 3) {
        printf("Error: not enough arguments given.\nUsage: 'online startpoint endpoint password'.\n'startpoint' is the path to the start points file.\n'endpoints' is the path to the end points file.\n'password' is the password you are looking to crack.");
        exit(1);
    }

    int mt = 100;

    Password *startPoints;
    Password *endPoints;

    // Read the rainbow table from its file
    fileToArray(argv[1], &startPoints, mt);
    fileToArray(argv[2], &endPoints, mt);

    // the password we will be looking to crack, after it's hashed
    Password *password = argv[2];
    if (strlen(password) != PASSWORD_LENGTH) {
        fprintf(stderr, "Error: password size should be %d.", PASSWORD_LENGTH);
        exit(EXIT_FAILURE);
    }

    // `digest` now contains the hashed password
    Digest digest;
    digest = ntlm(password);

    printf("\nLooking for password '%s', hashed into ", (char *) password);
    display_digest(digest, false);
    printf(".\nStarting attack...\n");

    // try to crack the password
    Password found;
    found = online(startPoints, endPoints, digest, found);

    // if `found` is not empty, then we successfully cracked the password
    if (!strcmp(found, "")) {
        printf("No password found for the given hash.\n");
    } else {
        printf("Password '%s' found for the given hash!\n", found);
    }

    return 0;
}