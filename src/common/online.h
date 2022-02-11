#ifndef GPUCRACK_ONLINE_H
#define GPUCRACK_ONLINE_H

#include "../gpu/constants.cuh"
#include "./reduction/reduction.h"
#include <iostream>
#include <fstream>

using namespace std;

/**
 * Hashes a char array using NTLM.
 * @param key the character array to sort.
 * @return a character array containing the NTLM hash of the given key.
 * https://github.com/tux-mind/autocrack/blob/master/common/crypto.c
 */
char *ntlm(char *key);

/**
 * Reads a file containing start or end points, and turns it into an array.
 * @param path the path of the start/end point file.
 * @param points the start/end point array.
 * @param mt the number of start/end point.
 */
void fileToArray(char *path, Password **points, int mt);

/**
 * Tries to find a given password in an array of end points.
 * @param password the password to find.
 * @param endPoints the end point array where the password is searched for.
 * @param mt the number of end points.
 * @return the index where the password was found, -1 if not found.
 */
int findPwdInEndpoints(Password password, Password *endPoints, int mt);

/**
 * Perform the online (or attack) phase, i.e. retrieve a password from its hash using rainbow tables.
 * @param startPoints the start point array.
 * @param endPoints the end point array.
 * @param digest the digest to crack.
 * @param password the retrieved password, if found.
 */
void online(Password *startPoints, Password *endPoints, Digest digest, Password password);

#endif //GPUCRACK_ONLINE_H
