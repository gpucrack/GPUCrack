#include "rainbow.cuh"

__device__ inline int pwdcmp(Password *p1, Password *p2) {
    for (int i = 0; i < CEILING(PASSWORD_LENGTH, 4); i++) {
        if (p1->i[i] != p2->i[i]) {
            return false;
        }
    }

    return true;
}

__device__ inline void pwdcpy(Password *p1, const Password *p2) {
    memcpy(p1, p2, PASSWORD_LENGTH);
}

__device__ int compare_rainbow_chains(const void *p1, const void *p2) {
    RainbowChain *chain1 = (RainbowChain *) p1;
    RainbowChain *chain2 = (RainbowChain *) p2;
    return pwdcmp(&chain1->endpoint, &chain2->endpoint);
}

__device__ char char_in_range(unsigned char n) {
    static const char *chars =
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";

    return chars[n];
}

__device__ void reduce_digest(Digest *digest, unsigned long iteration,
                              unsigned char table_number,
                              Password *plain_text) {
    // pseudo-random counter based on the hash
    unsigned long counter = digest->bytes[7];
    for (char i = 6; i >= 0; i--) {
        counter <<= 8;
        counter |= digest->bytes[i];
    }

    /*
        Get a plain text using the above counter.
        We multiply the table number by the iteration to have
        tables with reduction functions that are different enough
        (just adding the table number isn't optimal).

        The overflow that can happen on the counter variable isn't
        an issue since it happens reliably.

        https://www.gnu.org/software/autoconf/manual/autoconf-2.63/html_node/Integer-Overflow-Basics.html
    */
    create_startpoint(counter + iteration * table_number, plain_text);
}

__device__ void create_startpoint(unsigned long counter, Password *plain_text) {
    for (int i = PASSWORD_LENGTH - 1; i >= 0; i--) {
        plain_text->bytes[i] = char_in_range(counter % 64);
        counter /= 64;
    }
}

__device__ RainbowChain *binary_search(RainbowTable *table,
                                       Password *endpoint) {
    int start = 0;
    int end = table->length - 1;
    while (start <= end) {
        int middle = start + (end - start) / 2;
        if (!pwdcmp(&table->chains[middle].endpoint, endpoint)) {
            return &table->chains[middle];
        }

        if (!pwdcmp(&table->chains[middle].endpoint, endpoint) < 0) {
            start = middle + 1;
        } else {
            end = middle - 1;
        }
    }
    return NULL;
}

__device__ void dedup_endpoints(RainbowTable *table) {
    unsigned long dedup_index = 1;
    for (unsigned long i = 1; i < table->length; i++) {
        if (pwdcmp(&table->chains[i - 1].endpoint,
                   &table->chains[i].endpoint)) {
            table->chains[dedup_index] = table->chains[i];
            dedup_index++;
        }
    }

    table->length = dedup_index;
}

__global__ void ntlm_chain_kernel_old(RainbowTable *table) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    // generate the chain
    Password last_plain_text;
    Password startpoint;
    create_startpoint(index, &startpoint);
    pwdcpy(&last_plain_text, &startpoint);

    /*
        Apply a round of hash + reduce `TABLE_T - 1` times.
        The chain should look like this:

        n -> r0(h(n)) -> r1(h(r0(h(n))) -> ...
    */
    for (unsigned long j = 0; j < TABLE_T - 1; j++) {
        Digest digest;
        ntlm(&last_plain_text, &digest);
        // TODO : CHANGE 0 TO TABLE NUMBER 
        // OR USE DORIAN'S FUNCTION
        reduce_digest(&digest, j, 0, &last_plain_text);
    }

    pwdcpy(&table->chains[index].endpoint, &last_plain_text);
    pwdcpy(&table->chains[index].startpoint, &startpoint);
}

__host__ void print_table(const RainbowTable *table) {
    for (unsigned long i = 0; i < table->length; i++) {
        printf("%.*s -> ... -> %.*s\n", PASSWORD_LENGTH,
               table->chains[i].startpoint.bytes, PASSWORD_LENGTH,
               table->chains[i].endpoint.bytes);
    }
}
