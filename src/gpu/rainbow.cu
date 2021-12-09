#include "constants.cuh"
#include "ntlm.cuh"
#include "rainbow.cuh"

inline int pwdcmp(Password* p1, Password* p2) {
    return memcmp(p1, p2, PASSWORD_LENGTH);
}

inline int pwdcpy(Password* p1, const Password* p2) {
    memcpy(p1, p2, PASSWORD_LENGTH);
}

int compare_rainbow_chains(const void* p1, const void* p2) {
    RainbowChain* chain1 = (RainbowChain*)p1;
    RainbowChain* chain2 = (RainbowChain*)p2;
    return pwdcmp(&chain1->endpoint, &chain2->endpoint);
}

char char_in_range(unsigned char n) {
    assert(n >= 0 && n <= 63);
    static const char* chars =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";

    return chars[n];
}

void reduce_digest(Digest* digest, unsigned long iteration,
                   unsigned char table_number, Password* plain_text) {
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

void create_startpoint(unsigned long counter, Password* plain_text) {
    for (int i = PASSWORD_LENGTH - 1; i >= 0; i--) {
        plain_text->bytes[i] = char_in_range(counter % 64);
        counter /= 64;
    }
}

inline void insert_chain(RainbowTable* table, Password* startpoint,
                         Password* endpoint) {
    table->length++;
}

RainbowChain* binary_search(RainbowTable* table, Password* endpoint) {
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

void dedup_endpoints(RainbowTable* table) {
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

RainbowTable gen_table(unsigned char table_number, unsigned long m0) {
    DEBUG_PRINT("\nGenerating table %hhu\n", table_number);

    RainbowChain* chains = (RainbowChain*)malloc(sizeof(RainbowChain) * m0);
    if (!chains) {
        perror("Cannot allocate enough memory for the rainbow table");
        exit(EXIT_FAILURE);
    }

    RainbowTable table = {chains, 0, table_number};

    // generate all rows
    for (unsigned long i = 0; i < m0; i++) {
        // generate the chain
        Password last_plain_text;
        Password startpoint;
        create_startpoint(i, &startpoint);
        pwdcpy(&last_plain_text, &startpoint);

        /*
            Apply a round of hash + reduce `TABLE_T - 1` times.
            The chain should look like this:

            n -> r0(h(n)) -> r1(h(r0(h(n))) -> ...
        */
        for (unsigned long j = 0; j < TABLE_T - 1; j++) {
            Digest digest;
            ntlm(&last_plain_text, &digest);
            reduce_digest(&digest, j, table_number, &last_plain_text);
        }

        insert_chain(&table, &startpoint, &last_plain_text);

        if (i % 1000 == 0) {
            DEBUG_PRINT("\rprogress: %.2f%%", (float)(i + 1) / m0 * 100);
        }
    }
    // the debug macro requires at least one variadic argument
    DEBUG_PRINT("%s", " DONE\n");

    // sort the rainbow table by the endpoints
    DEBUG_PRINT("%s", "Sorting table...");
    qsort(table.chains, table.length, sizeof(RainbowChain),
          compare_rainbow_chains);
    DEBUG_PRINT("%s", " DONE\n");

    // deduplicates chains with similar endpoints
    DEBUG_PRINT("%s", "Deleting duplicate endpoints...");
    dedup_endpoints(&table);
    DEBUG_PRINT("%s", " DONE\n");

    return table;
}

void store_table(RainbowTable* table, const char* file_path) {
    FILE* file = fopen(file_path, "w");

    if (!file) {
        perror("Could not open file on disk");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "%lu %hhu\n", table->length, table->number);

    for (unsigned long i = 0; i < table->length; i++) {
        fprintf(file, "%s %s\n", table->chains[i].startpoint,
                table->chains[i].endpoint);
    }

    fclose(file);
}

RainbowTable load_table(const char* file_path) {
    FILE* file = fopen(file_path, "r");

    if (!file) {
        perror("Could not open file on disk");
        exit(EXIT_FAILURE);
    }

    RainbowTable table;
    table.length = 0;

    unsigned long table_length;
    fscanf(file, "%lu %hhu\n", &table_length, &table.number);

    RainbowChain* chains =
        (RainbowChain*)malloc(sizeof(RainbowChain) * table_length);
    table.chains = chains;
    if (!chains) {
        perror("Cannot allocate enough memory to load this rainbow table");
        exit(EXIT_FAILURE);
    }

    char startpoint[PASSWORD_LENGTH + 1];
    char endpoint[PASSWORD_LENGTH + 1];
    while (fscanf(file, "%s %s\n", startpoint, endpoint) != EOF) {
        insert_chain(&table, startpoint, endpoint);
    }
    assert(table.length == table_length);

    fclose(file);
    return table;
}

void del_table(RainbowTable* table) { free(table->chains); }

void offline(RainbowTable* rainbow_tables) {
    // the number of possible passwords
    unsigned long long n = pow(64, PASSWORD_LENGTH);

    // the expected number of unique chains
    unsigned long mtmax = 2 * n / (TABLE_T + 2);

    // the number of starting chains, given the alpha coefficient
    unsigned long m0 = TABLE_ALPHA / (1 - TABLE_ALPHA) * mtmax;

    DEBUG_PRINT(
        "Generating %hhu table(s)\n"
        "Password length = %hhu\n"
        "Chain length (t) = %lu\n"
        "Maximality factor (alpha) = %.3f\n"
        "Optimal number of starting chains given alpha (m0) = %lu\n",
        TABLE_COUNT, PASSWORD_LENGTH, TABLE_T, TABLE_ALPHA, m0);

    for (unsigned char i = 0; i < TABLE_COUNT; i++) {
        rainbow_tables[i] = gen_table(i + 1, m0);
    }
}

__global__ void ntlm_chain_kernel(RainbowTable* table) {
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
        reduce_digest(&digest, j, 0, &last_plain_text);
    }

    pwdcpy(&table->chains[index].endpoint, &last_plain_text);
    pwdcpy(&table->chains[index].startpoint, &startpoint);
}