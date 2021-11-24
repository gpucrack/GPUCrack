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
    const RainbowChain* chain1 = (RainbowChain*)p1;
    const RainbowChain* chain2 = (RainbowChain*)p2;
    return strcmp(chain1->endpoint, chain2->endpoint);
}

char char_in_range(unsigned char n) {
    assert(n >= 0 && n <= 63);
    static const char* chars =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";

    return chars[n];
}

void reduce_digest(unsigned char* digest, unsigned long iteration,
                   unsigned char table_number, char* plain_text) {
    // pseudo-random counter based on the hash
    unsigned long counter = digest[7];
    for (char i = 6; i >= 0; i--) {
        counter <<= 8;
        counter |= digest[i];
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
                         char* endpoint) {
    memcpy(&table->chains[table->length].endpoint, endpoint, PASSWORD_LENGTH);
    memcpy(&table->chains[table->length].startpoint, startpoint,
           PASSWORD_LENGTH);
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
        char startpoint[PASSWORD_LENGTH + 1];
        create_startpoint(i, startpoint);
        memcpy(last_plain_text, startpoint, PASSWORD_LENGTH);

        /*
            Apply a round of hash + reduce `TABLE_T - 1` times.
            The chain should look like this:

            n -> r0(h(n)) -> r1(h(r0(h(n))) -> ...
        */
        for (unsigned long j = 0; j < TABLE_T - 1; j++) {
            unsigned char digest[HASH_LENGTH];
            ntlm(&last_plain_text, &digest);
            reduce_digest(digest, j, table_number, last_plain_text);
        }

        insert_chain(&table, startpoint, last_plain_text);

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

    RainbowChain* chains = malloc(sizeof(RainbowChain) * table_length);
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

void print_hash(const unsigned char* digest) {
    for (int i = 0; i < HASH_LENGTH; i++) {
        printf("%02x", digest[i]);
    }
}

void print_table(const RainbowTable* table) {
    for (unsigned long i = 0; i < table->length; i++) {
        printf("%s -> ... -> %s\n", table->chains[i].startpoint,
               table->chains[i].endpoint);
    }
}

void print_matrix(const RainbowTable* table) {
    for (unsigned long i = 0; i < table->length; i++) {
        unsigned char plain_text[PASSWORD_LENGTH];
        unsigned char digest[HASH_LENGTH];
        memcpy(plain_text, table->chains[i].startpoint);
        strcpy(plain_text, table->chains[i].startpoint);

        for (unsigned long j = 0; j < TABLE_T - 1; j++) {
            ntlm(plain_text, strlen(plain_text), digest);
            printf("%s -> ", plain_text);
            print_hash(digest);
            printf(" -> ");
            reduce_digest(digest, j, table->number, plain_text);
        }

        assert(!strcmp(table->chains[i].endpoint, plain_text));
        printf("%s\n", plain_text);
    }
}

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

void online(RainbowTable* rainbow_tables, unsigned char* digest,
            char* password) {
    /*
        Iterate column by column, starting from the last digest.
        https://stackoverflow.com/questions/3623263/reverse-iteration-with-an-unsigned-loop-variable

        We iterate through all tables at the same time because it's faster
       to find a match in the last columns.
    */
    for (unsigned long i = TABLE_T - 1; i-- > 0;) {
        for (int j = 0; j < TABLE_COUNT; j++) {
            unsigned char tn = rainbow_tables[j].number;

            char column_plain_text[PASSWORD_LENGTH + 1];
            unsigned char column_digest[HASH_LENGTH];
            memcpy(column_digest, digest, HASH_LENGTH);

            // get the reduction corresponding to the current column
            for (unsigned long k = i; k < TABLE_T - 2; k++) {
                reduce_digest(column_digest, k, tn, column_plain_text);
                HASH(column_plain_text, strlen(column_plain_text),
                     column_digest);
            }
            reduce_digest(column_digest, TABLE_T - 2, tn, column_plain_text);

            RainbowChain* found =
                binary_search(&rainbow_tables[j], column_plain_text);

            if (!found) {
                continue;
            }

            // we found a matching endpoint, reconstruct the chain
            char chain_plain_text[PASSWORD_LENGTH + 1];
            unsigned char chain_digest[HASH_LENGTH];
            strcpy(chain_plain_text, found->startpoint);

            for (unsigned long k = 0; k < i; k++) {
                HASH(chain_plain_text, strlen(chain_plain_text), chain_digest);
                reduce_digest(chain_digest, k, tn, chain_plain_text);
            }
            HASH(chain_plain_text, strlen(chain_plain_text), chain_digest);

            /*
                The digest was indeed present in the chain, this was
                not a false positive from a reduction. We found a
                plain text that matches the digest!
            */
            if (!pwdcmp(chain_digest, digest, HASH_LENGTH)) {
                strcpy(password, chain_plain_text);
                return;
            }
        }
    }

    // no match found
    password[0] = '\0';
}

__global__ void ntlm_chain(Password* startpoints, Digest* digests) {
    for (unsigned long i = 0; i < m0; i++) {
        // generate the chain
        char last_plain_text[PASSWORD_LENGTH + 1];
        char startpoint[PASSWORD_LENGTH + 1];
        create_startpoint(i, startpoint);
        strcpy(last_plain_text, startpoint);

        /*
            Apply a round of hash + reduce `TABLE_T - 1` times.
            The chain should look like this:

            n -> r0(h(n)) -> r1(h(r0(h(n))) -> ...
        */
        for (unsigned long j = 0; j < TABLE_T - 1; j++) {
            unsigned char digest[HASH_LENGTH];
            ntlm(last_plain_text, strlen(last_plain_text), digest);
            reduce_digest(digest, j, table_number, last_plain_text);
        }

        insert_chain(&table, startpoint, last_plain_text);

        if (i % 1000 == 0) {
            DEBUG_PRINT("\rprogress: %.2f%%", (float)(i + 1) / m0 * 100);
        }
    }
}