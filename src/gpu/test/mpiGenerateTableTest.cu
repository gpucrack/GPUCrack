#include "mpiGenerateTableTest.cuh"

// nvcc -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi mpiGenerateTableTest.cu -o main.out

int my_rank;
MPI_Datatype response_type;

struct response_t
{
    long offset;
    uint8_t endpoints[BATCH_SIZE * HASH_LENGTH + 1];
};

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Error: not enough arguments given.\n"
               "Usage: 'generateTable c n mt (path)', where:\n"
               "     - c is the passwords' length (in characters).\n"
               "     - n is the number of tables to generate.\n"
               "     - mt is the number of end points to be generated.\n"
               "     - (optional) path is the path to the start and end points files to create.\n");
        exit(1);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Define custom MPI datatype for response
    int bytes_len[2] = {1, (BATCH_SIZE * HASH_LENGTH + 1)};
    MPI_Aint offsets[2];
    struct response_t dummy_response;
    MPI_Aint base_address;
    MPI_Get_address(&dummy_response, &base_address);
    MPI_Get_address(&dummy_response.offset, &offsets[0]);
    MPI_Get_address(&dummy_response.endpoints[0], &offsets[1]);
    offsets[0] = MPI_Aint_diff(offsets[0], base_address);
    offsets[1] = MPI_Aint_diff(offsets[1], base_address);

    MPI_Datatype types[2] = {MPI_LONG, MPI_UNSIGNED_CHAR};
    MPI_Type_create_struct(2, bytes_len, offsets, types, &response_type);
    MPI_Type_commit(&response_type);

    // Launch master or slave
    if (my_rank == MASTER)
    {
        printSignature();
        debug("struct response_t base_address: 0x%lx", base_address);
        debug("struct response_t offsets: [ %ld | %ld ]", offsets[0], offsets[1]);
        master();
    }
    else
    {
        slave();
    }

    MPI_Type_free(&response_type);
    MPI_Finalize();

    return 0;

    // unsigned long long parameters[5];
    // computeParameters(parameters, argc, argv, true);

    // unsigned long long passwordNumber = parameters[0];
    // unsigned long long t = parameters[2];

    // Password * passwords;

    // unsigned long long nbOp = t * passwordNumber;

    // std::cout << "Number of crypto op " << nbOp << std::endl;

    // MPI_Finalize();
    // return 0;
}

void nextTask(long task[2], long m0, long batch_size)
{
    long batch_remainder = m0 % batch_size;

    task[0] += task[1];

    if (task[0] + batch_size > m0)
    {
        task[1] = batch_remainder;
    }
}

void master()
{
    int ntasks, i;
    struct response_t response;
    int remain = 10;

    long job = 0;

    long m0 = 32;
    long batch_size = BATCH_SIZE;

    long batch_remainder = m0 % batch_size;
    int batch = m0 / batch_size + (batch_remainder == 0 ? 0 : 1);

    long task[2] = {0, batch_size};

    // First task
    if (task[0] + batch_size > m0)
    {
        task[1] = batch_remainder;
    }

    debug("Start space size: %d", m0);
    debug("Batch number: %d", batch);

    MPI_Status status;

    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    debug("Slaves number: %d", ntasks - 1);

    debug("Seeding slaves...");
    for (i = 1; i < ntasks; ++i)
    {
        MPI_Send(&task, 2, MPI_LONG, i, WORK_TAG, MPI_COMM_WORLD);

        nextTask(task, m0, batch_size);
        batch--;
    }
    debug("Seeding done.");
    debug("Jobs remaining: %d", batch);

    while (batch > 0)
    {
        MPI_Recv(&response, 1, response_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        debug("Result: [ %d | %s ] from slave [%d]", response.offset, response.endpoints, status.MPI_SOURCE);

        MPI_Send(&task, 2, MPI_LONG, status.MPI_SOURCE, WORK_TAG, MPI_COMM_WORLD);

        nextTask(task, m0, batch_size);
        batch--;
    }

    debug("All jobs sent.");

    debug("Waiting for slaves to finish...");
    for (i = 1; i < ntasks; ++i)
    {
        MPI_Recv(&response, 1, response_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        debug("Result: [ %d | %s ] from slave [%d]", response.offset, response.endpoints, status.MPI_SOURCE);
    }

    debug("Killing slaves");
    for (i = 1; i < ntasks; ++i)
    {
        MPI_Send(0, 0, MPI_LONG, i, TERMINATE_TAG, MPI_COMM_WORLD);
    }

    return;
}

void slave()
{
    int job, tag;
    struct response_t response;
    long task[2];
    MPI_Status status;

    int batch_size = BATCH_SIZE;

    for (;;)
    {

        MPI_Recv(&task, 2, MPI_LONG, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TERMINATE_TAG)
        {
            return;
        }

        debug("Received task, offset: %d, len: %d", task[0], task[1]);

        // Heavy task

        unsigned long long parameters[5];
        
        // EMULATE PARAMETERS
        parameters[0] = 100000; // 100 000
        parameters[1] = 4000;
        parameters[2] = 30;
        parameters[3] = 1;
        parameters[4] = 100001;

        unsigned long long passwordNumber = parameters[0];
        unsigned long long t = parameters[2];

        Password * passwords;

        // EMULATE SOME ARGS
        char buffer [128];
        int ret = snprintf(buffer, sizeof(buffer), "%ld.bin", task[0]);

        char * argv[5] = {"", "3", "1", "", buffer}; // pwd_len = 3, table = 1
        int argc = 5;

        generateTables(parameters, passwords, argc, argv);
        
        // DON'T RETURNS ANYTHING

        response.offset = task[0];

        int tmp = 0;
        int j = 0;
        int response_len = task[1] * HASH_LENGTH;

        for (int i = 0; i < response_len; ++i)
        {
            j++;
            if (i % HASH_LENGTH == 0)
            {
                tmp += 1;
            }
            response.endpoints[i] = (uint8_t)('a' + ((task[0] + tmp) % 26));
        }
        response.endpoints[j] = '\0';

        MPI_Send(&response, 1, response_type, MASTER, 0, MPI_COMM_WORLD);
    }

    return;
}

void debug(char *str, ...)
{
    va_list printfargs;

    if (DEBUG_MASTER || DEBUG_WORKER)
    {
        if (my_rank != MASTER)
        {
            printf("=> \t");
        }
        printf("[%d] ", my_rank);
    }

    if ((DEBUG_MASTER && my_rank == MASTER) || (DEBUG_WORKER && my_rank != MASTER))
    {
        va_start(printfargs, str);
        vprintf(str, printfargs);
        va_end(printfargs);
        printf("\n");
    }
}
