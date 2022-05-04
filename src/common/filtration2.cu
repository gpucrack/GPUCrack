#include "filtration2.cuh"

long *
filter2(char *start_path, char *end_path, const char *start_out_path, const char *end_out_path, char *path,
        unsigned long long mtMax) {

    FILE *start_file;
    FILE *end_file;

    start_file = fopen(start_path, "rb");
    end_file = fopen(end_path, "rb");

    if (start_file == NULL) {
        perror("Can't open start file.");
        exit(1);
    }

    if (end_file == NULL) {
        perror("Can't open end file.");
        exit(1);
    }

    char buff[255];

    // Retrieve the number of points
    unsigned long long mt;
    fscanf(start_file, "%s", buff);
    sscanf(buff, "%llu", &mt);
    fgets(buff, 255, (FILE *) start_file);

    // Retrieve the password length
    int pwd_length;
    fgets(buff, 255, (FILE *) start_file);
    sscanf(buff, "%d", &pwd_length);

    // Retrieve the chain length (t)
    int t;
    fgets(buff, 255, (FILE *) start_file);
    sscanf(buff, "%d", &t);

    std::unordered_map<std::string , std::string> map;

    // just to skip the first 3 rows of end_file (same as start_file)
    fgets(buff, 255, (FILE *) end_file);
    fgets(buff, 255, (FILE *) end_file);
    fgets(buff, 255, (FILE *) end_file);

    long sp_success = 0;
    long ep_success = 0;

    unsigned long long limit = (unsigned long long)mt * (unsigned long long)pwd_length;

    // Store all points inside hash table
    for (unsigned long long i = 0; i < limit; i = i + sizeof(char) * pwd_length) {

        std::string end(pwd_length, '\0');
        std::string start(pwd_length, '\0');
        fread(&end[0], pwd_length, 1, (FILE *) end_file);
        fread(&start[0], pwd_length, 1, (FILE *) start_file);

        auto success = map.insert({end, start}).second;

        if(!success){
            end = "";
            start = "";
        }

        if((i%1000000) == 0) printf("%llu\n", i);
    }

    printf("Sorted!\n");

    fclose(start_file);
    fclose(end_file);

    unsigned long long new_len = map.size();
    printf("New size: %llu\n", new_len);

    FILE * sp_out_file = fopen(start_out_path, "wb");
    FILE * ep_out_file = fopen(end_out_path, "wb");

    if (sp_out_file == NULL) {
        perror("Can't open start file.");
        exit(1);
    }

    if (ep_out_file == NULL) {
        perror("Can't open end file.");
        exit(1);
    }

    // Write the header
    fprintf(sp_out_file, "%llu\n", new_len);
    fprintf(ep_out_file, "%llu\n", new_len);
    fprintf(sp_out_file, "%d\n", pwd_length);
    fprintf(ep_out_file, "%d\n", pwd_length);
    fprintf(sp_out_file, "%d\n", t);
    fprintf(ep_out_file, "%d\n", t);

    for (std::unordered_map<std::string, std::string>::iterator it = map.begin(); it != map.end(); ++it) {

        std::string endPoint = it->first;
        std::string startPoint = it->second;

        char start[pwd_length];
        char end[pwd_length];

        for(int i=0; i<pwd_length; i++){
            end[i] = endPoint[i];
            start[i]= startPoint[i];
        }

        fwrite(start, pwd_length, 1, (FILE *) sp_out_file);
        sp_success++;

        fwrite(end, pwd_length, 1, (FILE *) ep_out_file);
        ep_success++;
    }

    map.clear();

    fclose(sp_out_file);
    fclose(ep_out_file);

    long *success = (long *) malloc(sizeof(long) * 4);
    success[0] = mt;
    success[1] = new_len;
    success[2] = sp_success;
    success[3] = ep_success;

    return success;

}