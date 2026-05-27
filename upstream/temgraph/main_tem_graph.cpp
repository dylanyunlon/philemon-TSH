

#include <iostream>
#include "tem_graph.cpp"

void usage()
{
    cerr << endl;
    cerr << "USAGE" << endl;
    cerr << "       ./temgraph [OPTION]... [DATA] [QUERIES]" << endl << endl;
    cerr << "DESCRIPTION" << endl;
    cerr << "       -? or -h" << endl;
    cerr << "              display this help message and exit" << endl;
    cerr << "       -q predicate" << endl;
    cerr << "              set predicate type: \"CONTAINS\" or \"CONTAINED\" "  << endl;
    cerr << "EXAMPLES" << endl;
    cerr << "       ./temgraph -q CONTAINS ../data/toy.dat ../data/query_toy.dat" << endl;
    cerr << "       ./temgraph -q CONTAINED ../data/toy.dat ../data/query_toy.dat" << endl;
}

int main(int argc, char** argv) {

    string strQuery = "", strPredicate = "", strOptimizations = "";
    string dataFile, queryFile;
    int indexType = 1, queryType = 0;
    char c;
    while ((c = getopt(argc, argv, "?hq:")) != -1)
    {
        switch (c)
        {
            case '?':
            case 'h':
                usage();
                return 0;

            case 'q':
                strPredicate = toUpperCase((char*)optarg);
                break;

            default:
                cerr << endl << "Error - unknown option '" << c << "'" << endl << endl;
                usage();
                return 1;
        }
    }

    dataFile = argv[optind];
    vector<string> queryFiles;
    for (int i = optind+1; i < argc; i++) {
        queryFiles.push_back(argv[i]);
    }


    TemGraph tt;
    double t_begin, t_stop, diff_in_second, diff_in_microsecond;

    
    if(strPredicate == "CONTAINS") {
            queryType = CONTAINS_QUERY;
            tt.load_intervals(queryType, dataFile);
    }
    else if(strPredicate == "CONTAINED") {
            queryType = OTHER_QUERY;
            tt.load_intervals(queryType, dataFile);
    }
    else {
        cout << "Invalid query type" << endl;
        return 0;
    }

    long long totalResult = 0, totalVisited = 0, result = 0;
    int cntqueryfile = queryFiles.size();
    for(int qcnt = 0; qcnt < cntqueryfile; qcnt++) {
        queryFile = queryFiles[qcnt];
        cout << queryFile << endl;

        // read query file
        vector<pair<Timestamp, Timestamp>> query;
        FILE* fquery = fopen(queryFile.c_str(), "r");
        Timestamp ql, qr;
        int test_cnt;
        while(fscanf(fquery, "%d %d\n", &ql, &qr) == 2) {
            query.push_back(make_pair(ql, qr));
        }
        test_cnt = query.size();
        if(strPredicate == "CONTAINS") {
            totalResult = 0; 
            t_begin = GetTime();
            
            for(int i = 0; i < test_cnt; i++) {
                ql = query[i].first;  qr = query[i].second;

                result = tt.contains_query(ql, qr);
                totalResult += result;
                totalVisited += visited_intervals_;

            }
            
            t_stop = GetTime();
            diff_in_microsecond = (t_stop - t_begin) * 1e6;


            std::printf("TEM-Graph average time for sub-valid query: %.4f microseconds  \navg #Result: %lu\n", diff_in_microsecond / test_cnt, totalResult / test_cnt);
            std::printf("avg #Result/Total: %.8f\n", (double)totalResult / test_cnt  / tt.total_intervals_);
            std::printf("avg #Result/Visited: %.8f\n", (double)totalResult / totalVisited);
            std::printf("Finish count: %d\n", test_cnt);
        }
        else if(strPredicate == "CONTAINED") {

            totalResult = 0; 
            t_begin = GetTime();
            for(int i = 0; i < test_cnt; i++) {
                ql = query[i].first;  qr = query[i].second;
                result = tt.contained_query(ql, qr);
                totalResult += result;
                totalVisited += visited_intervals_;

            }
            t_stop = GetTime();
            diff_in_microsecond = (t_stop - t_begin) * 1e6;


            std::printf("TEM-Graph average time for super-valid s query: %.4f microseconds  \navg #Result: %lu\n", diff_in_microsecond / test_cnt, totalResult / test_cnt);
            std::printf("avg #Result/Total: %.8f\n", (double)totalResult / test_cnt  / tt.total_intervals_);
            std::printf("avg #Result/Visited: %.8f\n", (double)totalResult / totalVisited);

        }
        else {
            cout << "Invalid query type" << endl;
        }
    
    cout << endl;
    }   

    return 0;
}