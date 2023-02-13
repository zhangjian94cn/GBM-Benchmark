#include <immintrin.h>

#include "rapidjson/filereadstream.h"
#include "rapidjson/document.h"

#include "treeagg_json.h"
#include "group_tree_acc.h"

#include "frontend.h"

void LoadTreeAggJSONModel(const char* filename, GBTreeModel& gbt) {

    char read_buffer[65536];

    #ifdef _WIN32
    FILE* fp = std::fopen(filename, "rb");
    #else
    FILE* fp = std::fopen(filename, "r");
    #endif

    rapidjson::FileReadStream is(fp, read_buffer, sizeof(read_buffer));
    
    rapidjson::Document d;
    if (d.ParseStream(is).HasParseError()) {
        printf("rapidjson parses file error");        
    }

    ParsedModelInfo treeInfo;
    {
        treeInfo.tree_num = 10;
        treeInfo.tree_depth = 8;
        // treeInfo.tree_depth = 8;
    }



    for (int k = 0; k < 100; ++ k) {
        std::vector<std::vector<float>> _weightAgg;
        std::vector<std::vector<int>>   _indexAgg;
        // __m512  _reg;
        __m512i _idx; 

        gbt.setTreeDepth(treeInfo.tree_depth);
        {
            const rapidjson::Value& reg = d["reg"].GetArray()[0].GetArray();
            // const rapidjson::Value& idx = d["idx"].GetArray()[0].GetArray();
            const rapidjson::Value& trees = d["trees"].GetArray();
            
            // float _r[16] = {}; 
            int   _i[16] = {};

            for (int i = 0; i < reg.Size(); ++ i) {
                // _i[i] = reg[i].GetFloat();
                _i[i] = reg[i].GetInt();
            }
            // _reg = _mm512_load_epi32(_r); 
            _idx = _mm512_load_epi32(_i);
            for (int i = 0; i < 10; ++ i) {
                const rapidjson::Value& weight = trees[i].GetObject()["weight"];
                const rapidjson::Value& index  = trees[i].GetObject()["index"];
                
                assert(weight.IsArray());
                assert(index.IsArray());
                std::vector<float> _weight;
                std::vector<int>   _index;

                // printf("tree %d : node number is %d \n", i, weight.Size());
                for (rapidjson::SizeType i = 0; i < weight.Size(); i++) {
                    _weight.push_back(weight[i].GetFloat());
                    _index.push_back(index[i].GetInt());
                    // printf("weight[%d] = %f \n", i, weight[i].GetFloat());
                    // printf("indices[%d] = %d \n", i, index[i].GetInt());
                }
                _weightAgg.push_back(_weight);
                _indexAgg.push_back(_index);
            }
        }
        gbt.pushTreeAgg(_idx, _weightAgg, _indexAgg);
    }

}
