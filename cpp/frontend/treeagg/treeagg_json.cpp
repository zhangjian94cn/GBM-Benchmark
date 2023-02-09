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

    gbt.setTreeDepth(treeInfo.tree_depth);
    {
        const rapidjson::Value& model = d["learner"].GetObject()["gradient_booster"].GetObject()["model"];
        const rapidjson::Value& trees = model.GetObject()["trees"].GetArray();
        
        for (int i = 0; i < treeInfo.tree_num; ++ i) {
            const rapidjson::Value& weight = trees[i].GetObject()["split_conditions"];
            const rapidjson::Value& index = trees[i].GetObject()["split_indices"];
            
            assert(weight.IsArray());
            assert(index.IsArray());
            std::vector<float> _weight;
            std::vector<int> _index;

            // printf("tree %d : node number is %d \n", i, weight.Size());
            for (rapidjson::SizeType i = 0; i < weight.Size(); i++) {
                // printf("weight[%d] = %f \n", i, weight[i].GetFloat());
                _weight.push_back(weight[i].GetFloat());
                // printf("indices[%d] = %d \n", i, index[i].GetInt());
                _index.push_back(index[i].GetInt());
            }
            __m512  r; 
            __m512i i;
            gbt.pushTreeAgg(r, i, _weight, _index);
        }
    }
}
