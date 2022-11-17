#include "rapidjson/filereadstream.h"
#include "rapidjson/document.h"

#include "frontend.h"
#include "xgboost_json.h"

void LoadXGBoostJSONModel(const char* filename, GBTreeModel& gbt) {

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
        const std::string& sklearn_info = d["learner"].GetObject()["attributes"].GetObject()["scikit_learn"].GetString();
        // parse info
        rapidjson::Document di;
        di.Parse<rapidjson::kParseNanAndInfFlag>(sklearn_info.c_str(), sklearn_info.length());
        treeInfo.tree_num = di["n_estimators"].GetInt();
        treeInfo.tree_depth = di["max_depth"].GetInt();
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
            for (rapidjson::SizeType i = 0; i < weight.Size(); i++) {
                printf("weight[%d] = %f \n", i, weight[i].GetFloat());
                _weight.push_back(weight[i].GetFloat());
                printf("indices[%d] = %d \n", i, index[i].GetInt());
                _index.push_back(index[i].GetInt());
            }
            gbt.pushTree(_weight, _index);
        }
    }
}
