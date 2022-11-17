
#pragma once
#include "group_tree.h"

#include <string>
#include <memory>
#include <vector>
#include <cstdint>

void LoadXGBoostJSONModel(const char* filename, GBTreeModel& gbt);
