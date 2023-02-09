
#pragma once

#include <string>
#include <memory>
#include <vector>
#include <cstdint>

void LoadXGBoostJSONModel(const char* filename, GBTreeModel& gbt);

void LoadTreeAggJSONModel(const char* filename, GBTreeModel& gbt);
