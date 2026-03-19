#include <algorithm>
/* Copyright 2018 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/primitive_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(PrimitiveUtilTest, StringToPrimitiveType) {
  auto expect_ok_and_equal = [](const std::string& str,
                                PrimitiveType expected) {
    TF_ASSERT_OK_AND_ASSIGN(PrimitiveType actual,
                            primitive_util::StringToPrimitiveType(str));
    EXPECT_EQ(expected, actual);
  };
  expect_ok_and_equal("f32", F32);
  expect_ok_and_equal("tuple", TUPLE);
  expect_ok_and_equal("pred", PRED);
  expect_ok_and_equal("s32", S32);

  EXPECT_IS_NOT_OK(primitive_util::StringToPrimitiveType("F32").status());
  EXPECT_IS_NOT_OK(primitive_util::StringToPrimitiveType("Pred").status());
  EXPECT_IS_NOT_OK(primitive_util::StringToPrimitiveType("preD").status());
}

TEST(PrimitiveUtilTest, FloatTypes) {
  EXPECT_EQ(primitive_util::SignificandWidth(F32), 24);
  EXPECT_EQ(primitive_util::SignificandWidth(BF16), 8);
  EXPECT_EQ(primitive_util::ExponentWidth(F32), 8);
  EXPECT_EQ(primitive_util::ExponentWidth(BF16), 8);
  EXPECT_EQ(primitive_util::UnderflowExponent(F32), -125);
  EXPECT_EQ(primitive_util::UnderflowExponent(BF16), -125);
  EXPECT_EQ(primitive_util::OverflowExponent(F32), 128);
  EXPECT_EQ(primitive_util::OverflowExponent(BF16), 128);
}

TEST(PrimitiveUtilTest, CastPreservesValues) {
  const std::string kMatrix = R"(
                   | ABCDEFGHIJKLMNOPQRSTUVWXYZabcd
A PRED             | xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
B S1               |  xxxxxxx       xxxxxxxxxxxxxx
C S2               |   xxxxxx       xxxxxxxxxxxxxx
D S4               |    xxxxx       xxxxxxxxxxxxx
E S8               |     xxxx       xxxxxx
F S16              |      xxx        xxx x
G S32              |       xx         x  x
H S64              |        x
I U1               |   xxxxxxxxxxxxxxxxxxxxxxxxxxx
J U2               |    xxxxx xxxxxxxxxxxxxxxxxxxx
K U4               |     xxxx  xxxxxxxxxxx xxx xx
L U8               |      xxx   xxxxxxxxxx
M U16              |       xx    xxx xxx x
N U32              |        x     xx  x  x
O U64              |               x
P F16              |                xxxx x
Q F32              |                 xxx x
R F64              |                  x  x
S C64              |                   x x
T BF16             |                 xxxxx
U C128             |                     x
V F8E5M2           |                xxxxxxx
W F8E4M3           |                xxxxxx x
X F8E4M3FN         |                xxxxxx  x
Y F8E4M3B11FNUZ    |                xxxxxx   x
Z F8E5M2FNUZ       |                xxxxxx    x
a F8E4M3FNUZ       |                xxxxxx     x
b F8E3M4           |                xxxxxx      x
c F4E2M1FN         |                xxxxxxxxx   xx
d F8E8M0FNU        |                 xxxxx        x
  )";

  bool expecteds[PrimitiveType_ARRAYSIZE][PrimitiveType_ARRAYSIZE] = {};
  std::istringstream iss(kMatrix);
  std::string line;
  std::getline(iss, line);  // skip empty line
  std::getline(iss, line);  // skip header

  struct RowInfo {
    PrimitiveType type;
    std::string marks;
  };
  std::vector<RowInfo> rows;
  std::vector<PrimitiveType> col_types;

  while (std::getline(iss, line)) {
    std::vector<absl::string_view> parts = absl::StrSplit(line, "| ");
    if (parts.size() < 2) {
      continue;
    }
    std::string marks(parts[1]);

    std::vector<absl::string_view> type_parts =
        absl::StrSplit(parts[0], ' ', absl::SkipEmpty());
    absl::string_view type_str = type_parts[1];

    TF_ASSERT_OK_AND_ASSIGN(
        PrimitiveType type,
        primitive_util::StringToPrimitiveType(absl::AsciiStrToLower(type_str)));
    col_types.push_back(type);
    rows.push_back({type, marks});
  }

  for (const auto& row : rows) {
    for (const auto& [mark, col_type] : llvm::zip(row.marks, col_types)) {
      expecteds[row.type][col_type] = (mark == 'x');
    }
  }

  for (int from_type_int = PrimitiveType_MIN;
       from_type_int < PrimitiveType_ARRAYSIZE; ++from_type_int) {
    auto from_type = static_cast<PrimitiveType>(from_type_int);
    if (!primitive_util::IsArrayType(from_type)) {
      continue;
    }
    for (int to_type_int = PrimitiveType_MIN;
         to_type_int < PrimitiveType_ARRAYSIZE; ++to_type_int) {
      auto to_type = static_cast<PrimitiveType>(to_type_int);
      if (!primitive_util::IsArrayType(to_type)) {
        continue;
      }
      bool expected = expecteds[from_type][to_type];
      bool actual = primitive_util::CastPreservesValues(from_type, to_type);
      EXPECT_EQ(expected, actual)
          << primitive_util::LowercasePrimitiveTypeName(from_type) << " -> "
          << primitive_util::LowercasePrimitiveTypeName(to_type);
    }
  }
}

}  // namespace
}  // namespace xla
