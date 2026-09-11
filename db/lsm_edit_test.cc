//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "rocksdb/lsm_edit.h"

#include <map>
#include <string>
#include <vector>

#include "db/db_test_util.h"
#include "port/stack_trace.h"
#include "rocksdb/metadata.h"
#include "rocksdb/options.h"
#include "rocksdb/sst_file_writer.h"
#include "test_util/testharness.h"

namespace ROCKSDB_NAMESPACE {

class LsmEditTest : public DBTestBase {
 public:
  LsmEditTest() : DBTestBase("lsm_edit_test", /*env_do_fsync=*/true) {
    sst_files_dir_ = dbname_ + "_sst_files/";
    EXPECT_OK(DestroyDir(env_, sst_files_dir_));
    EXPECT_OK(env_->CreateDir(sst_files_dir_));
  }

  ~LsmEditTest() override { EXPECT_OK(DestroyDir(env_, sst_files_dir_)); }

  // Writes `kvs` to a standalone SST file and returns an LsmEditFile that
  // installs it at `level`.
  LsmEditFile MakeFile(const std::string& name, int level,
                       const std::map<std::string, std::string>& kvs) {
    LsmEditFile file;
    file.path = sst_files_dir_ + name;
    file.level = level;
    SstFileWriter writer(EnvOptions(), CurrentOptions());
    EXPECT_OK(writer.Open(file.path));
    for (const auto& [key, value] : kvs) {
      EXPECT_OK(writer.Put(key, value));
    }
    EXPECT_OK(writer.Finish());
    return file;
  }

  static LsmEdit MakeEdit(ColumnFamilyHandle* cfh, const Slice& start,
                          const Slice& limit,
                          std::vector<LsmEditFile> add_files) {
    LsmEdit edit;
    edit.column_family = cfh;
    edit.range = RangeOpt(start, limit);
    edit.add_files = std::move(add_files);
    return edit;
  }

  // "<level>:<file count>" for each non-empty level of the default column
  // family, e.g. "0:2 1:1".
  static std::string ShapeOf(DB* db) {
    ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);
    std::string shape;
    for (const auto& level : cf_meta.levels) {
      if (level.files.empty()) {
        continue;
      }
      if (!shape.empty()) {
        shape += " ";
      }
      shape += std::to_string(level.level) + ":" +
               std::to_string(level.files.size());
    }
    return shape;
  }

  // File numbers currently in the default column family, per level. Used to
  // show that an edit left existing files alone rather than rewriting them.
  std::vector<std::vector<uint64_t>> FileNumbersPerLevel() {
    ColumnFamilyMetaData cf_meta;
    db_->GetColumnFamilyMetaData(&cf_meta);
    std::vector<std::vector<uint64_t>> result;
    for (const auto& level : cf_meta.levels) {
      result.emplace_back();
      for (const auto& file : level.files) {
        result.back().push_back(file.file_number);
      }
    }
    return result;
  }

 protected:
  std::string sst_files_dir_;
};

// The MyRocks index-rebuild shape: install files into a key range that holds
// no data but is not aligned to the boundaries of the file covering it.
TEST_F(LsmEditTest, GraftIntoEmptyRangeUnderStraddlingFile) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  // One bottom-level file spanning the whole key space, with a hole in the
  // middle where the edit will go.
  ASSERT_OK(Put("a1", "old_a"));
  ASSERT_OK(Put("z1", "old_z"));
  ASSERT_OK(Flush());
  MoveFilesToLevel(6);
  const std::vector<std::vector<uint64_t>> before = FileNumbersPerLevel();
  ASSERT_EQ(1, NumTableFilesAtLevel(6));

  // The straddling file sits at L6, so the graft declares L5. Two levels of
  // the source shape are preserved as two levels here.
  std::vector<LsmEditFile> files;
  files.push_back(MakeFile("graft_l5.sst", 5, {{"m1", "new_m1"}}));
  files.push_back(MakeFile("graft_l4.sst", 4, {{"m2", "new_m2"}}));
  ASSERT_OK(db_->ApplyLsmEdit(
      LsmEditOptions(),
      {MakeEdit(db_->DefaultColumnFamily(), "m", "n", std::move(files))}));

  EXPECT_EQ("old_a", Get("a1"));
  EXPECT_EQ("old_z", Get("z1"));
  EXPECT_EQ("new_m1", Get("m1"));
  EXPECT_EQ("new_m2", Get("m2"));
  EXPECT_EQ("0,0,0,0,1,1,1", FilesPerLevel());

  // The straddling file was retained untouched: no rewrite of live data was
  // needed to make the range file-aligned.
  const std::vector<std::vector<uint64_t>> after = FileNumbersPerLevel();
  EXPECT_EQ(before[6], after[6]);
}

TEST_F(LsmEditTest, RejectsRangeThatStillHoldsKeys) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  ASSERT_OK(Put("a1", "old_a"));
  ASSERT_OK(Put("m5", "old_m5"));
  ASSERT_OK(Put("z1", "old_z"));
  ASSERT_OK(Flush());
  MoveFilesToLevel(6);

  const Status s = db_->ApplyLsmEdit(
      LsmEditOptions(),
      {MakeEdit(db_->DefaultColumnFamily(), "m", "n",
                {MakeFile("graft.sst", 5, {{"m1", "new_m1"}})})});
  ASSERT_TRUE(s.IsInvalidArgument()) << s.ToString();
  EXPECT_NE(std::string::npos, s.ToString().find("range is not empty"));
  EXPECT_EQ("old_m5", Get("m5"));
  EXPECT_EQ("NOT_FOUND", Get("m1"));
}

TEST_F(LsmEditTest, RejectsRangeTombstoneReachingIntoRange) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  ASSERT_OK(Put("a1", "old_a"));
  ASSERT_OK(Put("z1", "old_z"));
  // Covers the edit range without leaving any point key in it. Grafted data
  // carries its own (lower) sequence numbers, so this tombstone would silently
  // delete it.
  ASSERT_OK(
      db_->DeleteRange(WriteOptions(), db_->DefaultColumnFamily(), "b", "y"));
  ASSERT_OK(Flush());
  MoveFilesToLevel(6);

  const Status s = db_->ApplyLsmEdit(
      LsmEditOptions(),
      {MakeEdit(db_->DefaultColumnFamily(), "m", "n",
                {MakeFile("graft.sst", 5, {{"m1", "new_m1"}})})});
  ASSERT_TRUE(s.IsInvalidArgument()) << s.ToString();
  EXPECT_NE(std::string::npos, s.ToString().find("range is not empty"));
}

TEST_F(LsmEditTest, RejectsStraddlingFileWhenProbeDisabled) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  ASSERT_OK(Put("a1", "old_a"));
  ASSERT_OK(Put("z1", "old_z"));
  ASSERT_OK(Flush());
  MoveFilesToLevel(6);

  LsmEditOptions edit_options;
  edit_options.probe_straddling_files = false;
  const Status s = db_->ApplyLsmEdit(
      edit_options, {MakeEdit(db_->DefaultColumnFamily(), "m", "n",
                              {MakeFile("graft.sst", 5, {{"m1", "v"}})})});
  ASSERT_TRUE(s.IsInvalidArgument()) << s.ToString();
  EXPECT_NE(std::string::npos, s.ToString().find("partially overlaps"));
}

TEST_F(LsmEditTest, RejectsDeclaredLevelOccupiedByRetainedFile) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  ASSERT_OK(Put("a1", "old_a"));
  ASSERT_OK(Put("z1", "old_z"));
  ASSERT_OK(Flush());
  MoveFilesToLevel(6);

  // The retained file's key range still covers L6 across the edit range, so
  // the grafted file cannot share that level even though the range is empty.
  const Status s = db_->ApplyLsmEdit(
      LsmEditOptions(), {MakeEdit(db_->DefaultColumnFamily(), "m", "n",
                                  {MakeFile("graft.sst", 6, {{"m1", "v"}})})});
  ASSERT_TRUE(s.IsInvalidArgument()) << s.ToString();
  EXPECT_NE(std::string::npos, s.ToString().find("declares level 6"));
}

// A file the edit removes must not block a declared level.
TEST_F(LsmEditTest, DeclaredLevelReusesSpaceOfRemovedFile) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  ASSERT_OK(Put("m1", "old_m1"));
  ASSERT_OK(Put("m9", "old_m9"));
  ASSERT_OK(Flush());
  MoveFilesToLevel(6);
  ASSERT_EQ(1, NumTableFilesAtLevel(6));

  ASSERT_OK(db_->ApplyLsmEdit(
      LsmEditOptions(),
      {MakeEdit(db_->DefaultColumnFamily(), "m", "n",
                {MakeFile("graft.sst", 6, {{"m5", "new_m5"}})})}));

  EXPECT_EQ("NOT_FOUND", Get("m1"));
  EXPECT_EQ("NOT_FOUND", Get("m9"));
  EXPECT_EQ("new_m5", Get("m5"));
  EXPECT_EQ("0,0,0,0,0,0,1", FilesPerLevel());
}

TEST_F(LsmEditTest, ReplaceWholeColumnFamily) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  ASSERT_OK(Put("a1", "old_a"));
  ASSERT_OK(Flush());
  ASSERT_OK(Put("b1", "old_b"));
  ASSERT_OK(Flush());
  ASSERT_EQ(2, NumTableFilesAtLevel(0));

  LsmEdit edit;
  edit.column_family = db_->DefaultColumnFamily();
  edit.add_files.push_back(MakeFile("all.sst", 6, {{"c1", "new_c"}}));
  ASSERT_OK(db_->ApplyLsmEdit(LsmEditOptions(), {edit}));

  EXPECT_EQ("NOT_FOUND", Get("a1"));
  EXPECT_EQ("NOT_FOUND", Get("b1"));
  EXPECT_EQ("new_c", Get("c1"));
  EXPECT_EQ("0,0,0,0,0,0,1", FilesPerLevel());
}

TEST_F(LsmEditTest, AtomicAcrossColumnFamilies) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  CreateAndReopenWithCF({"other"}, options);

  std::vector<LsmEdit> edits;
  edits.push_back(MakeEdit(handles_[0], "m", "n",
                           {MakeFile("cf0.sst", 6, {{"m1", "v0"}})}));
  edits.push_back(MakeEdit(handles_[1], "m", "n",
                           {MakeFile("cf1.sst", 6, {{"m1", "v1"}})}));
  ASSERT_OK(db_->ApplyLsmEdit(LsmEditOptions(), edits));

  EXPECT_EQ("v0", Get(0, "m1"));
  EXPECT_EQ("v1", Get(1, "m1"));
}

// One bad edit leaves every column family untouched.
TEST_F(LsmEditTest, RollsBackAllColumnFamiliesOnFailure) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  CreateAndReopenWithCF({"other"}, options);

  // A file straddling the edit range with a key inside it, so this column
  // family's edit cannot be satisfied.
  ASSERT_OK(Put(1, "a1", "old_a"));
  ASSERT_OK(Put(1, "m5", "old_m5"));
  ASSERT_OK(Put(1, "z1", "old_z"));
  ASSERT_OK(Flush(1));
  MoveFilesToLevel(6, 1);

  std::vector<LsmEdit> edits;
  edits.push_back(
      MakeEdit(handles_[0], "m", "n", {MakeFile("ok.sst", 6, {{"m1", "v0"}})}));
  edits.push_back(MakeEdit(handles_[1], "m", "n",
                           {MakeFile("bad.sst", 6, {{"m1", "v1"}})}));
  const Status s = db_->ApplyLsmEdit(LsmEditOptions(), edits);
  ASSERT_NOK(s);

  EXPECT_EQ("NOT_FOUND", Get(0, "m1"));
  EXPECT_EQ("NOT_FOUND", Get(1, "m1"));
  EXPECT_EQ("old_m5", Get(1, "m5"));
}

TEST_F(LsmEditTest, PreservesLevelZeroOrdering) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  // Two overlapping level 0 files: a shape file ingestion cannot express,
  // because it would have to assign them sequence numbers to disambiguate.
  std::vector<LsmEditFile> files;
  files.push_back(MakeFile("older.sst", 0, {{"m1", "older"}}));
  files.push_back(MakeFile("newer.sst", 0, {{"m1", "newer"}}));
  ASSERT_OK(db_->ApplyLsmEdit(
      LsmEditOptions(),
      {MakeEdit(db_->DefaultColumnFamily(), "m", "n", std::move(files))}));

  EXPECT_EQ(2, NumTableFilesAtLevel(0));
  EXPECT_EQ("newer", Get("m1"));
}

TEST_F(LsmEditTest, RejectsOverlappingFilesAtSameLevel) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  std::vector<LsmEditFile> files;
  files.push_back(MakeFile("one.sst", 6, {{"m1", "v"}, {"m5", "v"}}));
  files.push_back(MakeFile("two.sst", 6, {{"m3", "v"}}));
  const Status s = db_->ApplyLsmEdit(
      LsmEditOptions(),
      {MakeEdit(db_->DefaultColumnFamily(), "m", "n", std::move(files))});
  ASSERT_TRUE(s.IsInvalidArgument()) << s.ToString();
  EXPECT_NE(std::string::npos, s.ToString().find("overlap each other"));
}

TEST_F(LsmEditTest, RejectsFileOutsideRange) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  const Status s = db_->ApplyLsmEdit(
      LsmEditOptions(), {MakeEdit(db_->DefaultColumnFamily(), "m", "n",
                                  {MakeFile("out.sst", 6, {{"z1", "v"}})})});
  ASSERT_TRUE(s.IsInvalidArgument()) << s.ToString();
}

TEST_F(LsmEditTest, RejectsBadArguments) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  options.num_levels = 7;
  DestroyAndReopen(options);

  EXPECT_TRUE(db_->ApplyLsmEdit(LsmEditOptions(), {}).IsInvalidArgument());

  LsmEdit no_files;
  no_files.column_family = db_->DefaultColumnFamily();
  no_files.range = RangeOpt(Slice("m"), Slice("n"));
  EXPECT_TRUE(
      db_->ApplyLsmEdit(LsmEditOptions(), {no_files}).IsInvalidArgument());

  LsmEdit half_range;
  half_range.column_family = db_->DefaultColumnFamily();
  half_range.range.start = Slice("m");
  half_range.add_files.push_back(MakeFile("half.sst", 6, {{"m1", "v"}}));
  EXPECT_TRUE(
      db_->ApplyLsmEdit(LsmEditOptions(), {half_range}).IsNotSupported());

  const Status bad_level = db_->ApplyLsmEdit(
      LsmEditOptions(),
      {MakeEdit(db_->DefaultColumnFamily(), "m", "n",
                {MakeFile("deep.sst", options.num_levels, {{"m1", "v"}})})});
  ASSERT_TRUE(bad_level.IsInvalidArgument()) << bad_level.ToString();
  EXPECT_NE(std::string::npos, bad_level.ToString().find("outside [0, 7)"));
}

// Transplant a whole multi-level LSM, overlapping level 0 and all, from a
// separate DB. This is what lets a bulk loader skip flattening its scratch
// instance into a single sorted run before handing the files over.
TEST_F(LsmEditTest, GraftLsmBuiltInAnotherDb) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  const std::string src_dbname = dbname_ + "_source";
  ASSERT_OK(DestroyDB(src_dbname, options));
  std::vector<LsmEditFile> files;
  std::string src_shape;
  {
    Options src_options = options;
    src_options.create_if_missing = true;
    std::unique_ptr<DB> src;
    ASSERT_OK(DB::Open(src_options, src_dbname, &src));

    // Bottom level: the bulk of the data.
    for (int i = 0; i < 8; i++) {
      ASSERT_OK(src->Put(WriteOptions(), "m" + std::to_string(i), "base"));
    }
    ASSERT_OK(src->Flush(FlushOptions()));
    ASSERT_OK(src->CompactRange(CompactRangeOptions(), nullptr, nullptr));

    // Two overlapping level 0 runs on top of it, oldest written first.
    ASSERT_OK(src->Put(WriteOptions(), "m3", "middle"));
    ASSERT_OK(src->Flush(FlushOptions()));
    ASSERT_OK(src->Put(WriteOptions(), "m3", "newest"));
    ASSERT_OK(src->Put(WriteOptions(), "m5", "newest"));
    ASSERT_OK(src->Flush(FlushOptions()));

    ColumnFamilyMetaData src_meta;
    src->GetColumnFamilyMetaData(&src_meta);
    for (const auto& level : src_meta.levels) {
      std::vector<LsmEditFile> level_files;
      for (const auto& file : level.files) {
        LsmEditFile edit_file;
        edit_file.path = file.db_path + file.name;
        edit_file.level = level.level;
        level_files.push_back(std::move(edit_file));
      }
      // Level 0 metadata is newest first, but ApplyLsmEdit() treats later
      // entries as newer, so put the oldest run first.
      if (level.level == 0) {
        std::reverse(level_files.begin(), level_files.end());
      }
      for (auto& edit_file : level_files) {
        files.push_back(std::move(edit_file));
      }
    }
    ASSERT_EQ(3, files.size());
    src_shape = ShapeOf(src.get());
    ASSERT_OK(src->Close());
  }

  LsmEditOptions edit_options;
  edit_options.link_files = true;
  ASSERT_OK(db_->ApplyLsmEdit(
      edit_options,
      {MakeEdit(db_->DefaultColumnFamily(), "m", "n", std::move(files))}));

  EXPECT_EQ("base", Get("m0"));
  EXPECT_EQ("newest", Get("m3"));
  EXPECT_EQ("newest", Get("m5"));
  // The scratch instance's shape came across as-is: same files, same levels,
  // no compaction on either side.
  EXPECT_EQ(src_shape, ShapeOf(db_.get()));
}

TEST_F(LsmEditTest, SurvivesReopen) {
  Options options = CurrentOptions();
  options.disable_auto_compactions = true;
  DestroyAndReopen(options);

  ASSERT_OK(db_->ApplyLsmEdit(
      LsmEditOptions(), {MakeEdit(db_->DefaultColumnFamily(), "m", "n",
                                  {MakeFile("graft.sst", 4, {{"m1", "v"}})})}));
  Reopen(options);

  EXPECT_EQ("v", Get("m1"));
  EXPECT_EQ("0,0,0,0,1", FilesPerLevel());
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  RegisterCustomObjects(argc, argv);
  return RUN_ALL_TESTS();
}
