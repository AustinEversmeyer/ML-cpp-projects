cmake_minimum_required(VERSION 3.16)
project(bayes_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# BayesClassifier header directory.
# Pass -DBAYES_INCLUDE_DIR=<path> when invoking cmake from source_lib/build.xml.
# Falls back to a relative path for standalone in-tree builds.
# ---------------------------------------------------------------------------
if(NOT DEFINED BAYES_INCLUDE_DIR)
  set(BAYES_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../BayesClassifier/include")
endif()

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
add_library(bayes_pipeline STATIC
  src/DataSink.cpp
  src/BayesClassifierManager.cpp
  src/IngestQueue.cpp
  src/BayesRuntimeManager.cpp
)

target_include_directories(bayes_pipeline PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${BAYES_INCLUDE_DIR}
)
